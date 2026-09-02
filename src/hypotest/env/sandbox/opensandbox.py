# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OpenSandbox-backed execution using the raw asynchronous Python SDK.

OpenSandbox owns the remote container lifecycle. Hypotest supplies an explicit
entrypoint that starts its persistent kernel HTTP server, asks OpenSandbox for
that server's endpoint, and then uses the same ``HttpKernelClient`` protocol as
the Docker and agent-sandbox backends.

The optional SDK is imported lazily so local-only users do not need it.
``ConnectionConfig`` deliberately receives only explicit settings: when domain
or API key are omitted, the SDK reads its canonical ``OPEN_SANDBOX_DOMAIN`` and
``OPEN_SANDBOX_API_KEY`` environment variables.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from hypotest.env.interpreter import ExecutionResult
from hypotest.env.sandbox.base import ResourceSpec, Sandbox, SandboxConfig
from hypotest.env.sandbox.http_client import HttpKernelClient, ProtocolVersionError

logger = logging.getLogger(__name__)

_OPEN_SANDBOX_API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"
_IMAGE_PULL_POLICY_EXTENSION_KEY = "imagePullPolicy"
_IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY = "opensandbox.extensions.image-pull-policy"
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_ERROR_CODES = {
    "INTERNAL_UNKNOWN_ERROR",
    "POOL_ACQUIRE_FAILED",
    "POOL_EMPTY",
    "POOL_STATE_STORE_UNAVAILABLE",
    "READY_TIMEOUT",
    "UNHEALTHY",
}
_RETRYABLE_MESSAGE_MARKERS = (
    "connection refused",
    "connection reset",
    "imagepullbackoff",
    "pod failed",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
)
_ENVIRONMENT_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_ALLOCATION_CREDENTIAL_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_DEFAULT_REGION",
)


class OpenSandboxUnavailableError(RuntimeError):
    """The remote OpenSandbox placement could not be created or reached."""


class OpenSandboxImageAuth(BaseModel):
    """Credentials attached to the OpenSandbox image pull request.

    Exact ``${ENV_VAR}`` values are resolved while the server configuration is
    loaded. This provides the environment interpolation needed by Hypotest's
    plain YAML loader without placing registry passwords in configuration files.
    """

    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1)
    password: SecretStr

    @field_validator("username", "password", mode="before")
    @classmethod
    def resolve_environment_reference(cls, value: Any) -> Any:
        if not isinstance(value, str) or (match := _ENVIRONMENT_REFERENCE.fullmatch(value)) is None:
            return value
        variable = match.group(1)
        try:
            return os.environ[variable]
        except KeyError as exc:
            raise ValueError(f"environment variable {variable!r} is not set") from exc

    @model_validator(mode="after")
    def validate_non_blank(self) -> OpenSandboxImageAuth:
        if not self.username.strip():
            raise ValueError("registry username cannot be blank")
        if not self.password.get_secret_value().strip():
            raise ValueError("registry password cannot be blank")
        return self


class OpenSandboxSpec(BaseModel):
    """Lifecycle and capsule-delivery settings for one OpenSandbox server."""

    model_config = ConfigDict(extra="forbid")

    # Generic kernel image used by object-store/mounted-volume delivery and
    # no-data smoke tests.
    image: str
    # Private-registry credentials belong on the SDK's image specification,
    # not in container env or image layers. Public registries leave this unset.
    image_auth: OpenSandboxImageAuth | None = None

    # Leave these unset to use the OpenSandbox SDK's canonical OPEN_SANDBOX_*
    # environment variables and default protocol.
    domain: str | None = None
    api_key: str | None = None
    protocol: Literal["http", "https"] | None = None
    use_server_proxy: bool = True

    request_timeout_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    create_timeout_seconds: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    ready_timeout_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    health_poll_interval_seconds: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    # Poll immediately after submit, then use this value as the first delay.
    # Long-running cells back off with per-execution jitter to avoid synchronized
    # status-request bursts through the OpenSandbox proxy.
    execution_poll_interval_seconds: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    execution_poll_max_interval_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    execution_poll_backoff_multiplier: float = Field(default=2.0, ge=1, allow_inf_nan=False)
    execution_poll_jitter_ratio: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    # A status endpoint should answer immediately even while its cell is still
    # running. Bound one bad proxy request independently of the cell deadline.
    execution_poll_request_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    # None retries transient poll failures until the per-execution wire deadline.
    # Set an integer to impose an earlier consecutive-error cap.
    execution_poll_max_retries: int | None = Field(default=None, ge=0)

    # All OpenSandbox instances in one process/event loop share two transports:
    # a smaller lifecycle pool and an independently bounded hot-path kernel pool.
    lifecycle_max_connections: int = Field(default=64, ge=1)
    lifecycle_max_keepalive_connections: int = Field(default=32, ge=0)
    # Admit create operations before they enter httpx so excess allocations wait
    # outside the per-request pool timeout. OpenSandbox.create performs several
    # lifecycle calls, so one permit is held through the entire SDK operation.
    lifecycle_create_concurrency: int = Field(default=64, ge=1)
    kernel_max_connections: int = Field(default=256, ge=1)
    kernel_max_keepalive_connections: int = Field(default=128, ge=0)
    # Kernel calls are short submit/poll/control requests. Gate them before
    # httpx so many logical sandboxes cannot exhaust the proxy connection pool.
    kernel_request_concurrency: int = Field(default=128, ge=1)
    http_keepalive_expiry_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    create_attempts: int = Field(default=2, ge=1)
    create_retry_delay_seconds: float = Field(default=2.0, ge=0, allow_inf_nan=False)
    ttl_seconds: int | None = Field(default=5400, gt=0)
    kernel_port: int = Field(default=8000, ge=1, le=65535)
    # An inner RLIMIT_AS applied only to the Jupyter process. Keep this below
    # the outer ResourceSpec.mem_mb cgroup ceiling so Python receives ENOMEM
    # before Kubernetes OOM-kills the sandbox container.
    kernel_memory_limit_mb: int | None = Field(default=None, gt=0)
    # Remote sandboxes are already isolated at the machine/pod boundary, so
    # package-manager interception is off by default. Keep the shim available
    # for compatibility with colocated deployments that explicitly opt in.
    install_shim_enabled: bool = False
    # Operators can disable the colocated fallback for qualification runs (or
    # strict remote-only deployments) so capacity failures cannot be masked.
    local_fallback_enabled: bool = True

    capsule_mode: Literal["object_store", "mounted_volume", "large_bundle"] = "object_store"
    # Runtime values override ENV defaults baked into the image. ``capsule_key``
    # may be a literal relative object-store/mount path or contain
    # {capsule_uuid}; None leaves the corresponding image setting untouched.
    capsule_source: str | None = None
    capsule_key: str | None = "{capsule_uuid}"
    # Root of a capsule collection already mounted by the OpenSandbox cluster.
    # The selected capsule is copied from here into the sandbox-local
    # /workspace before the kernel and health endpoint start.
    mounted_capsule_root: str | None = None
    # In large-bundle mode, a selected image already contains this task's
    # capsule at /workspace, or contains a collection that the startup server
    # projects into /workspace. A per-capsule map wins over the format template,
    # which wins over one shared collection image.
    large_bundle_image: str | None = None
    large_bundle_images: dict[str, str] = Field(default_factory=dict)
    large_bundle_image_template: str | None = None

    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    extensions: dict[str, str] = Field(default_factory=dict)
    # Emit both extension spellings used by OpenSandbox deployments, matching
    # NeMo Gym's raw-provider behavior. Set None to leave extensions untouched.
    image_pull_policy: Literal["Always", "IfNotPresent", "Never"] | None = "IfNotPresent"
    secure_access: bool = False
    platform_os: Literal["linux", "windows"] | None = None
    platform_arch: Literal["amd64", "arm64"] | None = None
    # Advanced escape hatch. When omitted, Hypotest builds the kernel-server
    # command from the task language and safety settings.
    entrypoint: list[str] | None = None

    @model_validator(mode="after")
    def validate_platform(self) -> OpenSandboxSpec:
        if (self.platform_os is None) != (self.platform_arch is None):
            raise ValueError("platform_os and platform_arch must be set together")
        if self.entrypoint == []:
            raise ValueError("entrypoint must be non-empty when provided")
        if self.entrypoint is not None and self.kernel_memory_limit_mb is not None:
            raise ValueError("kernel_memory_limit_mb cannot be combined with a custom entrypoint")
        if self.execution_poll_max_interval_seconds < self.execution_poll_interval_seconds:
            raise ValueError("execution_poll_max_interval_seconds cannot be less than execution_poll_interval_seconds")
        if self.capsule_mode == "mounted_volume":
            if self.mounted_capsule_root is None:
                raise ValueError("mounted_volume capsule_mode requires mounted_capsule_root")
            if self.capsule_key is None:
                raise ValueError("mounted_volume capsule_mode requires a capsule_key template")
        if self.mounted_capsule_root is not None:
            if not self.mounted_capsule_root.strip():
                raise ValueError("mounted_capsule_root cannot be blank")
            if not self.mounted_capsule_root.startswith("/"):
                raise ValueError("mounted_capsule_root must be an absolute container path")
        return self

    @model_validator(mode="after")
    def validate_shared_client_limits(self) -> OpenSandboxSpec:
        if self.lifecycle_max_keepalive_connections > self.lifecycle_max_connections:
            raise ValueError("lifecycle_max_keepalive_connections cannot exceed lifecycle_max_connections")
        if self.lifecycle_create_concurrency > self.lifecycle_max_connections:
            raise ValueError("lifecycle_create_concurrency cannot exceed lifecycle_max_connections")
        if self.kernel_max_keepalive_connections > self.kernel_max_connections:
            raise ValueError("kernel_max_keepalive_connections cannot exceed kernel_max_connections")
        if self.kernel_request_concurrency > self.kernel_max_connections:
            raise ValueError("kernel_request_concurrency cannot exceed kernel_max_connections")
        return self

    def resolve_large_bundle_image(self, capsule_uuid: str) -> str:
        """Resolve the per-capsule or shared image used by large-bundle delivery."""
        if image := self.large_bundle_images.get(capsule_uuid):
            return image
        if self.large_bundle_image_template is not None:
            try:
                return self.large_bundle_image_template.format(capsule_uuid=capsule_uuid)
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError("large_bundle_image_template may only format the {capsule_uuid} value") from exc
        if self.large_bundle_image is not None:
            return self.large_bundle_image
        raise ValueError(
            f"No large-bundle image configured for capsule {capsule_uuid!r}; set "
            "large_bundle_images, large_bundle_image_template, or large_bundle_image"
        )

    def resolve_capsule_key(self, capsule_uuid: str) -> str | None:
        """Resolve the init-pull key, or preserve an image-baked key."""
        if self.capsule_key is None:
            return None
        unknown_braces = self.capsule_key.replace("{capsule_uuid}", "")
        if "{" in unknown_braces or "}" in unknown_braces:
            raise ValueError("capsule_key may only contain the {capsule_uuid} placeholder")
        return self.capsule_key.replace("{capsule_uuid}", capsule_uuid)

    def resolve_extensions(self) -> dict[str, str]:
        """Add a consistent image pull policy without overriding explicit extensions."""
        extensions = dict(self.extensions)
        if self.image_pull_policy is None:
            return extensions
        policy = (
            extensions.get(_IMAGE_PULL_POLICY_EXTENSION_KEY)
            or extensions.get(_IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY)
            or self.image_pull_policy
        )
        if policy not in {"Always", "IfNotPresent", "Never"}:
            raise ValueError(f"Invalid OpenSandbox image pull policy in extensions: {policy!r}")
        extensions.setdefault(_IMAGE_PULL_POLICY_EXTENSION_KEY, policy)
        extensions.setdefault(_IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY, policy)
        return extensions


def _require_opensandbox_sdk() -> tuple[Any, Any, Any, Any, Any]:
    """Import the optional OpenSandbox SDK at first use."""
    try:
        from opensandbox import Sandbox as OpenSandbox  # noqa: PLC0415
        from opensandbox.config import ConnectionConfig  # noqa: PLC0415
        from opensandbox.models.sandboxes import (  # noqa: PLC0415
            PlatformSpec,
            SandboxImageAuth,
            SandboxImageSpec,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The OpenSandbox backend requires the optional SDK; install hypotest[opensandbox]"
        ) from exc
    return OpenSandbox, ConnectionConfig, PlatformSpec, SandboxImageAuth, SandboxImageSpec


def _to_image_spec(image: str, auth: OpenSandboxImageAuth | None, auth_cls: Any, image_cls: Any) -> Any:
    """Attach registry credentials using the OpenSandbox SDK's image model."""
    if auth is None:
        return image
    sdk_auth = auth_cls(username=auth.username, password=auth.password.get_secret_value())
    return image_cls(image=image, auth=sdk_auth)


def _resource_quantity(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _resource_map(resources: ResourceSpec) -> dict[str, str]:
    """Map backend-neutral limits to OpenSandbox/Kubernetes quantities."""
    values: dict[str, str] = {}
    if resources.cpu is not None:
        values["cpu"] = _resource_quantity(resources.cpu)
    if resources.mem_mb is not None:
        values["memory"] = f"{resources.mem_mb}Mi"
    if resources.disk_gib is not None:
        values["ephemeral-storage"] = f"{resources.disk_gib}Gi"
    if resources.gpu is not None:
        values["gpu"] = str(resources.gpu)
    if resources.gpu_type is not None:
        values["gpu_type"] = resources.gpu_type
    return values


def _resource_request_map(resources: ResourceSpec) -> dict[str, str]:
    """Map the reserved CPU/memory floor to Kubernetes resource requests."""
    values: dict[str, str] = {}
    if resources.cpu_request is not None:
        values["cpu"] = _resource_quantity(resources.cpu_request)
    if resources.mem_request_mb is not None:
        values["memory"] = f"{resources.mem_request_mb}Mi"
    return values


def _endpoint_url(endpoint: str, protocol: str) -> str:
    """Normalize an SDK endpoint while preserving any server-proxy path."""
    endpoint = endpoint.strip()
    if endpoint.startswith(("http://", "https://")):
        return endpoint.rstrip("/") + "/"
    return f"{protocol}://{endpoint.rstrip('/')}/"


def _is_transient_remote_error(exc: Exception, *, endpoint_readiness: bool = False) -> bool:
    """Classify remote failures without hiding authentication/configuration errors."""
    if isinstance(exc, (ConnectionError, OSError, TimeoutError, httpx.TransportError)):
        return True
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code == 404 and endpoint_readiness:
        return True
    if status_code is not None:
        return status_code in _RETRYABLE_STATUS_CODES
    error = getattr(exc, "error", None)
    if getattr(error, "code", None) in _RETRYABLE_ERROR_CODES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MESSAGE_MARKERS)


@dataclass(frozen=True, slots=True)
class _OpenSandboxClientPoolKey:
    """Non-secret identity for one event-loop-local shared client pool."""

    connection_config_type: type
    base_url: str
    api_key_digest: bytes
    use_server_proxy: bool
    lifecycle_max_connections: int
    lifecycle_max_keepalive_connections: int
    lifecycle_create_concurrency: int
    kernel_max_connections: int
    kernel_max_keepalive_connections: int
    kernel_request_concurrency: int
    keepalive_expiry_seconds: float


@dataclass(slots=True)
class _OpenSandboxClientPool:
    """Shared connection pools; individual sandboxes retain only logical handles."""

    loop: asyncio.AbstractEventLoop
    key: _OpenSandboxClientPoolKey
    lifecycle_transport: httpx.AsyncBaseTransport
    lifecycle_create_semaphore: asyncio.Semaphore
    kernel_transport: httpx.AsyncBaseTransport
    kernel_client: httpx.AsyncClient
    kernel_request_semaphore: asyncio.Semaphore
    references: int = 0

    async def aclose(self) -> None:
        # AsyncClient owns/closes its configured kernel transport. Explicitly
        # close it too for custom transports whose client wrapper was replaced.
        with contextlib.suppress(Exception):
            await self.kernel_client.aclose()
        with contextlib.suppress(Exception):
            await self.kernel_transport.aclose()
        with contextlib.suppress(Exception):
            await self.lifecycle_transport.aclose()


_OPEN_SANDBOX_CLIENT_POOLS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[_OpenSandboxClientPoolKey, _OpenSandboxClientPool],
] = weakref.WeakKeyDictionary()


def _client_pool_key(connection_config: Any, spec: OpenSandboxSpec) -> _OpenSandboxClientPoolKey:
    get_base_url = getattr(connection_config, "get_base_url", None)
    if callable(get_base_url):
        base_url = str(get_base_url())
    else:
        protocol = getattr(connection_config, "protocol", None) or spec.protocol or "http"
        domain = (
            getattr(connection_config, "domain", None)
            or spec.domain
            or os.getenv(
                "OPEN_SANDBOX_DOMAIN",
                "localhost:8080",
            )
        )
        base_url = f"{protocol}://{domain}/v1"

    get_api_key = getattr(connection_config, "get_api_key", None)
    api_key = get_api_key() if callable(get_api_key) else (spec.api_key or os.getenv("OPEN_SANDBOX_API_KEY", ""))
    api_key_digest = hashlib.sha256(str(api_key).encode("utf-8")).digest()

    return _OpenSandboxClientPoolKey(
        connection_config_type=type(connection_config),
        base_url=base_url,
        api_key_digest=api_key_digest,
        use_server_proxy=spec.use_server_proxy,
        lifecycle_max_connections=spec.lifecycle_max_connections,
        lifecycle_max_keepalive_connections=spec.lifecycle_max_keepalive_connections,
        lifecycle_create_concurrency=spec.lifecycle_create_concurrency,
        kernel_max_connections=spec.kernel_max_connections,
        kernel_max_keepalive_connections=spec.kernel_max_keepalive_connections,
        kernel_request_concurrency=spec.kernel_request_concurrency,
        keepalive_expiry_seconds=spec.http_keepalive_expiry_seconds,
    )


async def _acquire_client_pool(
    connection_config: Any,
    spec: OpenSandboxSpec,
) -> _OpenSandboxClientPool:
    loop = asyncio.get_running_loop()
    key = _client_pool_key(connection_config, spec)
    loop_pools = _OPEN_SANDBOX_CLIENT_POOLS.setdefault(loop, {})
    if pool := loop_pools.get(key):
        pool.references += 1
        return pool

    lifecycle_transport = httpx.AsyncHTTPTransport(
        limits=httpx.Limits(
            max_connections=spec.lifecycle_max_connections,
            max_keepalive_connections=spec.lifecycle_max_keepalive_connections,
            keepalive_expiry=spec.http_keepalive_expiry_seconds,
        )
    )
    kernel_transport = httpx.AsyncHTTPTransport(
        limits=httpx.Limits(
            max_connections=spec.kernel_max_connections,
            max_keepalive_connections=spec.kernel_max_keepalive_connections,
            keepalive_expiry=spec.http_keepalive_expiry_seconds,
        )
    )
    try:
        kernel_client = httpx.AsyncClient(
            transport=kernel_transport,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    except BaseException:
        with contextlib.suppress(Exception):
            await kernel_transport.aclose()
        with contextlib.suppress(Exception):
            await lifecycle_transport.aclose()
        raise

    pool = _OpenSandboxClientPool(
        loop=loop,
        key=key,
        lifecycle_transport=lifecycle_transport,
        lifecycle_create_semaphore=asyncio.Semaphore(spec.lifecycle_create_concurrency),
        kernel_transport=kernel_transport,
        kernel_client=kernel_client,
        kernel_request_semaphore=asyncio.Semaphore(spec.kernel_request_concurrency),
        references=1,
    )
    loop_pools[key] = pool
    logger.debug(
        "Created shared OpenSandbox client pool (lifecycle=%d, create_admission=%d, kernel=%d, kernel_admission=%d)",
        spec.lifecycle_max_connections,
        spec.lifecycle_create_concurrency,
        spec.kernel_max_connections,
        spec.kernel_request_concurrency,
    )
    return pool


async def _request_with_kernel_admission(
    pool: _OpenSandboxClientPool,
    method: str,
    url: httpx.URL,
    *,
    record_wait: Callable[[float], None] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Queue a kernel-proxy request before its httpx timeout begins."""
    queue_started = time.perf_counter()
    async with pool.kernel_request_semaphore:
        if record_wait is not None:
            record_wait(time.perf_counter() - queue_started)
        return await pool.kernel_client.request(method, url, **kwargs)


async def _release_client_pool(pool: _OpenSandboxClientPool) -> None:
    if pool.references <= 0:
        return
    pool.references -= 1
    if pool.references:
        return

    loop_pools = _OPEN_SANDBOX_CLIENT_POOLS.get(pool.loop)
    if loop_pools is not None and loop_pools.get(pool.key) is pool:
        del loop_pools[pool.key]
        if not loop_pools:
            _OPEN_SANDBOX_CLIENT_POOLS.pop(pool.loop, None)
    await pool.aclose()
    logger.debug("Closed shared OpenSandbox client pool")


class OpenSandboxSandbox(Sandbox):
    """A fresh OpenSandbox container running Hypotest's persistent kernel server."""

    def __init__(self, config: SandboxConfig, spec: OpenSandboxSpec) -> None:
        if (
            spec.kernel_memory_limit_mb is not None
            and config.resources.mem_mb is not None
            and spec.kernel_memory_limit_mb >= config.resources.mem_mb
        ):
            raise ValueError("OpenSandbox kernel_memory_limit_mb must be less than the outer sandbox memory limit")
        self.work_dir = config.work_dir
        self.language = config.language
        self._config = config
        self._spec = spec
        self._ref = config.ref
        self._resources = config.resources
        self._sandbox: Any = None
        self._client: HttpKernelClient | None = None
        self._client_pool: _OpenSandboxClientPool | None = None
        self._startup_timings: dict[str, float] = {}
        self._kernel_request_wait_seconds = 0.0

    @property
    def startup_timings(self) -> dict[str, float]:
        """Return allocation/readiness phase timings for observability."""
        return dict(self._startup_timings)

    def _record_kernel_request_wait(self, seconds: float) -> None:
        self._kernel_request_wait_seconds += seconds

    async def start(self) -> None:
        if self._sandbox is not None:
            return

        startup_started = time.perf_counter()
        base_connection_config = self._make_connection_config()
        try:
            self._client_pool = await _acquire_client_pool(base_connection_config, self._spec)
            connection_config = self._make_connection_config(
                transport=self._client_pool.lifecycle_transport,
            )
            allocation_started = time.perf_counter()
            self._sandbox = await self._allocate(connection_config)
            self._startup_timings["allocation_seconds"] = time.perf_counter() - allocation_started
            connect_started = time.perf_counter()
            await self._connect_kernel(connection_config)
            self._startup_timings["kernel_connect_seconds"] = time.perf_counter() - connect_started
            assert self._client is not None
            if self._config.seed is not None:
                # Init-time capsule delivery completes before /health. Apply the
                # deterministic seed only after the ready kernel is reachable.
                reset_started = time.perf_counter()
                await self._client.reset(self._config.seed)
                self._startup_timings["seed_reset_seconds"] = time.perf_counter() - reset_started
            self._startup_timings["startup_seconds"] = time.perf_counter() - startup_started
        except BaseException:
            await self.close()
            raise

    def _make_connection_config(self, *, transport: httpx.AsyncBaseTransport | None = None) -> Any:
        _, ConnectionConfig, _, _, _ = _require_opensandbox_sdk()
        kwargs: dict[str, Any] = {
            "request_timeout": timedelta(seconds=self._spec.request_timeout_seconds),
            "use_server_proxy": self._spec.use_server_proxy,
        }
        if transport is not None:
            # OpenSandbox >=0.1.14 treats caller-supplied transports as
            # externally owned, so each Sandbox.close() leaves the shared pool
            # alive until the final Hypotest sandbox releases it.
            kwargs["transport"] = transport
        if self._spec.domain is not None:
            kwargs["domain"] = self._spec.domain
        if self._spec.api_key is not None:
            kwargs["api_key"] = self._spec.api_key
        if self._spec.protocol is not None:
            kwargs["protocol"] = self._spec.protocol
        return ConnectionConfig(**kwargs)

    def _allocation_env(self) -> dict[str, str]:
        """Build init-time capsule and caller environment for one allocation."""
        env = dict(self._spec.env)
        env.update(self._config.extra_envs)
        for variable in _ALLOCATION_CREDENTIAL_ENV_VARS:
            if (value := os.getenv(variable)) is not None:
                env.setdefault(variable, value)
        if self._ref.delivery == "object_store":
            source = self._ref.source or self._spec.capsule_source
            if source is not None:
                env["CAPSULE_SOURCE"] = source
            if self._ref.uuid and (capsule_key := self._spec.resolve_capsule_key(self._ref.uuid)) is not None:
                env["CAPSULE_KEY"] = capsule_key
        elif self._ref.delivery == "mounted_volume":
            # Clear any object-store defaults baked into the generic image so
            # mounted-volume delivery is unambiguous.
            env["CAPSULE_SOURCE"] = ""
            env["CAPSULE_KEY"] = ""
            root = self._ref.source or self._spec.mounted_capsule_root
            if root is not None:
                env["HYPOTEST_MOUNTED_CAPSULE_ROOT"] = root
            if self._ref.uuid and (capsule_key := self._spec.resolve_capsule_key(self._ref.uuid)) is not None:
                env["HYPOTEST_MOUNTED_CAPSULE_ID"] = capsule_key
        elif self._ref.delivery == "bundled" and self._ref.uuid:
            # Collection images use this to project just the requested capsule
            # into /workspace before the persistent kernel starts. Single-capsule
            # images ignore it because their workspace is already populated.
            env["HYPOTEST_BUNDLE_CAPSULE_ID"] = self._ref.uuid
        return env

    def _image_auth(self) -> OpenSandboxImageAuth | None:
        if self._spec.image_auth is not None:
            return self._spec.image_auth

        username = os.getenv("REGISTRY_USERNAME")
        password = os.getenv("REGISTRY_PASSWORD")
        if username is None and password is None:
            return None
        if not username or not password:
            raise ValueError("REGISTRY_USERNAME and REGISTRY_PASSWORD must both be set for image authentication")
        return OpenSandboxImageAuth(username=username, password=password)

    async def _allocate(self, connection_config: Any) -> Any:
        OpenSandbox, _, PlatformSpec, SandboxImageAuth, SandboxImageSpec = _require_opensandbox_sdk()
        image = self._selected_image()
        metadata = dict(self._spec.metadata)
        if self._config.job_id:
            metadata.setdefault("hypotest-job", self._config.job_id)

        kwargs: dict[str, Any] = {
            "image": _to_image_spec(image, self._image_auth(), SandboxImageAuth, SandboxImageSpec),
            "timeout": timedelta(seconds=self._spec.ttl_seconds) if self._spec.ttl_seconds is not None else None,
            "ready_timeout": timedelta(seconds=self._spec.ready_timeout_seconds),
            "env": self._allocation_env(),
            "metadata": metadata,
            "resource": _resource_map(self._resources),
            "resource_requests": _resource_request_map(self._resources),
            "extensions": self._spec.resolve_extensions(),
            "secure_access": self._spec.secure_access,
            "entrypoint": self._entrypoint(),
            "connection_config": connection_config,
            # OpenSandbox's default check probes execd. Hypotest instead waits
            # for its custom kernel endpoint and validates protocol_version.
            "skip_health_check": True,
        }
        if self._spec.platform_os is not None and self._spec.platform_arch is not None:
            kwargs["platform"] = PlatformSpec(os=self._spec.platform_os, arch=self._spec.platform_arch)

        last_error: Exception | None = None
        assert self._client_pool is not None
        create_semaphore = self._client_pool.lifecycle_create_semaphore
        for attempt in range(1, self._spec.create_attempts + 1):
            try:
                queue_started = time.perf_counter()
                async with create_semaphore:
                    queue_seconds = time.perf_counter() - queue_started
                    self._startup_timings["create_queue_seconds"] = (
                        self._startup_timings.get("create_queue_seconds", 0.0) + queue_seconds
                    )
                    sandbox = await asyncio.wait_for(
                        OpenSandbox.create(**kwargs),
                        timeout=self._spec.create_timeout_seconds,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _is_transient_remote_error(exc):
                    raise
                last_error = exc
                if attempt < self._spec.create_attempts:
                    delay = self._spec.create_retry_delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "OpenSandbox create attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt,
                        self._spec.create_attempts,
                        exc,
                        delay,
                    )
                    if delay:
                        await asyncio.sleep(delay)
            else:
                self._startup_timings["create_attempts"] = float(attempt)
                return sandbox

        assert last_error is not None
        raise OpenSandboxUnavailableError(
            f"OpenSandbox could not create image {image!r} after {self._spec.create_attempts} attempt(s): {last_error}"
        ) from last_error

    def _selected_image(self) -> str:
        if self._ref.delivery != "bundled":
            return self._spec.image
        if self._ref.image:
            return self._ref.image
        if not self._ref.uuid:
            raise ValueError("large-bundle delivery requires a capsule UUID")
        return self._spec.resolve_large_bundle_image(self._ref.uuid)

    def _entrypoint(self) -> list[str]:
        if self._spec.entrypoint is not None:
            return list(self._spec.entrypoint)
        command = [
            "sh",
            "/opt/entrypoint.sh",
            "/app/kernel_env/bin/python",
            "-m",
            "hypotest.kernel_capsule_server",
            "--port",
            str(self._spec.kernel_port),
            "--language",
            self.language.value,
        ]
        if self._spec.kernel_memory_limit_mb is not None:
            command.extend(("--kernel-memory-limit-mb", str(self._spec.kernel_memory_limit_mb)))
        if self._config.safe_execute:
            command.append("--safe-execute")
        if not self._spec.install_shim_enabled:
            command.append("--no-install-shim")
        return command

    async def _connect_kernel(self, connection_config: Any) -> None:
        """Resolve the custom port, then poll Hypotest's health endpoint."""
        assert self._sandbox is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._spec.ready_timeout_seconds
        endpoint: Any = None
        last_error: Exception | None = None

        while loop.time() < deadline and endpoint is None:
            try:
                endpoint = await self._sandbox.get_endpoint(self._spec.kernel_port)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _is_transient_remote_error(exc, endpoint_readiness=True):
                    raise
                last_error = exc
                await asyncio.sleep(min(self._spec.health_poll_interval_seconds, max(0.0, deadline - loop.time())))

        if endpoint is None:
            raise OpenSandboxUnavailableError(
                f"OpenSandbox kernel endpoint was unavailable for {self._spec.ready_timeout_seconds:g}s: {last_error}"
            ) from last_error

        protocol = getattr(connection_config, "protocol", None) or "http"
        headers = dict(getattr(endpoint, "headers", {}) or {})
        if self._spec.use_server_proxy:
            get_api_key = getattr(connection_config, "get_api_key", None)
            api_key = get_api_key() if callable(get_api_key) else self._spec.api_key
            if api_key:
                # Needed for the lifecycle server's /sandboxes/{id}/proxy route.
                # Never forward this credential to a direct sandbox endpoint.
                headers.setdefault(_OPEN_SANDBOX_API_KEY_HEADER, api_key)

        assert self._client_pool is not None
        client_pool = self._client_pool
        base_url = httpx.URL(_endpoint_url(str(endpoint.endpoint), protocol))

        async def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
            # Keep routing/auth headers sandbox-local while sharing only the
            # stateless AsyncClient and connection transport. A leading slash
            # would discard a server-proxy path, so always join a relative path.
            request_headers = httpx.Headers(headers)
            if extra_headers := kwargs.pop("headers", None):
                request_headers.update(extra_headers)
            return await _request_with_kernel_admission(
                client_pool,
                method,
                base_url.join(path.lstrip("/")),
                record_wait=self._record_kernel_request_wait,
                headers=request_headers,
                **kwargs,
            )

        self._client = HttpKernelClient(
            request,
            execution_timeout=self._config.execution_timeout,
            timeout_recovery=self._config.timeout_recovery,
            interrupt_grace_seconds=self._config.interrupt_grace_seconds,
            label=f"opensandbox:{getattr(self._sandbox, 'id', '?')}",
            execution_poll_interval_seconds=self._spec.execution_poll_interval_seconds,
            execution_poll_max_interval_seconds=self._spec.execution_poll_max_interval_seconds,
            execution_poll_backoff_multiplier=self._spec.execution_poll_backoff_multiplier,
            execution_poll_jitter_ratio=self._spec.execution_poll_jitter_ratio,
            execution_poll_max_retries=self._spec.execution_poll_max_retries,
            execution_poll_request_timeout_seconds=self._spec.execution_poll_request_timeout_seconds,
            infrastructure_wait_seconds=lambda: self._kernel_request_wait_seconds,
        )

        while loop.time() < deadline:
            try:
                if await self._client.health(raise_for_status=True):
                    return
            except ProtocolVersionError:
                raise
            except Exception as exc:
                if not _is_transient_remote_error(exc, endpoint_readiness=True):
                    raise
                last_error = exc
            await asyncio.sleep(min(self._spec.health_poll_interval_seconds, max(0.0, deadline - loop.time())))

        raise OpenSandboxUnavailableError(
            f"OpenSandbox kernel server did not become healthy within {self._spec.ready_timeout_seconds:g}s"
            + (f": {last_error}" if last_error is not None else "")
        ) from last_error

    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        if self._client is None:
            raise RuntimeError("OpenSandbox kernel client is not started")
        return await self._client.execute(code, timeout, req_uuid)

    async def reset(self) -> None:
        if self._client is None:
            raise RuntimeError("OpenSandbox kernel client is not started")
        await self._client.reset(self._config.seed)

    async def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        if self._client is None:
            raise RuntimeError("OpenSandbox kernel client is not started")
        return await self._client.list_dir(directory, max_files, show_hidden)

    async def health(self) -> bool:
        return self._client is not None and await self._client.health()

    async def close(self) -> None:
        client, self._client = self._client, None
        raw, self._sandbox = self._sandbox, None
        pool, self._client_pool = self._client_pool, None
        try:
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.aclose()
            if raw is not None:
                # A future artifact-retention policy belongs immediately before
                # destroy. For now the remote sandbox is intentionally ephemeral.
                try:
                    await raw.kill()
                except Exception:
                    logger.warning("Failed to kill OpenSandbox %s", getattr(raw, "id", "?"), exc_info=True)
                finally:
                    with contextlib.suppress(Exception):
                        await raw.close()
        finally:
            if pool is not None:
                await _release_client_pool(pool)
