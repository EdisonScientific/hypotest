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
import logging
import os
import re
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

    # Generic kernel image used by object-store delivery and no-data smoke tests.
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
    execution_poll_interval_seconds: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    create_attempts: int = Field(default=2, ge=1)
    create_retry_delay_seconds: float = Field(default=2.0, ge=0, allow_inf_nan=False)
    ttl_seconds: int | None = Field(default=5400, gt=0)
    kernel_port: int = Field(default=8000, ge=1, le=65535)

    capsule_mode: Literal["object_store", "large_bundle"] = "object_store"
    # Runtime values override ENV defaults baked into the image. ``capsule_key``
    # may be a literal relative S3 prefix or contain {capsule_uuid}; None leaves
    # the image's CAPSULE_KEY untouched.
    capsule_source: str | None = None
    capsule_key: str | None = "{capsule_uuid}"
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


class OpenSandboxSandbox(Sandbox):
    """A fresh OpenSandbox container running Hypotest's persistent kernel server."""

    def __init__(self, config: SandboxConfig, spec: OpenSandboxSpec) -> None:
        self.work_dir = config.work_dir
        self.language = config.language
        self._config = config
        self._spec = spec
        self._ref = config.ref
        self._resources = config.resources
        self._sandbox: Any = None
        self._client: HttpKernelClient | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._sandbox is not None:
            return

        connection_config = self._make_connection_config()
        self._sandbox = await self._allocate(connection_config)
        try:
            await self._connect_kernel(connection_config)
            assert self._client is not None
            if self._config.seed is not None:
                # Init-time capsule delivery completes before /health. Apply the
                # deterministic seed only after the ready kernel is reachable.
                await self._client.reset(self._config.seed)
        except BaseException:
            await self.close()
            raise

    def _make_connection_config(self) -> Any:
        _, ConnectionConfig, _, _, _ = _require_opensandbox_sdk()
        kwargs: dict[str, Any] = {
            "request_timeout": timedelta(seconds=self._spec.request_timeout_seconds),
            "use_server_proxy": self._spec.use_server_proxy,
        }
        if self._spec.domain is not None:
            kwargs["domain"] = self._spec.domain
        if self._spec.api_key is not None:
            kwargs["api_key"] = self._spec.api_key
        if self._spec.protocol is not None:
            kwargs["protocol"] = self._spec.protocol
        return ConnectionConfig(**kwargs)

    async def _allocate(self, connection_config: Any) -> Any:
        OpenSandbox, _, PlatformSpec, SandboxImageAuth, SandboxImageSpec = _require_opensandbox_sdk()
        image = self._selected_image()
        env = dict(self._spec.env)
        env.update(self._config.extra_envs)
        if self._ref.delivery == "object_store":
            source = self._ref.source or self._spec.capsule_source
            if source is not None:
                env["CAPSULE_SOURCE"] = source
            if self._ref.uuid and (capsule_key := self._spec.resolve_capsule_key(self._ref.uuid)) is not None:
                env["CAPSULE_KEY"] = capsule_key
        elif self._ref.delivery == "bundled" and self._ref.uuid:
            # Collection images use this to project just the requested capsule
            # into /workspace before the persistent kernel starts. Single-capsule
            # images ignore it because their workspace is already populated.
            env["HYPOTEST_BUNDLE_CAPSULE_ID"] = self._ref.uuid

        metadata = dict(self._spec.metadata)
        if self._config.job_id:
            metadata.setdefault("hypotest-job", self._config.job_id)

        kwargs: dict[str, Any] = {
            "image": _to_image_spec(image, self._spec.image_auth, SandboxImageAuth, SandboxImageSpec),
            "timeout": timedelta(seconds=self._spec.ttl_seconds) if self._spec.ttl_seconds is not None else None,
            "ready_timeout": timedelta(seconds=self._spec.ready_timeout_seconds),
            "env": env,
            "metadata": metadata,
            "resource": _resource_map(self._resources),
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
        for attempt in range(1, self._spec.create_attempts + 1):
            try:
                return await asyncio.wait_for(
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
        if self._config.safe_execute:
            command.append("--safe-execute")
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

        self._http_client = httpx.AsyncClient(
            base_url=_endpoint_url(str(endpoint.endpoint), protocol),
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        async def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
            assert self._http_client is not None
            # A leading slash would discard a server-proxy path in base_url.
            return await self._http_client.request(method, path.lstrip("/"), **kwargs)

        self._client = HttpKernelClient(
            request,
            execution_timeout=self._config.execution_timeout,
            timeout_recovery=self._config.timeout_recovery,
            interrupt_grace_seconds=self._config.interrupt_grace_seconds,
            label=f"opensandbox:{getattr(self._sandbox, 'id', '?')}",
            owns=self._http_client,
            execution_poll_interval_seconds=self._spec.execution_poll_interval_seconds,
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
        http_client, self._http_client = self._http_client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()
        elif http_client is not None:
            with contextlib.suppress(Exception):
                await http_client.aclose()
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
