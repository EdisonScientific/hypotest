# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""K8sSandbox — the agent-sandbox backend (primary fleet).

The kernel server (`kernel_capsule_server`) runs in an agent-sandbox pod, claimed
fresh-per-task from a warmpool (the pool pre-warms pods for fast cold-start; the
pod is dedicated to the claim and deleted on terminate — no reuse). We reach it
over HTTP via the SDK's `AsyncSandboxConnector.send_request`, which
`HttpKernelClient` rides as its `request_fn`, so the same kernel-server protocol
serves every backend.

The `k8s-agent-sandbox` SDK is OPTIONAL (the `k8s` extra) and imported lazily inside
`_allocate()`, so importing this module never requires it. The call SHAPE is from
the SDK source (`AsyncSandboxClient.create_sandbox(warmpool=..., namespace=...,
sandbox_ready_timeout=..., shutdown_after_seconds=...) -> AsyncSandbox` with
`.connector.send_request` and `.terminate()`); verify the exact import paths /
connection-config class names against the deployed SDK version. The image,
resources, and runtimeClass (gVisor) are defined in the WarmPool CRD, cluster-side.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from hypotest.env import config as cfg
from hypotest.env.interpreter import ExecutionResult
from hypotest.env.sandbox.base import Sandbox, SandboxConfig
from hypotest.env.sandbox.http_client import HttpKernelClient

logger = logging.getLogger(__name__)

# kernel_capsule_server's port inside the pod (proxied by the router via X-Sandbox-Port).
KERNEL_PORT = 8000

# Labels stamped on every claim we create: `managed-by` marks it as hypotest's; `job` carries an
# opaque caller identity (hashed to a valid label value) so the clean-on-startup sweep can reap a
# prior run's orphans WITHOUT touching a concurrent job's claims.
_MANAGED_BY_LABEL = "hypotest-managed-by"
_JOB_LABEL = "hypotest-job"


def _job_label_value(job_id: str) -> str:
    """Hash an opaque job id to a k8s-label-safe value (<=63 chars, alphanumeric)."""
    return hashlib.sha256(job_id.encode()).hexdigest()[:32]


def _claim_labels(job_id: str | None) -> dict[str, str]:
    """Labels for a new claim: always `managed-by`, plus a `job` label when a job id is set."""
    labels = {_MANAGED_BY_LABEL: "hypotest"}
    if job_id:
        labels[_JOB_LABEL] = _job_label_value(job_id)
    return labels


class NoCapacityError(Exception):
    """A warmpool/cluster had no capacity or didn't become ready — the scheduler tries the next placement."""


@dataclass
class K8sSandboxSpec:
    """Placement spec for one agent-sandbox target: which SandboxTemplate + how to reach the pod.

    These map directly onto the `k8s-agent-sandbox` SDK's
    `AsyncSandboxClient.create_sandbox(...)` call and its connection-config classes — see
    `K8sSandbox._allocate` for the wiring.
    """

    # The SandboxTemplate CRD name. REQUIRED — create_sandbox provisions the claim FROM this
    # template, which defines the pod (image, resources, runtimeClass/gVisor). This is the arg
    # that was previously missing.
    template: str
    # Optional warm-pool *adoption policy* ("default" / "none" / a custom pool name). When set, the
    # claim is fulfilled from a pre-warmed pod for fast cold-start instead of provisioning fresh.
    warmpool: str | None = None
    namespace: str = "default"  # namespace the claim lands in
    connection: str = "incluster"  # "incluster" | "gateway" | "direct" (async SDK has no local-tunnel)
    # Port the kernel server (kernel_capsule_server) listens on inside the pod. The SDK's connection
    # configs default to 8888; ours serves on KERNEL_PORT, so this must be threaded through.
    server_port: int = KERNEL_PORT
    use_pod_ip: bool = False  # incluster: reach the pod IP directly vs. the sandbox Service DNS
    gateway_name: str | None = None  # required for connection="gateway"
    gateway_namespace: str = "default"  # for connection="gateway"
    api_url: str | None = None  # required for connection="direct"
    ready_timeout: int = 180  # sandbox_ready_timeout — seconds to wait for the pod to report ready
    ttl_seconds: int | None = 5400  # shutdown_after_seconds — controller GC backstop (90m, > the 60m max session)
    labels: dict[str, str] | None = None  # optional k8s labels stamped on the claim
    # Multi-cluster: which cluster's control plane to create the claim in. The SDK loads these into
    # a PER-INSTANCE kube Configuration (never the process-global default), so specs targeting
    # different clusters never clobber each other's config. Both None => ambient / in-cluster
    # detection (single-cluster, today's behavior). For connection="direct", api_url must point at
    # the SAME cluster these select (control plane + data plane must agree).
    kubeconfig: str | None = None  # path to a kubeconfig file (None = default discovery / in-cluster)
    context: str | None = None  # named context within that kubeconfig (None = its current-context)


# Process-wide per-cluster client cache. One AsyncSandboxClient per (cluster, connection): its k8s
# ApiClient + httpx pool are reused across every env this process places on that cluster, instead of
# churning a fresh client (and kube-config load) per claim. Cached for the process lifetime and NEVER
# closed per-claim. This is a connection pool, not coordination state, so per-process is correct and
# replica-safe. The scheduler's placement reads and claim creation both borrow from here.
_CLIENT_CACHE: dict[tuple[Any, ...], Any] = {}
_CLIENT_LOCK = asyncio.Lock()


def _client_key(spec: K8sSandboxSpec) -> tuple[Any, ...]:
    return (
        spec.connection, spec.api_url, spec.server_port, spec.use_pod_ip,
        spec.gateway_name, spec.gateway_namespace, spec.kubeconfig, spec.context,
    )


def _build_connection_config(spec: K8sSandboxSpec) -> Any:
    """Map the spec's connection mode onto the SDK connection-config classes (lazy SDK import)."""
    from k8s_agent_sandbox.models import (  # noqa: PLC0415
        SandboxDirectConnectionConfig,
        SandboxGatewayConnectionConfig,
        SandboxInClusterConnectionConfig,
    )

    if spec.connection == "gateway":
        if not spec.gateway_name:
            raise ValueError("connection='gateway' requires gateway_name")
        return SandboxGatewayConnectionConfig(
            gateway_name=spec.gateway_name,
            gateway_namespace=spec.gateway_namespace,
            server_port=spec.server_port,
        )
    if spec.connection == "direct":
        if not spec.api_url:
            raise ValueError("connection='direct' requires api_url")
        return SandboxDirectConnectionConfig(api_url=spec.api_url, server_port=spec.server_port)
    return SandboxInClusterConnectionConfig(server_port=spec.server_port, use_pod_ip=spec.use_pod_ip)


async def _get_client(spec: K8sSandboxSpec) -> Any:
    """Return the process-cached AsyncSandboxClient for `spec`'s cluster, creating it once.

    kubeconfig/context select the cluster's control plane (loaded into a per-instance Configuration
    by the fork patch -- multi-cluster safe). Lazy-imports the optional SDK.
    """
    key = _client_key(spec)
    cached = _CLIENT_CACHE.get(key)
    if cached is not None:
        return cached
    async with _CLIENT_LOCK:
        cached = _CLIENT_CACHE.get(key)  # re-check under lock
        if cached is not None:
            return cached
        from k8s_agent_sandbox import AsyncSandboxClient  # noqa: PLC0415

        created = AsyncSandboxClient(
            connection_config=_build_connection_config(spec),
            kubeconfig=spec.kubeconfig,
            context=spec.context,
        )
        _CLIENT_CACHE[key] = created
        return created


async def warmpool_ready_replicas(spec: K8sSandboxSpec) -> int:
    """Free (claimable-now) warm pods on `spec`'s cluster -- the power-of-two-choices placement signal.

    Returns 0 when the spec has no warmpool (fresh-provision: zero instant-serve capacity) or on any
    read error, so a cluster we can't read / that has no warm headroom is deprioritized rather than
    blindly chosen. readyReplicas is reconciliation-lagged, so the scheduler relies on P2C's
    randomization (not a precise count) to stay balanced.
    """
    if not spec.warmpool:
        return 0
    try:
        sdk_client = await _get_client(spec)
        ready = await sdk_client.k8s_helper.get_warmpool_ready_replicas(spec.warmpool, spec.namespace)
    except Exception as e:
        # Any read failure (network/API/parse) -> treat as no headroom so P2C steers elsewhere.
        logger.warning("readyReplicas read failed for warmpool %r (ns=%s): %s", spec.warmpool, spec.namespace, e)
        return 0
    return ready or 0


async def aclose_clients() -> None:
    """Close every cached per-cluster client and clear the cache.

    The client cache is process-lived; call this at server/process shutdown to release the k8s API
    clients (aiohttp sessions) cleanly. Safe to call repeatedly.
    """
    async with _CLIENT_LOCK:
        for sdk_client in _CLIENT_CACHE.values():
            with contextlib.suppress(Exception):
                await sdk_client.close()
        _CLIENT_CACHE.clear()


async def sweep_stale_claims(specs: list[K8sSandboxSpec], job_id: str) -> int:
    """Delete this job's leftover SandboxClaims across all configured clusters (clean-on-startup).

    Job-scoped via the `hypotest-job` label, so it reaps only claims from a PRIOR incarnation of THIS
    job_id and never touches a concurrent job's claims. Intended to run once at orchestrator startup;
    the controller TTL backstops anything it misses. Returns the number deleted.

    The caller must scope `job_id` so it is stable across a restart of the same orchestrator instance
    but unique across instances meant to clean independently (else a restart could reap a live
    sibling's claims).
    """
    if not job_id:
        return 0
    selector = f"{_JOB_LABEL}={_job_label_value(job_id)}"
    seen: set[tuple[Any, ...]] = set()
    deleted = 0
    for spec in specs:
        dedup = (_client_key(spec), spec.namespace)
        if dedup in seen:  # one (cluster, namespace) swept once even if several specs share it
            continue
        seen.add(dedup)
        try:
            sdk_client = await _get_client(spec)
            names = await sdk_client.k8s_helper.list_sandbox_claims(spec.namespace, label_selector=selector)
        except Exception as e:  # one cluster being unreachable shouldn't block sweeping the others
            logger.warning("startup sweep: list failed for cluster api_url=%s: %s", spec.api_url, e)
            continue
        for name in names:
            with contextlib.suppress(Exception):
                await sdk_client.k8s_helper.delete_sandbox_claim(name, spec.namespace)
                deleted += 1
    return deleted


class K8sSandbox(Sandbox):
    """agent-sandbox execution backend. Fresh pod per task; capsule data via /load_capsule."""

    def __init__(self, config: SandboxConfig, spec: K8sSandboxSpec) -> None:
        self.work_dir = config.work_dir
        self.language = config.language
        self._ref = config.ref
        self._job_id = config.job_id
        self._execution_timeout = config.execution_timeout
        self._timeout_recovery = config.timeout_recovery
        self._interrupt_grace_seconds = config.interrupt_grace_seconds
        self._seed = config.seed
        self._spec = spec
        self._sandbox: Any = None  # agent-sandbox AsyncSandbox handle
        self._sdk_client: Any = None  # agent-sandbox AsyncSandboxClient (owns the k8s API client)
        self._client: HttpKernelClient | None = None

    async def start(self) -> None:
        self._sandbox = await self._allocate()
        try:
            self._client = HttpKernelClient(
                self._sandbox.connector.send_request,
                execution_timeout=self._execution_timeout,
                timeout_recovery=self._timeout_recovery,
                interrupt_grace_seconds=self._interrupt_grace_seconds,
                label=f"k8s:{getattr(self._sandbox, 'sandbox_id', '?')}",
            )
            await self._await_kernel_ready()
            if self._ref.uuid:
                await self._client.load_capsule(self._ref.uuid, self._seed)
            elif self._seed is not None:
                # Warm pods start before an env index is assigned. Configure and
                # restart the kernel now when there is no capsule-triggered reset.
                await self._client.reset(self._seed)
        except BaseException:
            # Failed after claiming the pod — terminate it so we don't leak (ADR §7).
            await self.close()
            raise

    async def _allocate(self) -> Any:
        """Claim a fresh sandbox pod from its SandboxTemplate, blocking until SandboxReady.

        Maps `K8sSandboxSpec` onto the SDK's
        `AsyncSandboxClient.create_sandbox(template, namespace=, sandbox_ready_timeout=, labels=,
        warmpool=, *, shutdown_after_seconds=)`:

        - ``template`` (positional, required): the SandboxTemplate to provision the claim from.
        - ``warmpool``: adoption policy — fulfil from a pre-warmed pod when set (fast cold-start).
        - ``namespace`` / ``labels``: where the claim lands and how it's tagged.
        - ``sandbox_ready_timeout``: seconds to wait for the pod to report ready.
        - ``shutdown_after_seconds`` (= ``ttl_seconds``): controller-side TTL so an orphaned claim
          is auto-deleted if this orchestrator dies before ``close()``.

        Borrows the process-cached AsyncSandboxClient for this cluster (see `_get_client`) — its
        ``server_port`` must match our kernel server (the SDK default is 8888). The client is shared
        and process-lived, so `close()` terminates only this claim, never the client.

        Tests override this to avoid the real SDK.
        """
        spec = self._spec
        self._sdk_client = await _get_client(spec)
        try:
            return await self._sdk_client.create_sandbox(
                spec.template,
                namespace=spec.namespace,
                sandbox_ready_timeout=spec.ready_timeout,
                labels={**(spec.labels or {}), **_claim_labels(self._job_id)},
                warmpool=spec.warmpool,
                shutdown_after_seconds=spec.ttl_seconds,
            )
        except TimeoutError as e:
            raise NoCapacityError(
                f"template {spec.template!r} (ns={spec.namespace!r}, warmpool={spec.warmpool!r}) "
                f"not ready within {spec.ready_timeout}s"
            ) from e

    async def _await_kernel_ready(self, interval: float = 0.5) -> None:
        """Poll the in-pod kernel /health until ready (SandboxReady != kernel up)."""
        assert self._client is not None
        deadline = time.monotonic() + cfg.KERNEL_SERVER_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if await self._client.health():
                return
            await asyncio.sleep(interval)
        raise NoCapacityError("kernel server did not become ready in the sandbox pod")

    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        assert self._client is not None
        return await self._client.execute(code, timeout, req_uuid)

    async def reset(self) -> None:
        assert self._client is not None
        await self._client.reset(self._seed)

    async def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        assert self._client is not None
        return await self._client.list_dir(directory, max_files, show_hidden)

    async def health(self) -> bool:
        return await self._client.health() if self._client is not None else False

    async def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
        if self._sandbox is not None:
            # terminate() closes the connector AND deletes the sandbox claim (fresh-per-task).
            with contextlib.suppress(Exception):
                await self._sandbox.terminate()
            self._sandbox = None
        # The SDK client is process-cached and shared across envs (see `_get_client`); drop our
        # reference but never close it here — terminate() above already released this claim.
        self._sdk_client = None
