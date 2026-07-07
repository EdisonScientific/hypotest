# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Core of the sandbox abstraction.

Defines the `Sandbox` ABC (the uniform backend interface the env holds), its
config/spec types (`SandboxConfig`, `CapsuleRef`, `ResourceSpec`), the pluggable
`ResourceLimiter` strategy, and the shared container infrastructure (free-port
allocation, kernel-server health polling, process-group teardown) used by the
docker and enroot backends.

This module must NOT import `interpreter_env` at load time (only lazily, inside
`PrlimitLimiter`), so that `interpreter_env` can re-import this infra without an
import cycle. See docs/adr/0001-sandbox-backend-abstraction.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel

from hypotest.env import config as cfg
from hypotest.env.interpreter import ExecutionResult
from hypotest.env.kernel_server import NBLanguage

logger = logging.getLogger(__name__)

# (method, endpoint, **kwargs) -> httpx.Response — an httpx-shaped request function.
# docker passes httpx.AsyncClient.request; k8s passes AsyncSandboxConnector.send_request.
# Both the kernel client and the startup health poller are transport-agnostic over this.
RequestFn = Callable[..., Awaitable[httpx.Response]]

# Container lifecycle log level: the root logger defaults to WARNING and we
# cannot reconfigure it, so use WARNING for all container diagnostics to ensure
# they are visible in production logs.
_CONTAINER_LOG_LEVEL = logging.WARNING

# ---- shared container infrastructure (moved verbatim from interpreter_env) ----
_USED_PORTS: set[int] = set()
used_ports_lock = asyncio.Lock()

# container launch semaphore to limit concurrency
CONTAINER_LAUNCH_SEM = asyncio.Semaphore(128)
MAX_CONTAINER_LAUNCH_RETRIES = int(os.getenv("MAX_CONTAINER_LAUNCH_RETRIES", "5"))
_RETRY_BASE_SLEEP = 1.0
_RETRY_MAX_SLEEP = 16.0
MAX_RAY_RESULT_WAIT_RETRIES = int(os.getenv("MAX_RAY_RESULT_WAIT_RETRIES", "3"))
_RAY_RESULT_WAIT_TIMEOUT_GRACE = float(os.getenv("RAY_RESULT_WAIT_TIMEOUT_GRACE", "30"))
_LIST_DIR_RAY_TIMEOUT = float(os.getenv("LIST_DIR_RAY_TIMEOUT", "30"))


async def get_free_port() -> int:
    """Get a free port for the kernel server container."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        async with used_ports_lock:
            if port not in _USED_PORTS:
                _USED_PORTS.add(port)
                return port


class _PortCollisionError(Exception):
    """Port already in use by another server — retry with a new port."""


async def _poll_kernel_health(  # noqa: PLR0912
    request: RequestFn,
    enroot_proc: asyncio.subprocess.Process | None,
    container_port: int | None,
    expected_startup_token: str | None,
    read_log_tail: Callable[[int], str],
    label: str,
) -> None:
    """Poll the kernel server's /health endpoint until ready.

    Transport-agnostic: ``request`` is any httpx-shaped request fn (docker's httpx
    client, the enroot actor's client, or the k8s connector's ``send_request``), so
    every backend shares one startup poller. The optional ``enroot_proc``
    (subprocess-death) and ``expected_startup_token`` (host port-collision) checks
    are used only by the docker/enroot backends; k8s passes ``None`` for both.
    """
    start_time = time.perf_counter()
    poll_count = 0
    last_status = "no_attempt"
    consecutive_token_mismatches = 0
    # Use a short per-request timeout for health checks so that a single
    # poll can never block longer than a few seconds.  Without this, the
    # client's default timeout (execution_timeout + 10 ≈ 190s) means one
    # health poll can exceed the entire KERNEL_SERVER_STARTUP_TIMEOUT,
    # e.g. when a port collision causes us to connect to the wrong server.
    health_timeout = httpx.Timeout(5.0, connect=3.0)

    while time.perf_counter() - start_time < cfg.KERNEL_SERVER_STARTUP_TIMEOUT:
        poll_count += 1
        elapsed = time.perf_counter() - start_time

        # Check if the enroot process has died before we even get an HTTP response
        if enroot_proc is not None and enroot_proc.returncode is not None:
            rc = enroot_proc.returncode
            log_tail = read_log_tail(1000)
            raise RuntimeError(
                f"Enroot process exited prematurely with returncode={rc} after {elapsed:.1f}s. log: {log_tail!r}"
            )

        try:
            response = await request("GET", "/health", timeout=health_timeout)
            if response.status_code == 200:
                if expected_startup_token is None:
                    logger.log(
                        _CONTAINER_LOG_LEVEL,
                        "[%s] Kernel server healthy after %.1fs (%d polls)",
                        label,
                        elapsed,
                        poll_count,
                    )
                    return
                payload = response.json()
                if payload.get("startup_token") == expected_startup_token:
                    logger.log(
                        _CONTAINER_LOG_LEVEL,
                        "[%s] Kernel server healthy (token matched) after %.1fs (%d polls)",
                        label,
                        elapsed,
                        poll_count,
                    )
                    return
                last_status = f"token_mismatch(got={payload.get('startup_token', '?')[:8]})"
                consecutive_token_mismatches += 1
                if consecutive_token_mismatches >= 3:
                    raise _PortCollisionError(
                        f"Port {container_port} appears to be owned by another server "
                        f"({consecutive_token_mismatches} consecutive token mismatches)"
                    )
            else:
                last_status = f"http_{response.status_code}"
                consecutive_token_mismatches = 0
        except httpx.ConnectError:
            last_status = "connect_error"
            consecutive_token_mismatches = 0
        except httpx.ReadError:
            last_status = "read_error"
            consecutive_token_mismatches = 0
        except httpx.TimeoutException:
            last_status = "timeout"
            consecutive_token_mismatches = 0
        except httpx.RemoteProtocolError:
            last_status = "protocol_error"
            consecutive_token_mismatches = 0

        # Log progress every 5s
        if poll_count % 10 == 0:
            proc_alive = enroot_proc.returncode is None if enroot_proc else False
            logger.log(
                _CONTAINER_LOG_LEVEL,
                "[%s] Health poll #%d at %.1fs: last_status=%s, process_alive=%s",
                label,
                poll_count,
                elapsed,
                last_status,
                proc_alive,
            )

        await asyncio.sleep(0.5)

    total_elapsed = time.perf_counter() - start_time
    if last_status.startswith("token_mismatch"):
        raise _PortCollisionError(
            f"Port {container_port} health-check timed out with token_mismatch "
            f"({poll_count} polls, elapsed={total_elapsed:.1f}s)"
        )
    log_tail = read_log_tail(500)
    raise TimeoutError(
        f"Kernel server did not become healthy within {cfg.KERNEL_SERVER_STARTUP_TIMEOUT}s "
        f"({poll_count} polls, last_status={last_status}, elapsed={total_elapsed:.1f}s)"
        f"{f' log_tail={log_tail!r}' if log_tail else ''}"
    )


async def _kill_process_group(
    proc: asyncio.subprocess.Process, label: str = "enroot", sigterm_timeout: float = 15
) -> None:
    """Safely terminate a process group, escalating from SIGTERM to SIGKILL.

    Handles all edge cases: already-dead process, missing process group, etc.
    """
    if proc.returncode is not None:
        logger.log(
            _CONTAINER_LOG_LEVEL,
            "[%s] Process pid=%d already exited with returncode=%d",
            label,
            proc.pid,
            proc.returncode,
        )
        return

    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        logger.log(_CONTAINER_LOG_LEVEL, "[%s] Process pid=%d vanished before we could get pgid", label, proc.pid)
        return

    # SIGTERM the whole group
    try:
        logger.log(_CONTAINER_LOG_LEVEL, "[%s] Sending SIGTERM to pgid=%d (pid=%d)", label, pgid, proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        logger.log(_CONTAINER_LOG_LEVEL, "[%s] Process group pgid=%d already gone after SIGTERM", label, pgid)
        return

    try:
        await asyncio.wait_for(proc.communicate(), timeout=sigterm_timeout)
    except TimeoutError:
        logger.warning(
            "[%s] Process pid=%d did not exit within %.1fs of SIGTERM, sending SIGKILL",
            label,
            proc.pid,
            sigterm_timeout,
        )
    else:
        logger.log(
            _CONTAINER_LOG_LEVEL,
            "[%s] Process pid=%d exited after SIGTERM with returncode=%d",
            label,
            proc.pid,
            proc.returncode,
        )
        return

    # SIGKILL the whole group
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        logger.log(_CONTAINER_LOG_LEVEL, "[%s] Process group pgid=%d already gone before SIGKILL", label, pgid)
        return

    try:
        await asyncio.wait_for(proc.communicate(), timeout=5)
        logger.log(
            _CONTAINER_LOG_LEVEL,
            "[%s] Process pid=%d exited after SIGKILL with returncode=%d",
            label,
            proc.pid,
            proc.returncode,
        )
    except TimeoutError:
        logger.exception("[%s] Process pid=%d still alive after SIGKILL — possible zombie", label, proc.pid)


# ---- pluggable resource limiting (prlimit default for enroot; cgroups opt-in) ----
class ResourceSpec(BaseModel):
    """Backend-agnostic resource limits.

    Mapped per-backend by a `ResourceLimiter` (enroot -> prlimit/cgroups;
    k8s -> pod resources.limits; local/docker -> noop).
    """

    mem_mb: int | None = None
    mem_high_mb: int | None = None  # cgroups soft throttle; ignored by prlimit
    max_pids: int | None = None


class ResourceLimiter(Protocol):
    def command_prefix(self, spec: ResourceSpec) -> list[str]: ...


class NoopLimiter:
    """No resource limiting (local / docker)."""

    def command_prefix(self, spec: ResourceSpec) -> list[str]:  # noqa: ARG002
        return []


def _build_resource_limit_prefix(
    memory_limit_mb: int | None,
    max_pids: int | None,
) -> list[str]:
    """Build a prlimit command prefix for resource-limited execution.

    Uses prlimit to set RLIMIT_AS (virtual address space) which is inherited
    by all child processes through the env -> enroot -> bash -> python chain.
    When a sandbox exceeds the limit, allocations fail with MemoryError rather
    than consuming all node memory.

    Returns an empty list if no limits are configured or prlimit is not available.
    """
    if memory_limit_mb is None and max_pids is None:
        return []

    if shutil.which("prlimit") is None:
        logger.warning(
            "prlimit not found on PATH; skipping resource limits (memory_limit_mb=%s, max_pids=%s)",
            memory_limit_mb,
            max_pids,
        )
        return []

    prefix = ["prlimit"]
    if memory_limit_mb is not None:
        prefix.append(f"--as={memory_limit_mb * 1024 * 1024}")
    if max_pids is not None:
        prefix.append(f"--nproc={max_pids}")

    prefix.append("--")
    return prefix


class PrlimitLimiter:
    """Default enroot limiter: prlimit RLIMIT_AS (virtual address space) + RLIMIT_NPROC."""

    def command_prefix(self, spec: ResourceSpec) -> list[str]:
        return _build_resource_limit_prefix(spec.mem_mb, spec.max_pids)


class CgroupsV2Limiter:
    """Opt-in cgroups v2 limiter via `systemd-run --scope`.

    Off by default — cgroups need delegation that some clusters lack (see ADR §6).
    Returns [] when unset.
    """

    def command_prefix(self, spec: ResourceSpec) -> list[str]:
        prefix = ["systemd-run", "--scope", "--quiet"]
        if spec.mem_mb is not None:
            prefix.append(f"--property=MemoryMax={spec.mem_mb}M")
        if spec.mem_high_mb is not None:
            prefix.append(f"--property=MemoryHigh={spec.mem_high_mb}M")
        if spec.max_pids is not None:
            prefix.append(f"--property=TasksMax={spec.max_pids}")
        if len(prefix) == 3:  # nothing configured
            return []
        return prefix


# ---- capsule reference + sandbox config ----
class CapsuleRef(BaseModel):
    """Uniform capsule identity.

    Each backend DELIVERS it its own way in start() (k8s pulls in-pod via
    /load_capsule; enroot/docker via the pre-populated mount; local into work_dir).
    `source=None` means work_dir is already populated.
    """

    source: str | None = None  # local dir or s3://bucket/prefix
    uuid: str | None = None


@dataclass
class SandboxConfig:
    """Everything a Sandbox needs to be constructed."""

    work_dir: Path
    language: NBLanguage
    execution_timeout: float = 600
    safe_execute: bool = True
    use_host_env_vars: bool = False
    extra_envs: dict[str, str] = field(default_factory=dict)
    container_sqsh_path: Path | None = None
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    ref: CapsuleRef = field(default_factory=CapsuleRef)
    # Opaque caller identity (e.g. a run-group) stamped on k8s claims for attribution + the
    # clean-on-startup sweep; hashed to a label value. None => claims carry only managed-by.
    job_id: str | None = None
    # Transitional backend selectors — interpreted ONLY in factory.make_sandbox.
    use_docker: bool = False
    use_enroot: bool = False
    use_ray: bool = True
    # Per-environment kernel RNG seed. None preserves the historical,
    # entropy-seeded behavior.
    seed: int | None = None


class Sandbox(ABC):
    """One execution backend behind a uniform interface.

    `InterpreterEnvState` holds exactly one `Sandbox` and never branches on backend
    type. Each implementation owns its placement, provisioning, lifecycle, and the
    transport to its kernel. Backend specifics (ray refs, aiodocker handles,
    agent-sandbox connectors) must never leak past this interface.
    """

    work_dir: Path
    language: NBLanguage

    @abstractmethod
    async def start(self) -> None:
        """Provision + place the kernel and make its capsule data ready."""

    @abstractmethod
    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        """Execute code and return the notebook outputs."""

    @abstractmethod
    async def reset(self) -> None:
        """Restart the kernel, clearing in-memory state."""

    @abstractmethod
    async def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        """List the workspace directory the kernel sees."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the kernel and its backing resources."""

    @abstractmethod
    async def health(self) -> bool:
        """Return whether the kernel is up and ready."""
