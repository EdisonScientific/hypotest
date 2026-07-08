# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""EnrootSandbox — the colocated enroot backend.

Two placement modes:

- **ray** (default): an `EnrootKernelServer` `@ray.remote` actor is created with
  ``SPREAD`` scheduling (ray is the placement layer that keeps colocated sandboxes
  off the head node); the env relays execute/reset/list_dir/close to it via
  ``.remote()`` + `_await_ray_ref`. The actor owns the kernel-server HTTP client.
- **non-ray**: an enroot subprocess is launched locally and reached over HTTP via
  `HttpKernelClient`; `list_dir` reads the bind-mounted host `work_dir`.

`EnrootKernelServer` (the ray actor) and the enroot command builders live here;
shared container infra (port allocation, health polling, process-group teardown,
the prlimit prefix) comes from `sandbox.base`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Awaitable
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, cast

import httpx
import ray

from hypotest.env import config as cfg
from hypotest.env.install_shim import bash_export_block, write_workspace_config
from hypotest.env.interpreter import ExecutionResult
from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox.base import (
    _CONTAINER_LOG_LEVEL,
    _LIST_DIR_RAY_TIMEOUT,
    _RAY_RESULT_WAIT_TIMEOUT_GRACE,
    _RETRY_BASE_SLEEP,
    _RETRY_MAX_SLEEP,
    _USED_PORTS,
    CONTAINER_LAUNCH_SEM,
    MAX_CONTAINER_LAUNCH_RETRIES,
    MAX_RAY_RESULT_WAIT_RETRIES,
    Sandbox,
    SandboxConfig,
    _build_resource_limit_prefix,
    _kill_process_group,
    _poll_kernel_health,
    _PortCollisionError,
    get_free_port,
    used_ports_lock,
)
from hypotest.env.sandbox.http_client import HttpKernelClient, execute_wire_timeout_seconds
from hypotest.env.tools.filesystem import FilesystemTool, list_dir_tool

logger = logging.getLogger(__name__)


def _prep_workspace_dir(work_dir: str, workspace_path: str = "/data_workspace") -> None:
    # Shared with the k8s kernel server (kernel_capsule_server) so the install
    # model can't drift. Enroot has no in-pod cutoff proxy, so index_url is None.
    write_workspace_config(Path(work_dir), runtime_path=workspace_path, index_url=None)


@ray.remote(  # type: ignore[call-overload]  # max_concurrency is a valid actor option missing from ray's stubs
    scheduling_strategy="SPREAD",
    max_concurrency=1,
    runtime_env={
        "py_executable": sys.executable,
    },
)
class EnrootKernelServer:
    def __init__(
        self,
        container_sqsh_path: Path,
        execution_timeout: float,
        safe_execute: bool = True,
        sandbox_memory_limit_mb: int | None = None,
        sandbox_max_pids: int | None = None,
        seed: int | None = None,
        timeout_recovery: Literal["none", "interrupt"] = "none",
        interrupt_grace_seconds: float = 10.0,
    ):
        self.container_sqsh_path = container_sqsh_path
        self.execution_timeout = execution_timeout
        self.safe_execute = safe_execute
        self.sandbox_memory_limit_mb = sandbox_memory_limit_mb
        self.sandbox_max_pids = sandbox_max_pids
        self.seed = seed
        self.timeout_recovery = timeout_recovery
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self._enroot_proc: asyncio.subprocess.Process | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._kernel_client: HttpKernelClient | None = None
        self._container_port: int | None = None
        self._container_log_path: Path | None = None
        self._container_log_file: Any | None = None
        self._node_workdir: Path | None = None

    def _proc_label(self) -> str:
        """Short label for log messages identifying this container."""
        port = self._container_port or "?"
        pid = self._enroot_proc.pid if self._enroot_proc else "?"
        return f"enroot(port={port}, pid={pid})"

    def _require_node_workdir(self) -> Path:
        if self._node_workdir is None:
            raise RuntimeError("Node-local workspace is not initialized")
        return self._node_workdir

    def _normalize_node_workspace_path(self, directory: str) -> Path:
        workspace_root = self._require_node_workdir().resolve()
        requested = Path(directory)
        workspace_alias = Path("/data_workspace")

        if requested.is_absolute():
            if requested == workspace_alias or workspace_alias in requested.parents:
                candidate = workspace_root / requested.relative_to(workspace_alias)
            elif requested == workspace_root or workspace_root in requested.parents:
                candidate = requested
            else:
                raise ValueError("Path must stay within the workspace root")
        else:
            candidate = workspace_root / requested

        candidate = candidate.resolve()
        if candidate != workspace_root and workspace_root not in candidate.parents:
            raise ValueError("Path must stay within the workspace root")
        return candidate

    @staticmethod
    def _build_kernel_bash_script(
        node_workdir: str,
        language: NBLanguage,
        port: int,
        startup_token: str,
        safe_execute: bool = True,
        seed: int | None = None,
    ) -> str:
        """Build the bash script that sets up the workspace and launches the kernel server."""
        exports = bash_export_block("$WORKDIR")
        seed_arg = f"--seed {seed}" if seed is not None else ""
        script = dedent(f"""\
            set -euo pipefail

            WORKDIR="{node_workdir}"
            trap 'rm -rf "$WORKDIR"' EXIT

            mkdir -p $WORKDIR
            cp -a /data_workspace/. $WORKDIR/

            cd $WORKDIR

            # pydeps / pip.conf / .install_shim (pip/conda/apt + R shim) / Rprofile /
            # r_libs are written host-side by write_workspace_config() and copied in
            # above; cp -a preserves execute bits but ensure them anyway.
            if [ -d "$WORKDIR/.install_shim/bin" ]; then
                chmod 755 "$WORKDIR/.install_shim/bin"/* 2>/dev/null || true
            fi

            __WORKSPACE_EXPORTS__
            export target_platform=${{target_platform:-linux-64}}

            source activate /app/kernel_env
            exec /app/kernel_env/bin/python /envs/kernel_server.py \\
                --work_dir $WORKDIR \\
                --language {language.value} \\
                --port {port} \\
                --startup-token {startup_token} {"--safe-execute" if safe_execute else ""} {seed_arg}
        """).strip()
        return script.replace("__WORKSPACE_EXPORTS__", exports)

    @staticmethod
    def _setup_enroot_env(startup_token: str) -> dict[str, str]:
        """Create enroot runtime directories and return env dict."""
        base = Path(f"/tmp/enroot_data/{startup_token}")  # noqa: S108
        subdirs = ["runtime", "config", "cache", "data", "tmp"]
        env_keys = [
            "ENROOT_RUNTIME_PATH",
            "ENROOT_CONFIG_PATH",
            "ENROOT_CACHE_PATH",
            "ENROOT_DATA_PATH",
            "ENROOT_TEMP_PATH",
        ]
        env: dict[str, str] = {}
        for subdir, key in zip(subdirs, env_keys, strict=True):
            p = base / subdir
            p.mkdir(parents=True, exist_ok=True)
            os.chmod(p, 0o700)
            env[key] = str(p)
        return env

    @staticmethod
    def _build_enroot_cmd(
        work_dir: Path,
        node_workdir: Path,
        kernel_server_path: Path,
        bash: str,
        enroot_env: dict[str, str],
        container_sqsh_path: Path,
        resource_prefix: list[str] | None = None,
    ) -> list[str]:
        """Assemble the full ``enroot start`` command, optionally prefixed with prlimit."""
        env_args = [f"{k}={v}" for k, v in enroot_env.items()]
        cmd = [
            "env",
            "-i",
            "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
            'HOME="$HOME"',
            'USER="$USER"',
            *env_args,
            "enroot",
            "start",
            "--rw",
            "--mount",
            f"{work_dir}:/data_workspace",
            "--mount",
            f"{node_workdir.resolve()}:{node_workdir}",
            "--mount",
            f"{kernel_server_path.resolve()}:/envs/kernel_server.py",
            str(container_sqsh_path.resolve()),
            "/bin/bash",
            "-lc",
            bash,
        ]
        if resource_prefix:
            return [*resource_prefix, *cmd]
        return cmd

    async def initialize(self, work_dir: Path, language: NBLanguage) -> None:
        startup_token = str(uuid.uuid4())
        node_workdir = Path(f"{cfg.CONTAINER_WORKSPACE_PREFIX}.{startup_token.split('-', maxsplit=1)[0]}")
        self._node_workdir = node_workdir

        _prep_workspace_dir(str(work_dir), workspace_path=str(node_workdir))
        logger.warning("[ray-enroot] prepared node-local workspace %s for host work_dir=%s", node_workdir, work_dir)

        kernel_server_path = Path(__file__).parent.parent / "kernel_server.py"
        assert kernel_server_path.is_file(), f"kernel server must be a valid path, found {kernel_server_path}"

        enroot_env = self._setup_enroot_env(startup_token)

        resource_prefix = _build_resource_limit_prefix(self.sandbox_memory_limit_mb, self.sandbox_max_pids)

        online = False
        attempt = 0
        last_err: Exception | None = None
        while not online:
            attempt += 1
            if attempt > MAX_CONTAINER_LAUNCH_RETRIES:
                log_tail = self._read_container_log_tail(500)
                raise RuntimeError(
                    f"Container failed to start after {MAX_CONTAINER_LAUNCH_RETRIES} attempts "
                    f"(last_error={last_err!r})"
                    f"{f' log_tail={log_tail!r}' if log_tail else ''}"
                )
            self._container_port = await get_free_port()

            node_workdir.mkdir(parents=True, exist_ok=True)
            bash = self._build_kernel_bash_script(
                str(node_workdir),
                language,
                self._container_port,
                startup_token,
                safe_execute=self.safe_execute,
                seed=self.seed,
            )
            cmd = self._build_enroot_cmd(
                work_dir,
                node_workdir,
                kernel_server_path,
                bash,
                enroot_env,
                self.container_sqsh_path,
                resource_prefix=resource_prefix,
            )

            async with CONTAINER_LAUNCH_SEM:
                launch_t0 = time.perf_counter()
                # Redirect container output to a log file instead of
                # subprocess.PIPE to avoid pipe-buffer deadlock (the 64KB
                # OS pipe buffer fills up when the kernel server produces
                # verbose DEBUG / uvicorn access logs, blocking the
                # container process on write() and freezing the kernel).
                log_dir = work_dir / ".container_logs"
                log_dir.mkdir(exist_ok=True)
                self._container_log_path = log_dir / "container.log"
                # Long-lived handle: the container subprocess writes its stdout/stderr here for its
                # whole lifetime and we close it in _close_container_log, so no context manager; the
                # open is a fast inline syscall (no thread).
                self._container_log_file = open(self._container_log_path, "w", encoding="utf-8")  # noqa: ASYNC230, SIM115
                self._enroot_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    start_new_session=True,
                    stdout=self._container_log_file,
                    stderr=subprocess.STDOUT,
                )
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container launch attempt #%d started (work_dir=%s, token=%s)",
                    self._proc_label(),
                    attempt,
                    work_dir,
                    startup_token[:8],
                )

            # Create HTTP client (outside semaphore — no need to hold the
            # concurrency slot while waiting for the container to come up)
            self._http_client = httpx.AsyncClient(
                base_url=f"http://localhost:{self._container_port}",
                timeout=httpx.Timeout(self.execution_timeout + 10, connect=30.0),
            )
            self._kernel_client = HttpKernelClient(
                self._http_client.request,
                execution_timeout=self.execution_timeout,
                timeout_recovery=self.timeout_recovery,
                interrupt_grace_seconds=self.interrupt_grace_seconds,
                label=self._proc_label(),
            )

            # Wait for health check
            try:
                await self._wait_for_health(expected_startup_token=startup_token)
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container online after %.1fms (attempt #%d)",
                    self._proc_label(),
                    launch_ms,
                    attempt,
                )
                online = True
            except Exception as e:
                last_err = e
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                await self._log_container_failure(attempt, launch_ms, e)
                await self._cleanup_failed_startup()
                if not isinstance(e, _PortCollisionError):
                    backoff = min(_RETRY_BASE_SLEEP * 2 ** (attempt - 1), _RETRY_MAX_SLEEP)
                    await asyncio.sleep(backoff)

    async def _log_container_failure(self, attempt: int, launch_ms: float, error: Exception) -> None:
        """Log detailed diagnostics when a container fails to start."""
        label = self._proc_label()
        proc = self._enroot_proc

        diag_parts = [
            f"attempt=#{attempt}",
            f"elapsed={launch_ms:.0f}ms",
            f"error={error!r}",
        ]

        if proc is not None:
            rc = proc.returncode
            diag_parts.append(f"process_alive={rc is None}")
            if rc is not None:
                diag_parts.append(f"returncode={rc}")

        logger.warning("[%s] Container startup FAILED: %s", label, ", ".join(diag_parts))

        # Log container output separately so tracebacks are readable
        log_tail = self._read_container_log_tail()
        if log_tail:
            logger.warning(
                "[%s] Container log output (last %d chars):\n%s",
                label,
                len(log_tail),
                log_tail,
            )

    def _read_container_log_tail(self, max_chars: int = 2000) -> str:
        """Read the tail of the container log file for diagnostics."""

        def _read() -> str:
            if self._container_log_path is None or not self._container_log_path.exists():
                return ""
            try:
                if self._container_log_file and not self._container_log_file.closed:
                    self._container_log_file.flush()
                text = self._container_log_path.read_text()
                return text[-max_chars:] if len(text) > max_chars else text
            except Exception:
                return ""

        return _read()

    def _close_container_log(self) -> None:
        """Close the container log file handle."""
        if self._container_log_file is not None:
            f = self._container_log_file
            self._container_log_file = None

            def _do_close() -> None:
                with contextlib.suppress(Exception):
                    f.close()

            _do_close()

    async def _cleanup_failed_startup(self) -> None:
        """Best-effort cleanup for failed startup attempts before retrying."""
        label = self._proc_label()

        if self._container_port is not None:
            async with used_ports_lock:
                _USED_PORTS.discard(self._container_port)
            self._container_port = None

        if self._http_client is not None:
            with contextlib.suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None

        if self._enroot_proc is not None:
            await _kill_process_group(self._enroot_proc, label=label)
            self._enroot_proc = None

        self._close_container_log()

        if self._node_workdir is not None:
            shutil.rmtree(self._node_workdir, ignore_errors=True)

    async def _wait_for_health(self, expected_startup_token: str | None = None) -> None:
        """Wait for the kernel server to become healthy."""
        assert self._http_client is not None
        await _poll_kernel_health(
            request=self._http_client.request,
            enroot_proc=self._enroot_proc,
            container_port=self._container_port,
            expected_startup_token=expected_startup_token,
            read_log_tail=self._read_container_log_tail,
            label=self._proc_label(),
        )

    async def _execute_via_http(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        """Execute code via the shared kernel HTTP client (timeout→error handling lives there)."""
        assert self._kernel_client is not None
        return await self._kernel_client.execute(code, timeout, req_uuid)

    async def _reset_via_http(self) -> None:
        """Reset the kernel via the shared kernel HTTP client."""
        assert self._kernel_client is not None
        await self._kernel_client.reset(self.seed)

    async def _list_dir_on_node(
        self,
        directory: str = ".",
        max_files: int = 20,
        show_hidden: bool = False,
        req_uuid: str = "",  # noqa: ARG002  (request-correlation id from the ray dispatch; unused in the body)
    ) -> str:
        """List contents of a directory with truncation protection.

        Recursively lists files in a directory, with built-in protection against
        overwhelming the context with too many files. Use this tool instead of
        writing code to list directories to avoid context bloat.

        Usage Examples:
            list_dir()                      # List working directory
            list_dir("data/")               # List specific folder
            list_dir(max_files=50)          # Show more files
            list_dir(show_hidden=True)      # Include hidden files

        Args:
            directory: Directory path to list (default: current working directory)
            max_files: Maximum number of files to display (default: 20)
            show_hidden: Whether to show hidden files starting with '.' (default: False)
            req_uuid: Request-correlation id from the ray dispatch (unused in the body).
        """
        try:
            normalized = self._normalize_node_workspace_path(directory)
        except ValueError:
            return f"Path must stay within the workspace root: {directory}"
        except Exception as e:
            return f"Error listing directory: {e!s}"

        # Bounded walk (capped by max_files); runs inline on the actor loop (concurrency=1) — no thread.
        return list_dir_tool(str(normalized), max_files=max_files, show_hidden=show_hidden)

    async def close(self):
        label = self._proc_label()
        logger.log(_CONTAINER_LOG_LEVEL, "[%s] Closing EnrootKernelServer", label)

        if self._container_port is not None:
            async with used_ports_lock:
                _USED_PORTS.discard(self._container_port)

        if self._http_client is not None:
            try:
                response = await self._http_client.post("/close")
                response.raise_for_status()
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException):
                logger.log(
                    _CONTAINER_LOG_LEVEL, "[%s] Graceful /close request failed (container may already be down)", label
                )
            except Exception:
                logger.warning("[%s] Unexpected error on /close request", label, exc_info=True)
            with contextlib.suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None

        if self._enroot_proc is not None:
            await _kill_process_group(self._enroot_proc, label=label)
            self._enroot_proc = None

        self._close_container_log()

        if self._node_workdir is not None:
            shutil.rmtree(self._node_workdir, ignore_errors=True)
            self._node_workdir = None

        logger.log(_CONTAINER_LOG_LEVEL, "[%s] EnrootKernelServer closed", label)


class EnrootSandbox(Sandbox):
    """Enroot execution backend (ray-placed actor, or local subprocess)."""

    def __init__(self, config: SandboxConfig) -> None:
        self.work_dir = config.work_dir
        self.language = config.language
        self._use_ray = config.use_ray
        self._container_sqsh_path = config.container_sqsh_path
        self._execution_timeout = config.execution_timeout
        self._safe_execute = config.safe_execute
        self._mem_mb = config.resources.mem_mb
        self._max_pids = config.resources.max_pids
        self._seed = config.seed
        self._timeout_recovery = config.timeout_recovery
        self._interrupt_grace_seconds = config.interrupt_grace_seconds

        # ray placement
        self.kernel_container: Any = None
        # non-ray subprocess
        self._enroot_proc: asyncio.subprocess.Process | None = None
        self._container_port: int | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._client: HttpKernelClient | None = None
        self._container_log_path: Path | None = None
        self._container_log_file: Any = None
        self._filesystem = FilesystemTool(config.work_dir)

    # ---- start --------------------------------------------------------------
    async def start(self) -> None:
        if self._use_ray:
            await self._start_ray()
        else:
            await self._start_subprocess()

    async def _start_ray(self) -> None:
        logger.warning("[ray-enroot] creating actor for work_dir=%s", self.work_dir)
        self.kernel_container = EnrootKernelServer.remote(  # type: ignore[attr-defined]
            self._container_sqsh_path,
            self._execution_timeout,
            safe_execute=self._safe_execute,
            sandbox_memory_limit_mb=self._mem_mb,
            sandbox_max_pids=self._max_pids,
            seed=self._seed,
            timeout_recovery=self._timeout_recovery,
            interrupt_grace_seconds=self._interrupt_grace_seconds,
        )
        init_ref = self.kernel_container.initialize.remote(self.work_dir, self.language)
        await self._await_ray_ref(
            init_ref,
            timeout=cfg.KERNEL_SERVER_STARTUP_TIMEOUT,
            req_uuid=f"init:{self.work_dir.name}",
            operation="initialize",
            max_wait_attempts=1,
        )
        logger.warning("[ray-enroot] initialize complete for work_dir=%s", self.work_dir)

    async def _start_subprocess(self) -> None:
        _prep_workspace_dir(str(self.work_dir))

        online = False
        attempt = 0
        last_err: Exception | None = None
        while not online:
            attempt += 1
            if attempt > MAX_CONTAINER_LAUNCH_RETRIES:
                log_tail = self._read_container_log_tail(500)
                raise RuntimeError(
                    f"Container failed to start after {MAX_CONTAINER_LAUNCH_RETRIES} attempts "
                    f"(last_error={last_err!r})"
                    f"{f' log_tail={log_tail!r}' if log_tail else ''}"
                )
            self._container_port = await get_free_port()
            startup_token = str(uuid.uuid4())

            resource_prefix = _build_resource_limit_prefix(self._mem_mb, self._max_pids)

            exports = bash_export_block("/data_workspace")
            seed_arg = f"--seed {self._seed}" if self._seed is not None else ""
            bash = (
                dedent(f"""\
                set -euo pipefail
                cd /data_workspace

                if [ -d /data_workspace/.install_shim/bin ]; then
                    chmod 755 /data_workspace/.install_shim/bin/* 2>/dev/null || true
                fi
                __WORKSPACE_EXPORTS__
                exec /app/kernel_env/bin/python /envs/kernel_server.py \\
                    --work_dir /data_workspace \\
                    --language {self.language.value} \\
                    --port {self._container_port} \\
                    --startup-token {startup_token} {"--safe-execute" if self._safe_execute else ""} {seed_arg}
            """)
                .strip()
                .replace("__WORKSPACE_EXPORTS__", exports)
            )

            kernel_server_path = Path(__file__).parent.parent / "kernel_server.py"
            assert kernel_server_path.is_file(), f"kernel server must be a valid path, found {kernel_server_path}"
            assert self._container_sqsh_path is not None, "container_sqsh_path must be set when using enroot container"

            cmd = [
                *resource_prefix,
                "env",
                "-i",
                "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
                'HOME="$HOME"',
                'USER="$USER"',
                "enroot",
                "start",
                "--mount",
                f"{self.work_dir}:/data_workspace",
                "--mount",
                f"{kernel_server_path.resolve()}:/envs/kernel_server.py",
                str(self._container_sqsh_path.resolve()),
                "/bin/bash",
                "-lc",
                bash,
            ]

            async with CONTAINER_LAUNCH_SEM:
                launch_t0 = time.perf_counter()
                (self.work_dir / ".container_logs").mkdir(exist_ok=True)
                self._container_log_path = self.work_dir / ".container_logs" / "container.log"
                # Long-lived handle: the container subprocess writes its stdout/stderr here for its
                # whole lifetime and we close it in _close_container_log, so no context manager; the
                # open is a fast inline syscall (no thread).
                self._container_log_file = open(self._container_log_path, "w", encoding="utf-8")  # noqa: ASYNC230, SIM115
                self._enroot_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    start_new_session=True,
                    stdout=self._container_log_file,
                    stderr=subprocess.STDOUT,
                )
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container launch attempt #%d (work_dir=%s, token=%s)",
                    self._label(),
                    attempt,
                    self.work_dir,
                    startup_token[:8],
                )

            self._http_client = httpx.AsyncClient(
                base_url=f"http://localhost:{self._container_port}",
                timeout=httpx.Timeout(self._execution_timeout + 10, connect=30.0),
            )
            self._client = HttpKernelClient(
                self._http_client.request,
                execution_timeout=self._execution_timeout,
                timeout_recovery=self._timeout_recovery,
                interrupt_grace_seconds=self._interrupt_grace_seconds,
                label=self._label(),
            )

            try:
                await _poll_kernel_health(
                    request=self._http_client.request,
                    enroot_proc=self._enroot_proc,
                    container_port=self._container_port,
                    expected_startup_token=startup_token,
                    read_log_tail=self._read_container_log_tail,
                    label=self._label(),
                )
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container online after %.1fms (attempt #%d)",
                    self._label(),
                    launch_ms,
                    attempt,
                )
                online = True
            except Exception as e:
                last_err = e
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                await self._log_container_failure(attempt, launch_ms, e)
                await self._cleanup_failed_startup()
                if not isinstance(e, _PortCollisionError):
                    backoff = min(_RETRY_BASE_SLEEP * 2 ** (attempt - 1), _RETRY_MAX_SLEEP)
                    await asyncio.sleep(backoff)

    # ---- dispatch -----------------------------------------------------------
    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        if self._use_ray:
            ref = self.kernel_container._execute_via_http.remote(code, timeout, req_uuid=req_uuid)
            return cast(
                ExecutionResult,
                await self._await_ray_ref(
                    ref,
                    timeout=timeout,
                    req_uuid=req_uuid,
                    operation="_execute_via_http",
                    execution_request=True,
                ),
            )
        assert self._client is not None
        return await self._client.execute(code, timeout, req_uuid)

    async def reset(self) -> None:
        if self._use_ray:
            await self.kernel_container._reset_via_http.remote()
            return
        assert self._client is not None
        await self._client.reset(self._seed)

    async def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        if self._use_ray:
            if self.kernel_container is None:
                return "Error listing directory: node-local workspace is unavailable"
            list_dir_uuid = str(uuid.uuid4())
            ref = self.kernel_container._list_dir_on_node.remote(
                directory=directory, max_files=max_files, show_hidden=show_hidden, req_uuid=list_dir_uuid
            )
            return cast(
                str,
                await self._await_ray_ref(
                    ref,
                    timeout=_LIST_DIR_RAY_TIMEOUT,
                    req_uuid=list_dir_uuid,
                    operation="list_dir",
                    max_wait_attempts=2,
                ),
            )
        return self._filesystem.list_dir(directory, max_files, show_hidden)

    async def health(self) -> bool:
        if self._use_ray:
            return self.kernel_container is not None
        return await self._client.health() if self._client is not None else False

    async def close(self) -> None:
        if self._use_ray:
            if self.kernel_container is not None:
                await self.kernel_container.close.remote()
                self.kernel_container = None
            return
        if self._container_port is not None:
            async with used_ports_lock:
                _USED_PORTS.discard(self._container_port)
            self._container_port = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._http_client is not None:
            with contextlib.suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None
        if self._enroot_proc is not None:
            await _kill_process_group(self._enroot_proc, label=self._label())
            self._enroot_proc = None
        self._close_container_log()

    # ---- helpers ------------------------------------------------------------
    def _label(self) -> str:
        port = self._container_port or "?"
        pid = self._enroot_proc.pid if self._enroot_proc else "?"
        return f"enroot(port={port}, pid={pid})"

    async def _await_ray_ref(
        self,
        ref: Awaitable[Any],
        *,
        timeout: float | None,  # noqa: ASYNC109
        req_uuid: str,
        operation: str,
        max_wait_attempts: int = MAX_RAY_RESULT_WAIT_RETRIES,
        execution_request: bool = False,
    ) -> Any:
        effective_timeout = timeout if timeout is not None else self._execution_timeout
        if execution_request:
            wait_timeout = (
                execute_wire_timeout_seconds(
                    effective_timeout,
                    self._timeout_recovery,
                    self._interrupt_grace_seconds,
                )
                + _RAY_RESULT_WAIT_TIMEOUT_GRACE
            )
        else:
            wait_timeout = effective_timeout + _RAY_RESULT_WAIT_TIMEOUT_GRACE
        last_timeout: TimeoutError | None = None
        for attempt in range(1, max_wait_attempts + 1):
            try:
                # Retry waiting on the same underlying, shielded ObjectRef. Never call the
                # actor method again: that could execute a non-idempotent cell
                # multiple times and must not create additional time charges.
                return await asyncio.wait_for(asyncio.shield(ref), timeout=wait_timeout)
            except TimeoutError as exc:
                last_timeout = exc
                if attempt >= max_wait_attempts:
                    logger.exception(
                        "[ray-enroot] req %s exhausted waits for %s on work_dir=%s (attempts=%d, timeout=%.1fs)",
                        req_uuid,
                        operation,
                        self.work_dir,
                        max_wait_attempts,
                        wait_timeout,
                    )
                    break
                backoff = min(_RETRY_BASE_SLEEP * 2 ** (attempt - 1), _RETRY_MAX_SLEEP)
                logger.warning(
                    "[ray-enroot] req %s timed out waiting for %s (wait #%d/%d); waiting again in %.1fs",
                    req_uuid,
                    operation,
                    attempt,
                    max_wait_attempts,
                    backoff,
                )
                await asyncio.sleep(backoff)
        raise TimeoutError(
            f"Timed out waiting for ray {operation} after {max_wait_attempts} wait attempts "
            f"(req={req_uuid}, timeout_per_attempt={wait_timeout:.1f}s)"
        ) from last_timeout

    def _read_container_log_tail(self, max_chars: int = 2000) -> str:
        def _read() -> str:
            if self._container_log_path is None or not self._container_log_path.exists():
                return ""
            try:
                if self._container_log_file and not self._container_log_file.closed:
                    self._container_log_file.flush()
                text = self._container_log_path.read_text()
                return text[-max_chars:] if len(text) > max_chars else text
            except Exception:
                return ""

        return _read()

    def _close_container_log(self) -> None:
        if self._container_log_file is not None:
            f = self._container_log_file
            self._container_log_file = None

            def _do_close() -> None:
                with contextlib.suppress(Exception):
                    f.close()

            _do_close()

    async def _cleanup_failed_startup(self) -> None:
        if self._container_port is not None:
            async with used_ports_lock:
                _USED_PORTS.discard(self._container_port)
            self._container_port = None
        if self._http_client is not None:
            with contextlib.suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None
        if self._enroot_proc is not None:
            await _kill_process_group(self._enroot_proc, label=self._label())
            self._enroot_proc = None
        self._close_container_log()

    async def _log_container_failure(self, attempt: int, launch_ms: float, error: Exception) -> None:
        diag = [f"attempt=#{attempt}", f"elapsed={launch_ms:.0f}ms", f"error={error!r}"]
        if self._enroot_proc is not None:
            rc = self._enroot_proc.returncode
            diag.append(f"process_alive={rc is None}")
            if rc is not None:
                diag.append(f"returncode={rc}")
        logger.warning("[%s] Container startup FAILED: %s", self._label(), ", ".join(diag))
        log_tail = self._read_container_log_tail()
        if log_tail:
            logger.warning("[%s] Container log (last %d chars):\n%s", self._label(), len(log_tail), log_tail)
