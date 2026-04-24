"""InterpreterEnv: Standalone code execution environment for data analysis.

This module provides a lightweight, execution-focused environment for running
code in Jupyter kernels. It focuses on direct code execution via run_cell().
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import mkdtemp
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID

import aiodocker
import httpx
import nbformat
import numpy as np
import tenacity
from aviary.core import (
    EnvStateMessage,
    Frame,
    Message,
    Messages,
    Tool,
    ToolCall,
    ToolRequestMessage,
)
from aviary.env import Environment
from lmi import LiteLLMModel
from nbformat import NotebookNode
from pydantic import BaseModel, Field, JsonValue, model_validator

from . import config as cfg
from .code_safety import check_code_safety
from .config import ExecutionConfig
from .hybrid_gate import (
    HYBRID_GATE_PROMPT,
    hybrid_reward,
    parse_hybrid_response,
    synthesize_per_item_awards,
)
from .install_shim import _write_install_shims
from .interpreter import ExecutionResult, Interpreter
from .wager import (
    WAGER_BETA_DEFAULT,
    WAGER_GAMMA_DEFAULT,
    clamp_confidence,
    score_with_wager,
)
from .prompts import (
    CORRECT_MSG,
    FAITHFULNESS_GATE_PROMPT,
    HYPOTHESIS_TASK_DESC,
    INCORRECT_MSG,
    RUBRIC_SCORE_PROMPT,
    PromptingConfig,
)
from .tools.filesystem import FilesystemTool, list_dir_tool
from .utils import NBLanguage, view_notebook

RAY_INSTALLED = True
try:
    import ray
except ImportError:
    RAY_INSTALLED = False

if TYPE_CHECKING:
    from aiodocker.containers import DockerContainer


# Port management for Docker containers
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

_warned_unsafe_execution: set[str] = set()
_BACKGROUND_CLEANUP_TASKS: set[asyncio.Task[None]] = set()


def _make_cleanup_path(path: Path) -> Path:
    return path.with_name(f".cleanup-{path.name}-{uuid.uuid4().hex}")


def _detach_dir_for_cleanup(path: Path) -> Path | None:
    if not path.exists():
        return None

    cleanup_path = _make_cleanup_path(path)
    while cleanup_path.exists():
        cleanup_path = _make_cleanup_path(path)

    path.replace(cleanup_path)
    return cleanup_path


def _schedule_dir_cleanup(path: Path) -> None:
    async def _cleanup() -> None:
        try:
            await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
        except Exception:
            logger.warning("Background cleanup failed for %s", path, exc_info=True)

    task = asyncio.create_task(_cleanup())
    _BACKGROUND_CLEANUP_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_CLEANUP_TASKS.discard)


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


logger = logging.getLogger(__name__)

# Container lifecycle log level: root logger defaults to WARNING and we
# cannot reconfigure it, so use WARNING for all container diagnostics to
# ensure they are visible in production logs.
_CONTAINER_LOG_LEVEL = logging.WARNING


class _PortCollisionError(Exception):
    """Port already in use by another server — retry with a new port."""


async def _poll_kernel_health(  # noqa: PLR0912
    http_client: httpx.AsyncClient,
    enroot_proc: asyncio.subprocess.Process | None,
    container_port: int | None,
    expected_startup_token: str | None,
    read_log_tail: Callable[[int], Awaitable[str]],
    label: str,
) -> None:
    """Poll kernel server /health endpoint until ready, with token validation and timeout."""
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
            log_tail = await read_log_tail(1000)
            raise RuntimeError(
                f"Enroot process exited prematurely with returncode={rc} after {elapsed:.1f}s. log: {log_tail!r}"
            )

        try:
            response = await http_client.get("/health", timeout=health_timeout)
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
    log_tail = await read_log_tail(500)
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
            "prlimit not found on PATH; skipping resource limits "
            "(memory_limit_mb=%s, max_pids=%s)",
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


class ProblemInstance(BaseModel):
    id: UUID
    hypothesis: str
    protocol: str
    accepted: bool = Field(alias="answer")
    rubric: str
    max_score: int = Field(alias="max_points")
    input_data_path: str = ""
    faithfulness_rubric: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    nb_primary_language: str = Field(default=str(NBLanguage.PYTHON))

    @model_validator(mode="before")
    @classmethod
    def handle_language(cls, data: dict) -> dict:
        if data.get("nb_primary_language") is None:
            data["nb_primary_language"] = str(NBLanguage.PYTHON)
        return data


def _prep_workspace_dir(work_dir: str, workspace_path: str = "/data_workspace") -> None:
    wd = Path(work_dir)
    (wd / "pydeps").mkdir(parents=True, exist_ok=True)
    (wd / "pip-cache").mkdir(parents=True, exist_ok=True)

    (wd / "pip.conf").write_text(
        "[global]\n"
        "disable-pip-version-check = true\n"
        "no-input = true\n"
        f"cache-dir = {workspace_path}/pip-cache\n"
        f"target = {workspace_path}/pydeps\n"
    )

    _write_install_shims(wd)


@ray.remote(
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
    ):
        self.container_sqsh_path = container_sqsh_path
        self.execution_timeout = execution_timeout
        self.safe_execute = safe_execute
        self.sandbox_memory_limit_mb = sandbox_memory_limit_mb
        self.sandbox_max_pids = sandbox_max_pids
        self._enroot_proc: asyncio.subprocess.Process | None = None
        self._http_client: httpx.AsyncClient | None = None
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
        node_workdir: str, language: NBLanguage, port: int, startup_token: str, safe_execute: bool = True
    ) -> str:
        """Build the bash script that sets up the workspace and launches the kernel server."""
        return dedent(f"""\
            set -euo pipefail

            WORKDIR="{node_workdir}"
            trap 'rm -rf "$WORKDIR"' EXIT

            mkdir -p $WORKDIR
            cp -a /data_workspace/. $WORKDIR/

            cd $WORKDIR

            # Install-shim wrappers (pip / conda / apt-get) + R shim + JSONL log
            # all live under $WORKDIR/.install_shim/ (hidden from the agent's
            # list_dir default view). Pre-written by _write_install_shims().
            # cp -a preserves execute bits but ensure anyway.
            if [ -d "$WORKDIR/.install_shim/bin" ]; then
                chmod 755 "$WORKDIR/.install_shim/bin"/* 2>/dev/null || true
            fi

            mkdir -p "$WORKDIR/r_libs"
            cat >"$WORKDIR/Rprofile" <<'EOF'
            .local_lib <- Sys.getenv("R_LIBS_USER")
            if (nzchar(.local_lib)) .libPaths(unique(c(.local_lib, .libPaths())))
            EOF
            # Append the R install-shim (pre-written by _write_install_shims).
            if [ -f "$WORKDIR/.install_shim/r_shim.R" ]; then
                cat "$WORKDIR/.install_shim/r_shim.R" >> "$WORKDIR/Rprofile"
            fi

            mkdir -p $WORKDIR/r_libs
            export R_LIBS_USER="$WORKDIR/r_libs"
            export R_PROFILE_USER="$WORKDIR/Rprofile"

            # Prepend the shim bin dir so pip/conda/apt-get resolve to our wrappers.
            # The wrappers exec the real installer at their absolute paths, so no recursion.
            export PATH="$WORKDIR/.install_shim/bin:$PATH"
            export INSTALL_SHIM_LOG="$WORKDIR/.install_shim/log"

            export PYTHONPATH="$WORKDIR/pydeps:${{PYTHONPATH}}"
            export PIP_CONFIG_FILE=$WORKDIR/pip.conf
            export target_platform=${{target_platform:-linux-64}}

            source activate /app/kernel_env
            exec /app/kernel_env/bin/python /envs/kernel_server.py \\
                --work_dir $WORKDIR \\
                --language {language.value} \\
                --port {port} \\
                --startup-token {startup_token} {"--safe-execute" if safe_execute else ""}
        """).strip()

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

        kernel_server_path = Path(__file__).parent / "kernel_server.py"
        assert kernel_server_path.is_file(), f"kernel server must be a valid path, found {kernel_server_path}"

        enroot_env = self._setup_enroot_env(startup_token)

        resource_prefix = _build_resource_limit_prefix(
            self.sandbox_memory_limit_mb, self.sandbox_max_pids
        )

        online = False
        attempt = 0
        last_err: Exception | None = None
        while not online:
            attempt += 1
            if attempt > MAX_CONTAINER_LAUNCH_RETRIES:
                log_tail = await self._read_container_log_tail(500)
                raise RuntimeError(
                    f"Container failed to start after {MAX_CONTAINER_LAUNCH_RETRIES} attempts "
                    f"(last_error={last_err!r})"
                    f"{f' log_tail={log_tail!r}' if log_tail else ''}"
                )
            self._container_port = await get_free_port()

            await asyncio.to_thread(node_workdir.mkdir, parents=True, exist_ok=True)
            bash = self._build_kernel_bash_script(
                str(node_workdir), language, self._container_port, startup_token, safe_execute=self.safe_execute
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
                self._container_log_file = await asyncio.to_thread(
                    open, self._container_log_path, "w", encoding="utf-8"
                )
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
        log_tail = await self._read_container_log_tail()
        if log_tail:
            logger.warning(
                "[%s] Container log output (last %d chars):\n%s",
                label,
                len(log_tail),
                log_tail,
            )

    async def _read_container_log_tail(self, max_chars: int = 2000) -> str:
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

        return await asyncio.to_thread(_read)

    async def _close_container_log(self) -> None:
        """Close the container log file handle."""
        if self._container_log_file is not None:
            f = self._container_log_file
            self._container_log_file = None

            def _do_close() -> None:
                with contextlib.suppress(Exception):
                    f.close()

            await asyncio.to_thread(_do_close)

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

        await self._close_container_log()

        if self._node_workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self._node_workdir, ignore_errors=True)

    async def _wait_for_health(self, expected_startup_token: str | None = None) -> None:
        """Wait for the kernel server to become healthy."""
        assert self._http_client is not None
        await _poll_kernel_health(
            http_client=self._http_client,
            enroot_proc=self._enroot_proc,
            container_port=self._container_port,
            expected_startup_token=expected_startup_token,
            read_log_tail=self._read_container_log_tail,
            label=self._proc_label(),
        )

    async def _execute_via_http(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        """Execute code via HTTP to the containerized kernel server.

        Handles httpx.TimeoutException (including ReadTimeout) by converting to
        an error ExecutionResult. This happens when the kernel server's internal
        asyncio.timeout doesn't cancel the ZMQ recv promptly, causing the HTTP
        read to time out before the kernel server responds.
        """
        assert self._http_client is not None

        try:
            response = await self._http_client.post(
                "/execute",
                json={"code": code, "timeout": timeout},
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            effective_timeout = timeout if timeout is not None else self.execution_timeout
            logger.warning(
                "[%s] HTTP %s during /execute (requested kernel timeout=%.1fs): %s",
                self._proc_label(),
                type(e).__name__,
                effective_timeout,
                e,
            )
            timeout_output = nbformat.v4.new_output(
                output_type="error",
                ename="TimeoutError",
                evalue=f"Code execution timed out after {effective_timeout}s (HTTP layer)",
                traceback=[f"TimeoutError: Code execution timed out after {effective_timeout}s (HTTP layer)"],
            )
            return ExecutionResult(
                notebook_outputs=[timeout_output],
                error_occurred=True,
                execution_time=effective_timeout,
            )

        data = response.json()

        # Convert serialized outputs back to NotebookNode
        notebook_outputs = [nbformat.from_dict(o) for o in data["notebook_outputs"]]

        return ExecutionResult(
            notebook_outputs=notebook_outputs,
            error_occurred=data["error_occurred"],
            execution_time=data.get("execution_time"),
        )

    async def _reset_via_http(self) -> None:
        """Reset the kernel via HTTP."""
        assert self._http_client is not None
        try:
            response = await self._http_client.post("/reset")
            response.raise_for_status()
        except httpx.TimeoutException as e:
            logger.warning(
                "[%s] HTTP %s during /reset: %s",
                self._proc_label(),
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"Kernel reset timed out: {e}") from e

    async def _list_dir_on_node(
        self,
        directory: str = ".",
        max_files: int = 20,
        show_hidden: bool = False,
        req_uuid: str = "",
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
        """
        try:
            normalized = self._normalize_node_workspace_path(directory)
        except ValueError:
            return f"Path must stay within the workspace root: {directory}"
        except Exception as e:
            return f"Error listing directory: {e!s}"

        result = await asyncio.to_thread(
            list_dir_tool,
            str(normalized),
            max_files=max_files,
            show_hidden=show_hidden,
        )

        return result

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

        await self._close_container_log()

        if self._node_workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self._node_workdir, ignore_errors=True)
            self._node_workdir = None

        logger.log(_CONTAINER_LOG_LEVEL, "[%s] EnrootKernelServer closed", label)


class InterpreterEnvState:
    """State container for the InterpreterEnv.

    Manages the kernel, notebook state, and execution tracking.
    Supports both local kernel execution and Docker-based execution.
    """

    def __init__(
        self,
        work_dir: Path,
        language: NBLanguage,
        execution_timeout: int = 600,
        safe_execute: bool = True,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
        use_docker: bool = cfg.USE_DOCKER,
        use_enroot: bool = False,
        use_ray: bool = True,
        container_sqsh_path: Path | None = None,
        save_dir: Path | None = None,
        sandbox_memory_limit_mb: int | None = None,
        sandbox_max_pids: int | None = None,
    ):
        self.work_dir = work_dir
        self.language = language
        self.execution_timeout = execution_timeout
        self.safe_execute = safe_execute
        self.total_reward = 0.0
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}
        self.answer: str | None = None
        self.actions: list[str] = []
        self.done = False
        self.use_docker = use_docker
        self.use_enroot = use_enroot
        self.container_sqsh_path = container_sqsh_path
        self.save_dir = save_dir
        self.use_ray = use_ray if RAY_INSTALLED else False
        self.sandbox_memory_limit_mb = sandbox_memory_limit_mb
        self.sandbox_max_pids = sandbox_max_pids

        # Local interpreter (only used when use_docker=False)
        self.interpreter: Interpreter | None = None
        if not use_docker and not use_enroot:
            self.interpreter = Interpreter(
                work_dir=work_dir,
                language=language,
                execution_timeout=execution_timeout,
                use_host_env_vars=use_host_env_vars,
                extra_envs=extra_envs,
            )

        # Docker/Enroot container state (only used when use_docker=True or use_enroot=True)
        self._docker_client: aiodocker.Docker | None = None
        self._container: DockerContainer | None = None
        self._enroot_proc: asyncio.subprocess.Process | None = None
        self._container_port: int | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._container_log_path: Path | None = None
        self._container_log_file: Any | None = None
        self.kernel_container: EnrootKernelServer | None = None

        # Initialize notebook structure for state tracking
        self.nb: NotebookNode = nbformat.v4.new_notebook()
        self.nb.metadata.kernelspec = language.make_kernelspec()
        self.notebook_runtime_errors: list[str] = []
        self._execution_count = 0

        self.raw_score: int = 0
        self.score: float = 0.0
        self.score_metadata: dict[str, str | int] = {}
        self.faithfulness_passed: bool | None = None
        self.faithfulness_metadata: dict[str, str] = {}
        self.rubric_reward_raw: float = 0.0
        self.hybrid_reward_value: float = 0.0
        self.hybrid_metadata: dict[str, Any] = {}
        self.wager: float = 0.0
        self.wager_reward_shadow: float = 0.0
        self.wager_metadata: dict[str, Any] = {}
        self.cell_timeout_override_requests: list[float] = []

    async def start(self):
        """Start the interpreter (local or Docker-based)."""
        if self.use_enroot:
            assert self.container_sqsh_path is not None, "container_sqsh_path must be set in config to use enroot"
            if self.use_ray:
                await self._start_ray_enroot_container()
            else:
                await self._start_enroot_container()
        elif self.use_docker:
            await self._start_docker_container()
        else:
            assert self.interpreter is not None
            await self.interpreter.start()

    async def _start_ray_enroot_container(self) -> None:
        logger.warning("[ray-enroot] creating actor for work_dir=%s", self.work_dir)
        self.kernel_container = EnrootKernelServer.remote(  # type: ignore[attr-defined]
            self.container_sqsh_path,
            self.execution_timeout,
            safe_execute=self.safe_execute,
            sandbox_memory_limit_mb=self.sandbox_memory_limit_mb,
            sandbox_max_pids=self.sandbox_max_pids,
        )
        logger.warning("[ray-enroot] actor created, calling initialize for work_dir=%s", self.work_dir)
        init_ref = self.kernel_container.initialize.remote(self.work_dir, self.language)  # type: ignore[union-attr]
        await self._await_ray_ref(
            init_ref,
            timeout=cfg.KERNEL_SERVER_STARTUP_TIMEOUT,
            req_uuid=f"init:{self.work_dir.name}",
            operation="initialize",
            max_retries=1,
        )
        logger.warning("[ray-enroot] initialize complete for work_dir=%s", self.work_dir)

    async def _await_ray_ref(
        self,
        ref: Awaitable[Any],
        *,
        timeout: float | None,
        req_uuid: str,
        operation: str,
        max_retries: int = MAX_RAY_RESULT_WAIT_RETRIES,
    ) -> Any:
        effective_timeout = timeout if timeout is not None else self.execution_timeout
        wait_timeout = effective_timeout + _RAY_RESULT_WAIT_TIMEOUT_GRACE
        last_timeout: TimeoutError | None = None

        for attempt in range(1, max_retries + 1):
            try:
                return await asyncio.wait_for(asyncio.shield(ref), timeout=wait_timeout)
            except TimeoutError as exc:
                last_timeout = exc
                if attempt >= max_retries:
                    logger.error(
                        "[ray-enroot] req %s exhausted waits for %s on work_dir=%s "
                        "(attempts=%d, timeout_per_attempt=%.1fs)",
                        req_uuid,
                        operation,
                        self.work_dir,
                        max_retries,
                        wait_timeout,
                    )
                    break

                backoff = min(_RETRY_BASE_SLEEP * 2 ** (attempt - 1), _RETRY_MAX_SLEEP)
                logger.warning(
                    "[ray-enroot] req %s timed out waiting for %s on work_dir=%s "
                    "(attempt #%d/%d, timeout=%.1fs); retrying in %.1fs",
                    req_uuid,
                    operation,
                    self.work_dir,
                    attempt,
                    max_retries,
                    wait_timeout,
                    backoff,
                )
                await asyncio.sleep(backoff)

        raise TimeoutError(
            f"Timed out waiting for ray {operation} after {max_retries} attempts "
            f"(req={req_uuid}, timeout_per_attempt={wait_timeout:.1f}s)"
        ) from last_timeout

    def _enroot_label(self) -> str:
        port = self._container_port or "?"
        pid = self._enroot_proc.pid if self._enroot_proc else "?"
        return f"enroot-state(port={port}, pid={pid})"

    async def _start_enroot_container(self) -> None:
        _prep_workspace_dir(str(self.work_dir))

        online = False
        attempt = 0
        last_err: Exception | None = None
        while not online:
            attempt += 1
            if attempt > MAX_CONTAINER_LAUNCH_RETRIES:
                log_tail = await self._read_container_log_tail(500)
                raise RuntimeError(
                    f"Container failed to start after {MAX_CONTAINER_LAUNCH_RETRIES} attempts "
                    f"(last_error={last_err!r})"
                    f"{f' log_tail={log_tail!r}' if log_tail else ''}"
                )
            self._container_port = await get_free_port()
            startup_token = str(uuid.uuid4())

            resource_prefix = _build_resource_limit_prefix(
                self.sandbox_memory_limit_mb, self.sandbox_max_pids
            )

            bash = dedent(f"""\
                set -euo pipefail
                cd /data_workspace

                if [ -d /data_workspace/.install_shim/bin ]; then
                    chmod 755 /data_workspace/.install_shim/bin/* 2>/dev/null || true
                fi
                export PATH="/data_workspace/.install_shim/bin:$PATH"
                export INSTALL_SHIM_LOG="/data_workspace/.install_shim/log"

                export PYTHONPATH="/data_workspace/pydeps:${{PYTHONPATH}}"
                export PIP_CONFIG_FILE=/data_workspace/pip.conf
                exec /app/kernel_env/bin/python /envs/kernel_server.py \\
                    --work_dir /data_workspace \\
                    --language {self.language.value} \\
                    --port {self._container_port} \\
                    --startup-token {startup_token} {"--safe-execute" if self.safe_execute else ""}
            """).strip()

            kernel_server_path = Path(__file__).parent / "kernel_server.py"
            assert kernel_server_path.is_file(), f"kernel server must be a valid path, found {kernel_server_path}"
            assert self.container_sqsh_path is not None, "container_sqsh_path must be set when using enroot container"

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
                str(self.container_sqsh_path.resolve()),
                "/bin/bash",
                "-lc",
                bash,
            ]

            async with CONTAINER_LAUNCH_SEM:
                launch_t0 = time.perf_counter()
                # Redirect container output to a log file (see EnrootKernelServer
                # for detailed rationale on why we avoid subprocess.PIPE here).
                (self.work_dir / ".container_logs").mkdir(exist_ok=True)
                self._container_log_path = self.work_dir / ".container_logs" / "container.log"
                self._container_log_file = await asyncio.to_thread(
                    open, self._container_log_path, "w", encoding="utf-8"
                )
                self._enroot_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    start_new_session=True,
                    stdout=self._container_log_file,
                    stderr=subprocess.STDOUT,
                )
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container launch attempt #%d (work_dir=%s, token=%s)",
                    self._enroot_label(),
                    attempt,
                    self.work_dir,
                    startup_token[:8],
                )

            # Create HTTP client (outside semaphore — no need to hold the
            # concurrency slot while waiting for the container to come up)
            self._http_client = httpx.AsyncClient(
                base_url=f"http://localhost:{self._container_port}",
                timeout=httpx.Timeout(self.execution_timeout + 10, connect=30.0),
            )

            # Wait for health check
            try:
                await self._wait_for_health(expected_startup_token=startup_token)
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container online after %.1fms (attempt #%d)",
                    self._enroot_label(),
                    launch_ms,
                    attempt,
                )
                online = True
            except Exception as e:
                last_err = e
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                await self._log_enroot_container_failure(attempt, launch_ms, e)
                await self._cleanup_failed_startup()
                if not isinstance(e, _PortCollisionError):
                    backoff = min(_RETRY_BASE_SLEEP * 2 ** (attempt - 1), _RETRY_MAX_SLEEP)
                    await asyncio.sleep(backoff)

    async def _log_enroot_container_failure(self, attempt: int, launch_ms: float, error: Exception) -> None:
        """Log detailed diagnostics when a container fails to start."""
        label = self._enroot_label()
        diag = [f"attempt=#{attempt}", f"elapsed={launch_ms:.0f}ms", f"error={error!r}"]
        if self._enroot_proc is not None:
            rc = self._enroot_proc.returncode
            diag.append(f"process_alive={rc is None}")
            if rc is not None:
                diag.append(f"returncode={rc}")
        logger.warning("[%s] Container startup FAILED: %s", label, ", ".join(diag))
        log_tail = await self._read_container_log_tail()
        if log_tail:
            logger.warning(
                "[%s] Container log output (last %d chars):\n%s",
                label,
                len(log_tail),
                log_tail,
            )

    async def _read_container_log_tail(self, max_chars: int = 2000) -> str:
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

        return await asyncio.to_thread(_read)

    async def _close_container_log(self) -> None:
        """Close the container log file handle."""
        if self._container_log_file is not None:
            f = self._container_log_file
            self._container_log_file = None

            def _do_close() -> None:
                with contextlib.suppress(Exception):
                    f.close()

            await asyncio.to_thread(_do_close)

    async def _cleanup_failed_startup(self) -> None:
        """Best-effort cleanup for failed startup attempts before retrying."""
        label = self._enroot_label()

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

        await self._close_container_log()

    async def _start_docker_container(self) -> None:
        """Start a Docker container with the kernel server."""
        self._docker_client = aiodocker.Docker()
        self._container_port = await get_free_port()
        startup_token = str(uuid.uuid4())

        cmd_list = [
            "/app/kernel_env/bin/python",
            "/envs/kernel_server.py",
            "--work_dir",
            "/data_workspace",
            "--language",
            self.language.value,
            "--startup-token",
            startup_token,
        ]
        if self.safe_execute:
            cmd_list += ["--safe-execute"]

        docker_config = {
            "Image": cfg.NB_ENVIRONMENT_DOCKER_IMAGE,
            "Cmd": cmd_list,
            "HostConfig": {
                "Binds": [f"{self.work_dir}:/data_workspace"],
                "PortBindings": {f"{cfg.KERNEL_SERVER_PORT}/tcp": [{"HostPort": str(self._container_port)}]},
            },
            "WorkingDir": "/data_workspace",
            "Tty": True,
            "ExposedPorts": {f"{cfg.KERNEL_SERVER_PORT}/tcp": {}},
        }

        self._container = await self._docker_client.containers.run(config=cast(dict[str, Any], docker_config))
        logger.log(_CONTAINER_LOG_LEVEL, "Started docker container on port %s", self._container_port)

        # Create HTTP client
        self._http_client = httpx.AsyncClient(
            base_url=f"http://localhost:{self._container_port}",
            timeout=httpx.Timeout(self.execution_timeout + 10, connect=30.0),
        )

        # Wait for health check
        await self._wait_for_health(expected_startup_token=startup_token)

    async def _wait_for_health(self, expected_startup_token: str | None = None) -> None:
        """Wait for the kernel server to become healthy."""
        assert self._http_client is not None
        await _poll_kernel_health(
            http_client=self._http_client,
            enroot_proc=self._enroot_proc,
            container_port=self._container_port,
            expected_startup_token=expected_startup_token,
            read_log_tail=self._read_container_log_tail,
            label=self._enroot_label(),
        )

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.ReadError)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _execute_via_http(self, code: str, timeout: float | None = None) -> ExecutionResult:  # noqa: ASYNC109
        """Execute code via HTTP to the containerized kernel server.

        Handles httpx.TimeoutException (including ReadTimeout) by converting to
        an error ExecutionResult, since the kernel server's asyncio.timeout may
        not cancel the ZMQ recv promptly.
        """
        assert self._http_client is not None

        try:
            response = await self._http_client.post(
                "/execute",
                json={"code": code, "timeout": timeout},
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            effective_timeout = timeout if timeout is not None else self.execution_timeout
            logger.warning(
                "[%s] HTTP %s during /execute (requested kernel timeout=%.1fs): %s",
                self._enroot_label(),
                type(e).__name__,
                effective_timeout,
                e,
            )
            timeout_output = nbformat.v4.new_output(
                output_type="error",
                ename="TimeoutError",
                evalue=f"Code execution timed out after {effective_timeout}s (HTTP layer)",
                traceback=[f"TimeoutError: Code execution timed out after {effective_timeout}s (HTTP layer)"],
            )
            return ExecutionResult(
                notebook_outputs=[timeout_output],
                error_occurred=True,
                execution_time=effective_timeout,
            )

        data = response.json()

        # Convert serialized outputs back to NotebookNode
        notebook_outputs = [nbformat.from_dict(o) for o in data["notebook_outputs"]]

        return ExecutionResult(
            notebook_outputs=notebook_outputs,
            error_occurred=data["error_occurred"],
            execution_time=data.get("execution_time"),
        )

    async def _reset_via_http(self) -> None:
        """Reset the kernel via HTTP."""
        assert self._http_client is not None
        try:
            response = await self._http_client.post("/reset")
            response.raise_for_status()
        except httpx.TimeoutException as e:
            logger.warning(
                "HTTP %s during /reset: %s",
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"Kernel reset timed out: {e}") from e

    async def close(self):
        """Save the notebook and close the interpreter or container."""
        nbformat.write(self.nb, self.work_dir / "notebook.ipynb")

        if self.use_ray and self.use_enroot:
            if self.kernel_container is not None:
                close_ref = self.kernel_container.close.remote()  # type: ignore[attr-defined]
                await close_ref
                self.kernel_container = None

        elif self.use_docker or self.use_enroot:
            if self._container_port is not None:
                async with used_ports_lock:
                    _USED_PORTS.discard(self._container_port)

            if self._http_client is not None:
                with contextlib.suppress(Exception):
                    await self._http_client.aclose()
                self._http_client = None

            if self._container is not None:
                try:
                    await self._container.stop()
                    await self._container.delete()
                except Exception as e:
                    logger.warning(f"Failed to stop/delete container: {e}")
                self._container = None

            if self._docker_client is not None:
                await self._docker_client.close()
                self._docker_client = None

            if self._enroot_proc is not None:
                await _kill_process_group(self._enroot_proc, label=self._enroot_label())
                self._enroot_proc = None

            await self._close_container_log()

        elif self.interpreter is not None:
            await self.interpreter.close()

        if self.save_dir is not None and self.work_dir.exists():
            self.save_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.save_dir.exists():
                try:
                    cleanup_path = _detach_dir_for_cleanup(self.save_dir)
                except Exception as e:
                    logger.warning("Failed to detach existing save_dir %s for cleanup: %s", self.save_dir, e)
                else:
                    if cleanup_path is not None:
                        logger.warning("Detached existing save_dir %s to %s for background cleanup", self.save_dir, cleanup_path)
                        _schedule_dir_cleanup(cleanup_path)
            try:
                self.work_dir.replace(self.save_dir)
            except Exception as e:
                logger.warning("Failed to move work_dir %s to save_dir %s: %s", self.work_dir, self.save_dir, e)
        elif self.work_dir.exists():
            try:
                cleanup_path = _detach_dir_for_cleanup(self.work_dir)
            except Exception as e:
                logger.warning("Failed to detach workspace %s for background cleanup: %s", self.work_dir, e)
            else:
                if cleanup_path is not None:
                    logger.warning("Detached workspace %s to %s for background cleanup", self.work_dir, cleanup_path)
                    _schedule_dir_cleanup(cleanup_path)

    def _add_cell(self, code: str, result: "ExecutionResult") -> int:
        """Add a new code cell to the notebook with execution results.

        Args:
            code: The code that was executed
            result: The execution result

        Returns:
            The cell index of the added cell
        """
        self._execution_count += 1

        cell = nbformat.v4.new_code_cell(
            source=code,
            outputs=result.notebook_outputs,
            execution_count=self._execution_count,
        )

        self.nb.cells.append(cell)
        cell_idx = len(self.nb.cells) - 1

        # Track errors if any
        if result.error_occurred:
            error_msg = result.get_error_message()
            if error_msg:
                self.notebook_runtime_errors.append(f"Cell {self._execution_count}: {error_msg}")

        return cell_idx

    def _update_cell(self, idx: int, code: str, result: "ExecutionResult") -> None:
        """Update an existing cell's source and outputs.

        Args:
            idx: The cell index to update
            code: The new code
            result: The execution result
        """
        cell = self.nb.cells[idx]
        cell.source = code
        cell.outputs = result.notebook_outputs

        # Update error tracking - remove old error for this cell, add new if any
        if result.error_occurred:
            error_msg = result.get_error_message()
            if error_msg:
                # Remove any existing error for this cell
                self.notebook_runtime_errors = [
                    err for err in self.notebook_runtime_errors if not err.startswith(f"Cell {idx + 1}:")
                ]
                self.notebook_runtime_errors.append(f"Cell {idx + 1}: {error_msg}")

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of execution history and current state.

        Works in both local and Docker modes.
        """
        if not self.use_docker and self.interpreter is not None:
            return self.interpreter.get_execution_summary()

        # For Docker mode, build summary from notebook state
        error_count = len(self.notebook_runtime_errors)
        recent_errors = self.notebook_runtime_errors[-3:] if self.notebook_runtime_errors else []

        return {
            "total_executions": self._execution_count,
            "error_count": error_count,
            "recent_errors": recent_errors,
            "last_execution": None,  # Not tracked in Docker mode
            "is_ready": self._http_client is not None,
            "language": self.language.value,
            "work_dir": str(self.work_dir),
        }

    async def execute_and_add_cell(
        self,
        code: str,
        cell_idx: int | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
        req_uuid: str = "",
    ) -> tuple[ExecutionResult, int]:
        """Execute code and atomically update notebook.

        Args:
            code: Code to execute
            cell_idx: Cell index to update (None = append new cell)
            timeout: Optional execution timeout

        Returns:
            Tuple of (ExecutionResult, actual_cell_index)
        """
        if self.safe_execute:
            block_reason = check_code_safety(code, self.language)
            if block_reason is not None:
                logger.warning("Blocked code execution in execute_and_add_cell: %s", block_reason)
                error_output = nbformat.v4.new_output(
                    output_type="error",
                    ename="SecurityError",
                    evalue=block_reason,
                    traceback=[f"SecurityError: {block_reason}"],
                )
                result = ExecutionResult(
                    notebook_outputs=[error_output],
                    error_occurred=True,
                    execution_time=0.0,
                )
                if cell_idx is None or cell_idx >= len(self.nb.cells):
                    actual_idx = self._add_cell(code, result)
                else:
                    self._update_cell(cell_idx, code, result)
                    actual_idx = cell_idx
                return result, actual_idx
        elif "unsafe_execution" not in _warned_unsafe_execution:
            logger.warning(
                "Running code sandbox without safety filter, may result in destructive code running on the node"
            )
            _warned_unsafe_execution.add("unsafe_execution")

        if self.use_ray and self.use_enroot:
            result_ref = self.kernel_container._execute_via_http.remote(code, timeout, req_uuid=req_uuid)  # type: ignore[union-attr]
            try:
                result = await self._await_ray_ref(
                    result_ref,
                    timeout=timeout,
                    req_uuid=req_uuid,
                    operation="_execute_via_http",
                )
            except Exception:
                logger.exception("req %s failed waiting for ray execute_via_http", req_uuid)
                raise
        elif self.use_docker or self.use_enroot:
            result = await self._execute_via_http(code, timeout)
        else:
            assert self.interpreter is not None
            result = await self.interpreter.execute_code(code, timeout)

        if cell_idx is None or cell_idx >= len(self.nb.cells):
            actual_idx = self._add_cell(code, result)
        else:
            self._update_cell(cell_idx, code, result)
            actual_idx = cell_idx

        return result, actual_idx


class InterpreterEnvConfig(BaseModel):
    """Configuration for preparing the InterpreterEnv during task creation."""

    language: NBLanguage = Field(default=NBLanguage.PYTHON)
    prompting_config: PromptingConfig = Field(default_factory=PromptingConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    max_steps: int = cfg.AGENT_MAX_STEPS
    use_ray: bool = False
    use_docker: bool = cfg.USE_DOCKER
    use_enroot: bool = False
    container_sqsh_path: Path | None = None
    normalize_reward: bool = True
    enable_faithfulness_gate: bool = False
    faithfulness_mode: Literal["off", "binary", "shadow", "hybrid"] = "off"
    wager_mode: Literal["off", "shadow", "active"] = "off"
    wager_beta: float = WAGER_BETA_DEFAULT
    wager_gamma: float = WAGER_GAMMA_DEFAULT
    cell_timeout_override_mode: Literal["off", "on"] = "off"
    cell_timeout_min: float = 60.0
    cell_timeout_max: float = 1200.0

    @model_validator(mode="after")
    def _migrate_enable_faithfulness_gate(self) -> "InterpreterEnvConfig":
        if self.enable_faithfulness_gate and self.faithfulness_mode == "off":
            warnings.warn(
                "enable_faithfulness_gate=True is deprecated; use faithfulness_mode='binary' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.faithfulness_mode = "binary"
        return self

    @model_validator(mode="after")
    def _validate_wager_requires_gate(self) -> "InterpreterEnvConfig":
        if self.wager_mode != "off" and self.faithfulness_mode == "off":
            raise ValueError(
                f"wager_mode={self.wager_mode!r} requires faithfulness_mode "
                "∈ {'shadow', 'hybrid'}; got 'off'. Wager uses the gate's "
                "correct signal; it cannot operate standalone."
            )
        return self


class InterpreterEnv(Environment[InterpreterEnvState]):
    """Standalone environment for code execution and data analysis.

    This environment provides direct code execution via run_cell() without
    requiring notebook file I/O. It maintains an in-memory notebook for
    trajectory tracking and state export.
    """

    def __init__(
        self,
        *,
        problem: ProblemInstance,
        work_dir: Path,
        rubric_model: LiteLLMModel | None = None,
        config: InterpreterEnvConfig | None = None,
        input_data: list[dict[str, str | int | None]] | None = None,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
        include_env_state_msg: bool = False,
        save_dir: Path | None = None,
    ):
        self.config = config or InterpreterEnvConfig()
        self.work_dir = work_dir
        self.rubric_model = rubric_model
        self.done = False
        self.problem = problem
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}
        self.save_dir = save_dir

        # Execution config for timeouts and capabilities
        self.execution_config = self.config.execution_config
        self.execution_timeout = self.execution_config.cell_execution_timeout
        self.max_steps = self.config.max_steps

        self.input_data = input_data
        self.output_data: list[dict[str, str | int]] = []
        self.logger = logger
        self.start_time: float | None = None
        self.step_count = 0
        self.include_env_state_msg = include_env_state_msg
        self.state: InterpreterEnvState
        # prompting_config is set during reset() after language resolution
        self.prompting_config: PromptingConfig

        if self.score_info_path.exists():
            self.score_info_path.unlink()

        nb_path = self.work_dir / "notebook.ipynb"
        if nb_path.exists():
            nb_path.unlink()

    @property
    def language(self) -> NBLanguage:
        return self.config.language

    async def close(self) -> None:
        """Save notebook, shut down interpreter/container."""
        self.logger.info("Closing environment")
        await self.state.close()

    async def reset(self) -> tuple[Messages, list[Tool]]:
        """Reset the environment and prepare for execution."""
        reset_id = getattr(self, "_nemo_env_id", "?")[:8]
        logger.warning("[reset:%s] building state for work_dir=%s", reset_id, self.work_dir)

        # Format environment capabilities with job_timeout
        env_capabilities = self.execution_config.environment_capabilities_prompt.format(
            job_timeout=self.execution_config.job_timeout
        )

        self.prompting_config = self.config.prompting_config.interpolate(
            language=self.language.value.capitalize(),
            environment_capabilities=env_capabilities,
            job_timeout=self.execution_config.job_timeout,
        )

        # Use kernel environment paths for isolated execution
        kernel_env_path = Path(cfg.KERNEL_ENV_PATH)
        kernel_site_packages = kernel_env_path / "lib" / "python3.12" / "site-packages"

        self.state = InterpreterEnvState(
            work_dir=self.work_dir,
            language=self.language,
            execution_timeout=self.execution_timeout,
            safe_execute=self.execution_config.safe_execute,
            use_host_env_vars=self.use_host_env_vars,
            extra_envs={
                # Point to kernel environment's site-packages
                "PYTHONPATH": str(kernel_site_packages),
                # Include kernel environment bin in PATH
                "PATH": (str(kernel_env_path / "bin") + os.pathsep + os.environ.get("PATH", "")),
                # R library path for user-installed packages
                "R_LIBS_USER": str(kernel_env_path / "lib" / "R" / "library"),
            }
            | self.extra_envs,
            save_dir=self.save_dir,
            use_docker=self.config.use_docker,
            use_enroot=self.config.use_enroot,
            container_sqsh_path=self.config.container_sqsh_path,
            sandbox_memory_limit_mb=self.execution_config.sandbox_memory_limit_mb,
            sandbox_max_pids=self.execution_config.sandbox_max_pids,
        )
        logger.warning("[reset:%s] starting container", reset_id)
        await self.state.start()
        logger.warning("[reset:%s] container started, building tools", reset_id)

        # Record start time for timeout tracking
        self.start_time = time.perf_counter()

        messages = []
        if self.prompting_config.system_prompt:
            messages.append(Message(role="system", content=self.prompting_config.system_prompt))

        self._filesystem_tool = FilesystemTool(self.work_dir)

        # Reproducibility: wager_mode='off' runs expose the IDENTICAL submit_answer
        # schema they saw before the wager patch landed. Only wager_mode ∈ {shadow,
        # active} builds the closure that adds a `confidence` field. The closure is
        # renamed to "submit_answer" so the tool-call name the policy sees is stable
        # across modes.
        if self.config.wager_mode == "off":
            submit_tool = Tool.from_function(self.submit_answer)
        else:
            base_submit = self.submit_answer  # bound method; captured in closure

            async def _submit_answer_with_wager(answer: str, confidence: float = 0.0) -> str:
                """Submit your response to the research question.

                Note that this tool may only be called once and ends the episode.

                Args:
                    answer: Your final response to the research question.
                    confidence: A wager value in [0.0, 1.0] reflecting how strongly
                        your work supports the answer. 0.0 = fully hedged (you
                        submit an answer but aren't willing to wager on it; full
                        credit if correct, no extra cost if wrong). Larger values
                        stake more on the answer: larger bonus if correct, larger
                        reduction of procedural credit if wrong. 1.0 = maximum
                        wager. Choose the value that reflects how strongly your
                        work supports the answer. Wagering high on answers you
                        cannot defend will cost more than it gains; wagering low
                        on answers you can defend leaves value on the table. If
                        you are unsure, the safe default is a low confidence.
                """
                self.state.wager = clamp_confidence(confidence)
                return await base_submit(answer)

            _submit_answer_with_wager.__name__ = "submit_answer"
            submit_tool = Tool.from_function(_submit_answer_with_wager)

        # Same reproducibility principle for run_cell: when the cell-timeout
        # override is off, the exposed tool is the plain `self.run_cell` whose
        # schema is identical to pre-patch. When on, the closure adds a
        # `timeout_seconds` kwarg clamped to [cell_timeout_min, cell_timeout_max]
        # (default [60, 1200]) and delegates to `_run_cell_with_cap` with the
        # clamped cap.
        if self.config.cell_timeout_override_mode == "off":
            run_cell_tool = Tool.from_function(self.run_cell)
        else:
            ct_min = float(self.config.cell_timeout_min)
            ct_max = float(self.config.cell_timeout_max)
            env_default_cap = float(self.execution_timeout)
            run_cell_impl = self._run_cell_with_cap  # bound method captured in closure

            async def _run_cell_with_timeout(
                code: str,
                idx: int | None = None,
                timeout_seconds: float | None = None,
            ) -> Message | str | list[dict[str, Any]]:
                """Run code in a notebook cell and return the execution output.

                This method allows running code in a new cell (append) or re-running
                an existing cell with updated code.

                Usage Examples:
                    run_cell("print('Hello, world!')")
                    run_cell("print('Hello, world!')", idx=0)
                    run_cell("slow_op()", timeout_seconds=900)

                Error Recovery:
                    When a cell fails with an error, you MUST fix it by calling
                    run_cell with the corrected code and the SAME idx as the failed
                    cell:

                    run_cell("corrected_code", idx=3)  # Fix error in Cell #3

                    The cell number is shown in the output prefix (e.g., "[Cell #3]").
                    Do NOT create a new cell to fix an error - always edit the
                    failed cell.

                Args:
                    code: Code to execute.
                    idx: Cell index to run. If None or >= len(cells), appends a new
                        cell. If provided, updates and re-runs the existing cell at
                        that index. Use this to fix errors in existing cells.
                    timeout_seconds: Optional per-cell execution cap, in seconds.
                        Use this if you expect a long-running cell (e.g., a large
                        DE analysis, a permutation test) to exceed the default cap.
                        Values below the minimum (60s) or above the maximum (1200s)
                        are silently clamped. Leave unset for most cells to use the
                        default cap. A cell that hits its cap returns a TimeoutError
                        output just like any other timeout.

                Returns:
                    Message with multimodal content if images present, otherwise
                    string. The response includes the cell number (e.g.,
                    "[Cell #0] output...").

                Related tools:
                    `reset_kernel` and `list_dir` are separate tools, NOT Python
                    symbols in the kernel namespace. Do NOT write `reset_kernel()`
                    or `list_dir()` inside a `run_cell` call — invoke them as
                    separate tool calls instead. A `reset_kernel` tool call after
                    a `TimeoutError` is the supported way to recover from a locked
                    kernel.

                Installing packages:
                    Install commands (`pip install`, `conda install`, `apt-get install`,
                    `BiocManager::install`, `install.packages`) are intercepted: if
                    the package is already present, the call returns quickly with a
                    "[pre-installed]" message. To force a fresh install of a
                    specific version, use the installer's native force flag
                    (`pip install --force-reinstall`, `BiocManager::install(..., force=TRUE)`,
                    `conda install --force-reinstall`, `apt-get install --reinstall`).
                    Version pins without a force flag are treated as informational;
                    the existing install is used and a "[version-mismatch]" message
                    is printed.
                """
                if timeout_seconds is None:
                    cap = env_default_cap
                else:
                    try:
                        cap = float(timeout_seconds)
                    except (TypeError, ValueError):
                        cap = env_default_cap
                    cap = max(ct_min, min(ct_max, cap))
                    self.state.cell_timeout_override_requests.append(cap)
                return await run_cell_impl(code, idx=idx, timeout_cap=cap)

            _run_cell_with_timeout.__name__ = "run_cell"
            run_cell_tool = Tool.from_function(_run_cell_with_timeout)

        self.tools = [
            run_cell_tool,
            Tool.from_function(self.reset_kernel),
            submit_tool,
            Tool.from_function(self.list_dir),
        ]

        messages.append(
            Message(
                content=HYPOTHESIS_TASK_DESC.format(
                    language=self.language.value.capitalize(),
                    hypothesis=self.problem.hypothesis,
                    protocol=self.problem.protocol,
                )
            )
        )

        if self.include_env_state_msg:
            messages.append(self.get_env_state_msg())

        # Always show initial directory listing (with truncation protection)
        messages.append(Message(content=await self.list_dir()))

        logger.warning("[reset:%s] reset fully complete", reset_id)
        return messages, self.tools

    async def step(self, action: ToolRequestMessage) -> tuple[Messages, float, bool, bool]:
        """Execute a step in the environment."""
        self.step_count += 1
        obs = cast(
            Messages,
            await self.exec_tool_calls(action, concurrency=False, handle_tool_exc=True),
        )

        obs = [*obs]
        if self.include_env_state_msg:
            obs.append(self.get_env_state_msg())

        time_msg = self.get_time_management_message()
        if time_msg is not None:
            obs.append(time_msg)

        if self.step_count >= (self.max_steps - 1):
            obs.append(Message(content=cfg.FORCE_MSG))

        self.state.actions.append(str(action))
        reward = self.state.score if self.state.done else 0.0
        return obs, reward, self.state.done, False

    # ========== Tools ==========

    async def list_dir(
        self,
        directory: str = ".",
        max_files: int = 20,
        show_hidden: bool = False,
    ) -> str:
        """List contents of a directory with truncation protection.

        Recursively lists files in a directory, with built-in protection against
        overwhelming the context with too many files. Use this tool instead of
        writing code to list directories to avoid context bloat. This is a tool
        — do NOT call it as code (e.g., `list_dir()`) inside a `run_cell` call;
        invoke it as a separate tool call.

        Usage Examples:
            list_dir()                      # List working directory
            list_dir("data/")               # List specific folder
            list_dir(max_files=50)          # Show more files
            list_dir(show_hidden=True)      # Include hidden files

        Args:
            directory: Directory path to list (default: current working directory)
            max_files: Maximum number of files to display (default: 20)
            show_hidden: Whether to show hidden files starting with '.' (default: False)
        """
        if self.state.use_ray and self.state.use_enroot:
            if self.state.kernel_container is None:
                return "Error listing directory: node-local workspace is unavailable"

            list_dir_uuid = str(uuid.uuid4())
            list_ref = self.state.kernel_container._list_dir_on_node.remote(  # type: ignore[attr-defined]
                directory=directory,
                max_files=max_files,
                show_hidden=show_hidden,
                req_uuid=list_dir_uuid,
            )
            result = await self.state._await_ray_ref(
                list_ref,
                timeout=_LIST_DIR_RAY_TIMEOUT,
                req_uuid=list_dir_uuid,
                operation="list_dir",
                max_retries=2,
            )
            return cast(str, result)

        return self._filesystem_tool.list_dir(directory, max_files, show_hidden)

    async def run_cell(
        self,
        code: str,
        idx: int | None = None,
    ) -> Message | str | list[dict[str, Any]]:
        """Run code in a notebook cell and return the execution output.

        This method allows running code in a new cell (append) or re-running
        an existing cell with updated code.

        Usage Examples:
            run_cell("print('Hello, world!')")           # Run code in new cell
            run_cell("print('Hello, world!')", idx=0)    # Run code in existing cell at index 0

        Error Recovery:
            When a cell fails with an error, you MUST fix it by calling run_cell
            with the corrected code and the SAME idx as the failed cell:

            run_cell("corrected_code", idx=3)  # Fix error in Cell #3

            The cell number is shown in the output prefix (e.g., "[Cell #3]").
            Do NOT create a new cell to fix an error - always edit the failed cell.

        Args:
            code: Code to execute
            idx: Cell index to run. If None or >= len(cells), appends a new cell.
                If provided, updates and re-runs the existing cell at that index.
                Use this to fix errors in existing cells.

        Returns:
            Message with multimodal content if images present, otherwise string.
            The response includes the cell number (e.g., "[Cell #0] output...").

        Related tools:
            `reset_kernel` and `list_dir` are separate tools, NOT Python symbols
            in the kernel namespace. Do NOT write `reset_kernel()` or `list_dir()`
            inside a `run_cell` call — invoke them as separate tool calls instead.
            A `reset_kernel` tool call after a `TimeoutError` is the supported way
            to recover from a locked kernel.

        Installing packages:
            Install commands (`pip install`, `conda install`, `apt-get install`,
            `BiocManager::install`, `install.packages`) are intercepted: if the
            package is already present, the call returns quickly with a
            "[pre-installed]" message. To force a fresh install of a specific
            version, use the installer's native force flag
            (`pip install --force-reinstall`, `BiocManager::install(..., force=TRUE)`,
            `conda install --force-reinstall`, `apt-get install --reinstall`).
            Version pins without a force flag are treated as informational; the
            existing install is used and a "[version-mismatch]" message is printed.
        """
        return await self._run_cell_with_cap(code, idx=idx, timeout_cap=self.execution_timeout)

    async def _run_cell_with_cap(
        self,
        code: str,
        idx: int | None = None,
        timeout_cap: float | None = None,
    ) -> Message | str | list[dict[str, Any]]:
        """Implementation shared by `run_cell` and the timeout-override closure.

        `timeout_cap` caps the per-cell execution time. `run_cell` passes
        `self.execution_timeout` (the config default). The override closure
        passes the model's clamped requested value.
        """
        if timeout_cap is None:
            timeout_cap = self.execution_timeout

        run_cell_uuid = str(uuid.uuid4())
        remaining_seconds = self.get_remaining_time()

        if remaining_seconds <= self.execution_config.force_submit_threshold:
            self.logger.warning(
                f"Refusing cell execution with {remaining_seconds:.1f}s remaining "
                f"(force threshold: {self.execution_config.force_submit_threshold}s)"
            )
            return cfg.FORCE_MSG

        dynamic_timeout = remaining_seconds - self.execution_config.force_submit_threshold
        effective_timeout = min(timeout_cap, dynamic_timeout)

        self.logger.info(
            f"Cell execution with dynamic timeout: {effective_timeout:.1f}s "
            f"(remaining: {remaining_seconds:.1f}s, cap: {timeout_cap}s)"
        )

        # Parse idx (handle string input from LLM)
        cell_idx: int | None = None
        if idx is not None:
            try:
                cell_idx = int(idx)
            except (ValueError, TypeError):
                cell_idx = None

        # Execute code and update notebook atomically
        result, actual_cell_idx = await self.state.execute_and_add_cell(
            code, cell_idx=cell_idx, timeout=effective_timeout, req_uuid=run_cell_uuid,
        )

        # Build response with cell number
        cell_info = f"[Cell #{actual_cell_idx}] "

        if result.has_images():
            # Format images as data URLs for Message. Aviary validates the image
            # via PIL on construction; a figure with >178M pixels trips
            # PIL.Image.DecompressionBombError. Reshape that to a cell-level
            # error matching the `[Cell #N] Error: ...` shape the model sees for
            # every other kernel error, so the framework's generic
            # "Encountered exception during tool call:" wrapper doesn't fire.
            try:
                image_urls = [f"data:{mime_type};base64,{base64_data}" for mime_type, base64_data in result.get_images()]
                return Message.create_message(
                    role="tool",
                    text=cell_info + result.get_truncated_text(),
                    images=cast(list[np.ndarray | str], image_urls),
                )
            except Exception as e:
                if type(e).__name__ != "DecompressionBombError" and "DecompressionBombError" not in str(e):
                    raise
                self.logger.warning(
                    "DecompressionBombError on image output for cell %d: %s",
                    actual_cell_idx, e,
                )
                hint = "Hint: reduce the figure size or dpi on plt.savefig / fig.savefig."
                # Replace the cell's image output with an error output so the
                # notebook state is consistent with what the model sees in text.
                if 0 <= actual_cell_idx < len(self.state.nb.cells):
                    self.state.nb.cells[actual_cell_idx].outputs = [
                        nbformat.v4.new_output(
                            output_type="error",
                            ename="DecompressionBombError",
                            evalue=str(e),
                            traceback=[f"DecompressionBombError: {e}", hint],
                        )
                    ]
                return (
                    f"{cell_info}Error: DecompressionBombError\n"
                    f"Message: {e}\n"
                    f"Traceback: {hint}"
                )

        return cell_info + result.get_truncated_text()

    async def reset_kernel(self) -> str:
        """Reset the kernel to a clean state.

        This clears all variables and execution state. This is a tool — do NOT
        call it as code (e.g., `reset_kernel()`) inside a `run_cell` call;
        invoke it as a separate tool call. After a `TimeoutError` in a prior
        cell, this is the supported way to unlock a frozen kernel.
        """
        if self.state.use_ray and self.state.kernel_container is not None:
            reset_ref = self.state.kernel_container._reset_via_http.remote()  # type: ignore[attr-defined]
            await reset_ref
        elif self.state.use_docker or self.state.use_enroot:
            await self.state._reset_via_http()
        else:
            assert self.state.interpreter is not None
            await self.state.interpreter.reset()

        # Reset notebook state to match kernel reset
        self.state.nb = nbformat.v4.new_notebook()
        self.state.nb.metadata.kernelspec = self.state.language.make_kernelspec()
        self.state.notebook_runtime_errors = []
        self.state._execution_count = 0

        return "Kernel reset successfully."

    @property
    def score_info_path(self) -> Path:
        return self.work_dir / "score_info.json"

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(ValueError))
    async def _evaluate_rubric(self, solution: str, nb_content: str) -> int:
        """Evaluate the solution against the rubric. Returns raw integer score."""
        assert self.rubric_model is not None

        prompt = self.state.score_metadata["prompt"] = RUBRIC_SCORE_PROMPT.format(
            hypothesis=self.problem.hypothesis,
            accepted=self.problem.accepted,
            rubric=self.problem.rubric,
            notebook=nb_content,
            proposed_solution=solution,
        )

        resp = await self.rubric_model.call_single(prompt, timeout=3 * 60)
        if not resp.text:
            raise ValueError("No response from rubric model")
        self.state.score_metadata["response"] = resp.text

        try:
            return int(resp.text.split("<score>")[1].split("</score>")[0])
        except Exception as e:
            raise ValueError("Failed to parse score from response") from e

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(ValueError))
    async def _evaluate_faithfulness_gate(self, solution: str, nb_content: str) -> bool:
        """Evaluate whether the conclusion is supported by notebook state. Returns True if faithful."""
        assert self.rubric_model is not None

        additional = ""
        if self.problem.faithfulness_rubric:
            additional = f"Additional task-specific criteria:\n{self.problem.faithfulness_rubric}"

        prompt = FAITHFULNESS_GATE_PROMPT.format(
            hypothesis=self.problem.hypothesis,
            notebook=nb_content,
            proposed_solution=solution,
            additional_criteria=additional,
        )
        self.state.faithfulness_metadata["prompt"] = prompt

        resp = await self.rubric_model.call_single(prompt, timeout=3 * 60)
        if not resp.text:
            raise ValueError("No response from faithfulness gate")
        self.state.faithfulness_metadata["response"] = resp.text

        if "<verdict>PASS</verdict>" in resp.text:
            return True
        if "<verdict>FAIL</verdict>" in resp.text:
            return False
        raise ValueError("Failed to parse verdict from faithfulness gate response")

    async def _evaluate_hybrid_gate(self, solution: str, nb_content: str) -> dict[str, Any]:
        """Per-item hybrid faithfulness judge. Fail-open on any error.

        The judge reads the raw rubric text and is responsible for numbering
        items and echoing each item's weight inline. We do NOT parse the
        rubric on the client — formats in the dataset are too varied for
        regex to handle reliably.

        Returns a dict with parse_hybrid_response fields plus:
            item_weights, prompt, response, judge_call_failed, parse_failed,
            weights_mismatch (True if sum(weights) != problem.max_score).
        """
        assert self.rubric_model is not None

        prompt = HYBRID_GATE_PROMPT.format(
            hypothesis=self.problem.hypothesis,
            notebook=nb_content,
            proposed_solution=solution,
            rubric=self.problem.rubric,
        )

        try:
            resp = await self.rubric_model.call_single(prompt, timeout=3 * 60)
        except Exception as e:
            self.logger.exception("Hybrid judge call failed — failing open")
            return {
                "per_item": [], "proc_present_pts": 0, "proc_max_pts": 0,
                "concl_present_pts": 0, "concl_max_pts": 0,
                "item_weights": [], "prompt": prompt, "response": "",
                "judge_call_failed": True, "parse_failed": False,
                "weights_mismatch": False, "error": repr(e),
            }

        if not resp.text:
            self.logger.warning("Hybrid judge returned empty response — failing open")
            return {
                "per_item": [], "proc_present_pts": 0, "proc_max_pts": 0,
                "concl_present_pts": 0, "concl_max_pts": 0,
                "item_weights": [], "prompt": prompt, "response": "",
                "judge_call_failed": True, "parse_failed": False,
                "weights_mismatch": False, "error": "empty response",
            }

        parsed = parse_hybrid_response(resp.text)

        # Build the 1-indexed weight list the rubric-award synthesis needs.
        # Items may come out of order; fill gaps with 0 (treated as "no such item").
        item_weights: list[int] = []
        if parsed["per_item"]:
            max_idx = max(idx for idx, _, _, _ in parsed["per_item"])
            by_idx = {idx: w for idx, w, _, _ in parsed["per_item"]}
            item_weights = [by_idx.get(i, 0) for i in range(1, max_idx + 1)]

        total_weight = sum(item_weights)
        weights_mismatch = bool(item_weights) and total_weight != self.problem.max_score
        if weights_mismatch:
            self.logger.warning(
                "Hybrid judge weights sum to %d but problem.max_score=%d — failing open on scoring",
                total_weight, self.problem.max_score,
            )

        return {
            **parsed,
            "item_weights": item_weights,
            "prompt": prompt,
            "response": resp.text,
            "judge_call_failed": False,
            "parse_failed": not parsed["per_item"],
            "weights_mismatch": weights_mismatch,
        }

    async def _score_solution(self, solution: str) -> bool:
        assert self.rubric_model is not None
        nb_content, _ = view_notebook(self.state.nb.cells, self.language.value)

        mode = self.config.faithfulness_mode
        faith_result: dict[str, Any] | None = None

        if mode == "binary":
            rubric_task = asyncio.ensure_future(self._evaluate_rubric(solution, nb_content))
            gate_task = asyncio.ensure_future(self._evaluate_faithfulness_gate(solution, nb_content))
            try:
                raw_score = await rubric_task
            except Exception:
                gate_task.cancel()
                raise
            try:
                self.state.faithfulness_passed = await gate_task
            except Exception:
                self.logger.exception("Binary faithfulness gate failed — falling back to rubric-only scoring")
                self.state.faithfulness_passed = None

        elif mode in ("shadow", "hybrid"):
            rubric_task = asyncio.ensure_future(self._evaluate_rubric(solution, nb_content))
            hybrid_task = asyncio.ensure_future(self._evaluate_hybrid_gate(solution, nb_content))
            try:
                raw_score = await rubric_task
            except Exception:
                hybrid_task.cancel()
                raise
            try:
                faith_result = await hybrid_task
            except Exception:
                self.logger.exception("Hybrid gate failed — failing open to rubric-only scoring")
                faith_result = {
                    "per_item": [], "item_weights": [],
                    "judge_call_failed": True, "parse_failed": False,
                    "weights_mismatch": False,
                }

        else:  # "off"
            raw_score = await self._evaluate_rubric(solution, nb_content)

        try:
            self.state.raw_score = raw_score
            correct = raw_score == self.problem.max_score
            rubric_score = raw_score / self.problem.max_score if self.config.normalize_reward else raw_score
            rubric_score = max(
                0.0,
                min(1.0 if self.config.normalize_reward else self.problem.max_score, rubric_score),
            )
            self.state.rubric_reward_raw = float(rubric_score)

            if mode == "binary":
                if self.state.faithfulness_passed is False:
                    self.logger.info("Binary faithfulness gate FAILED — zeroing reward")
                    applied = 0.0
                else:
                    applied = rubric_score

            elif mode in ("shadow", "hybrid"):
                assert faith_result is not None
                item_weights = faith_result.get("item_weights", [])
                judge_broken = (
                    faith_result.get("judge_call_failed", False)
                    or faith_result.get("parse_failed", False)
                    or faith_result.get("weights_mismatch", False)
                )
                if judge_broken or not item_weights:
                    # Fail-open: hybrid reward equals rubric reward, no items stripped.
                    self.state.hybrid_reward_value = float(rubric_score)
                    self.state.hybrid_metadata = {**faith_result, "strip_reason": "judge_unavailable"}
                else:
                    rubric_awards = synthesize_per_item_awards(raw_score, item_weights)
                    hybrid_value, breakdown = hybrid_reward(rubric_awards, faith_result, self.problem.max_score)
                    self.state.hybrid_reward_value = float(hybrid_value)
                    self.state.hybrid_metadata = {**faith_result, **breakdown}
                applied = rubric_score if mode == "shadow" else self.state.hybrid_reward_value

            else:  # "off"
                applied = rubric_score

            # Scheme D: wager-shaped reward. Runs AFTER faithfulness-mode scoring
            # and consumes the gate's correct signal via hybrid_metadata. Shadow
            # mode computes but does not apply. Active mode applies and relaxes
            # the upper clamp so the upside bonus can lift reward above 1.0.
            wager_mode = self.config.wager_mode
            if wager_mode != "off":
                hm = self.state.hybrid_metadata or {}
                proc_max = int(hm.get("proc_max_pts", 0))
                concl_max_hm = int(hm.get("concl_max_pts", 0))
                proc_credited = float(hm.get("proc_pts_credited", 0.0))
                concl_credited = float(hm.get("concl_pts_credited", 0.0))
                gate_unavailable = (
                    (proc_max + concl_max_hm) <= 0
                    or hm.get("strip_reason") == "judge_unavailable"
                )
                if gate_unavailable:
                    self.state.wager_reward_shadow = float(applied)
                    self.state.wager_metadata = {
                        "skipped_reason": "gate_unavailable",
                        "wager": self.state.wager,
                    }
                else:
                    gate_correct = concl_credited >= concl_max_hm and concl_max_hm > 0
                    wager_value, wager_breakdown = score_with_wager(
                        proc_credit=proc_credited,
                        proc_max=proc_max,
                        concl_credit=concl_credited,
                        concl_max=concl_max_hm,
                        correct=gate_correct,
                        wager=self.state.wager,
                        beta=self.config.wager_beta,
                        gamma=self.config.wager_gamma,
                    )
                    self.state.wager_reward_shadow = float(wager_value)
                    self.state.wager_metadata = wager_breakdown

                if wager_mode == "active":
                    applied = self.state.wager_reward_shadow

            # Upper clamp relaxes only when wager is active — the bonus can
            # legitimately lift reward above 1.0, and downstream (NeMo-RL
            # advantage computation) does not re-clamp.
            if wager_mode == "active":
                applied = max(0.0, applied)
            # In off/shadow the existing clamp is preserved implicitly (the
            # computed `applied` already came out of rubric_score/hybrid paths
            # which were clamped above).

            self.state.score = applied
            self.state.total_reward += applied
            return correct

        finally:
            score_info = {
                **self.state.score_metadata,
                "score": self.state.score,
                "raw_score": self.state.raw_score,
                "max_score": self.problem.max_score,
                "faithfulness_passed": self.state.faithfulness_passed,
                "faithfulness_mode": self.config.faithfulness_mode,
                "rubric_reward_raw": self.state.rubric_reward_raw,
                "hybrid_reward_value": self.state.hybrid_reward_value,
                "hybrid_metadata": self.state.hybrid_metadata,
                "wager_mode": self.config.wager_mode,
                "wager": self.state.wager,
                "wager_reward_shadow": self.state.wager_reward_shadow,
                "wager_metadata": self.state.wager_metadata,
            }
            with self.score_info_path.open("w") as f:
                json.dump(score_info, f, indent=2, default=str)

            self.logger.info(f"Received solution ({self.state.raw_score}/{self.problem.max_score}): {solution!r}.")

    async def submit_answer(self, answer: str) -> str:
        """Submit your response to the research question.

        Note that this tool may only be called once and ends the episode.

        Args:
            answer: Your final response to the research question
        """
        if self.state.done:
            return "Episode already finished."

        self.state.answer = answer
        self.state.done = True

        if self.rubric_model is None:
            self.logger.warning("No rubric_model configured, skipping scoring")
            return answer

        correct = await self._score_solution(answer)
        return CORRECT_MSG if correct else INCORRECT_MSG

    # ========== Time Management ==========

    def get_remaining_time(self) -> int:
        """Get remaining execution time in seconds."""
        elapsed = 0 if self.start_time is None else time.perf_counter() - self.start_time
        return int(self.execution_config.job_timeout - elapsed)

    def get_time_management_message(self) -> Message | None:
        """Get a time management message if thresholds are reached."""
        remaining = self.get_remaining_time()

        if remaining <= self.execution_config.force_submit_threshold:
            self.logger.warning(
                f"Forcing answer submission with {remaining}s remaining "
                f"(threshold: {self.execution_config.force_submit_threshold}s)"
            )
            return Message(content=cfg.FORCE_MSG.format(remaining=remaining))

        if remaining <= self.execution_config.warn_submit_threshold:
            self.logger.info(
                f"Warning agent about timeout with {remaining}s remaining "
                f"(threshold: {self.execution_config.warn_submit_threshold}s)"
            )
            return Message(content=cfg.WARN_MSG.format(remaining=remaining))

        return None

    # ========== State Export ==========

    def export_frame(self) -> Frame:
        """Export the current environment state as a Frame."""
        return Frame(
            state={
                "last_action": self.state.actions[-1] if self.state.actions else None,
                "answer": self.state.answer,
                "done": self.state.done,
                "nb_state": self.state.nb,
                "nb_runtime_errors": self.state.notebook_runtime_errors,
                "raw_score": self.state.raw_score,
                "score": self.state.score,
                "score_metadata": self.state.score_metadata,
                "total_reward": self.state.total_reward,
                "faithfulness_passed": self.state.faithfulness_passed,
                "faithfulness_metadata": self.state.faithfulness_metadata,
                "rubric_reward_raw": self.state.rubric_reward_raw,
                "hybrid_reward_value": self.state.hybrid_reward_value,
                "hybrid_metadata": self.state.hybrid_metadata,
                "faithfulness_mode": self.config.faithfulness_mode,
                "wager": self.state.wager,
                "wager_reward_shadow": self.state.wager_reward_shadow,
                "wager_metadata": self.state.wager_metadata,
                "wager_mode": self.config.wager_mode,
            },
            info={
                "language": self.state.language,
                "problem": self.problem,
                "work_dir": self.work_dir,
                "input_data": self.input_data,
                "output_data": self.output_data,
            },
        )

    def get_env_state_msg(self) -> EnvStateMessage:
        """Get the current environment state message."""
        summary = self.state.get_execution_summary()

        state_summary = (
            f"{summary['language']} Interpreter Environment\n"
            f"Working Directory: {summary['work_dir']}\n"
            f"Execution History: {summary['total_executions']} commands executed\n"
        )

        if summary["recent_errors"]:
            state_summary += "\nRecent Errors:\n"
            for error in summary["recent_errors"]:
                state_summary += f"- {error}\n"

        if summary["last_execution"]:
            max_len = 200
            state_summary += "\nLast Execution:\n"
            last_exec = summary["last_execution"]
            # Get code from the last notebook cell (ExecutionResult doesn't store code)
            if self.state.nb.cells:
                last_cell = self.state.nb.cells[-1]
                code_source = last_cell.get("source", "")
                code = code_source[:max_len] + "..." if len(code_source) > max_len else code_source
                state_summary += f"Code: {code}\n"
            # Use ExecutionResult methods
            text_outputs = last_exec.get_text_outputs()
            if text_outputs:
                output = text_outputs[0]
                output = output[:max_len] + "..." if len(output) > max_len else output
                state_summary += f"Output: {output}\n"
            if last_exec.has_images():
                images_count = len(last_exec.get_images())
                state_summary += f"Images generated: {images_count}\n"

        return EnvStateMessage.create_message(text=state_summary, images=[])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir")
    args = parser.parse_args()

    work_dir = Path(args.work_dir or mkdtemp())
    print(f"Working directory: {work_dir}")

    problem = ProblemInstance(
        id="",
        hypothesis="",
        protocol="",
        answer=False,
        rubric="",
        max_points=0,
        metadata={},
    )

    env = InterpreterEnv(problem=problem, work_dir=work_dir, config=InterpreterEnvConfig(use_docker=True))
    await env.reset()

    code = ""
    done = False
    while not done:
        breakpoint()  # noqa: T100
        action = ToolRequestMessage(tool_calls=[ToolCall.from_name("run_cell", code=code)])
        obs, *_ = await env.step(action)
        for msg in obs:
            print(msg.content)

    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
