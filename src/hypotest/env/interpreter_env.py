"""InterpreterEnv: Standalone code execution environment for data analysis.

This module provides a lightweight, execution-focused environment for running
code in Jupyter kernels. It focuses on direct code execution via run_cell().
"""

import sys
import ray
import signal
import argparse
import asyncio
import json
import logging
import os
import shutil
import socket
import statistics
import threading
import time
import random
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID
import uuid

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
from pydantic import BaseModel, Field, JsonValue

import subprocess
from textwrap import dedent

from . import config as cfg
from .config import ExecutionConfig
from .interpreter import ExecutionResult, Interpreter
from .prompts import CORRECT_MSG, HYPOTHESIS_TASK_DESC, INCORRECT_MSG, RUBRIC_SCORE_PROMPT, PromptingConfig
from .tools.filesystem import FilesystemTool
from .utils import NBLanguage, view_notebook

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


def _kill_process_group(proc: subprocess.Popen, label: str = "enroot", sigterm_timeout: float = 15) -> None:
    """Safely terminate a process group, escalating from SIGTERM to SIGKILL.

    Handles all edge cases: already-dead process, missing process group, etc.
    """
    if proc.poll() is not None:
        logger.log(
            _CONTAINER_LOG_LEVEL,
            "[%s] Process pid=%d already exited with returncode=%d",
            label, proc.pid, proc.returncode,
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
        # proc.wait(timeout=sigterm_timeout)
        proc.communicate(timeout=sigterm_timeout)
        logger.log(
            _CONTAINER_LOG_LEVEL,
            "[%s] Process pid=%d exited after SIGTERM with returncode=%d",
            label, proc.pid, proc.returncode,
        )
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "[%s] Process pid=%d did not exit within %.1fs of SIGTERM, sending SIGKILL",
            label, proc.pid, sigterm_timeout,
        )

    # SIGKILL the whole group
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        logger.log(_CONTAINER_LOG_LEVEL, "[%s] Process group pgid=%d already gone before SIGKILL", label, pgid)
        return

    try:
        # proc.wait(timeout=5)
        proc.communicate(timeout=5)
        logger.log(
            _CONTAINER_LOG_LEVEL,
            "[%s] Process pid=%d exited after SIGKILL with returncode=%d",
            label, proc.pid, proc.returncode,
        )
    except subprocess.TimeoutExpired:
        logger.error("[%s] Process pid=%d still alive after SIGKILL — possible zombie", label, proc.pid)

PERF_METRICS_ENABLED = os.getenv("NEMO_GYM_PERF_METRICS", "0") == "1"
PERF_DETAIL_SAMPLE_RATE = float(os.getenv("NEMO_GYM_PERF_DETAIL_SAMPLE_RATE", "0.2"))
PERF_SLOW_STEP_MS = float(os.getenv("NEMO_GYM_PERF_SLOW_STEP_MS", "5000"))
PERF_AGGREGATE_INTERVAL_S = float(os.getenv("NEMO_GYM_PERF_AGGREGATE_INTERVAL_S", "30"))
PERF_LOG_LEVEL = getattr(logging, os.getenv("NEMO_GYM_PERF_LOG_LEVEL", "WARNING").upper(), logging.WARNING)

_PERF_METRIC_SAMPLES: dict[str, list[float]] = {}
_PERF_LAST_FLUSH_TS = time.perf_counter()
_PERF_LOCK = threading.Lock()


def _log_perf(payload: dict[str, Any]) -> None:
    logger.log(PERF_LOG_LEVEL, json.dumps(payload))


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _maybe_flush_perf_aggregates() -> None:
    global _PERF_LAST_FLUSH_TS, _PERF_METRIC_SAMPLES
    now = time.perf_counter()
    if now - _PERF_LAST_FLUSH_TS < PERF_AGGREGATE_INTERVAL_S:
        return

    with _PERF_LOCK:
        now = time.perf_counter()
        if now - _PERF_LAST_FLUSH_TS < PERF_AGGREGATE_INTERVAL_S:
            return
        snapshots = _PERF_METRIC_SAMPLES
        _PERF_METRIC_SAMPLES = {}
        _PERF_LAST_FLUSH_TS = now

    for metric_name, values in snapshots.items():
        if not values:
            continue
        ordered = sorted(values)
        payload = {
            "component": "aviary_hypotest.interpreter_env",
            "event": "perf_aggregate",
            "metric": metric_name,
            "count": len(values),
            "mean_ms": round(statistics.fmean(values), 3),
            "p50_ms": round(_percentile(ordered, 0.50), 3),
            "p90_ms": round(_percentile(ordered, 0.90), 3),
            "p95_ms": round(_percentile(ordered, 0.95), 3),
            "p99_ms": round(_percentile(ordered, 0.99), 3),
            "max_ms": round(ordered[-1], 3),
        }
        _log_perf(payload)


def _record_perf_metric(metric_name: str, value_ms: float) -> None:
    if not PERF_METRICS_ENABLED:
        return
    with _PERF_LOCK:
        _PERF_METRIC_SAMPLES.setdefault(metric_name, []).append(value_ms)
    _maybe_flush_perf_aggregates()

class ProblemInstance(BaseModel):
    uuid: UUID
    hypothesis: str
    objective: str
    accepted: bool = Field(alias="answer")
    rubric: str
    max_score: int = Field(alias="max_points")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    nb_primary_language: str = Field(default=str(NBLanguage.PYTHON))

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

@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
class EnrootKernelServer:
    def __init__(self, container_sqsh_path: Path, execution_timeout: float):
        self.container_sqsh_path = container_sqsh_path
        self.execution_timeout = execution_timeout
        self._enroot_proc: subprocess.Popen[str] | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._container_port: int | None = None
        self._container_log_path: Path | None = None
        self._container_log_file: Any | None = None

    def _proc_label(self) -> str:
        """Short label for log messages identifying this container."""
        port = self._container_port or "?"
        pid = self._enroot_proc.pid if self._enroot_proc else "?"
        return f"enroot(port={port}, pid={pid})"

    async def initialize(self, work_dir: Path, language: NBLanguage) -> None:
        startup_token = str(uuid.uuid4())
        node_workdir = f"/tmp/data_workspace.{startup_token.split('-')[0]}"

        _prep_workspace_dir(work_dir, workspace_path=node_workdir)

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
            
            bash = dedent(f"""\
                set -euo pipefail

                WORKDIR="{node_workdir}"
                trap 'rm -rf "$WORKDIR"' EXIT

                mkdir -p $WORKDIR
                cp -a /data_workspace/. $WORKDIR/

                cd $WORKDIR

                mkdir -p "$WORKDIR/r_libs"
                cat >"$WORKDIR/Rprofile" <<'EOF'
                .local_lib <- Sys.getenv("R_LIBS_USER")
                if (nzchar(.local_lib)) .libPaths(unique(c(.local_lib, .libPaths())))
                EOF

                mkdir -p $WORKDIR/r_libs
                export R_LIBS_USER="$WORKDIR/r_libs"
                export R_PROFILE_USER="$WORKDIR/Rprofile"

                export PYTHONPATH="$WORKDIR/pydeps:${{PYTHONPATH}}"
                export PIP_CONFIG_FILE=$WORKDIR/pip.conf
                export target_platform=${{target_platform:-linux-64}}

                source activate /app/kernel_env
                exec /app/kernel_env/bin/python /envs/kernel_server.py \\
                    --work_dir $WORKDIR \\
                    --language {language.value} \\
                    --port {self._container_port} \\
                    --startup-token {startup_token}
            """).strip()

            env_dir = Path(__file__).parent
            kernel_server_path = env_dir / "kernel_server.py"
            assert kernel_server_path.is_file(), f"kernel server must be a valid path, found {kernel_server_path}"

            cmd = [
                "env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", 'HOME="$HOME"', 'USER="$USER"',
                "enroot", "start",
                "--mount", f"{work_dir}:/data_workspace",
                "--mount", f"{kernel_server_path.resolve()}:/envs/kernel_server.py",
                str(self.container_sqsh_path.resolve()),
                "/bin/bash", "-lc", bash,
            ]

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
                self._container_log_file = open(self._container_log_path, "w")  # noqa: SIM115
                self._enroot_proc = subprocess.Popen(
                    cmd, text=True, start_new_session=True,
                    stdout=self._container_log_file, stderr=subprocess.STDOUT,
                )
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container launch attempt #%d started (work_dir=%s, token=%s)",
                    self._proc_label(), attempt, work_dir, startup_token[:8],
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
                    self._proc_label(), launch_ms, attempt,
                )
                online = True
            except Exception as e:
                last_err = e
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                self._log_container_failure(attempt, launch_ms, e)
                await self._cleanup_failed_startup()
                if not isinstance(e, _PortCollisionError):
                    backoff = min(_RETRY_BASE_SLEEP * 2 ** (attempt - 1), _RETRY_MAX_SLEEP)
                    await asyncio.sleep(backoff)

    def _log_container_failure(self, attempt: int, launch_ms: float, error: Exception) -> None:
        """Log detailed diagnostics when a container fails to start."""
        label = self._proc_label()
        proc = self._enroot_proc

        diag_parts = [
            f"attempt=#{attempt}",
            f"elapsed={launch_ms:.0f}ms",
            f"error={error!r}",
        ]

        if proc is not None:
            rc = proc.poll()
            diag_parts.append(f"process_alive={rc is None}")
            if rc is not None:
                diag_parts.append(f"returncode={rc}")

        logger.warning("[%s] Container startup FAILED: %s", label, ", ".join(diag_parts))

        # Log container output separately so tracebacks are readable
        log_tail = self._read_container_log_tail()
        if log_tail:
            logger.warning(
                "[%s] Container log output (last %d chars):\n%s",
                label, len(log_tail), log_tail,
            )

    def _read_container_log_tail(self, max_chars: int = 2000) -> str:
        """Read the tail of the container log file for diagnostics."""
        if self._container_log_path is None or not self._container_log_path.exists():
            return ""
        try:
            # Flush so any buffered output is written to disk
            if self._container_log_file and not self._container_log_file.closed:
                self._container_log_file.flush()
            text = self._container_log_path.read_text()
            return text[-max_chars:] if len(text) > max_chars else text
        except Exception:
            return ""

    def _close_container_log(self) -> None:
        """Close the container log file handle."""
        if self._container_log_file is not None:
            try:
                self._container_log_file.close()
            except Exception:
                pass
            self._container_log_file = None

    async def _cleanup_failed_startup(self) -> None:
        """Best-effort cleanup for failed startup attempts before retrying."""
        label = self._proc_label()

        if self._container_port is not None:
            async with used_ports_lock:
                _USED_PORTS.discard(self._container_port)
            self._container_port = None

        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

        if self._enroot_proc is not None:
            _kill_process_group(self._enroot_proc, label=label)
            self._enroot_proc = None

        self._close_container_log()

    async def _wait_for_health(self, expected_startup_token: str | None = None) -> None:
        """Wait for the kernel server to become healthy."""
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
            if self._enroot_proc is not None and self._enroot_proc.poll() is not None:
                rc = self._enroot_proc.returncode
                log_tail = self._read_container_log_tail(1000)
                raise RuntimeError(
                    f"Enroot process exited prematurely with returncode={rc} "
                    f"after {elapsed:.1f}s. log: {log_tail!r}"
                )

            try:
                assert self._http_client is not None
                response = await self._http_client.get("/health", timeout=health_timeout)
                if response.status_code == 200:
                    if expected_startup_token is None:
                        logger.log(
                            _CONTAINER_LOG_LEVEL,
                            "[%s] Kernel server healthy after %.1fs (%d polls)",
                            self._proc_label(), elapsed, poll_count,
                        )
                        return
                    payload = response.json()
                    if payload.get("startup_token") == expected_startup_token:
                        logger.log(
                            _CONTAINER_LOG_LEVEL,
                            "[%s] Kernel server healthy (token matched) after %.1fs (%d polls)",
                            self._proc_label(), elapsed, poll_count,
                        )
                        return
                    else:
                        last_status = f"token_mismatch(got={payload.get('startup_token', '?')[:8]})"
                        consecutive_token_mismatches += 1
                        if consecutive_token_mismatches >= 3:
                            raise _PortCollisionError(
                                f"Port {self._container_port} appears to be owned by another server "
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
                proc_alive = self._enroot_proc.poll() is None if self._enroot_proc else False
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Health poll #%d at %.1fs: last_status=%s, process_alive=%s",
                    self._proc_label(), poll_count, elapsed, last_status, proc_alive,
                )

            await asyncio.sleep(0.5)

        total_elapsed = time.perf_counter() - start_time
        if last_status.startswith("token_mismatch"):
            raise _PortCollisionError(
                f"Port {self._container_port} health-check timed out with token_mismatch "
                f"({poll_count} polls, elapsed={total_elapsed:.1f}s)"
            )
        log_tail = self._read_container_log_tail(500)
        raise TimeoutError(
            f"Kernel server did not become healthy within {cfg.KERNEL_SERVER_STARTUP_TIMEOUT}s "
            f"({poll_count} polls, last_status={last_status}, elapsed={total_elapsed:.1f}s)"
            f"{f' log_tail={log_tail!r}' if log_tail else ''}"
        )

    # @tenacity.retry(
    #     retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.ReadError)),
    #     stop=tenacity.stop_after_attempt(3),
    #     wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    #     reraise=True,
    # )
    async def _execute_via_http(self, code: str, timeout: float | None = None) -> ExecutionResult:  # noqa: ASYNC109
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
                self._proc_label(), type(e).__name__, effective_timeout, e,
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
                self._proc_label(), type(e).__name__, e,
            )
            raise RuntimeError(f"Kernel reset timed out: {e}") from e

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
                logger.log(_CONTAINER_LOG_LEVEL, "[%s] Graceful /close request failed (container may already be down)", label)
            except Exception:
                logger.warning("[%s] Unexpected error on /close request", label, exc_info=True)
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

        if self._enroot_proc is not None:
            _kill_process_group(self._enroot_proc, label=label)
            self._enroot_proc = None

        self._close_container_log()
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
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
        use_docker: bool = cfg.USE_DOCKER,
        use_enroot: bool = False,
        use_ray: bool = True,
        container_sqsh_path: Path | None = None,
        save_dir: Path | None = None,
    ):
        self.work_dir = work_dir
        self.language = language
        self.execution_timeout = execution_timeout
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
        self.use_ray = use_ray

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
        self._enroot_proc = None
        self._container_port: int | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._container_log_path: Path | None = None
        self._container_log_file: Any | None = None

        # Initialize notebook structure for state tracking
        self.nb: NotebookNode = nbformat.v4.new_notebook()
        self.nb.metadata.kernelspec = language.make_kernelspec()
        self.notebook_runtime_errors: list[str] = []
        self._execution_count = 0

        self.raw_score: int = 0
        self.score: float = 0.0
        self.score_metadata: dict[str, str | int] = {}

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
        self.kernel_container = EnrootKernelServer.remote(self.container_sqsh_path, self.execution_timeout)
        init_ref = self.kernel_container.initialize.remote(self.work_dir, self.language)
        await init_ref

    def _enroot_label(self) -> str:
        port = self._container_port or "?"
        pid = self._enroot_proc.pid if self._enroot_proc else "?"
        return f"enroot-state(port={port}, pid={pid})"

    async def _start_enroot_container(self) -> None:
        _prep_workspace_dir(self.work_dir)

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

            bash = dedent(f"""\
                set -euo pipefail
                cd /data_workspace

                export PYTHONPATH="/data_workspace/pydeps:${{PYTHONPATH}}"
                export PIP_CONFIG_FILE=/data_workspace/pip.conf
                exec /app/kernel_env/bin/python /envs/kernel_server.py \\
                    --work_dir /data_workspace \\
                    --language {self.language.value} \\
                    --port {self._container_port} \\
                    --startup-token {startup_token}
            """).strip()

            env_dir = Path(__file__).parent
            kernel_server_path = env_dir / "kernel_server.py"
            assert kernel_server_path.is_file(), f"kernel server must be a valid path, found {kernel_server_path}"

            cmd = [
                "env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", 'HOME="$HOME"', 'USER="$USER"',
                "enroot", "start",
                "--mount", f"{self.work_dir}:/data_workspace",
                "--mount", f"{kernel_server_path.resolve()}:/envs/kernel_server.py",
                str(self.container_sqsh_path.resolve()),
                "/bin/bash", "-lc", bash,
            ]

            async with CONTAINER_LAUNCH_SEM:
                launch_t0 = time.perf_counter()
                # Redirect container output to a log file (see EnrootKernelServer
                # for detailed rationale on why we avoid subprocess.PIPE here).
                log_dir = self.work_dir / ".container_logs"
                log_dir.mkdir(exist_ok=True)
                self._container_log_path = log_dir / "container.log"
                self._container_log_file = open(self._container_log_path, "w")  # noqa: SIM115
                self._enroot_proc = subprocess.Popen(
                    cmd, text=True, start_new_session=True,
                    stdout=self._container_log_file, stderr=subprocess.STDOUT,
                )
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Container launch attempt #%d (work_dir=%s, token=%s)",
                    self._enroot_label(), attempt, self.work_dir, startup_token[:8],
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
                    self._enroot_label(), launch_ms, attempt,
                )
                online = True
            except Exception as e:
                last_err = e
                launch_ms = (time.perf_counter() - launch_t0) * 1000.0
                label = self._enroot_label()
                diag = [f"attempt=#{attempt}", f"elapsed={launch_ms:.0f}ms", f"error={e!r}"]
                if self._enroot_proc is not None:
                    rc = self._enroot_proc.poll()
                    diag.append(f"process_alive={rc is None}")
                    if rc is not None:
                        diag.append(f"returncode={rc}")
                logger.warning("[%s] Container startup FAILED: %s", label, ", ".join(diag))
                log_tail = self._read_container_log_tail()
                if log_tail:
                    logger.warning(
                        "[%s] Container log output (last %d chars):\n%s",
                        label, len(log_tail), log_tail,
                    )
                await self._cleanup_failed_startup()
                if not isinstance(e, _PortCollisionError):
                    backoff = min(_RETRY_BASE_SLEEP * 2 ** (attempt - 1), _RETRY_MAX_SLEEP)
                    await asyncio.sleep(backoff)

    def _read_container_log_tail(self, max_chars: int = 2000) -> str:
        """Read the tail of the container log file for diagnostics."""
        if self._container_log_path is None or not self._container_log_path.exists():
            return ""
        try:
            if self._container_log_file and not self._container_log_file.closed:
                self._container_log_file.flush()
            text = self._container_log_path.read_text()
            return text[-max_chars:] if len(text) > max_chars else text
        except Exception:
            return ""

    def _close_container_log(self) -> None:
        """Close the container log file handle."""
        if self._container_log_file is not None:
            try:
                self._container_log_file.close()
            except Exception:
                pass
            self._container_log_file = None

    async def _cleanup_failed_startup(self) -> None:
        """Best-effort cleanup for failed startup attempts before retrying."""
        label = self._enroot_label()

        if self._container_port is not None:
            async with used_ports_lock:
                _USED_PORTS.discard(self._container_port)
            self._container_port = None

        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

        if self._enroot_proc is not None:
            _kill_process_group(self._enroot_proc, label=label)
            self._enroot_proc = None

        self._close_container_log()

    async def _start_docker_container(self) -> None:
        """Start a Docker container with the kernel server."""
        self._docker_client = aiodocker.Docker()
        self._container_port = await get_free_port()
        startup_token = str(uuid.uuid4())

        docker_config = {
            "Image": cfg.NB_ENVIRONMENT_DOCKER_IMAGE,
            "Cmd": [
                "/app/kernel_env/bin/python",
                "/envs/kernel_server.py",
                "--work_dir",
                "/data_workspace",
                "--language",
                self.language.value,
                "--startup-token",
                startup_token,
            ],
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
        start_time = time.perf_counter()
        poll_count = 0
        last_status = "no_attempt"
        consecutive_token_mismatches = 0
        # Short per-request timeout (see EnrootKernelServer._wait_for_health
        # for rationale).
        health_timeout = httpx.Timeout(5.0, connect=3.0)

        while time.perf_counter() - start_time < cfg.KERNEL_SERVER_STARTUP_TIMEOUT:
            poll_count += 1
            elapsed = time.perf_counter() - start_time

            # Check if the enroot process has died
            if self._enroot_proc is not None and self._enroot_proc.poll() is not None:
                rc = self._enroot_proc.returncode
                log_tail = self._read_container_log_tail(1000)
                raise RuntimeError(
                    f"Enroot process exited prematurely with returncode={rc} "
                    f"after {elapsed:.1f}s. log: {log_tail!r}"
                )

            try:
                assert self._http_client is not None
                response = await self._http_client.get("/health", timeout=health_timeout)
                if response.status_code == 200:
                    if expected_startup_token is None:
                        logger.log(
                            _CONTAINER_LOG_LEVEL,
                            "[%s] Kernel server healthy after %.1fs (%d polls)",
                            self._enroot_label(), elapsed, poll_count,
                        )
                        return
                    payload = response.json()
                    if payload.get("startup_token") == expected_startup_token:
                        logger.log(
                            _CONTAINER_LOG_LEVEL,
                            "[%s] Kernel server healthy (token matched) after %.1fs (%d polls)",
                            self._enroot_label(), elapsed, poll_count,
                        )
                        return
                    else:
                        last_status = f"token_mismatch(got={payload.get('startup_token', '?')[:8]})"
                        consecutive_token_mismatches += 1
                        if consecutive_token_mismatches >= 3:
                            raise _PortCollisionError(
                                f"Port {self._container_port} appears to be owned by another server "
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
                proc_alive = self._enroot_proc.poll() is None if self._enroot_proc else False
                logger.log(
                    _CONTAINER_LOG_LEVEL,
                    "[%s] Health poll #%d at %.1fs: last_status=%s, process_alive=%s",
                    self._enroot_label(), poll_count, elapsed, last_status, proc_alive,
                )

            await asyncio.sleep(0.5)

        total_elapsed = time.perf_counter() - start_time
        if last_status.startswith("token_mismatch"):
            raise _PortCollisionError(
                f"Port {self._container_port} health-check timed out with token_mismatch "
                f"({poll_count} polls, elapsed={total_elapsed:.1f}s)"
            )
        log_tail = self._read_container_log_tail(500)
        raise TimeoutError(
            f"Kernel server did not become healthy within {cfg.KERNEL_SERVER_STARTUP_TIMEOUT}s "
            f"({poll_count} polls, last_status={last_status}, elapsed={total_elapsed:.1f}s)"
            f"{f' log_tail={log_tail!r}' if log_tail else ''}"
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
                self._enroot_label(), type(e).__name__, effective_timeout, e,
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
                "HTTP %s during /reset: %s", type(e).__name__, e,
            )
            raise RuntimeError(f"Kernel reset timed out: {e}") from e

    async def close(self):
        """Save the notebook and close the interpreter or container."""
        nbformat.write(self.nb, self.work_dir / "notebook.ipynb")

        if self.save_dir is not None:
            shutil.rmtree(self.save_dir, ignore_errors=True)
            self.save_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(self.work_dir, self.save_dir)

        if self.use_ray and self.use_enroot:
            if self.kernel_container is not None:
                close_ref = self.kernel_container.close.remote()
                await close_ref
                self.kernel_container = None

        elif self.use_docker or self.use_enroot:
            if self._container_port is not None:
                async with used_ports_lock:
                    _USED_PORTS.discard(self._container_port)

            if self._http_client is not None:
                try:
                    await self._http_client.aclose()
                except Exception:
                    pass
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
                _kill_process_group(self._enroot_proc, label=self._enroot_label())
                self._enroot_proc = None

            self._close_container_log()

        elif self.interpreter is not None:
            await self.interpreter.close()

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
    ) -> tuple[ExecutionResult, int]:
        """Execute code and atomically update notebook.

        Args:
            code: Code to execute
            cell_idx: Cell index to update (None = append new cell)
            timeout: Optional execution timeout

        Returns:
            Tuple of (ExecutionResult, actual_cell_index)
        """
        # Safety check: block dangerous code before execution
        from .code_safety import check_code_safety
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

        execute_start = time.perf_counter()
        if self.use_ray and self.use_enroot:
            rpc_wait_start = time.perf_counter()
            result_ref = self.kernel_container._execute_via_http.remote(code, timeout)
            result = await result_ref
            _record_perf_metric(
                "execute_path_ms.ray_enroot_rpc_wait",
                (time.perf_counter() - rpc_wait_start) * 1000.0,
            )
        elif self.use_docker or self.use_enroot:
            http_exec_start = time.perf_counter()
            result = await self._execute_via_http(code, timeout)
            _record_perf_metric(
                "execute_path_ms.http_kernel_execute",
                (time.perf_counter() - http_exec_start) * 1000.0,
            )
        else:
            assert self.interpreter is not None
            local_exec_start = time.perf_counter()
            result = await self.interpreter.execute_code(code, timeout)
            _record_perf_metric(
                "execute_path_ms.local_interpreter_execute",
                (time.perf_counter() - local_exec_start) * 1000.0,
            )

        if cell_idx is None or cell_idx >= len(self.nb.cells):
            actual_idx = self._add_cell(code, result)
        else:
            self._update_cell(cell_idx, code, result)
            actual_idx = cell_idx

        _record_perf_metric("execute_and_add_cell_total_ms", (time.perf_counter() - execute_start) * 1000.0)
        return result, actual_idx


class InterpreterEnvConfig(BaseModel):
    """Configuration for preparing the InterpreterEnv during task creation."""

    language: NBLanguage = Field(default=NBLanguage.PYTHON)
    prompting_config: PromptingConfig = Field(default_factory=PromptingConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    max_steps: int = cfg.AGENT_MAX_STEPS
    use_docker: bool = cfg.USE_DOCKER
    use_enroot: bool = False
    use_ray: bool = False
    container_sqsh_path: Path | None = None
    normalize_reward: bool = True


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

        logger.error(f"EXECUTION TIMEOUT SET TO {self.execution_timeout} SECONDS")

        self.input_data = input_data
        self.output_data: list[dict[str, str | int]] = []
        self.logger = logger
        self.start_time: float | None = None
        self.step_count = 0
        self.include_env_state_msg = include_env_state_msg
        self.state: InterpreterEnvState
        # prompting_config is set during reset() after language resolution
        self.prompting_config: PromptingConfig
        self._perf_sample_step = False

    def _emit_perf_event(self, event: str, **fields: Any) -> None:
        if not PERF_METRICS_ENABLED:
            return
        payload: dict[str, Any] = {
            "component": "aviary_hypotest.interpreter_env",
            "event": event,
            "env_id": getattr(self, "_nemo_env_id", None),
            "step_idx": self.step_count,
        }
        payload.update(fields)
        _log_perf(payload)

    @property
    def language(self) -> NBLanguage:
        return self.config.language

    async def close(self) -> None:
        """Save notebook, shut down interpreter/container."""
        self.logger.info("Closing environment")
        await self.state.close()

    async def reset(self) -> tuple[Messages, list[Tool]]:
        """Reset the environment and prepare for execution."""
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
            use_ray=self.config.use_ray,
            use_docker=self.config.use_docker,
            use_enroot=self.config.use_enroot,
            container_sqsh_path=self.config.container_sqsh_path
        )
        await self.state.start()

        # Record start time for timeout tracking
        self.start_time = time.perf_counter()

        messages = []
        if self.prompting_config.system_prompt:
            messages.append(Message(role="system", content=self.prompting_config.system_prompt))

        self._filesystem_tool = FilesystemTool(self.work_dir)
        self.tools = [
            Tool.from_function(self.run_cell),
            Tool.from_function(self.reset_kernel),
            Tool.from_function(self.submit_answer),
            Tool.from_function(self.list_dir),
        ]

        messages.append(
            Message(
                content=HYPOTHESIS_TASK_DESC.format(
                    language=self.language.value.capitalize(),
                    hypothesis=self.problem.hypothesis,
                    objective=self.problem.objective,
                )
            )
        )

        if self.include_env_state_msg:
            messages.append(self.get_env_state_msg())

        # Always show initial directory listing (with truncation protection)
        messages.append(Message(content=self.list_dir()))

        return messages, self.tools

    async def step(self, action: ToolRequestMessage) -> tuple[Messages, float, bool, bool]:
        """Execute a step in the environment."""
        self.step_count += 1
        self._perf_sample_step = PERF_METRICS_ENABLED and random.random() < PERF_DETAIL_SAMPLE_RATE
        step_start = time.perf_counter()
        exec_tool_calls_start = time.perf_counter()
        obs = cast(
            Messages,
            await self.exec_tool_calls(action, concurrency=False, handle_tool_exc=True),
        )
        exec_tool_calls_ms = (time.perf_counter() - exec_tool_calls_start) * 1000.0
        _record_perf_metric("exec_tool_calls_ms", exec_tool_calls_ms)

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
        step_total_ms = (time.perf_counter() - step_start) * 1000.0
        _record_perf_metric("env_step_ms", step_total_ms)
        should_log_detail = self._perf_sample_step or step_total_ms >= PERF_SLOW_STEP_MS
        if should_log_detail:
            obs_payload_chars = sum(len(str(getattr(m, "content", ""))) for m in obs)
            self._emit_perf_event(
                "env_step_end",
                duration_ms=round(step_total_ms, 3),
                exec_tool_calls_ms=round(exec_tool_calls_ms, 3),
                action_tool_calls_count=len(action.tool_calls),
                obs_messages_count=len(obs),
                obs_payload_chars=obs_payload_chars,
                slow=step_total_ms >= PERF_SLOW_STEP_MS,
            )
        return obs, reward, self.state.done, False

    # ========== Tools ==========

    async def run_cell(
        self,
        code: str,
        idx: int | None = None,
    ) -> Message | str | list[dict[str, Any]]:
        """Run code in a notebook cell and return the execution output.

        This method allows running code in a new cell (append) or re-running
        an existing cell with updated code.

        Error Recovery:
            When a cell fails with an error, you MUST fix it by calling run_cell
            with the corrected code and the SAME idx as the failed cell.

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
        """
        remaining_seconds = self.get_remaining_time()

        if remaining_seconds <= self.execution_config.force_submit_threshold:
            self.logger.warning(
                f"Refusing cell execution with {remaining_seconds:.1f}s remaining "
                f"(force threshold: {self.execution_config.force_submit_threshold}s)"
            )
            return cfg.FORCE_MSG

        dynamic_timeout = remaining_seconds - self.execution_config.force_submit_threshold
        effective_timeout = min(self.execution_timeout, dynamic_timeout)

        self.logger.info(
            f"Cell execution with dynamic timeout: {effective_timeout:.1f}s "
            f"(remaining: {remaining_seconds:.1f}s, default: {self.execution_timeout}s)"
        )

        # Parse idx (handle string input from LLM)
        cell_idx: int | None = None
        if idx is not None:
            try:
                cell_idx = int(idx)
            except (ValueError, TypeError):
                cell_idx = None

        # Execute code and update notebook atomically
        exec_start = time.perf_counter()
        result, actual_cell_idx = await self.state.execute_and_add_cell(
            code, cell_idx=cell_idx, timeout=effective_timeout
        )
        execute_and_add_cell_ms = (time.perf_counter() - exec_start) * 1000.0
        _record_perf_metric("execute_and_add_cell_ms", execute_and_add_cell_ms)

        # Build response with cell number
        cell_info = f"[Cell #{actual_cell_idx}] "

        if result.has_images():
            # Format images as data URLs for Message
            image_urls = [f"data:{mime_type};base64,{base64_data}" for mime_type, base64_data in result.get_images()]

            return Message.create_message(
                role="tool",
                text=cell_info + result.get_truncated_text(),
                images=cast(list[np.ndarray | str], image_urls),
            )

        return cell_info + result.get_truncated_text()

    async def reset_kernel(self) -> str:
        """Reset the kernel to a clean state.

        This clears all variables and execution state.
        """
        start = time.perf_counter()
        if self.state.use_ray:
            reset_ref = self.state.kernel_container._reset_via_http.remote()
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

        duration_ms = (time.perf_counter() - start) * 1000.0
        _record_perf_metric("tool_call_ms.reset_kernel", duration_ms)
        if self._perf_sample_step:
            self._emit_perf_event(
                "tool_call_end",
                tool_name="reset_kernel",
                duration_ms=round(duration_ms, 3),
            )
        return "Kernel reset successfully."

    @property
    def score_info_path(self) -> Path:
        return self.work_dir / "score_info.json"

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(ValueError))
    async def _score_solution(self, solution: str) -> bool:
        assert self.rubric_model is not None
        nb_content, _ = view_notebook(self.state.nb.cells, self.language.value)

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
            raw_score = int(resp.text.split("<score>")[1].split("</score>")[0])
            self.state.raw_score = raw_score

        except Exception as e:
            raise ValueError("Failed to parse score from response") from e

        else:
            correct = raw_score == self.problem.max_score
            score = raw_score / self.problem.max_score if self.config.normalize_reward else raw_score
            score = max(
                0.0,
                min(1.0 if self.config.normalize_reward else self.problem.max_score, score),
            )

            self.state.score = score
            self.state.total_reward += score
            return correct

        finally:
            score_info = {
                **self.state.score_metadata,
                "score": self.state.score,
                "raw_score": self.state.raw_score,
                "max_score": self.problem.max_score,
            }
            with self.score_info_path.open("w") as f:
                json.dump(score_info, f, indent=2)

            self.logger.info(f"Received solution ({self.state.raw_score}/{self.problem.max_score}): {solution!r}.")

    async def submit_answer(self, answer: str) -> str:
        """Submit your response to the research question.

        Note that this tool may only be called once and ends the episode.

        Args:
            answer: Your final response to the research question
        """
        start = time.perf_counter()
        if self.state.done:
            duration_ms = (time.perf_counter() - start) * 1000.0
            _record_perf_metric("tool_call_ms.submit_answer", duration_ms)
            return "Episode already finished."

        self.state.answer = answer
        self.state.done = True

        if self.rubric_model is None:
            self.logger.warning("No rubric_model configured, skipping scoring")
            duration_ms = (time.perf_counter() - start) * 1000.0
            _record_perf_metric("tool_call_ms.submit_answer", duration_ms)
            return answer

        correct = await self._score_solution(answer)
        duration_ms = (time.perf_counter() - start) * 1000.0
        _record_perf_metric("tool_call_ms.submit_answer", duration_ms)
        if self._perf_sample_step:
            self._emit_perf_event(
                "tool_call_end",
                tool_name="submit_answer",
                duration_ms=round(duration_ms, 3),
            )
        return CORRECT_MSG if correct else INCORRECT_MSG

    def list_dir(self) -> str:
        """List contents of a directory with truncation protection.

        Recursively lists files in a directory, with built-in protection against
        overwhelming the context with too many files. Use this tool instead of
        writing code to list directories to avoid context bloat.
    
        Args:
            directory: Directory path to list (default: current working directory)
            max_files: Maximum number of files to display (default: 20)
            show_hidden: Whether to show hidden files starting with '.' (default: False)
        """
        start = time.perf_counter()
        result = self._filesystem_tool.list_dir()
        duration_ms = (time.perf_counter() - start) * 1000.0
        _record_perf_metric("tool_call_ms.list_dir", duration_ms)
        if self._perf_sample_step:
            self._emit_perf_event(
                "tool_call_end",
                tool_name="list_dir",
                duration_ms=round(duration_ms, 3),
            )
        return result

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
        uuid="",
        hypothesis="",
        objective="",
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
