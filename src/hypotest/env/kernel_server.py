"""Standalone kernel server for Docker-based execution.

This module runs inside the container and provides an HTTP API for code execution.
It also contains shared types (NBLanguage, MessageType) that hypotest imports.

IMPORTANT: This module must be standalone with no imports from hypotest package,
as it gets copied into the Docker image and run independently.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from queue import Empty
from typing import Annotated, Any, Literal, assert_never

import nbformat
import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import AsyncKernelManager
from nbformat import NotebookNode
from pydantic import BaseModel, Field


class DeadlineExceededError(Exception):
    """Raised when a cooperative deadline check expires."""

    def __init__(self, msg_id: str):
        super().__init__(msg_id)
        self.msg_id = msg_id


class KernelDiedError(RuntimeError):
    """Raised when the Jupyter kernel exits before returning to idle."""

    def __init__(self, msg_id: str, exit_code: int | None):
        self.msg_id = msg_id
        self.exit_code = exit_code
        detail = f" with exit code {exit_code}" if exit_code is not None else ""
        super().__init__(f"Jupyter kernel exited unexpectedly{detail}")


class KernelExecutionState(StrEnum):
    """Admission state for the single Jupyter execution channel."""

    IDLE = auto()
    EXECUTING = auto()
    INTERRUPTING = auto()
    RECOVERING = auto()
    WEDGED = auto()
    CLOSED = auto()


logger = logging.getLogger(__name__)


class _PrlimitAsyncKernelManager(AsyncKernelManager):
    """Prefix only the Jupyter child with a Linux virtual-memory limit."""

    def __init__(self, *args: Any, kernel_memory_limit_mb: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.kernel_memory_limit_mb = kernel_memory_limit_mb

    def format_kernel_cmd(self, extra_arguments: list[str] | None = None) -> list[str]:
        command = super().format_kernel_cmd(extra_arguments)
        if self.kernel_memory_limit_mb is None:
            return command

        prlimit = shutil.which("prlimit")
        if prlimit is None:
            raise RuntimeError("kernel_memory_limit_mb requires the util-linux prlimit executable")
        limit_bytes = self.kernel_memory_limit_mb * 1024 * 1024
        return [prlimit, f"--as={limit_bytes}", "--", *command]


def deterministic_kernel_env(seed: int) -> dict[str, str]:
    """Environment variables that must be fixed before the kernel process starts."""
    return {
        "PYTHONHASHSEED": str(seed),
        "HYPOTEST_SEED": str(seed),
        # Required by deterministic CUDA matrix operations when CUDA is used.
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def rng_bootstrap_code(language: NBLanguage, seed: int) -> str:
    """Return hidden startup code that seeds common RNGs without a notebook cell."""
    if language == NBLanguage.R:
        return (
            'RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")\n'
            f"set.seed({seed})"
        )

    return f"""
def _hypotest_seed_all_rngs():
    import random

    try:
        import numpy
    except ImportError:
        numpy = None

    # Import first, then seed: module import side effects cannot advance the
    # initialized streams before the policy's first cell.
    random.seed({seed})
    if numpy is not None:
        numpy.random.seed({seed})

_hypotest_seed_all_rngs()
del _hypotest_seed_all_rngs
""".strip()


# Wire-protocol version for the kernel-server HTTP API. Bump on a breaking change
# to /execute, /reset, /list_dir, /load_capsule, or /health; the client
# (HttpKernelClient) reads it from /health to detect deploy skew.
PROTOCOL_VERSION = 2


# =============================================================================
# Shared Types (imported by hypotest)
# =============================================================================


class NBLanguage(StrEnum):
    """Supported notebook languages."""

    PYTHON = auto()
    R = auto()

    def make_kernelspec(self) -> dict[str, str]:
        match self:
            case NBLanguage.PYTHON:
                kspec = {"name": "python", "display_name": "Python 3 (ipykernel)"}
            case NBLanguage.R:
                kspec = {"name": "ir", "display_name": "R"}
            case _:
                assert_never(self)

        return kspec | {"language": self.value}

    @classmethod
    def from_string(cls, s: str) -> NBLanguage | None:
        """Parse language string, returning None for AUTO."""
        s = s.upper()
        if s == "AUTO":
            return None
        try:
            return cls[s]
        except KeyError:
            logger.warning(f"Invalid language '{s}', defaulting to PYTHON")
            return cls.PYTHON


class MessageType(StrEnum):
    """Jupyter kernel IOPub message types.

    See: https://jupyter-client.readthedocs.io/en/latest/messaging.html#messages-on-the-iopub-pub-sub-channel
    """

    STREAM = "stream"
    EXECUTE_RESULT = "execute_result"
    DISPLAY_DATA = "display_data"
    ERROR = "error"
    STATUS = "status"

    @classmethod
    def from_string(cls, value: str) -> MessageType | None:
        """Convert string to MessageType, returning None for unknown types."""
        try:
            return cls(value)
        except ValueError:
            return None

    def to_notebook_output(self, content: dict[str, Any]) -> NotebookNode | None:
        """Convert this message type to an nbformat output node.

        Args:
            content: The message content dictionary from the kernel

        Returns:
            NotebookNode output or None if this message type doesn't produce output
        """
        match self:
            case MessageType.STREAM:
                return nbformat.v4.new_output(
                    output_type="stream",
                    name=content.get("name", "stdout"),
                    text=content.get("text", ""),
                )
            case MessageType.EXECUTE_RESULT:
                return nbformat.v4.new_output(
                    output_type="execute_result",
                    data=content.get("data", {}),
                    metadata=content.get("metadata", {}),
                    execution_count=content.get("execution_count"),
                )
            case MessageType.DISPLAY_DATA:
                return nbformat.v4.new_output(
                    output_type="display_data",
                    data=content.get("data", {}),
                    metadata=content.get("metadata", {}),
                )
            case MessageType.ERROR:
                return nbformat.v4.new_output(
                    output_type="error",
                    ename=content.get("ename", ""),
                    evalue=content.get("evalue", ""),
                    traceback=content.get("traceback", []),
                )
            case MessageType.STATUS:
                return None


# =============================================================================
# Server-only Code (not imported by hypotest)
# =============================================================================


# ---------------------------------------------------------------------------
# Lightweight regex safety check (defense-in-depth, standalone — no hypotest imports)
# ---------------------------------------------------------------------------
_KERNEL_SAFETY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Process killing
    (re.compile(r"\bos\s*\.\s*kill\s*\("), "restricted function"),
    (re.compile(r"\bos\s*\.\s*killpg\s*\("), "restricted function"),
    (re.compile(r"\bos\s*\.\s*system\s*\("), "restricted function"),
    (re.compile(r"\bos\s*\.\s*popen\s*\("), "restricted function"),
    (re.compile(r"\bos\s*\.\s*fork\s*\("), "restricted function"),
    (re.compile(r"\bos\s*\.\s*exec\w*\s*\("), "restricted function"),
    # (re.compile(r"\bsubprocess\s*\.\s*(run|Popen|call|check_call|check_output)\s*\("), "restricted function"),
    # Blocked modules
    (re.compile(r"\bimport\s+ctypes\b"), "restricted module"),
    (re.compile(r"\bimport\s+signal\b"), "restricted module"),
    (re.compile(r"\bfrom\s+ctypes\b"), "restricted module"),
    (re.compile(r"\bfrom\s+signal\b"), "restricted module"),
    # Shell commands
    (re.compile(r"\bkillall\b"), "restricted shell command"),
    (re.compile(r"\bpkill\b"), "restricted shell command"),
]


def _kernel_check_code_safety(code: str) -> str | None:
    """Lightweight regex safety check for the kernel server.

    Returns None if safe, or a message if blocked.
    """
    for pattern, category in _KERNEL_SAFETY_PATTERNS:
        if pattern.search(code):
            return f"Code blocked: calls a {category}."
    return None


class ExecuteRequest(BaseModel):
    """Request model for /execute endpoint."""

    code: str
    timeout: float | None = None
    timeout_recovery: Literal["none", "interrupt"] = "none"
    interrupt_grace_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)


class ExecuteResponse(BaseModel):
    """Response model for /execute endpoint.

    Contains serialized notebook outputs that can be deserialized back to NotebookNode.
    """

    notebook_outputs: list[dict[str, Any]]
    error_occurred: bool
    execution_time: float | None
    timed_out: bool = False
    timeout_recovery: Literal["interrupted", "wedged"] | None = None
    interrupt_seconds: float | None = None
    kernel_restarted: bool = False
    kernel_state_lost: bool = False
    kernel_exit_code: int | None = None


def _require_successful_bootstrap(result: ExecuteResponse) -> None:
    if result.error_occurred:
        raise RuntimeError("Deterministic kernel RNG bootstrap failed")


class ExecuteJobStatus(StrEnum):
    """Transport-level state for an asynchronously submitted cell."""

    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class ExecuteSubmissionResponse(BaseModel):
    """Immediate response returned after accepting an execution request."""

    execution_id: str
    status: ExecuteJobStatus


class ExecutePollResponse(ExecuteSubmissionResponse):
    """Current state of a submitted execution, including its terminal payload."""

    result: ExecuteResponse | None = None
    error: str | None = None


@dataclass
class _ExecutionJob:
    """Server-internal execution record retained so clients can reconnect and poll."""

    execution_id: str
    request_id: str
    request: ExecuteRequest
    status: ExecuteJobStatus
    created_at: float
    completed_at: float | None = None
    result: ExecuteResponse | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None


class IdempotencyConflictError(ValueError):
    """A request UUID was reused for a different execution payload."""


class ResetResponse(BaseModel):
    """Response model for /reset endpoint."""

    success: bool
    # Echo the seed actually retained by the server. Deterministic clients
    # validate it so a stale additive-only server cannot ignore their seed.
    seed: int | None = None


class ResetRequest(BaseModel):
    """Optional deterministic seed update for /reset."""

    seed: int | None = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    startup_token: str
    kernel_ready: bool
    protocol_version: int = PROTOCOL_VERSION


class ListDirResponse(BaseModel):
    """Response model for /list_dir endpoint."""

    listing: str


def _collect_dir_paths(path: Path, prefix: str = "", show_hidden: bool = False) -> list[str]:
    """Recursively collect file paths relative to `path` (mirrors env.tools.filesystem)."""
    paths: list[str] = []
    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    except PermissionError:
        rel = f"{prefix}{path.name}/" if prefix else f"{path.name}/"
        return [f"# {rel} (permission denied)"]
    for item in items:
        if not show_hidden and item.name.startswith("."):
            continue
        rel = f"{prefix}{item.name}" if prefix else item.name
        if item.is_dir():
            paths.extend(_collect_dir_paths(item, prefix=f"{rel}/", show_hidden=show_hidden))
        else:
            paths.append(rel)
    return paths


class KernelServer:
    """Manages a persistent Jupyter kernel and exposes it via HTTP."""

    def __init__(
        self,
        work_dir: Path,
        language: NBLanguage,
        default_timeout: float = 600,
        startup_token: str = "",
        safe_execute: bool = True,
        seed: int | None = None,
        kernel_memory_limit_mb: int | None = None,
        execution_result_ttl_seconds: float = 3600,
        max_retained_executions: int = 256,
    ):
        if kernel_memory_limit_mb is not None and kernel_memory_limit_mb <= 0:
            raise ValueError("kernel_memory_limit_mb must be positive")
        self.work_dir = work_dir
        self.language = language
        self.default_timeout = default_timeout
        self.startup_token = startup_token
        self.safe_execute = safe_execute
        self.seed = seed
        self.kernel_memory_limit_mb = kernel_memory_limit_mb
        self.execution_result_ttl_seconds = execution_result_ttl_seconds
        self.max_retained_executions = max_retained_executions

        self._kernel_manager: AsyncKernelManager | None = None
        self._client: AsyncKernelClient | None = None
        self._is_ready = False
        self._execution_state = KernelExecutionState.IDLE
        self._state_lock = asyncio.Lock()
        self._active_msg_id: str | None = None
        self._execution_jobs: dict[str, _ExecutionJob] = {}
        self._execution_ids_by_request: dict[str, str] = {}
        self._execution_jobs_lock = asyncio.Lock()
        self._kernel_lifecycle_lock = asyncio.Lock()
        self._kernel_runtime_dir: Path | None = None

    async def start(self) -> None:
        """Start the Jupyter kernel."""
        async with self._kernel_lifecycle_lock:
            if self._is_ready and await self._kernel_process_is_alive():
                return
            if self._kernel_manager is not None or self._client is not None:
                await self._dispose_kernel_locked()
            await self._launch_kernel_locked()
        await self._set_execution_state(KernelExecutionState.IDLE)

    async def _launch_kernel_locked(self) -> None:
        """Launch one kernel while the caller holds the lifecycle lock."""
        kernel_name = self.language.make_kernelspec()["name"]
        # ZeroMQ appends an IPC channel suffix to this path. Keeping it under a
        # short private runtime directory avoids macOS/Linux sockaddr_un limits
        # when the task workspace itself has a long path.
        runtime_root = Path(os.getenv("HYPOTEST_KERNEL_RUNTIME_ROOT", tempfile.gettempdir()))
        self._kernel_runtime_dir = Path(tempfile.mkdtemp(prefix="hk-", dir=runtime_root))
        kernel_connect_file = (self._kernel_runtime_dir / "c.json").resolve()

        self._kernel_manager = _PrlimitAsyncKernelManager(
            kernel_name=kernel_name,
            transport="ipc",
            connection_file=str(kernel_connect_file),
            kernel_memory_limit_mb=self.kernel_memory_limit_mb,
        )
        kernel_env = os.environ.copy()
        if self.seed is not None:
            kernel_env.update(deterministic_kernel_env(self.seed))
        try:
            await self._kernel_manager.start_kernel(cwd=str(self.work_dir), env=kernel_env)
            self._client = self._kernel_manager.client()
            self._client.start_channels()
            await self._client.wait_for_ready()
            if self.seed is not None:
                bootstrap = await self._execute_code(
                    rng_bootstrap_code(self.language, self.seed),
                    deadline=time.perf_counter() + self.default_timeout,
                    store_history=False,
                )
                _require_successful_bootstrap(bootstrap)
            self._is_ready = True
            logger.info(
                "Kernel %s started in %s (memory limit: %s MiB)",
                kernel_name,
                self.work_dir,
                self.kernel_memory_limit_mb if self.kernel_memory_limit_mb is not None else "unlimited",
            )
        except Exception as e:
            await self._dispose_kernel_locked()
            raise RuntimeError(f"Kernel failed to start: {e}") from e

    async def _dispose_kernel_locked(self) -> None:
        """Dispose the current kernel while the caller holds the lifecycle lock."""
        client, self._client = self._client, None
        manager, self._kernel_manager = self._kernel_manager, None
        self._is_ready = False
        self._active_msg_id = None
        if client is not None:
            with contextlib.suppress(Exception):
                client.stop_channels()
        if manager is not None:
            with contextlib.suppress(Exception):
                await manager.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                await manager.cleanup_resources(restart=False)
        self._cleanup_kernel_runtime_dir()

    async def _kernel_process_is_alive(self) -> bool:
        manager = self._kernel_manager
        if manager is None:
            return False
        try:
            return bool(await manager.is_alive())
        except Exception:
            logger.exception("Failed to inspect Jupyter kernel liveness")
            return False

    async def kernel_ready(self) -> bool:
        """Return whether the server has a live, initialized Jupyter kernel."""
        return self._is_ready and await self._kernel_process_is_alive()

    def _kernel_exit_code(self) -> int | None:
        manager = self._kernel_manager
        provisioner = getattr(manager, "provisioner", None) if manager is not None else None
        process = getattr(provisioner, "process", None)
        return getattr(process, "returncode", None)

    def _cleanup_kernel_runtime_dir(self) -> None:
        runtime_dir, self._kernel_runtime_dir = self._kernel_runtime_dir, None
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=True)

    async def submit_execution(self, req: ExecuteRequest, request_id: str | None = None) -> ExecuteSubmissionResponse:
        """Accept one cell and start it in a detached task.

        ``request_id`` is an idempotency key. Repeating the same request returns
        the original execution id; reusing the key for a different payload is a
        conflict. This prevents a lost submit response from executing a cell twice.
        """
        request_id = request_id or str(uuid.uuid4())
        now = time.monotonic()
        async with self._execution_jobs_lock:
            self._prune_execution_jobs_locked(now)
            existing_id = self._execution_ids_by_request.get(request_id)
            if existing_id is not None:
                existing = self._execution_jobs.get(existing_id)
                if existing is not None:
                    if existing.request != req:
                        raise IdempotencyConflictError(
                            f"X-Req-UUID {request_id!r} was already used for a different execution"
                        )
                    return ExecuteSubmissionResponse(
                        execution_id=existing.execution_id,
                        status=existing.status,
                    )
                # A stale reverse index should never survive pruning, but heal it
                # defensively rather than making this request permanently unusable.
                self._execution_ids_by_request.pop(request_id, None)

            execution_id = str(uuid.uuid4())
            job = _ExecutionJob(
                execution_id=execution_id,
                request_id=request_id,
                request=req,
                status=ExecuteJobStatus.QUEUED,
                created_at=now,
            )
            self._execution_jobs[execution_id] = job
            self._execution_ids_by_request[request_id] = execution_id
            job.task = asyncio.create_task(
                self._run_execution_job(job),
                name=f"kernel-execute:{execution_id}",
            )
            return ExecuteSubmissionResponse(execution_id=execution_id, status=job.status)

    async def get_execution(self, execution_id: str) -> ExecutePollResponse | None:
        """Return one retained execution, or ``None`` when it is unknown/expired."""
        async with self._execution_jobs_lock:
            self._prune_execution_jobs_locked(time.monotonic())
            job = self._execution_jobs.get(execution_id)
            if job is None:
                return None
            return ExecutePollResponse(
                execution_id=job.execution_id,
                status=job.status,
                result=job.result,
                error=job.error,
            )

    async def cancel_execution(self, execution_id: str) -> ExecutePollResponse | None:
        """Cancel an active execution and return its resulting state."""
        async with self._execution_jobs_lock:
            job = self._execution_jobs.get(execution_id)
            task = job.task if job is not None else None
        if job is None:
            return None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return await self.get_execution(execution_id)

    async def _run_execution_job(self, job: _ExecutionJob) -> None:
        async with self._execution_jobs_lock:
            job.status = ExecuteJobStatus.RUNNING
        try:
            job.result = await self.execute(
                job.request.code,
                job.request.timeout,
                timeout_recovery=job.request.timeout_recovery,
                interrupt_grace_seconds=job.request.interrupt_grace_seconds,
            )
        except asyncio.CancelledError:
            async with self._execution_jobs_lock:
                job.status = ExecuteJobStatus.CANCELLED
                job.error = "Execution was cancelled"
                job.completed_at = time.monotonic()
            raise
        except Exception as exc:
            logger.exception("Detached kernel execution %s failed", job.execution_id)
            async with self._execution_jobs_lock:
                job.status = ExecuteJobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.completed_at = time.monotonic()
        else:
            async with self._execution_jobs_lock:
                job.status = ExecuteJobStatus.COMPLETED
                job.completed_at = time.monotonic()

    def _prune_execution_jobs_locked(self, now: float) -> None:
        """Drop expired results and bound retained terminal executions."""
        expired = [
            execution_id
            for execution_id, job in self._execution_jobs.items()
            if job.completed_at is not None and now - job.completed_at >= self.execution_result_ttl_seconds
        ]
        for execution_id in expired:
            self._drop_execution_job_locked(execution_id)

        overflow = len(self._execution_jobs) - self.max_retained_executions
        if overflow <= 0:
            return
        terminal = sorted(
            (job for job in self._execution_jobs.values() if job.completed_at is not None),
            key=lambda job: job.completed_at or job.created_at,
        )
        for job in terminal[:overflow]:
            self._drop_execution_job_locked(job.execution_id)

    def _drop_execution_job_locked(self, execution_id: str) -> None:
        job = self._execution_jobs.pop(execution_id, None)
        if job is not None:
            self._execution_ids_by_request.pop(job.request_id, None)

    async def _cancel_active_execution_jobs(self) -> None:
        """Cancel every detached execution before resetting or closing the kernel."""
        async with self._execution_jobs_lock:
            tasks = [job.task for job in self._execution_jobs.values() if job.task is not None and not job.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def execute(
        self,
        code: str,
        timeout: float | None = None,  # noqa: ASYNC109
        *,
        timeout_recovery: Literal["none", "interrupt"] = "none",
        interrupt_grace_seconds: float = 10.0,
    ) -> ExecuteResponse:
        """Execute code and return the result."""
        if not self._client or not self._is_ready:
            raise RuntimeError("Kernel not ready")

        if safety_block := self._safety_block_response(code):
            return safety_block

        unavailable = await self._claim_execution()
        if unavailable is not None:
            return unavailable

        effective_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.perf_counter()

        try:
            result = await self._execute_code(
                code,
                deadline=start_time + effective_timeout,
            )
        except KernelDiedError as exc:
            result = await self._recover_kernel_death(exc, start_time)
        except DeadlineExceededError as exc:
            result = await self._recover_deadline(
                exc,
                start_time=start_time,
                effective_timeout=effective_timeout,
                timeout_recovery=timeout_recovery,
                interrupt_grace_seconds=interrupt_grace_seconds,
            )
        except asyncio.CancelledError:
            await self._recover_cancelled_execution(timeout_recovery, interrupt_grace_seconds)
            raise
        except Exception as e:
            result = await self._recover_execution_exception(e, start_time)
        else:
            await self._set_execution_state(KernelExecutionState.IDLE)
            self._active_msg_id = None

        return result

    def _safety_block_response(self, code: str) -> ExecuteResponse | None:
        if not self.safe_execute:
            return None
        block_reason = _kernel_check_code_safety(code)
        if block_reason is None:
            return None
        logger.warning("Kernel safety block: %s code=%r", block_reason, code[:200])
        error_output = MessageType.ERROR.to_notebook_output({
            "ename": "SecurityError",
            "evalue": block_reason,
            "traceback": [f"SecurityError: {block_reason}"],
        })
        return ExecuteResponse(
            notebook_outputs=[dict(error_output)] if error_output else [],
            error_occurred=True,
            execution_time=0.0,
        )

    async def _recover_deadline(
        self,
        exc: DeadlineExceededError,
        *,
        start_time: float,
        effective_timeout: float,
        timeout_recovery: Literal["none", "interrupt"],
        interrupt_grace_seconds: float,
    ) -> ExecuteResponse:
        recovered = False
        interrupt_seconds: float | None = None
        if timeout_recovery == "interrupt":
            await self._set_execution_state(KernelExecutionState.INTERRUPTING)
            interrupt_started = time.perf_counter()
            recovered = await self._interrupt_and_drain(exc.msg_id, interrupt_grace_seconds)
            interrupt_seconds = time.perf_counter() - interrupt_started
        await self._set_execution_state(KernelExecutionState.IDLE if recovered else KernelExecutionState.WEDGED)
        if recovered:
            self._active_msg_id = None
        timeout_output = MessageType.ERROR.to_notebook_output({
            "ename": "TimeoutError",
            "evalue": (
                f"Code execution timed out after {effective_timeout} seconds; "
                + ("the cell was interrupted and the kernel is ready" if recovered else "the kernel is unresponsive")
            ),
            "traceback": [f"TimeoutError: Code execution timed out after {effective_timeout} seconds"],
        })
        return ExecuteResponse(
            notebook_outputs=[dict(timeout_output)] if timeout_output else [],
            error_occurred=True,
            execution_time=time.perf_counter() - start_time,
            timed_out=True,
            timeout_recovery="interrupted" if recovered else "wedged",
            interrupt_seconds=interrupt_seconds,
        )

    async def _recover_cancelled_execution(
        self,
        timeout_recovery: Literal["none", "interrupt"],
        interrupt_grace_seconds: float,
    ) -> None:
        recovered = False
        if timeout_recovery == "interrupt" and self._active_msg_id is not None:
            await self._set_execution_state(KernelExecutionState.INTERRUPTING)
            recovered = await asyncio.shield(self._interrupt_and_drain(self._active_msg_id, interrupt_grace_seconds))
        await self._set_execution_state(KernelExecutionState.IDLE if recovered else KernelExecutionState.WEDGED)
        if recovered:
            self._active_msg_id = None

    async def _recover_execution_exception(self, exc: Exception, start_time: float) -> ExecuteResponse:
        if not await self._kernel_process_is_alive():
            died = KernelDiedError(self._active_msg_id or "", self._kernel_exit_code())
            return await self._recover_kernel_death(died, start_time)
        await self._set_execution_state(KernelExecutionState.WEDGED)
        error_output = MessageType.ERROR.to_notebook_output({
            "ename": type(exc).__name__,
            "evalue": str(exc),
            "traceback": [f"{type(exc).__name__}: {exc}"],
        })
        return ExecuteResponse(
            notebook_outputs=[dict(error_output)] if error_output else [],
            error_occurred=True,
            execution_time=time.perf_counter() - start_time,
        )

    async def _recover_kernel_death(self, exc: KernelDiedError, start_time: float) -> ExecuteResponse:
        await self._set_execution_state(KernelExecutionState.RECOVERING)
        recovered = await self._restart_after_kernel_death()
        await self._set_execution_state(KernelExecutionState.IDLE if recovered else KernelExecutionState.WEDGED)
        self._active_msg_id = None
        recovery_message = (
            "the kernel was restarted; in-memory variables were lost, but workspace files were preserved"
            if recovered
            else "automatic kernel restart failed; the kernel is unavailable"
        )
        exit_detail = f" (exit code {exc.exit_code})" if exc.exit_code is not None else ""
        message = f"Jupyter kernel exited during code execution{exit_detail}; {recovery_message}"
        error_output = MessageType.ERROR.to_notebook_output({
            "ename": "KernelDiedError",
            "evalue": message,
            "traceback": [f"KernelDiedError: {message}"],
        })
        return ExecuteResponse(
            notebook_outputs=[dict(error_output)] if error_output else [],
            error_occurred=True,
            execution_time=time.perf_counter() - start_time,
            kernel_restarted=recovered,
            kernel_state_lost=True,
            kernel_exit_code=exc.exit_code,
        )

    async def _claim_execution(self) -> ExecuteResponse | None:
        """Claim the Jupyter channel without queuing behind active work."""
        async with self._state_lock:
            if self._execution_state == KernelExecutionState.IDLE:
                self._execution_state = KernelExecutionState.EXECUTING
                return None
            if self._execution_state == KernelExecutionState.WEDGED:
                error_name = "KernelUnresponsiveError"
                message = "Kernel did not return to idle after a previous timeout"
            else:
                error_name = "KernelBusyError"
                message = f"Kernel is {self._execution_state.value}; request was not queued"

        output = MessageType.ERROR.to_notebook_output({
            "ename": error_name,
            "evalue": message,
            "traceback": [f"{error_name}: {message}"],
        })
        return ExecuteResponse(
            notebook_outputs=[dict(output)] if output else [],
            error_occurred=True,
            execution_time=0.0,
        )

    async def _set_execution_state(self, state: KernelExecutionState) -> None:
        async with self._state_lock:
            self._execution_state = state

    async def _interrupt_and_drain(self, msg_id: str, grace_seconds: float) -> bool:
        """Interrupt one execution and consume its IOPub stream through idle."""
        if not self._kernel_manager or not self._client:
            return False

        deadline = time.perf_counter() + grace_seconds
        try:
            async with asyncio.timeout(grace_seconds):
                await self._kernel_manager.interrupt_kernel()
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return False
                    try:
                        msg = await self._client.get_iopub_msg(timeout=min(0.5, remaining))
                    except Empty:
                        continue
                    if msg["parent_header"].get("msg_id") != msg_id:
                        continue
                    if (
                        MessageType.from_string(msg["msg_type"]) == MessageType.STATUS
                        and msg["content"].get("execution_state") == "idle"
                    ):
                        return True
        except TimeoutError:
            return False
        except Exception:
            logger.exception("Failed to interrupt Jupyter request %s", msg_id)
            return False

    async def _restart_after_kernel_death(self) -> bool:
        """Replace only the dead Jupyter process, preserving the sandbox workspace."""
        async with self._kernel_lifecycle_lock:
            await self._dispose_kernel_locked()
            try:
                await self._launch_kernel_locked()
            except Exception:
                logger.exception("Failed to restart dead Jupyter kernel")
                return False
        return True

    async def _execute_code(self, code: str, deadline: float, *, store_history: bool = True) -> ExecuteResponse:
        """Internal method to execute code and collect outputs.

        Uses cooperative deadline checking instead of asyncio.timeout, because
        ZMQ socket operations may not respond to asyncio cancellation promptly.
        Each get_iopub_msg call uses a short poll timeout so we can check the
        deadline between messages.
        """
        if not self._client:
            raise RuntimeError("Kernel client not initialized")

        # How long each ZMQ poll waits before we re-check the deadline.
        # Shorter = more responsive timeout, slightly more overhead.
        POLL_INTERVAL_S = 2.0

        start_time = time.perf_counter()
        # Bootstrap calls disable history so they do not consume a visible
        # notebook execution count. Keep IOPub enabled so startup errors are
        # still observable and can fail the kernel start.
        msg_id = self._client.execute(code, store_history=store_history)
        self._active_msg_id = msg_id

        notebook_outputs: list[dict[str, Any]] = []
        error_occurred = False

        while True:
            # Check deadline before each poll
            if time.perf_counter() >= deadline:
                raise DeadlineExceededError(msg_id)

            # Use a bounded poll so we never block longer than _POLL_INTERVAL_S.
            # get_iopub_msg(timeout=T) raises queue.Empty if no message arrives
            # within T seconds.
            try:
                msg = await self._client.get_iopub_msg(timeout=POLL_INTERVAL_S)
            except Empty:
                if not await self._kernel_process_is_alive():
                    raise KernelDiedError(msg_id, self._kernel_exit_code()) from None
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = MessageType.from_string(msg["msg_type"])
            if msg_type is None:
                continue

            content = msg["content"]

            if msg_type == MessageType.STATUS and content.get("execution_state") == "idle":
                break

            if msg_type == MessageType.ERROR:
                logger.debug(f"Error Message:\n{content}")

            output = msg_type.to_notebook_output(content)
            if output:
                notebook_outputs.append(dict(output))
                if msg_type == MessageType.ERROR:
                    error_occurred = True

        execution_time = time.perf_counter() - start_time

        return ExecuteResponse(
            notebook_outputs=notebook_outputs,
            error_occurred=error_occurred,
            execution_time=execution_time,
        )

    async def reset(self, seed: int | None = None) -> ResetResponse:
        """Reset the kernel to a clean state."""
        await self._cancel_active_execution_jobs()
        if seed is not None:
            self.seed = seed
        async with self._kernel_lifecycle_lock:
            await self._dispose_kernel_locked()
            await self._launch_kernel_locked()
        await self._set_execution_state(KernelExecutionState.IDLE)
        return ResetResponse(success=True, seed=self.seed)

    async def close(self) -> None:
        """Shutdown the kernel."""
        await self._cancel_active_execution_jobs()
        async with self._kernel_lifecycle_lock:
            await self._dispose_kernel_locked()
        await self._set_execution_state(KernelExecutionState.CLOSED)
        logger.info("Kernel shutdown complete")

    def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        """List the workspace directory (confined to work_dir) with truncation protection."""
        try:
            max_files = int(max_files)
        except (TypeError, ValueError):
            max_files = 20
        show_hidden = bool(show_hidden)

        root = self.work_dir.resolve()
        requested = Path(directory)
        candidate = (requested if requested.is_absolute() else root / requested).resolve()
        if candidate != root and root not in candidate.parents:
            return f"Path must stay within the workspace root: {directory}"
        if not candidate.exists() or not candidate.is_dir():
            return f"Path is not a directory: {directory}"

        paths = _collect_dir_paths(candidate, show_hidden=show_hidden)
        if not paths:
            return "Directory is empty."
        if len(paths) > max_files:
            shown = paths[:max_files]
            return (
                "Files in directory:\n"
                + "\n".join(f"  {p}" for p in shown)
                + f"\n  ({len(paths) - max_files} more files not shown)"
            )
        return "Files in directory:\n" + "\n".join(f"  {p}" for p in paths)


def create_app(server: KernelServer) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Kernel Server")

    @app.post(
        "/execute",
        response_model=ExecuteSubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def execute(
        req: ExecuteRequest,
        request_id: Annotated[str | None, Header(alias="X-Req-UUID")] = None,
    ) -> ExecuteSubmissionResponse:
        try:
            return await server.submit_execution(req, request_id)
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.get("/execute/{execution_id}", response_model=ExecutePollResponse)
    async def get_execution(execution_id: str) -> ExecutePollResponse:
        execution = await server.get_execution(execution_id)
        if execution is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown or expired execution")
        return execution

    @app.post("/execute/{execution_id}/cancel", response_model=ExecutePollResponse)
    async def cancel_execution(execution_id: str) -> ExecutePollResponse:
        execution = await server.cancel_execution(execution_id)
        if execution is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown or expired execution")
        return execution

    @app.post("/reset")
    async def reset(req: ResetRequest | None = None) -> ResetResponse:
        return await server.reset(req.seed if req is not None else None)

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(
            status="OK",
            startup_token=server.startup_token,
            kernel_ready=await server.kernel_ready(),
        )

    @app.get("/list_dir")
    async def list_dir(directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> ListDirResponse:
        return ListDirResponse(listing=server.list_dir(directory, max_files, show_hidden))

    @app.post("/close")
    async def close() -> dict[str, bool]:
        await server.close()
        return {"success": True}

    return app


async def run_server(
    work_dir: Path,
    language: NBLanguage,
    port: int = 8000,
    startup_token: str = "",
    seed: int | None = None,
    kernel_memory_limit_mb: int | None = None,
) -> None:
    """Start the kernel server."""
    server = KernelServer(
        work_dir,
        language,
        startup_token=startup_token,
        seed=seed,
        kernel_memory_limit_mb=kernel_memory_limit_mb,
    )
    await server.start()

    app = create_app(server)

    config = uvicorn.Config(app, host="0.0.0.0", port=port, loop="asyncio")  # noqa: S104
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Kernel server for Docker-based execution")
    parser.add_argument("--work_dir", type=Path, default=Path("/"))
    parser.add_argument("--language", type=str, default="python", choices=["python", "r"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-token", type=str, default="")
    parser.add_argument("--safe-execute", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--kernel-memory-limit-mb", type=int)
    args = parser.parse_args()

    language = NBLanguage.PYTHON if args.language == "python" else NBLanguage.R
    asyncio.run(
        run_server(
            args.work_dir,
            language,
            args.port,
            args.startup_token,
            args.seed,
            args.kernel_memory_limit_mb,
        )
    )
