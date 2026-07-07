"""Standalone kernel server for Docker-based execution.

This module runs inside the container and provides an HTTP API for code execution.
It also contains shared types (NBLanguage, MessageType) that hypotest imports.

IMPORTANT: This module must be standalone with no imports from hypotest package,
as it gets copied into the Docker image and run independently.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import time
import uuid
from enum import StrEnum, auto
from pathlib import Path
from queue import Empty
from typing import Any, assert_never

import nbformat
import uvicorn
from fastapi import FastAPI
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import AsyncKernelManager
from nbformat import NotebookNode
from pydantic import BaseModel


class DeadlineExceededError(Exception):
    """Raised when a cooperative deadline check expires."""


logger = logging.getLogger(__name__)


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
PROTOCOL_VERSION = 1


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


class ExecuteResponse(BaseModel):
    """Response model for /execute endpoint.

    Contains serialized notebook outputs that can be deserialized back to NotebookNode.
    """

    notebook_outputs: list[dict[str, Any]]
    error_occurred: bool
    execution_time: float | None


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
    ):
        self.work_dir = work_dir
        self.language = language
        self.default_timeout = default_timeout
        self.startup_token = startup_token
        self.safe_execute = safe_execute
        self.seed = seed

        self._kernel_manager: AsyncKernelManager | None = None
        self._client: AsyncKernelClient | None = None
        self._is_ready = False

    async def start(self) -> None:
        """Start the Jupyter kernel."""
        if self._is_ready:
            return

        kernel_name = self.language.make_kernelspec()["name"]
        kernel_runtime_dir = self.work_dir / ".jupyter_runtime"
        kernel_runtime_dir.mkdir(exist_ok=True)

        conn_uuid = uuid.uuid4()
        kernel_connect_file = (kernel_runtime_dir / f"conn_{conn_uuid}.json").resolve()

        self._kernel_manager = AsyncKernelManager(
            kernel_name=kernel_name, transport="ipc", connection_file=str(kernel_connect_file)
        )
        kernel_env = os.environ.copy()
        if self.seed is not None:
            kernel_env.update(deterministic_kernel_env(self.seed))
        await self._kernel_manager.start_kernel(cwd=str(self.work_dir), env=kernel_env)

        self._client = self._kernel_manager.client()
        self._client.start_channels()

        try:
            await self._client.wait_for_ready()
            if self.seed is not None:
                bootstrap = await self._execute_code(
                    rng_bootstrap_code(self.language, self.seed),
                    deadline=time.perf_counter() + self.default_timeout,
                    store_history=False,
                )
                if bootstrap.error_occurred:
                    raise RuntimeError("Deterministic kernel RNG bootstrap failed")
            self._is_ready = True
            logger.info(f"Kernel {kernel_name} started in {self.work_dir}")
        except Exception as e:
            raise RuntimeError(f"Kernel failed to start: {e}") from e

    async def execute(self, code: str, timeout: float | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        """Execute code and return the result."""
        if not self._client or not self._is_ready:
            raise RuntimeError("Kernel not ready")

        if self.safe_execute:
            # Defense-in-depth: lightweight regex safety check
            block_reason = _kernel_check_code_safety(code)
            if block_reason is not None:
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

        effective_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.perf_counter()

        try:
            result = await self._execute_code(
                code,
                deadline=start_time + effective_timeout,
            )
        except DeadlineExceededError:
            timeout_output = MessageType.ERROR.to_notebook_output({
                "ename": "TimeoutError",
                "evalue": f"Code execution timed out after {effective_timeout} seconds",
                "traceback": [f"TimeoutError: Code execution timed out after {effective_timeout} seconds"],
            })
            result = ExecuteResponse(
                notebook_outputs=[dict(timeout_output)] if timeout_output else [],
                error_occurred=True,
                execution_time=time.perf_counter() - start_time,
            )
        except Exception as e:
            error_output = MessageType.ERROR.to_notebook_output({
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [f"{type(e).__name__}: {e}"],
            })
            result = ExecuteResponse(
                notebook_outputs=[dict(error_output)] if error_output else [],
                error_occurred=True,
                execution_time=time.perf_counter() - start_time,
            )

        return result

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

        notebook_outputs: list[dict[str, Any]] = []
        error_occurred = False

        while True:
            # Check deadline before each poll
            if time.perf_counter() >= deadline:
                raise DeadlineExceededError

            # Use a bounded poll so we never block longer than _POLL_INTERVAL_S.
            # get_iopub_msg(timeout=T) raises queue.Empty if no message arrives
            # within T seconds.
            try:
                msg = await self._client.get_iopub_msg(timeout=POLL_INTERVAL_S)
            except Empty:
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
        if seed is not None:
            self.seed = seed
        if self._is_ready and self._kernel_manager:
            if self._client:
                self._client.stop_channels()
                self._client = None

            await self._kernel_manager.shutdown_kernel(now=True)
            self._is_ready = False

        await self.start()
        return ResetResponse(success=True, seed=self.seed)

    async def close(self) -> None:
        """Shutdown the kernel."""
        if self._is_ready and self._kernel_manager:
            if self._client:
                self._client.stop_channels()
                self._client = None

            await self._kernel_manager.shutdown_kernel(now=True)
            await self._kernel_manager.cleanup_resources(restart=False)
            self._is_ready = False
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

    @app.post("/execute")
    async def execute(req: ExecuteRequest) -> ExecuteResponse:
        return await server.execute(req.code, req.timeout)

    @app.post("/reset")
    async def reset(req: ResetRequest | None = None) -> ResetResponse:
        return await server.reset(req.seed if req is not None else None)

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(status="OK", startup_token=server.startup_token, kernel_ready=server._is_ready)

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
) -> None:
    """Start the kernel server."""
    server = KernelServer(work_dir, language, startup_token=startup_token, seed=seed)
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
    args = parser.parse_args()

    language = NBLanguage.PYTHON if args.language == "python" else NBLanguage.R
    asyncio.run(run_server(args.work_dir, language, args.port, args.startup_token, args.seed))
