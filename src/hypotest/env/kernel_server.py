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
import time
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, assert_never

import nbformat
import uvicorn
from fastapi import FastAPI
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import AsyncKernelManager
from nbformat import NotebookNode
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


class KernelServer:
    """Manages a persistent Jupyter kernel and exposes it via HTTP."""

    def __init__(
        self,
        work_dir: Path,
        language: NBLanguage,
        default_timeout: float = 600,
    ):
        self.work_dir = work_dir
        self.language = language
        self.default_timeout = default_timeout

        self._kernel_manager: AsyncKernelManager | None = None
        self._client: AsyncKernelClient | None = None
        self._is_ready = False

    async def start(self) -> None:
        """Start the Jupyter kernel."""
        if self._is_ready:
            return

        kernel_name = self.language.make_kernelspec()["name"]
        self._kernel_manager = AsyncKernelManager(kernel_name=kernel_name)
        await self._kernel_manager.start_kernel(cwd=str(self.work_dir))

        self._client = self._kernel_manager.client()
        self._client.start_channels()

        try:
            await self._client.wait_for_ready()
            self._is_ready = True
            logger.info(f"Kernel {kernel_name} started in {self.work_dir}")
        except Exception as e:
            raise RuntimeError(f"Kernel failed to start: {e}") from e

    async def execute(self, code: str, timeout: float | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        """Execute code and return the result."""
        if not self._client or not self._is_ready:
            raise RuntimeError("Kernel not ready")

        effective_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.perf_counter()

        try:
            async with asyncio.timeout(effective_timeout):
                result = await self._execute_code(code)
        except TimeoutError:
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

    async def _execute_code(self, code: str) -> ExecuteResponse:
        """Internal method to execute code and collect outputs."""
        if not self._client:
            raise RuntimeError("Kernel client not initialized")

        start_time = time.perf_counter()
        msg_id = self._client.execute(code)
        logger.debug(f"Executing code with message ID: {msg_id}")

        notebook_outputs: list[dict[str, Any]] = []
        error_occurred = False

        while True:
            msg = await self._client.get_iopub_msg()
            logger.debug(f"Received message type: {msg['msg_type']}")

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = MessageType.from_string(msg["msg_type"])
            if msg_type is None:
                continue

            content = msg["content"]

            if msg_type == MessageType.STATUS and content.get("execution_state") == "idle":
                break

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

    async def reset(self) -> ResetResponse:
        """Reset the kernel to a clean state."""
        if self._is_ready and self._kernel_manager:
            if self._client:
                self._client.stop_channels()
                self._client = None

            await self._kernel_manager.shutdown_kernel(now=True)
            self._is_ready = False

        await self.start()
        return ResetResponse(success=True)

    async def close(self) -> None:
        """Shutdown the kernel."""
        if self._is_ready and self._kernel_manager:
            if self._client:
                self._client.stop_channels()
                self._client = None

            await self._kernel_manager.shutdown_kernel(now=True)
            self._is_ready = False
            logger.info("Kernel shutdown complete")


def create_app(server: KernelServer) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Kernel Server")

    @app.post("/execute")
    async def execute(req: ExecuteRequest) -> ExecuteResponse:
        return await server.execute(req.code, req.timeout)

    @app.post("/reset")
    async def reset() -> ResetResponse:
        return await server.reset()

    @app.get("/health")
    async def health() -> str:
        return "OK"

    @app.post("/close")
    async def close() -> dict[str, bool]:
        await server.close()
        return {"success": True}

    return app


async def run_server(work_dir: Path, language: NBLanguage, port: int = 8000) -> None:
    """Start the kernel server."""
    server = KernelServer(work_dir, language)
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
    parser.add_argument("--work_dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--language", type=str, default="python", choices=["python", "r"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    language = NBLanguage.PYTHON if args.language == "python" else NBLanguage.R
    asyncio.run(run_server(args.work_dir, language, args.port))
