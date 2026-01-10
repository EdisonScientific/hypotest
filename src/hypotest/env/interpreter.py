"""Unified kernel management module for code interpretation.

This module provides the Interpreter class for managing Jupyter kernels
and executing code, with optional security isolation via SecureKernelManager.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import nbformat
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import AsyncKernelManager
from nbformat import NotebookNode
from pydantic import BaseModel, ConfigDict, Field

from . import config as cfg
from . import utils

logger = logging.getLogger(__name__)


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


class ExecutionResult(BaseModel):
    """Structured result from kernel code execution.

    Stores notebook outputs in nbformat format as the single source of truth.
    Text and images are derived lazily from notebook_outputs when needed.
    """

    notebook_outputs: list[NotebookNode] = Field(default_factory=list)
    error_occurred: bool = False
    execution_time: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _extract_text_from_output(output: NotebookNode) -> str | None:
        """Extract text from a single notebook output.

        Args:
            output: A NotebookNode output from cell execution

        Returns:
            Formatted text string or None if no text available
        """
        output_type = output.get("output_type", "")

        if output_type == MessageType.STREAM:
            name = output.get("name", "stdout")
            text = output.get("text", "")
            return f"[{name}]\n{text}"

        if output_type in {MessageType.EXECUTE_RESULT, MessageType.DISPLAY_DATA}:
            data = output.get("data", {})
            # Check for images first to add placeholder text
            text_parts = ["[Image generated]"] if utils.JUPYTER_IMAGE_OUTPUT_TYPES.intersection(data.keys()) else []
            # Add text/plain if available
            if "text/plain" in data:
                text_parts.append(data["text/plain"])
            return "\n".join(text_parts) if text_parts else None

        if output_type == MessageType.ERROR:
            traceback = output.get("traceback", [])
            traceback_str = "\n".join(traceback) if isinstance(traceback, list) else traceback
            return (
                f"Error: {output.get('ename', 'Unknown')}\n"
                f"Message: {output.get('evalue', 'No error message')}\n"
                f"Traceback:\n{traceback_str}"
            )

        return None

    @staticmethod
    def _extract_images_from_output(output: NotebookNode) -> list[tuple[str, str]]:
        """Extract images from a single notebook output.

        Args:
            output: A NotebookNode output from cell execution

        Returns:
            List of (mime_type, base64_data) tuples
        """
        images: list[tuple[str, str]] = []
        output_type = output.get("output_type", "")

        if output_type in {MessageType.EXECUTE_RESULT, MessageType.DISPLAY_DATA}:
            data = output.get("data", {})
            for img_type in utils.JUPYTER_IMAGE_OUTPUT_TYPES:
                if img_type in data:
                    try:
                        encoded = utils.encode_image_to_base64(data[img_type])
                        images.append((img_type, encoded))
                    except RuntimeError:
                        logger.exception("Error encoding image.")

        return images

    def get_text_outputs(self) -> list[str]:
        """Extract formatted text from all notebook outputs.

        Returns:
            List of text strings extracted from outputs
        """
        return [text for output in self.notebook_outputs if (text := self._extract_text_from_output(output))]

    def get_images(self) -> list[tuple[str, str]]:
        """Extract images from all notebook outputs.

        Returns:
            List of (mime_type, base64_data) tuples
        """
        return [img for output in self.notebook_outputs for img in self._extract_images_from_output(output)]

    def get_combined_text(self) -> str:
        """Get all text outputs combined as a single string."""
        text_outputs = self.get_text_outputs()
        if not text_outputs:
            return "Code executed successfully (no output)"
        return "\n".join(text_outputs)

    def has_images(self) -> bool:
        """Check if execution result contains images."""
        return any(self._extract_images_from_output(output) for output in self.notebook_outputs)

    def get_truncated_text(self) -> str:
        """Get the combined text, truncated to the output limit."""
        return utils.limit_notebook_output(self.get_combined_text())

    def get_error_message(self) -> str | None:
        """Extract the error message from outputs if an error occurred.

        Returns:
            Formatted error message or None if no error
        """
        if not self.error_occurred:
            return None

        for output in self.notebook_outputs:
            if output.get("output_type") == MessageType.ERROR:
                return self._extract_text_from_output(output)
        return None

    def to_message(self) -> dict[str, Any]:
        """Convert ExecutionResult to MCP tool result format.

        Returns a dict with 'content' array containing text and/or images.
        The SDK will automatically wrap this in a ToolResultBlock.
        """
        content: list[dict[str, Any]] = []

        # Always include text output (even if there are images)
        text = self.get_combined_text()
        if text:
            content.append({"type": "text", "text": text})

        # Add any images - now using extracted tuples directly
        for mime_type, base64_data in self.get_images():
            content.append({
                "type": "image",
                "mimeType": mime_type,
                "data": base64_data,
            })

        return {"content": content}


class Interpreter:
    """Manages Python/R interpreter kernels for code execution.

    This class handles kernel lifecycle, code execution, and maintains
    execution history and error tracking. It supports both AsyncKernelManager
    (development) and SecureKernelManager (production with isolation).
    """

    def __init__(
        self,
        work_dir: Path,
        language: utils.NBLanguage = utils.NBLanguage.PYTHON,
        *,
        execution_timeout: float = 600,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
        kernel_meta_dir: Path | None = None,
    ):
        """Initialize the interpreter.

        Args:
            work_dir: Working directory for the kernel
            language: Programming language (Python or R)
            execution_timeout: Timeout for code execution in seconds
            use_host_env_vars: Whether to use host environment variables
            extra_envs: Additional environment variables to pass to the kernel
            kernel_meta_dir: Directory for kernel metadata (connection files, etc.)
        """
        self.work_dir = work_dir
        self.language = language
        self.execution_timeout = execution_timeout
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}
        self.kernel_meta_dir = kernel_meta_dir

        # Execution state
        self.execution_history: list[ExecutionResult] = []

        # Kernel state
        self.kernel_manager: AsyncKernelManager
        self.client: AsyncKernelClient | None = None
        self._is_ready = False

    async def start(self) -> None:
        """Start the kernel and prepare for execution."""
        if self._is_ready:
            return

        kernel_name = self.language.make_kernelspec()["name"]
        self.kernel_manager = AsyncKernelManager(kernel_name=kernel_name)

        # Prepare kernel startup kwargs with environment variables
        kwargs: dict[str, Any] = {"cwd": str(self.work_dir)}
        if not self.use_host_env_vars:
            kwargs["env"] = {
                required_env_var: os.environ[required_env_var]
                for required_env_var in cfg.REQUIRED_PATH_ENV_VARS
                if os.environ.get(required_env_var)
            } | self.extra_envs
        else:
            kwargs["env"] = os.environ | self.extra_envs

        await self.kernel_manager.start_kernel(**kwargs)
        self.client = self.kernel_manager.client()
        self.client.start_channels()

        try:
            await self.client.wait_for_ready()
            self._is_ready = True
            logger.debug(f"Kernel {kernel_name} started successfully in {self.work_dir}")
        except Exception as e:
            raise RuntimeError(f"Kernel failed to start: {e}") from e

    async def _execute_code(self, code: str) -> ExecutionResult:
        """Internal method to execute code and collect outputs.

        Uses MessageType.to_notebook_output to convert kernel messages
        to nbformat outputs, storing them as the single source of truth.
        """
        if not self.client:
            raise ValueError("Kernel client not initialized")

        start_time = time.perf_counter()
        msg_id = self.client.execute(code)
        logger.debug(f"Executing code with message ID: {msg_id}")

        notebook_outputs: list[NotebookNode] = []
        error_occurred = False

        while True:
            msg = await self.client.get_iopub_msg()
            logger.debug(f"Received message type: {msg['msg_type']}")

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = MessageType.from_string(msg["msg_type"])
            if msg_type is None:
                continue  # Unknown message type, skip

            content = msg["content"]

            if msg_type == MessageType.STATUS and content.get("execution_state") == "idle":
                break

            output = msg_type.to_notebook_output(content)
            if output:
                notebook_outputs.append(output)
                if msg_type == MessageType.ERROR:
                    error_occurred = True

        execution_time = time.perf_counter() - start_time

        return ExecutionResult(
            notebook_outputs=notebook_outputs,
            error_occurred=error_occurred,
            execution_time=execution_time,
        )

    async def execute_code(
        self,
        code: str,
        execution_timeout: float | None = None,
        extract_code: bool = False,
    ) -> ExecutionResult:
        r"""Execute code in the kernel session.

        The code will be executed in the current session context, maintaining
        all variables, imports, and state from previous executions.

        Code wrapped in markdown backticks (```code```, ```\ncode\n```, or
        ```language\ncode\n```) will be extracted if extract_code is True.

        Args:
            code: Code to execute
            execution_timeout: Optional timeout in seconds. If None, uses the
                instance's execution_timeout. Useful for dynamic timeout based
                on remaining job time.
            extract_code: Whether to extract code from md backticks and language identifiers.

        Returns:
            ExecutionResult containing text outputs and images
        """
        # Preprocess code to extract from markdown backticks and language identifiers
        if extract_code:
            code = utils.extract_code_from_markdown(code)

        if not self._is_ready:
            await self.start()

        # Use provided timeout or fall back to instance default
        timeout = execution_timeout if execution_timeout is not None else self.execution_timeout

        try:
            async with asyncio.timeout(timeout):
                result = await self._execute_code(code)
        except TimeoutError:
            timeout_output = MessageType.ERROR.to_notebook_output({
                "ename": "TimeoutError",
                "evalue": f"Code execution timed out after {timeout} seconds",
                "traceback": [f"TimeoutError: Code execution timed out after {timeout} seconds"],
            })
            result = ExecutionResult(
                notebook_outputs=[cast(NotebookNode, timeout_output)],
                error_occurred=True,
            )
        except Exception as e:
            error_output = MessageType.ERROR.to_notebook_output({
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [f"{type(e).__name__}: {e}"],
            })
            result = ExecutionResult(
                notebook_outputs=[cast(NotebookNode, error_output)],
                error_occurred=True,
            )

        self.execution_history.append(result)
        return result

    async def execute_cells(self, cells: list[NotebookNode], cell_idx: int | None = None) -> list[str]:
        """Execute notebook cells using the kernel.

        This method supports the notebook workflow by executing cells
        and handling notebook-specific output formats.

        Args:
            cells: List of notebook cells
            cell_idx: Specific cell index to execute, or None to execute all

        Returns:
            List of error messages (empty if no errors)
        """
        if not self._is_ready:
            await self.start()

        if not self.client:
            raise ValueError("Kernel client not initialized")

        try:
            async with asyncio.timeout(self.execution_timeout):
                error_messages = await utils.nbformat_run_notebook(cells=cells, client=self.client, cell_idx=cell_idx)
        except TimeoutError as err:
            raise TimeoutError(f"Cell execution timed out after {self.execution_timeout} seconds") from err

        return error_messages

    async def reset(self) -> None:
        """Reset the kernel to a clean state."""
        if self._is_ready:
            await self.close()
        await self.start()

        # Clear execution history
        self.execution_history.clear()

    async def close(self) -> None:
        """Shutdown the kernel and cleanup resources."""
        if self._is_ready:
            # Properly stop client channels first
            if self.client:
                self.client.stop_channels()
                self.client = None

            # Then shutdown the kernel
            await self.kernel_manager.shutdown_kernel(now=True)

            # Clean up the kernel manager
            self._is_ready = False
            logger.debug("Kernel shutdown complete")

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of execution history and current state.

        Returns:
            Dictionary with execution statistics and recent activity
        """
        error_count = sum(1 for r in self.execution_history if r.error_occurred)
        recent_errors = [
            r.get_error_message() for r in self.execution_history[-3:] if r.error_occurred and r.get_error_message()
        ]

        return {
            "total_executions": len(self.execution_history),
            "error_count": error_count,
            "recent_errors": recent_errors,
            "last_execution": (self.execution_history[-1] if self.execution_history else None),
            "is_ready": self._is_ready,
            "language": self.language.value,
            "work_dir": str(self.work_dir),
        }

    @property
    def is_ready(self) -> bool:
        """Check if the kernel is ready for execution."""
        return self._is_ready
