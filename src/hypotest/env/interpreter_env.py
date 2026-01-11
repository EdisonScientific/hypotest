"""InterpreterEnv: Standalone code execution environment for data analysis.

This module provides a lightweight, execution-focused environment for running
code in Jupyter kernels. It focuses on direct code execution via run_cell().
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import socket
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, cast
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
from pydantic import BaseModel, Field, JsonValue

from . import config as cfg
from .config import ExecutionConfig
from .interpreter import ExecutionResult, Interpreter
from .prompts import CORRECT_MSG, INCORRECT_MSG, RUBRIC_SCORE_PROMPT, PromptingConfig
from .tools.filesystem import list_dir_tool
from .utils import NBLanguage, view_notebook

if TYPE_CHECKING:
    from aiodocker.containers import DockerContainer


# Port management for Docker containers
_USED_PORTS: set[int] = set()


def get_free_port() -> int:
    """Get a free port for the kernel server container."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        if port not in _USED_PORTS:
            _USED_PORTS.add(port)
            return port


logger = logging.getLogger(__name__)


class ProblemInstance(BaseModel):
    uuid: UUID
    hypothesis: str
    objective: str
    accepted: bool = Field(alias="answer")
    rubric: str
    max_score: int = Field(alias="max_points")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


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
        self.save_dir = save_dir

        # Local interpreter (only used when use_docker=False)
        self.interpreter: Interpreter | None = None
        if not use_docker:
            self.interpreter = Interpreter(
                work_dir=work_dir,
                language=language,
                execution_timeout=execution_timeout,
                use_host_env_vars=use_host_env_vars,
                extra_envs=extra_envs,
            )

        # Docker container state (only used when use_docker=True)
        self._docker_client: aiodocker.Docker | None = None
        self._container: DockerContainer | None = None
        self._container_port: int | None = None
        self._http_client: httpx.AsyncClient | None = None

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
        if self.use_docker:
            await self._start_container()
        else:
            assert self.interpreter is not None
            await self.interpreter.start()

    async def _start_container(self) -> None:
        """Start a Docker container with the kernel server."""
        self._docker_client = aiodocker.Docker()
        self._container_port = get_free_port()

        docker_config = {
            "Image": cfg.NB_ENVIRONMENT_DOCKER_IMAGE,
            "Cmd": [
                "/app/kernel_env/bin/python",
                "/envs/kernel_server.py",
                "--work_dir",
                "/workspace",
                "--language",
                self.language.value,
            ],
            "HostConfig": {
                "Binds": [f"{self.work_dir}:/workspace"],
                "PortBindings": {f"{cfg.KERNEL_SERVER_PORT}/tcp": [{"HostPort": str(self._container_port)}]},
            },
            "WorkingDir": "/workspace",
            "Tty": True,
            "ExposedPorts": {f"{cfg.KERNEL_SERVER_PORT}/tcp": {}},
        }

        self._container = await self._docker_client.containers.run(config=cast(dict[str, Any], docker_config))
        logger.info(f"Started container on port {self._container_port}")

        # Create HTTP client
        self._http_client = httpx.AsyncClient(
            base_url=f"http://localhost:{self._container_port}",
            timeout=httpx.Timeout(self.execution_timeout + 10, connect=30.0),
        )

        # Wait for health check
        await self._wait_for_health()

    async def _wait_for_health(self) -> None:
        """Wait for the kernel server to become healthy."""
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < cfg.KERNEL_SERVER_STARTUP_TIMEOUT:
            try:
                assert self._http_client is not None
                response = await self._http_client.get("/health")
                if response.status_code == 200:
                    logger.info("Kernel server is healthy")
                    return
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Kernel server did not become healthy within {cfg.KERNEL_SERVER_STARTUP_TIMEOUT}s")

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.ReadError)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _execute_via_http(self, code: str, timeout: float | None = None) -> ExecutionResult:  # noqa: ASYNC109
        """Execute code via HTTP to the containerized kernel server."""
        assert self._http_client is not None

        response = await self._http_client.post(
            "/execute",
            json={"code": code, "timeout": timeout},
        )
        response.raise_for_status()
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
        response = await self._http_client.post("/reset")
        response.raise_for_status()

    async def close(self):
        """Save the notebook and close the interpreter or container."""
        nbformat.write(self.nb, self.work_dir / "notebook.ipynb")

        if self.save_dir is not None:
            shutil.rmtree(self.save_dir, ignore_errors=True)
            self.save_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(self.work_dir, self.save_dir)

        if self.use_docker:
            if self._container_port is not None:
                _USED_PORTS.discard(self._container_port)

            if self._http_client is not None:
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
        if self.use_docker:
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
    use_docker: bool = cfg.USE_DOCKER
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

        self.input_data = input_data
        self.output_data: list[dict[str, str | int]] = []
        self.logger = logger
        self.start_time: float | None = None
        self.step_count = 0
        self.include_env_state_msg = include_env_state_msg
        self.state: InterpreterEnvState
        # prompting_config is set during reset() after language resolution
        self.prompting_config: PromptingConfig

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
            use_docker=self.config.use_docker,
        )
        await self.state.start()

        # Record start time for timeout tracking
        self.start_time = time.perf_counter()

        messages = []
        if self.prompting_config.system_prompt:
            messages.append(Message(role="system", content=self.prompting_config.system_prompt))

        # Define core tools
        self.tools = [
            Tool.from_function(self.run_cell),
            Tool.from_function(self.reset_kernel),
            Tool.from_function(self.submit_answer),
            Tool.from_function(list_dir_tool),
        ]

        # TODO: FORMAT THIS PROPELRY
        messages.append(Message(content=self.problem.hypothesis))

        if self.include_env_state_msg:
            messages.append(self.get_env_state_msg())

        # Always show initial directory listing (with truncation protection)
        messages.append(Message(content=list_dir_tool(str(self.work_dir))))

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
        result, actual_cell_idx = await self.state.execute_and_add_cell(
            code, cell_idx=cell_idx, timeout=effective_timeout
        )

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
        if self.state.use_docker:
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
