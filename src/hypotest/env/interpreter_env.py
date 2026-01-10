"""InterpreterEnv: Standalone code execution environment for data analysis.

This module provides a lightweight, execution-focused environment for running
code in Jupyter kernels. It focuses on direct code execution via run_cell().
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import nbformat
import numpy as np
from aviary.core import (
    EnvStateMessage,
    Frame,
    Message,
    Messages,
    Tool,
    ToolRequestMessage,
)
from aviary.env import Environment
from heron.utils.workspace_utils import (
    collect_input_data,
    fetch_continuation_data,
    install_config,
    resolve_auto_language,
    validate_workspace_path,
)
from lmi.cost_tracker import GLOBAL_COST_TRACKER, enable_cost_tracking
from nbformat import NotebookNode
from pydantic import BaseModel, Field, model_validator

from . import config as cfg
from . import prompts
from .config import ExecutionConfig
from .interpreter import Interpreter
from .prompts import LLMConfig, PromptingConfig
from .tools.filesystem import list_dir_tool
from .tools.registry import resolve_tools
from .utils import (
    NBLanguage,
    TrajectoryLoggerAdapter,
    download_files_from_data_storage_service,
    upload_files_from_trajectory_run,
)

if TYPE_CHECKING:
    from .interpreter import ExecutionResult

logger = logging.getLogger(__name__)


class InterpreterConfig(BaseModel):
    """Configuration for preparing the InterpreterEnv during task creation."""

    # Core configs
    user_id: str | None = None
    trajectory_id: str | None = None
    language: NBLanguage | None = Field(default=None, description="None means 'auto' - to be resolved via LLM")
    data_storage_uris: list[str] = Field(default_factory=list)

    # Continuation-specific fields
    continued_trajectory_id: str | None = None
    previous_research_question: str | None = None
    previous_final_answer: str | None = None

    # Nested configs
    prompting_config: PromptingConfig = Field(default_factory=PromptingConfig)
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Agent behavior limits
    max_steps: int = cfg.AGENT_MAX_STEPS

    # Environment kwargs extracted from environment_config
    additional_tools: list[str] | None = Field(
        default=None,
        description="Additional tools to register beyond the defaults",
    )

    @model_validator(mode="after")
    def validate_config_requirements(self) -> "InterpreterConfig":
        """Validate that required conditions are met."""
        if not self.continued_trajectory_id and not self.data_storage_uris:
            logger.warning("Running jobs without data_storage_uris")
        return self


class InterpreterEnvState:
    """State container for the InterpreterEnv.

    Manages the kernel, notebook state, and execution tracking.
    """

    def __init__(
        self,
        work_dir: Path,
        language: NBLanguage,
        execution_timeout: int = 600,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
    ):
        self.work_dir = work_dir
        self.language = language
        self.total_reward = 0.0
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}
        self.answer: str | float | int | dict[str, Any] | None = None
        self.actions: list[str] = []
        self.done = False

        self.use_kernel_isolation = cfg.USE_KERNEL_ISOLATION
        # Create kernel metadata directory OUTSIDE work_dir
        self.kernel_meta_dir = work_dir.parent / f".kernel_meta_{work_dir.name}"
        self.kernel_meta_dir.mkdir(exist_ok=True)

        # Initialize interpreter (kernel manager)
        self.interpreter = Interpreter(
            work_dir=work_dir,
            language=language,
            execution_timeout=execution_timeout,
            use_host_env_vars=use_host_env_vars,
            extra_envs=extra_envs,
            use_kernel_isolation=self.use_kernel_isolation,
            kernel_meta_dir=self.kernel_meta_dir,
        )

        # Initialize notebook structure for state tracking
        self.nb: NotebookNode = nbformat.v4.new_notebook()
        self.nb.metadata.kernelspec = language.make_kernelspec()
        self.notebook_runtime_errors: list[str] = []
        self._execution_count = 0

    async def start(self):
        """Start the interpreter."""
        await self.interpreter.start()

    async def close(self):
        """Close the interpreter."""
        await self.interpreter.close()
        if self.kernel_meta_dir.exists():
            shutil.rmtree(self.kernel_meta_dir)

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

    async def execute_and_add_cell(
        self,
        code: str,
        cell_idx: int | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> tuple["ExecutionResult", int]:
        """Execute code and atomically update notebook.

        Args:
            code: Code to execute
            cell_idx: Cell index to update (None = append new cell)
            timeout: Optional execution timeout

        Returns:
            Tuple of (ExecutionResult, actual_cell_index)
        """
        result = await self.interpreter.execute_code(code, timeout)

        if cell_idx is None or cell_idx >= len(self.nb.cells):
            actual_idx = self._add_cell(code, result)
        else:
            self._update_cell(cell_idx, code, result)
            actual_idx = cell_idx

        return result, actual_idx


class InterpreterEnv(Environment[InterpreterEnvState]):
    """Standalone environment for code execution and data analysis.

    This environment provides direct code execution via run_cell() without
    requiring notebook file I/O. It maintains an in-memory notebook for
    trajectory tracking and state export.
    """

    def __init__(
        self,
        *,
        problem: str,
        work_dir: Path,
        config: InterpreterConfig | None = None,
        additional_tools: list[str] | None = None,
        enabled_subagents: list[str] | None = None,
        input_data: list[dict[str, str | int | None]] | None = None,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
        persist_outputs: bool = True,
        include_env_state_msg: bool = False,
    ):
        if config is None:
            config = InterpreterConfig()
        self.config = config
        self.work_dir = work_dir
        self.language = config.language  # Convenience attribute, may be None for auto
        self.done = False
        self.problem = problem
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}

        # Execution config for timeouts and capabilities
        self.execution_config = self.config.execution_config
        self.execution_timeout = self.execution_config.cell_execution_timeout
        self.max_steps = self.config.max_steps

        self.additional_tools = additional_tools if additional_tools is not None else cfg.DEFAULT_ADDITIONAL_TOOLS
        self.enabled_subagents = enabled_subagents
        self.input_data = input_data
        self.output_data: list[dict[str, str | int]] = []
        self.trajectory_id = config.trajectory_id
        self.logger = TrajectoryLoggerAdapter(logger, {"trajectory_id": self.trajectory_id or "unknown"})
        self.start_time: float | None = None
        self.step_count = 0
        self.persist_outputs = persist_outputs
        self.include_env_state_msg = include_env_state_msg
        self.state: InterpreterEnvState
        # prompting_config is set during reset() after language resolution
        self.prompting_config: PromptingConfig

    async def close(self) -> None:
        """Close the environment, save notebook, and upload files."""
        await super().close()

        # Save notebook to disk
        if self.state is not None:
            nb_dir = self.work_dir / "notebooks"
            nb_dir.mkdir(exist_ok=True)
            nb_path = nb_dir / f"{self.trajectory_id}.ipynb"
            try:
                nbformat.write(self.state.nb, nb_path)
                self.logger.info(f"Saved notebook to {nb_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save notebook: {e}")

            await self.state.close()

        self.logger.info("Closing environment")
        if self.persist_outputs:
            self.logger.info("Uploading files to data storage service")
            self.output_data = upload_files_from_trajectory_run(
                input_data=self.input_data or [],
                work_dir=self.work_dir,
                trajectory_id=self.trajectory_id or "",
            )

    async def reset(self) -> tuple[Messages, list[Tool]]:
        """Reset the environment and prepare for execution."""
        # Handle language resolution (must happen before state creation
        # because InterpreterEnvState needs the resolved language)
        if self.language is None:
            self.language = await resolve_auto_language(
                self.language, self.problem, self.config.llm_config.utility_model
            )
            self.logger.info(f"Language resolved to: {self.language}")

        # Interpolate prompting config now that language is resolved
        language_display = self.language.value.capitalize()

        # Format environment capabilities with job_timeout
        env_capabilities = self.execution_config.environment_capabilities_prompt.format(
            job_timeout=self.execution_config.job_timeout
        )

        self.prompting_config = self.config.prompting_config.interpolate(
            language=language_display,
            environment_capabilities=env_capabilities,
            job_timeout=self.execution_config.job_timeout,
        )

        # Handle continuation prompts if applicable
        if self.config.continued_trajectory_id:
            self.problem = prompts.INTERPRETER_CONTINUATION_PROMPT_TEMPLATE.format(
                previous_research_question=self.config.previous_research_question or "",
                previous_final_answer=self.config.previous_final_answer or "",
                query=self.problem,
            )
            self.logger.info("Applied continuation prompt template")

        # Set up config directory (must be writable)
        config_dir = self.work_dir / ".config"
        if config_dir.exists():
            shutil.rmtree(config_dir)

        install_config(config_dir)

        # Use kernel environment paths for isolated execution
        kernel_env_path = Path(cfg.KERNEL_ENV_PATH)
        kernel_site_packages = kernel_env_path / "lib" / "python3.12" / "site-packages"

        self.state = InterpreterEnvState(
            work_dir=self.work_dir,
            language=self.language,
            execution_timeout=self.execution_timeout,
            use_host_env_vars=self.use_host_env_vars,
            extra_envs={
                "XDG_CONFIG_HOME": str(config_dir),
                # on non-linux, matploltlib will ignore XDG_CONFIG_HOME
                # (this is for unit tests on mac)
                "MPLCONFIGDIR": str(config_dir / "matplotlib"),
                # Point to kernel environment's site-packages
                "PYTHONPATH": str(kernel_site_packages),
                # Include kernel environment bin in PATH
                "PATH": (str(kernel_env_path / "bin") + os.pathsep + os.environ.get("PATH", "")),
                # R library path for user-installed packages
                "R_LIBS_USER": str(kernel_env_path / "lib" / "R" / "library"),
            }
            | self.extra_envs,
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
        ]

        # Add additional tools from registry (includes filesystem tools with list_dir)
        if self.additional_tools:
            additional = resolve_tools(
                self.additional_tools,
                self.work_dir,
                enabled_subagents=self.enabled_subagents,
            )
            self.tools.extend(additional)

        messages.append(Message(content=self.problem))

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
        return obs, 0, self.state.done, False

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
        await self.state.interpreter.reset()

        # Reset notebook state to match kernel reset
        self.state.nb = nbformat.v4.new_notebook()
        self.state.nb.metadata.kernelspec = self.state.language.make_kernelspec()
        self.state.notebook_runtime_errors = []
        self.state._execution_count = 0

        return "Kernel reset successfully."

    async def submit_answer(self, answer: str) -> str:
        """Submit your response to the research question.

        Args:
            answer: Your final response to the research question
        """
        self.state.answer = answer
        self.state.done = True
        self.logger.info("Submitting answer: %s", answer)
        return answer

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
            },
            info={
                "language": self.state.language,
                "problem": self.problem,
                "cost": GLOBAL_COST_TRACKER.lifetime_cost_usd,
                "work_dir": self.work_dir,
                "input_data": self.input_data,
                "output_data": self.output_data,
            },
        )

    def get_env_state_msg(self) -> EnvStateMessage:
        """Get the current environment state message."""
        summary = self.state.interpreter.get_execution_summary()

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

    # ========== Factory Methods ==========

    @staticmethod
    def _build_config(
        trajectory_id: str | None = None,
        user_id: str | None = None,
        environment_config: dict[str, Any] | None = None,
        continued_trajectory_id: str | None = None,
    ) -> InterpreterConfig:
        """Build and validate config given task specification."""
        safe_env_config = environment_config or {}
        safe_user_id = user_id or "default_user"
        safe_trajectory_id = trajectory_id or f"trajectory-{time.time()}"

        data_storage_uris = safe_env_config.get("data_storage_uris", [])
        language_str = safe_env_config.get("language", "AUTO").upper()
        prompting_config = PromptingConfig(**safe_env_config.get("prompting_config", {}))
        llm_config = LLMConfig(**safe_env_config.get("llm_config", {}))

        # Build execution config from deployment profile
        profile = os.getenv("DEPLOYMENT_PROFILE") or safe_env_config.get("deployment_profile", "standard")
        exec_config = ExecutionConfig.from_profile(profile)

        language = NBLanguage.from_string(language_str)

        # Config validation is performed automatically by Pydantic model validators
        return InterpreterConfig(
            user_id=safe_user_id,
            trajectory_id=safe_trajectory_id,
            continued_trajectory_id=continued_trajectory_id,
            language=language,
            data_storage_uris=data_storage_uris,
            additional_tools=safe_env_config.get("additional_tools"),
            prompting_config=prompting_config,
            llm_config=llm_config,
            execution_config=exec_config,
        )

    @classmethod
    def from_task(
        cls,
        task: str,
        *,
        trajectory_id: str | None = None,
        user_id: str | None = None,
        environment_config: dict[str, Any] | None = None,
        continued_trajectory_id: str | None = None,
        user_jwt: str | None = None,
    ) -> "InterpreterEnv":
        """Create an InterpreterEnv from a task specification.

        Args:
            task: The user query/research question
            trajectory_id: Unique identifier for this trajectory
            user_id: Unique identifier for the user
            environment_config: Environment configuration dict
            continued_trajectory_id: ID of trajectory to continue from
            user_jwt: JWT token for the user

        Returns:
            Configured InterpreterEnv instance
        """
        logger.info(
            "\n".join([
                f"User task: {task[:50]}",
                f"environment_config: {environment_config}",
                f"trajectory_id: {trajectory_id}",
                f"user_id: {user_id}",
                f"continued_trajectory_id: {continued_trajectory_id}",
            ])
        )

        enable_cost_tracking()

        # Build configuration
        config = cls._build_config(
            trajectory_id=trajectory_id,
            user_id=user_id,
            environment_config=environment_config,
            continued_trajectory_id=continued_trajectory_id,
        )

        config = fetch_continuation_data(config)
        workspace_path = Path(cfg.DATA_STORAGE_PATH) if cfg.STAGE == "local" else cfg.DATA_STORAGE_PATH / "workspace"

        logger.info("Creating InterpreterEnv with workspace_path: %s", workspace_path)

        # Download files
        workspace_path, files_downloaded = download_files_from_data_storage_service(
            workspace_path,
            config.data_storage_uris,
            user_jwt=user_jwt,
            trajectory_id=trajectory_id,
        )

        validate_workspace_path(workspace_path)

        input_data = collect_input_data(
            workspace_path,
            files_downloaded,
            config.data_storage_uris,
        )

        # Create environment - language resolution and prompt interpolation happen in reset()
        return cls(
            problem=task,
            work_dir=workspace_path,
            config=config,
            additional_tools=config.additional_tools,
            user_jwt=user_jwt,
            input_data=input_data,
        )
