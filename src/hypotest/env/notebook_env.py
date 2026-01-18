from typing import cast

from aviary.core import EnvStateMessage, Messages, ToolRequestMessage

from .interpreter_env import InterpreterEnv, cfg
from .utils import view_notebook


class NotebookEnv(InterpreterEnv):
    async def run_cell(
        self,
        code: str,
        idx: int | None = None,
    ) -> str:
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

        if result.error_occurred:
            return f"Failed to execute cell {actual_cell_idx}. Please fix the error and try again."
        return f"Code executed successfully in cell {actual_cell_idx}."

    async def step(self, action: ToolRequestMessage) -> tuple[Messages, float, bool, bool]:
        """Execute a step in the environment."""
        self.step_count += 1
        obs = cast(
            Messages,
            await self.exec_tool_calls(action, concurrency=False, handle_tool_exc=True),
        )

        obs = [*obs, self.get_env_state_msg()]

        time_msg = self.get_time_management_message()
        if time_msg is not None:
            obs.append(time_msg)

        # if self.step_count >= (self.max_steps - 1):
        #     obs.append(Message(content=cfg.FORCE_MSG))

        self.state.actions.append(str(action))
        reward = self.state.score if self.state.done else 0.0
        return obs, reward, self.state.done, False

    def get_env_state_msg(self) -> EnvStateMessage:
        return EnvStateMessage(content=view_notebook(self.state.nb.cells, self.language.value))
