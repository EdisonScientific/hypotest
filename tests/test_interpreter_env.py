"""Tests for the InterpreterEnv class."""

import base64
import json
import os
import pathlib
import tempfile
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch
from uuid import UUID

import pytest
import pytest_asyncio
from aviary.core import Message, ToolCall, ToolRequestMessage, ToolResponseMessage
from lmi import LiteLLMModel

from hypotest.env import config as cfg
from hypotest.env.config import ExecutionConfig
from hypotest.env.interpreter_env import (
    InterpreterEnv,
    InterpreterEnvConfig,
    InterpreterEnvState,
    ProblemInstance,
)
from hypotest.env.kernel_server import NBLanguage

from .conftest import requires_matplotlib, should_skip_docker_test


@pytest_asyncio.fixture
async def interpreter_env(default_problem: ProblemInstance) -> AsyncGenerator[InterpreterEnv, Any]:
    """Fixture that creates and cleans up InterpreterEnv (local mode)."""
    with tempfile.TemporaryDirectory() as tmp:
        env = InterpreterEnv(
            problem=default_problem,
            work_dir=pathlib.Path(tmp),
            config=InterpreterEnvConfig(language=NBLanguage.PYTHON),
        )
        await env.reset()
        try:
            yield env
        finally:
            await env.close()


class TestInterpreterConfig:
    """Tests for the InterpreterConfig class."""

    def test_interpreter_config_defaults(self):
        """Test InterpreterConfig has sensible defaults."""
        config = InterpreterEnvConfig()
        assert config.language == NBLanguage.PYTHON
        assert isinstance(config.execution_config, ExecutionConfig)
        assert config.max_steps == cfg.AGENT_MAX_STEPS

    def test_interpreter_config_with_language(self):
        """Test InterpreterConfig with explicit language."""
        config = InterpreterEnvConfig(language=NBLanguage.R)
        assert config.language == NBLanguage.R

    def test_interpreter_config_has_execution_config(self):
        """Test InterpreterConfig has execution_config field with defaults."""
        config = InterpreterEnvConfig()
        assert hasattr(config, "execution_config")
        assert isinstance(config.execution_config, ExecutionConfig)
        assert config.execution_config.job_timeout == 60 * 60
        assert config.execution_config.force_submit_threshold == 10 * 60
        assert config.execution_config.warn_submit_threshold == 20 * 60
        assert config.execution_config.cell_execution_timeout == 15 * 60
        assert not config.execution_config.has_gpu

    def test_interpreter_config_max_steps_custom(self):
        """Test InterpreterConfig accepts custom max_steps value."""
        config = InterpreterEnvConfig(max_steps=30)
        assert config.max_steps == 30


class TestInterpreterEnvState:
    """Tests for InterpreterEnvState class."""

    @pytest.mark.asyncio
    async def test_interpreter_env_state_creation(self):
        """Test InterpreterEnvState can be created and started."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=False,
            )
            await state.start()

            try:
                assert state.work_dir == work_dir
                assert state.language == NBLanguage.PYTHON
                assert state.answer is None
                assert not state.done
                assert len(state.nb.cells) == 0
            finally:
                await state.close()

    @pytest.mark.asyncio
    async def test_interpreter_env_state_notebook_metadata(self):
        """Test that notebook has correct kernelspec metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=False,
            )
            await state.start()

            try:
                assert state.nb.metadata.kernelspec is not None
                assert state.nb.metadata.kernelspec["language"] == "python"
            finally:
                await state.close()

    @pytest.mark.asyncio
    async def test_interpreter_env_state_execute_and_add_cell(self):
        """Test execute_and_add_cell adds cells correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=False,
            )
            await state.start()

            try:
                result, idx = await state.execute_and_add_cell("print('hello')")
                assert idx == 0
                assert not result.error_occurred
                assert len(state.nb.cells) == 1
                assert state.nb.cells[0].source == "print('hello')"

                _result2, idx2 = await state.execute_and_add_cell("x = 42")
                assert idx2 == 1
                assert len(state.nb.cells) == 2
            finally:
                await state.close()

    @pytest.mark.asyncio
    async def test_interpreter_env_state_edit_cell(self):
        """Test execute_and_add_cell can edit existing cells."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=False,
            )
            await state.start()

            try:
                await state.execute_and_add_cell("x = 1")
                _result, idx = await state.execute_and_add_cell("x = 42", cell_idx=0)
                assert idx == 0
                assert len(state.nb.cells) == 1
                assert state.nb.cells[0].source == "x = 42"
            finally:
                await state.close()

    @pytest.mark.asyncio
    async def test_interpreter_env_state_get_execution_summary(self):
        """Test get_execution_summary returns correct data."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=False,
            )
            await state.start()

            try:
                await state.execute_and_add_cell("print('hello')")
                summary = state.get_execution_summary()
                assert summary["total_executions"] >= 1
                assert summary["is_ready"] is True
                assert summary["language"] == "python"
            finally:
                await state.close()


class TestInterpreterEnv:
    """Tests for InterpreterEnv class."""

    @pytest.mark.asyncio
    async def test_interpreter_env_reset(self, interpreter_env: InterpreterEnv):
        """Test that reset creates tools and state."""
        messages, tools = await interpreter_env.reset()

        assert len(messages) >= 2
        assert any("Test hypothesis" in str(m.content) for m in messages)

        tool_names = {t.info.name for t in tools}
        assert "run_cell" in tool_names
        assert "submit_answer" in tool_names
        assert "reset_kernel" in tool_names
        assert "list_dir" in tool_names

    @pytest.mark.asyncio
    async def test_interpreter_env_run_cell_append(self, interpreter_env: InterpreterEnv):
        """Test run_cell appends new cells."""
        result = await interpreter_env.run_cell("print('hello')")

        assert "[Cell #0]" in str(result)
        assert "hello" in str(result)

        assert len(interpreter_env.state.nb.cells) == 1
        assert interpreter_env.state.nb.cells[0].source == "print('hello')"

    @pytest.mark.asyncio
    async def test_interpreter_env_run_cell_edit(self, interpreter_env: InterpreterEnv):
        """Test run_cell edits existing cells."""
        await interpreter_env.run_cell("x = 1")
        assert len(interpreter_env.state.nb.cells) == 1

        await interpreter_env.run_cell("print(x)")
        assert len(interpreter_env.state.nb.cells) == 2

        result = await interpreter_env.run_cell("x = 42", idx=0)
        assert "[Cell #0]" in str(result)
        assert len(interpreter_env.state.nb.cells) == 2
        assert interpreter_env.state.nb.cells[0].source == "x = 42"

    @requires_matplotlib
    @pytest.mark.asyncio
    async def test_interpreter_env_run_cell_with_images(self, interpreter_env: InterpreterEnv):
        """Test run_cell returns Message when images are present."""
        await interpreter_env.reset()

        code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 2, 3])
plt.show()
"""
        result = await interpreter_env.run_cell(code)

        assert isinstance(result, Message)
        assert isinstance(result.content, str)
        assert result.content_is_json_str is True

        parsed = json.loads(result.content)
        assert isinstance(parsed, list)
        content_types = {item.get("type") for item in parsed}
        assert "text" in content_types
        assert "image_url" in content_types

    @pytest.mark.asyncio
    async def test_interpreter_env_reset_kernel(self, interpreter_env: InterpreterEnv):
        """Test reset_kernel clears state."""
        await interpreter_env.run_cell("x = 42")
        assert len(interpreter_env.state.nb.cells) == 1

        result = await interpreter_env.reset_kernel()
        assert "reset" in result.lower()
        assert len(interpreter_env.state.nb.cells) == 0

    @pytest.mark.asyncio
    async def test_interpreter_env_submit_answer(self, interpreter_env: InterpreterEnv):
        """Test submit_answer sets state correctly."""
        await interpreter_env.reset()

        result = await interpreter_env.submit_answer("The answer is 42")
        assert result == "The answer is 42"
        assert interpreter_env.state.answer == "The answer is 42"
        assert interpreter_env.state.done

    @pytest.mark.asyncio
    async def test_interpreter_env_initial_dir_listing_in_reset(self, default_problem: ProblemInstance):
        """Test that reset() includes initial directory listing in messages."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            (work_dir / "test.txt").write_text("test content")
            (work_dir / "data.csv").write_text("a,b,c")

            env = InterpreterEnv(
                problem=default_problem,
                work_dir=work_dir,
                config=InterpreterEnvConfig(language=NBLanguage.PYTHON),
            )

            try:
                messages, _ = await env.reset()

                dir_listing = None
                for msg in messages:
                    content = str(msg.content)
                    if "test.txt" in content or "data.csv" in content:
                        dir_listing = content
                        break

                assert dir_listing is not None
                assert "test.txt" in dir_listing
                assert "data.csv" in dir_listing
            finally:
                await env.close()

    @pytest.mark.asyncio
    async def test_interpreter_env_get_env_state_msg(self, interpreter_env: InterpreterEnv):
        """Test get_env_state_msg returns execution summary."""
        await interpreter_env.run_cell("print('hello')")

        msg = interpreter_env.get_env_state_msg()
        content = str(msg.content)

        assert "Interpreter Environment" in content
        assert "Execution History: 1" in content

    @pytest.mark.asyncio
    async def test_interpreter_env_time_management(self, default_problem: ProblemInstance):
        """Test time management message generation."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            env = InterpreterEnv(
                problem=default_problem,
                work_dir=work_dir,
                config=InterpreterEnvConfig(language=NBLanguage.PYTHON),
            )

            try:
                await env.reset()

                remaining = env.get_remaining_time()
                assert remaining > 0

                msg = env.get_time_management_message()
                assert msg is None
            finally:
                await env.close()


class TestInterpreterEnvRunCell:
    """Tests for the run_cell method in InterpreterEnv."""

    @pytest.mark.asyncio
    async def test_run_cell_append_new_cell(self, interpreter_env: InterpreterEnv):
        """Test that run_cell appends a new cell when idx is None."""
        result = await interpreter_env.run_cell(code="print('hello')")

        assert "[Cell #0]" in str(result)
        assert "hello" in str(result)

        assert interpreter_env.state is not None
        assert len(interpreter_env.state.nb.cells) == 1
        assert interpreter_env.state.nb.cells[0].source == "print('hello')"

    @pytest.mark.asyncio
    async def test_run_cell_append_multiple_cells(self, interpreter_env: InterpreterEnv):
        """Test that run_cell appends cells sequentially with correct indices."""
        result1 = await interpreter_env.run_cell(code="x = 1")
        assert "[Cell #0]" in str(result1)

        result2 = await interpreter_env.run_cell(code="y = 2")
        assert "[Cell #1]" in str(result2)

        result3 = await interpreter_env.run_cell(code="print(x + y)")
        assert "[Cell #2]" in str(result3)
        assert "3" in str(result3)

        assert interpreter_env.state is not None
        assert len(interpreter_env.state.nb.cells) == 3

    @pytest.mark.asyncio
    async def test_run_cell_edit_existing_cell(self, interpreter_env: InterpreterEnv):
        """Test that run_cell can edit an existing cell by index."""
        result1 = await interpreter_env.run_cell(code="print(undefined_var)")
        assert "[Cell #0]" in str(result1)
        assert "NameError" in str(result1) or "Error" in str(result1)

        result2 = await interpreter_env.run_cell(code="print('fixed')", idx=0)
        assert "[Cell #0]" in str(result2)
        assert "fixed" in str(result2)

        assert interpreter_env.state is not None
        assert len(interpreter_env.state.nb.cells) == 1
        assert interpreter_env.state.nb.cells[0].source == "print('fixed')"

    @pytest.mark.asyncio
    async def test_run_cell_out_of_bounds_idx_appends(self, interpreter_env: InterpreterEnv):
        """Test that run_cell appends when idx is out of bounds."""
        result = await interpreter_env.run_cell(code="print('appended')", idx=5)

        assert "[Cell #0]" in str(result)
        assert "appended" in str(result)
        assert interpreter_env.state is not None
        assert len(interpreter_env.state.nb.cells) == 1

    @requires_matplotlib
    @pytest.mark.asyncio
    async def test_run_cell_with_images(self, interpreter_env: InterpreterEnv):
        """Test that run_cell returns Message with images when plot is generated."""
        plot_code = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [1, 2, 3])\nplt.show()"

        result = await interpreter_env.run_cell(code=plot_code)

        assert isinstance(result, Message)
        assert isinstance(result.content, str)
        assert result.content_is_json_str is True

        parsed = json.loads(result.content)
        assert isinstance(parsed, list)
        content_types = {item.get("type") for item in parsed}
        assert "text" in content_types
        assert "image_url" in content_types

        text_item = next(item for item in parsed if item.get("type") == "text")
        assert "[Cell #0]" in text_item.get("text", "")

    @pytest.mark.asyncio
    async def test_run_cell_registered_as_tool(self, interpreter_env: InterpreterEnv):
        """Test that run_cell is registered as a tool after reset."""
        tool_names = [t.info.name for t in interpreter_env.tools]
        assert "run_cell" in tool_names

    @pytest.mark.asyncio
    async def test_run_cell_error_tracking(self, interpreter_env: InterpreterEnv):
        """Test that errors are tracked in notebook_runtime_errors."""
        await interpreter_env.run_cell(code="raise ValueError('test error')")

        assert interpreter_env.state.notebook_runtime_errors
        assert any("Error" in err for err in interpreter_env.state.notebook_runtime_errors)

    @pytest.mark.asyncio
    async def test_run_cell_returns_force_msg_when_time_low(self, interpreter_env: InterpreterEnv):
        """Test that run_cell returns FORCE_MSG when remaining time is below threshold."""
        force_threshold = interpreter_env.execution_config.force_submit_threshold
        with patch.object(
            interpreter_env,
            "get_remaining_time",
            return_value=force_threshold - 10,
        ):
            result = await interpreter_env.run_cell(code="print('hello')")
            assert result == cfg.FORCE_MSG

    @pytest.mark.asyncio
    async def test_run_cell_passes_dynamic_timeout_to_execute_code(self, interpreter_env: InterpreterEnv):
        """Test that run_cell passes the effective timeout to execute_code."""
        force_threshold = interpreter_env.execution_config.force_submit_threshold
        remaining_time = force_threshold + 200.0
        expected_effective_timeout = remaining_time - force_threshold

        captured_timeout = None

        original_execute = interpreter_env.state.execute_and_add_cell

        async def mock_execute_and_add_cell(code, cell_idx=None, timeout=None):  # noqa: ASYNC109
            nonlocal captured_timeout
            captured_timeout = timeout
            return await original_execute(code, cell_idx, timeout)

        with (
            patch.object(interpreter_env, "get_remaining_time", return_value=remaining_time),
            patch.object(
                interpreter_env.state,
                "execute_and_add_cell",
                side_effect=mock_execute_and_add_cell,
            ),
        ):
            await interpreter_env.run_cell(code="print('test')")
            assert captured_timeout is not None
            assert captured_timeout == expected_effective_timeout


# ========== Helper Functions for Multimodal Tests ==========


def create_tool_request(tool_name: str, **kwargs) -> ToolRequestMessage:
    """Create a ToolRequestMessage for a single tool call."""
    tool_call = ToolCall.from_name(tool_name, **kwargs)
    return ToolRequestMessage(content="", tool_calls=[tool_call])


def create_test_png_file(path: pathlib.Path) -> None:
    """Create a minimal PNG file for testing."""
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    path.write_bytes(png_data)


class TestMultimodalToolOutputs:
    """Tests for multimodal tool outputs via step() method."""

    @requires_matplotlib
    @pytest.mark.asyncio
    async def test_step_run_cell_with_plot_returns_correct_multimodal_format(self, interpreter_env: InterpreterEnv):
        """Test run_cell with plot returns properly formatted multimodal response."""
        plot_code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 2, 3])
plt.title('Test Plot')
plt.show()
"""
        action = create_tool_request("run_cell", code=plot_code)

        obs, _reward, _done, _truncated = await interpreter_env.step(action)

        response = obs[0]
        assert isinstance(response, ToolResponseMessage)
        assert response.name == "run_cell"

        assert isinstance(response.content, str)
        assert response.content_is_json_str is True

        parsed = json.loads(response.content)
        assert isinstance(parsed, list)

        content_types = {item.get("type") for item in parsed}
        assert "text" in content_types
        assert "image_url" in content_types

        text_item = next(item for item in parsed if item.get("type") == "text")
        assert "[Cell #" in text_item.get("text", "")

        image_item = next(item for item in parsed if item.get("type") == "image_url")
        assert image_item.get("image_url", {}).get("url", "").startswith("data:image/")

    @pytest.mark.asyncio
    async def test_step_run_cell_text_only_returns_string_content(self, interpreter_env: InterpreterEnv):
        """Test that run_cell with text-only output returns plain string content."""
        action = create_tool_request("run_cell", code="print('hello world')")

        obs, _reward, _done, _truncated = await interpreter_env.step(action)

        response = obs[0]
        assert isinstance(response, ToolResponseMessage)
        assert isinstance(response.content, str)

        assert response.content_is_json_str is False

        assert "[Cell #" in response.content
        assert "hello world" in response.content

    @requires_matplotlib
    @pytest.mark.asyncio
    async def test_step_multimodal_response_has_tool_call_id(self, interpreter_env: InterpreterEnv):
        """Test that multimodal response has correct tool_call_id."""
        plot_code = """
import matplotlib.pyplot as plt
plt.plot([1, 2], [1, 2])
plt.show()
"""
        tool_call = ToolCall.from_name("run_cell", code=plot_code)
        action = ToolRequestMessage(content="", tool_calls=[tool_call])

        obs, _reward, _done, _truncated = await interpreter_env.step(action)

        response = obs[0]
        assert isinstance(response, ToolResponseMessage)
        assert response.tool_call_id == tool_call.id


# ========== Docker Mode Tests ==========


class TestInterpreterEnvDocker:
    """Tests for InterpreterEnv in Docker mode.

    These tests are parameterized to run in both local and Docker modes.
    Docker tests will be skipped if Docker is not available or the image is not found.
    """

    @pytest.mark.parametrize("use_docker", [False, True])
    @pytest.mark.asyncio
    async def test_interpreter_env_basic_execution(self, use_docker: bool):
        """Test basic code execution in both local and Docker modes."""
        if should_skip_docker_test(use_docker):
            pytest.skip("Docker not available or image not found")

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            # Create state directly with use_docker parameter
            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=use_docker,
            )

            try:
                await state.start()

                result, idx = await state.execute_and_add_cell("print('hello')")
                assert idx == 0
                assert not result.error_occurred
                assert "hello" in result.get_combined_text()

                assert len(state.nb.cells) == 1
            finally:
                await state.close()

    @pytest.mark.parametrize("use_docker", [False, True])
    @pytest.mark.asyncio
    async def test_interpreter_env_state_persistence(self, use_docker: bool):
        """Test that state persists across executions in both modes."""
        if should_skip_docker_test(use_docker):
            pytest.skip("Docker not available or image not found")

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=use_docker,
            )

            try:
                await state.start()

                await state.execute_and_add_cell("x = 42")
                result, _ = await state.execute_and_add_cell("print(x)")

                assert not result.error_occurred
                assert "42" in result.get_combined_text()
            finally:
                await state.close()

    @pytest.mark.parametrize("use_docker", [False, True])
    @pytest.mark.asyncio
    async def test_interpreter_env_error_handling(self, use_docker: bool):
        """Test error handling in both local and Docker modes."""
        if should_skip_docker_test(use_docker):
            pytest.skip("Docker not available or image not found")

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=use_docker,
            )

            try:
                await state.start()

                result, _ = await state.execute_and_add_cell("raise ValueError('test error')")

                assert result.error_occurred
                combined = result.get_combined_text()
                assert "ValueError" in combined
                assert "test error" in combined
            finally:
                await state.close()

    @requires_matplotlib
    @pytest.mark.parametrize("use_docker", [False, True])
    @pytest.mark.asyncio
    async def test_interpreter_env_plot_generation(self, use_docker: bool):
        """Test plot generation in both local and Docker modes."""
        if should_skip_docker_test(use_docker):
            pytest.skip("Docker not available or image not found")

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=use_docker,
            )

            try:
                await state.start()

                code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 2, 3])
plt.show()
"""
                result, _ = await state.execute_and_add_cell(code)

                assert not result.error_occurred
                assert result.has_images()
            finally:
                await state.close()

    @pytest.mark.parametrize("use_docker", [False, True])
    @pytest.mark.asyncio
    async def test_interpreter_env_file_access(self, use_docker: bool):
        """Test file access from kernel in both local and Docker modes."""
        if should_skip_docker_test(use_docker):
            pytest.skip("Docker not available or image not found")

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            # Create a test file in the work directory
            test_file = work_dir / "test_data.txt"
            test_file.write_text("hello from file")

            state = InterpreterEnvState(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
                use_docker=use_docker,
            )

            try:
                await state.start()

                # Read the file from within the kernel
                code = """
with open('test_data.txt', 'r') as f:
    print(f.read())
"""
                result, _ = await state.execute_and_add_cell(code)

                assert not result.error_occurred
                assert "hello from file" in result.get_combined_text()
            finally:
                await state.close()

    @pytest.mark.parametrize("use_docker", [False, True])
    @pytest.mark.asyncio
    async def test_interpreter_env_full_workflow(self, use_docker: bool, default_problem: ProblemInstance):
        """Test complete InterpreterEnv workflow in both modes."""
        if should_skip_docker_test(use_docker):
            pytest.skip("Docker not available or image not found")

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            (work_dir / "data.txt").write_text("test data")

            # Patch USE_DOCKER config to control mode
            with patch.object(cfg, "USE_DOCKER", use_docker):
                env = InterpreterEnv(
                    problem=default_problem,
                    work_dir=work_dir,
                    config=InterpreterEnvConfig(language=NBLanguage.PYTHON),
                )

                try:
                    messages, tools = await env.reset()

                    assert len(messages) >= 2
                    tool_names = {t.info.name for t in tools}
                    assert "run_cell" in tool_names

                    result = await env.run_cell("print('hello')")
                    assert "[Cell #0]" in str(result)
                    assert "hello" in str(result)

                    result = await env.reset_kernel()
                    assert "reset" in result.lower()
                    assert len(env.state.nb.cells) == 0
                finally:
                    await env.close()


class TestListDirTool:
    """Tests for list_dir_tool behavior in InterpreterEnv."""

    @pytest.mark.asyncio
    async def test_list_dir_respects_work_dir(self, default_problem: ProblemInstance):
        """Test that list_dir called with relative path uses work_dir, not process CWD."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            (work_dir / "work_dir_file.txt").write_text("in work_dir")
            (work_dir / "data").mkdir()
            (work_dir / "data" / "nested.csv").write_text("nested")

            env = InterpreterEnv(
                problem=default_problem,
                work_dir=work_dir,
                config=InterpreterEnvConfig(language=NBLanguage.PYTHON),
            )

            try:
                await env.reset()

                original_cwd = os.getcwd()
                assert pathlib.Path(original_cwd) != work_dir, "Test requires CWD != work_dir"

                action = ToolRequestMessage(tool_calls=[ToolCall.from_name("list_dir", directory=".")])
                obs, _, _, _ = await env.step(action)
                result = obs[0].content
                assert result is not None

                assert "work_dir_file.txt" in result, (
                    f"list_dir('.') should list work_dir contents, not CWD. Got: {result}"
                )
                assert "data/nested.csv" in result

                action_subdir = ToolRequestMessage(tool_calls=[ToolCall.from_name("list_dir", directory="data")])
                obs_subdir, _, _, _ = await env.step(action_subdir)
                result_subdir = obs_subdir[0].content
                assert result_subdir is not None

                assert "nested.csv" in result_subdir, (
                    f"list_dir('data') should list work_dir/data contents. Got: {result_subdir}"
                )

            finally:
                await env.close()


class TestToolSchemas:
    """Tests for tool schema stability and correctness."""

    EXPECTED_SCHEMAS_PATH = pathlib.Path(__file__).parent / "fixtures" / "expected_tool_schemas.json"

    @pytest.mark.asyncio
    async def test_tool_schemas_match_expected(self, default_problem: ProblemInstance):
        """Test that tool schemas returned by reset() match expected schemas.

        This test validates that docstrings and annotations are properly propagated
        to the tool schemas. The expected schemas are stored in a JSON file and
        should remain stable across refactors (e.g., switching to FilesystemTool).
        """
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            env = InterpreterEnv(
                problem=default_problem,
                work_dir=work_dir,
                config=InterpreterEnvConfig(language=NBLanguage.PYTHON),
            )

            try:
                _, tools = await env.reset()

                with self.EXPECTED_SCHEMAS_PATH.open() as f:
                    expected_schemas = json.load(f)

                actual_schemas = {tool.info.name: tool.info.model_dump() for tool in tools}

                assert set(actual_schemas.keys()) == set(expected_schemas.keys()), (
                    f"Tool names mismatch. "
                    f"Expected: {set(expected_schemas.keys())}, "
                    f"Actual: {set(actual_schemas.keys())}"
                )

                for tool_name, expected_schema in expected_schemas.items():
                    actual_schema = actual_schemas[tool_name]
                    assert actual_schema == expected_schema, (
                        f"Schema mismatch for tool '{tool_name}'.\n"
                        f"Expected:\n{json.dumps(expected_schema, indent=2)}\n"
                        f"Actual:\n{json.dumps(actual_schema, indent=2)}"
                    )

            finally:
                await env.close()


class TestRubricGrading:
    """Tests for rubric-based grading."""

    GRADING_PROBLEM = ProblemInstance(
        uuid=UUID("12345678-1234-5678-1234-567812345678"),
        hypothesis="Numbers greater than 10 are large",
        objective="Load data, compute statistics, determine if hypothesis holds",
        answer=True,
        rubric="""* 1 point: Data is loaded correctly
* 1 point: Statistics are computed
* 1 point: Correct conclusion is reached""",
        max_points=3,
    )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("code", "answer"),
        [
            pytest.param([], "True", id="empty_notebook_correct_answer"),
            pytest.param(["data = [15, 20, 25]"], "True", id="partial_steps_correct_answer"),
            pytest.param(
                ["data = [15, 20, 25]", "mean = sum(data) / len(data)\nprint(f'Mean: {mean}')"],
                "True",
                id="all_steps_correct_answer",
            ),
            pytest.param(
                ["data = [15, 20, 25]", "mean = sum(data) / len(data)\nprint(f'Mean: {mean}')"],
                "False",
                id="all_steps_incorrect_answer",
            ),
        ],
    )
    async def test_rubric_grading(self, code: list[str], answer: str):
        """Test rubric-based grading with gpt-5-mini."""
        rubric_model = LiteLLMModel(name="openai/gpt-5-mini")

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            env = InterpreterEnv(
                problem=self.GRADING_PROBLEM,
                work_dir=work_dir,
                rubric_model=rubric_model,
                config=InterpreterEnvConfig(language=NBLanguage.PYTHON, normalize_reward=False),
            )

            try:
                await env.reset()

                for cell_code in code:
                    action = ToolRequestMessage(tool_calls=[ToolCall.from_name("run_cell", code=cell_code)])
                    await env.step(action)

                submit_action = ToolRequestMessage(tool_calls=[ToolCall.from_name("submit_answer", answer=answer)])
                _, reward, done, _ = await env.step(submit_action)

                assert done is True
                assert 0 <= env.state.raw_score <= self.GRADING_PROBLEM.max_score
                assert env.state.score == env.state.raw_score
                assert env.state.total_reward == env.state.score
                assert reward == env.state.score

                assert env.score_info_path.exists()
                score_info = json.loads(env.score_info_path.read_text())
                assert score_info["raw_score"] == env.state.raw_score
                assert score_info["max_score"] == self.GRADING_PROBLEM.max_score
                assert "prompt" in score_info
                assert "response" in score_info

            finally:
                await env.close()
