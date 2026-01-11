"""Tests for the Interpreter class and related types."""

import pathlib
import tempfile

import nbformat
import pytest
from conftest import requires_matplotlib

from hypotest.env.interpreter import ExecutionResult, Interpreter
from hypotest.env.kernel_server import MessageType, NBLanguage


class TestExecutionResult:
    """Tests for the ExecutionResult class."""

    def test_get_combined_text_empty(self):
        """Test get_combined_text with no outputs."""
        result = ExecutionResult()
        assert result.get_combined_text() == "Code executed successfully (no output)"

    def test_get_combined_text_with_outputs(self):
        """Test get_combined_text with multiple outputs via notebook_outputs."""
        stream_output = nbformat.v4.new_output(output_type="stream", name="stdout", text="line1")
        execute_result = nbformat.v4.new_output(
            output_type="execute_result",
            data={"text/plain": "line2"},
            metadata={},
            execution_count=1,
        )
        combined = ExecutionResult(notebook_outputs=[stream_output, execute_result]).get_combined_text()
        assert "[stdout]\nline1" in combined
        assert "line2" in combined

    def test_get_combined_text_with_image_output(self):
        """Test get_combined_text includes [Image generated] for image outputs."""
        display_data = nbformat.v4.new_output(
            output_type="display_data",
            data={"image/png": "base64data", "text/plain": "<Figure>"},
            metadata={},
        )
        combined = ExecutionResult(notebook_outputs=[display_data]).get_combined_text()
        assert "[Image generated]" in combined
        assert "<Figure>" in combined

    def test_has_images_false(self):
        """Test has_images when no images present."""
        result = ExecutionResult()
        assert not result.has_images()

    def test_has_images_true(self):
        """Test has_images when images present in notebook_outputs."""
        # Minimal valid base64 PNG (1x1 transparent pixel)
        display_data = nbformat.v4.new_output(
            output_type="display_data",
            data={
                "image/png": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                )
            },
            metadata={},
        )
        result = ExecutionResult(notebook_outputs=[display_data])
        assert result.has_images()

    def test_get_truncated_text(self):
        """Test that get_truncated_text truncates long output."""
        # Create output longer than NB_OUTPUT_LIMIT (3000 chars)
        long_output = "x" * 5000
        stream_output = nbformat.v4.new_output(output_type="stream", name="stdout", text=long_output)
        truncated = ExecutionResult(notebook_outputs=[stream_output]).get_truncated_text()
        assert len(truncated) < 5000
        assert "truncated" in truncated

    def test_to_message_text_only(self):
        """Test to_message with text output only."""
        stream_output = nbformat.v4.new_output(output_type="stream", name="stdout", text="hello world")
        message = ExecutionResult(notebook_outputs=[stream_output]).to_message()
        assert "content" in message
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "text"
        assert "hello world" in message["content"][0]["text"]

    def test_to_message_with_images(self):
        """Test to_message with text and images."""
        # Minimal valid base64 PNG for testing (1x1 transparent pixel)
        test_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        display_data = nbformat.v4.new_output(
            output_type="display_data",
            data={"text/plain": "hello world", "image/png": test_png},
            metadata={},
        )
        message = ExecutionResult(notebook_outputs=[display_data]).to_message()
        assert "content" in message
        assert len(message["content"]) == 2
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image"
        assert message["content"][1]["mimeType"] == "image/png"

    def test_error_occurred_default(self):
        """Test that error_occurred defaults to False."""
        result = ExecutionResult()
        assert not result.error_occurred

    def test_get_error_message(self):
        """Test get_error_message extracts error from notebook_outputs."""
        error_output = nbformat.v4.new_output(
            output_type="error",
            ename="ValueError",
            evalue="test error",
            traceback=["Traceback...", "ValueError: test error"],
        )
        error_msg = ExecutionResult(notebook_outputs=[error_output], error_occurred=True).get_error_message()
        assert error_msg is not None
        assert "ValueError" in error_msg
        assert "test error" in error_msg

    def test_get_images_returns_tuples(self):
        """Test that get_images returns list of (mime_type, base64_data) tuples."""
        test_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        display_data = nbformat.v4.new_output(
            output_type="display_data",
            data={"image/png": test_png},
            metadata={},
        )
        images = ExecutionResult(notebook_outputs=[display_data]).get_images()
        assert len(images) == 1
        mime_type, base64_data = images[0]
        assert mime_type == "image/png"
        assert base64_data


class TestMessageType:
    """Tests for the MessageType enum."""

    def test_from_string_valid_types(self):
        """Test from_string returns correct MessageType for known types."""
        assert MessageType.from_string("stream") == MessageType.STREAM
        assert MessageType.from_string("execute_result") == MessageType.EXECUTE_RESULT
        assert MessageType.from_string("display_data") == MessageType.DISPLAY_DATA
        assert MessageType.from_string("error") == MessageType.ERROR
        assert MessageType.from_string("status") == MessageType.STATUS

    def test_from_string_unknown_returns_none(self):
        """Test from_string returns None for unknown message types."""
        assert MessageType.from_string("unknown_type") is None
        assert MessageType.from_string("") is None
        assert MessageType.from_string("execute_input") is None

    def test_to_notebook_output_stream(self):
        """Test to_notebook_output for stream message type."""
        content = {"name": "stdout", "text": "hello world"}
        output = MessageType.STREAM.to_notebook_output(content)
        assert output is not None
        assert output["output_type"] == "stream"
        assert output["name"] == "stdout"
        assert output["text"] == "hello world"

    def test_to_notebook_output_execute_result(self):
        """Test to_notebook_output for execute_result message type."""
        content = {
            "data": {"text/plain": "42"},
            "metadata": {},
            "execution_count": 1,
        }
        output = MessageType.EXECUTE_RESULT.to_notebook_output(content)
        assert output is not None
        assert output["output_type"] == "execute_result"
        assert output["data"]["text/plain"] == "42"
        assert output["execution_count"] == 1

    def test_to_notebook_output_display_data(self):
        """Test to_notebook_output for display_data message type."""
        content = {
            "data": {"image/png": "base64data"},
            "metadata": {"width": 100},
        }
        output = MessageType.DISPLAY_DATA.to_notebook_output(content)
        assert output is not None
        assert output["output_type"] == "display_data"
        assert output["data"]["image/png"] == "base64data"

    def test_to_notebook_output_error(self):
        """Test to_notebook_output for error message type."""
        content = {
            "ename": "ValueError",
            "evalue": "test error",
            "traceback": ["line 1", "line 2"],
        }
        output = MessageType.ERROR.to_notebook_output(content)
        assert output is not None
        assert output["output_type"] == "error"
        assert output["ename"] == "ValueError"
        assert output["evalue"] == "test error"
        assert output["traceback"] == ["line 1", "line 2"]

    def test_to_notebook_output_status_returns_none(self):
        """Test to_notebook_output returns None for status message type."""
        content = {"execution_state": "idle"}
        output = MessageType.STATUS.to_notebook_output(content)
        assert output is None

    def test_to_notebook_output_defaults(self):
        """Test to_notebook_output uses defaults for missing content fields."""
        output = MessageType.STREAM.to_notebook_output({})
        assert output is not None
        assert output["name"] == "stdout"
        assert not output["text"]

        output = MessageType.ERROR.to_notebook_output({})
        assert output is not None
        assert not output["ename"]
        assert not output["evalue"]
        assert output["traceback"] == []


class TestNBLanguage:
    """Tests for the NBLanguage enum."""

    def test_make_kernelspec_python(self):
        """Test make_kernelspec for Python."""
        spec = NBLanguage.PYTHON.make_kernelspec()
        assert spec["name"] == "python"
        assert spec["language"] == "python"
        assert "Python" in spec["display_name"]

    def test_make_kernelspec_r(self):
        """Test make_kernelspec for R."""
        spec = NBLanguage.R.make_kernelspec()
        assert spec["name"] == "ir"
        assert spec["language"] == "r"

    def test_from_string_valid(self):
        """Test from_string with valid language strings."""
        assert NBLanguage.from_string("PYTHON") == NBLanguage.PYTHON
        assert NBLanguage.from_string("python") == NBLanguage.PYTHON
        assert NBLanguage.from_string("R") == NBLanguage.R
        assert NBLanguage.from_string("r") == NBLanguage.R

    def test_from_string_auto_returns_none(self):
        """Test from_string returns None for AUTO."""
        assert NBLanguage.from_string("AUTO") is None
        assert NBLanguage.from_string("auto") is None

    def test_from_string_invalid_defaults_to_python(self):
        """Test from_string with invalid language returns PYTHON."""
        assert NBLanguage.from_string("INVALID") == NBLanguage.PYTHON
        assert NBLanguage.from_string("javascript") == NBLanguage.PYTHON


class TestInterpreter:
    """Tests for the Interpreter class."""

    @pytest.mark.asyncio
    async def test_interpreter_start_stop(self):
        """Test basic interpreter lifecycle."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            assert not interpreter.is_ready

            await interpreter.start()
            assert interpreter.is_ready

            await interpreter.close()  # type: ignore[unreachable]
            assert not interpreter.is_ready

    @pytest.mark.asyncio
    async def test_interpreter_execute_code(self):
        """Test simple code execution."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                result = await interpreter.execute_code("print('hello')")
                assert not result.error_occurred
                assert any("hello" in output for output in result.get_text_outputs())
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_execute_code_with_error(self):
        """Test code execution with runtime error."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                result = await interpreter.execute_code("raise ValueError('test error')")
                assert result.error_occurred
                text_outputs = result.get_text_outputs()
                assert any("ValueError" in output for output in text_outputs)
                assert any("test error" in output for output in text_outputs)
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_execute_code_with_return_value(self):
        """Test code execution with return value."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                result = await interpreter.execute_code("2 + 2")
                assert not result.error_occurred
                assert any("4" in output for output in result.get_text_outputs())
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_state_persistence(self):
        """Test that state persists across executions."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                await interpreter.execute_code("x = 42")

                result = await interpreter.execute_code("print(x)")
                assert not result.error_occurred
                assert any("42" in output for output in result.get_text_outputs())
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_reset(self):
        """Test kernel reset clears state."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                await interpreter.execute_code("x = 42")

                await interpreter.reset()

                result = await interpreter.execute_code("print(x)")
                assert result.error_occurred
                assert any("NameError" in output for output in result.get_text_outputs())
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_execution_history(self):
        """Test execution history tracking."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                await interpreter.execute_code("x = 1")
                await interpreter.execute_code("y = 2")
                await interpreter.execute_code("z = x + y")

                assert len(interpreter.execution_history) == 3
                assert all(isinstance(r, ExecutionResult) for r in interpreter.execution_history)
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_get_execution_summary(self):
        """Test get_execution_summary returns correct data."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                await interpreter.execute_code("print('hello')")
                await interpreter.execute_code("raise ValueError('test')")

                summary = interpreter.get_execution_summary()

                assert summary["total_executions"] == 2
                assert summary["error_count"] == 1
                assert summary["is_ready"] is True
                assert summary["language"] == "python"
                assert str(work_dir) in summary["work_dir"]
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_notebook_outputs(self):
        """Test that notebook_outputs are created for notebook cells."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                await interpreter.start()

                result = await interpreter.execute_code("print('hello')")
                assert result.notebook_outputs
                assert result.notebook_outputs[0]["output_type"] == "stream"
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_interpreter_auto_start(self):
        """Test that execute_code auto-starts the kernel if not ready."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )

            try:
                assert not interpreter.is_ready

                result = await interpreter.execute_code("print('auto start')")
                assert interpreter.is_ready
                assert any(  # type: ignore[unreachable]
                    "auto start" in output for output in result.get_text_outputs()
                )
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_execute_code_with_custom_timeout(self):
        """Test execute_code with explicit timeout parameter."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                result = await interpreter.execute_code("print('hello')", execution_timeout=5.0)
                assert not result.error_occurred
                assert "hello" in result.get_combined_text()
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_execute_code_timeout_triggers(self):
        """Test that custom timeout actually triggers."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                result = await interpreter.execute_code("import time; time.sleep(5)", execution_timeout=0.1)
                assert result.error_occurred
                assert "timed out" in result.get_combined_text()
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_execute_code_stderr_output(self):
        """Test that stderr output is captured with correct stream name."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                result = await interpreter.execute_code("import sys; sys.stderr.write('error message')")
                assert not result.error_occurred
                combined = result.get_combined_text()
                assert "error message" in combined
                assert "[stderr]" in combined
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_execute_code_multiple_outputs(self):
        """Test execution that produces multiple output types."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                code = """
print('stream output')
x = 42
x
"""
                result = await interpreter.execute_code(code)
                assert not result.error_occurred
                combined = result.get_combined_text()
                assert "stream output" in combined
                assert "42" in combined
                assert len(result.notebook_outputs) >= 2
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_execute_code_long_output_truncation(self):
        """Test that very long outputs are truncated by get_truncated_text."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                result = await interpreter.execute_code("print('x' * 10000)")
                assert not result.error_occurred
                combined = result.get_combined_text()
                assert len(combined) > 5000
                truncated = result.get_truncated_text()
                assert len(truncated) < len(combined)
                assert "truncated" in truncated
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_execute_code_long_error_traceback(self):
        """Test that long error tracebacks are captured."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                code = """
def level1():
    level2()
def level2():
    level3()
def level3():
    level4()
def level4():
    level5()
def level5():
    raise ValueError("Deep error")
level1()
"""
                result = await interpreter.execute_code(code)
                assert result.error_occurred
                combined = result.get_combined_text()
                assert "ValueError" in combined
                assert "Deep error" in combined
                assert "level1" in combined
                assert "level5" in combined
            finally:
                await interpreter.close()

    @pytest.mark.asyncio
    async def test_execute_code_display_data_with_text(self):
        """Test display_data output with text/plain content."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                code = """
from IPython.display import display
display("Hello from display")
"""
                result = await interpreter.execute_code(code)
                assert not result.error_occurred
                combined = result.get_combined_text()
                assert "Hello from display" in combined
            finally:
                await interpreter.close()

    @requires_matplotlib
    @pytest.mark.asyncio
    async def test_execute_code_with_plot(self):
        """Test code execution with matplotlib plot."""
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)
            interpreter = Interpreter(
                work_dir=work_dir,
                language=NBLanguage.PYTHON,
            )
            await interpreter.start()
            try:
                code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 2, 3])
plt.show()
"""
                result = await interpreter.execute_code(code)
                assert not result.error_occurred
                assert result.has_images()
                images = result.get_images()
                assert len(images) >= 1
            finally:
                await interpreter.close()
