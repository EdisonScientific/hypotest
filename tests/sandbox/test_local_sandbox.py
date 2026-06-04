"""Unit tests for LocalSandbox (in-process Interpreter backend)."""

import pytest

from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import LocalSandbox, SandboxConfig


@pytest.mark.asyncio
async def test_local_sandbox_full_lifecycle(tmp_path):
    sb = LocalSandbox(SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON, execution_timeout=30))
    await sb.start()
    try:
        assert await sb.health() is True

        result = await sb.execute("x = 6 * 7\nprint(x)")
        assert result.error_occurred is False
        assert "42" in result.get_combined_text()

        # state persists across executions
        persisted = await sb.execute("print(x + 1)")
        assert "43" in persisted.get_combined_text()

        assert isinstance(await sb.list_dir("."), str)

        # reset clears kernel state
        await sb.reset()
        after_reset = await sb.execute("print(x)")
        assert after_reset.error_occurred is True
    finally:
        await sb.close()
    assert await sb.health() is False
