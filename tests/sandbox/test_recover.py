"""Tests for InterpreterEnvState session recovery (swap + replay, clip, surfaced notice).

These exercise the in-process LocalSandbox backend (``use_docker=False``) so no container,
ray, or k8s is needed — recovery logic lives above the Sandbox seam and is backend-agnostic.
"""

import httpx
import pytest

import hypotest.env.interpreter_env as ie
from hypotest.env.kernel_server import NBLanguage


async def _boom(code, timeout=None, req_uuid=""):  # noqa: ASYNC109, RUF029
    """Stand-in sandbox.execute that simulates a transport failure (kernel/pod gone)."""
    raise httpx.ConnectError("kernel connection lost")


@pytest.mark.asyncio
async def test_recover_replays_history_and_preserves_state(tmp_path):
    state = ie.InterpreterEnvState(work_dir=tmp_path, language=NBLanguage.PYTHON, use_docker=False)
    await state.start()
    try:
        await state.execute_and_add_cell("x = 41")
        await state.execute_and_add_cell("x = x + 1")
        assert len(state.nb.cells) == 2

        old_sandbox = state.sandbox
        recovered = await state.recover()

        assert recovered == 2
        assert state.sandbox is not old_sandbox  # a fresh sandbox was swapped in
        # The fresh kernel was rebuilt from the cell history: x == 42.
        result, _ = await state.execute_and_add_cell("print(x)")
        assert "42" in result.get_combined_text()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_recover_clips_notebook_when_replay_budget_exceeded(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "_REPLAY_BUDGET", 1)
    state = ie.InterpreterEnvState(work_dir=tmp_path, language=NBLanguage.PYTHON, use_docker=False)
    await state.start()
    try:
        await state.execute_and_add_cell("a = 1")
        await state.execute_and_add_cell("b = 2")
        await state.execute_and_add_cell("c = 3")
        assert len(state.nb.cells) == 3

        recovered = await state.recover()

        assert recovered == 1  # replay capped at the budget
        assert len(state.nb.cells) == 1  # notebook clipped to match the rebuilt kernel
        assert state._execution_count == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_transport_failure_triggers_recovery_and_surfaces_notice(tmp_path, monkeypatch):
    state = ie.InterpreterEnvState(
        work_dir=tmp_path, language=NBLanguage.PYTHON, use_docker=False, enable_recovery=True
    )
    await state.start()
    try:
        await state.execute_and_add_cell("y = 7")
        # The next execute on the current sandbox fails; recovery swaps in a fresh one mid-call.
        monkeypatch.setattr(state.sandbox, "execute", _boom)

        result, _ = await state.execute_and_add_cell("print(y)")

        combined = result.get_combined_text()
        assert "session recovered" in combined  # the recovery notice is surfaced to the agent
        assert "7" in combined  # replayed y=7 on the fresh kernel, then re-ran the failed cell
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_transport_failure_propagates_when_recovery_disabled(tmp_path, monkeypatch):
    # Recovery is dark-launched: with enable_recovery=False (default) the error propagates.
    state = ie.InterpreterEnvState(work_dir=tmp_path, language=NBLanguage.PYTHON, use_docker=False)
    await state.start()
    try:
        monkeypatch.setattr(state.sandbox, "execute", _boom)
        with pytest.raises(httpx.ConnectError):
            await state.execute_and_add_cell("print(1)")
    finally:
        await state.close()
