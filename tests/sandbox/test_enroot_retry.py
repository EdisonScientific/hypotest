"""Regression tests for Ray/Enroot retry identity."""

import asyncio

import pytest

from hypotest.env.interpreter import ExecutionResult
from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import EnrootSandbox, SandboxConfig


@pytest.mark.asyncio
async def test_ray_wait_retries_same_remote_reference(tmp_path, monkeypatch) -> None:
    """Wait retries must not issue the actor method (and execute the cell) again."""
    sandbox = EnrootSandbox(
        SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON, use_enroot=True, use_ray=True)
    )
    stable_ref = object()
    remote_calls: list[tuple] = []
    waited_refs: list[object] = []

    class RemoteExecute:
        def remote(self, *args, **kwargs):
            remote_calls.append((args, kwargs))
            return stable_ref

    class Actor:
        def __init__(self) -> None:
            self._execute_via_http = RemoteExecute()

    async def fake_wait_for(ref, *, timeout):  # noqa: RUF029
        waited_refs.append(ref)
        if len(waited_refs) < 2:
            raise TimeoutError
        return ExecutionResult(execution_time=2.0)

    async def skip_backoff(delay):  # noqa: RUF029
        return None

    monkeypatch.setattr(asyncio, "shield", lambda ref: ref)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(asyncio, "sleep", skip_backoff)
    sandbox.kernel_container = Actor()

    result = await sandbox.execute("1 + 1", timeout=1, req_uuid="logical-request")

    assert result.execution_time == 2.0
    assert len(remote_calls) == 1
    assert waited_refs == [stable_ref, stable_ref]
