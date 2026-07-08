"""Tests for the standalone kernel server."""

import asyncio
from queue import Empty

import httpx
import pytest

from hypotest.env.kernel_server import (
    PROTOCOL_VERSION,
    HealthResponse,
    KernelExecutionState,
    KernelServer,
    NBLanguage,
    ResetResponse,
    create_app,
)
from hypotest.env.tools.filesystem import list_dir_tool


def _server(work_dir):
    return KernelServer(work_dir, NBLanguage.PYTHON)


class _InterruptibleClient:
    def __init__(self):
        self.execute_count = 0
        self.active_msg_id = ""
        self.interrupted = False
        self.started = asyncio.Event()

    def execute(self, code, store_history=True):  # noqa: ARG002
        self.execute_count += 1
        self.active_msg_id = f"msg-{self.execute_count}"
        self.interrupted = False
        self.started.set()
        return self.active_msg_id

    async def get_iopub_msg(self, timeout=None):  # noqa: ARG002, ASYNC109
        await asyncio.sleep(0)
        if self.execute_count == 1 and not self.interrupted:
            raise Empty
        return {
            "parent_header": {"msg_id": self.active_msg_id},
            "msg_type": "status",
            "content": {"execution_state": "idle"},
        }


class _InterruptManager:
    def __init__(self, client):
        self.client = client
        self.interrupt_count = 0

    async def interrupt_kernel(self):
        self.interrupt_count += 1
        self.client.interrupted = True


def _ready_server(tmp_path):
    server = _server(tmp_path)
    client = _InterruptibleClient()
    manager = _InterruptManager(client)
    server._client = client
    server._kernel_manager = manager
    server._is_ready = True
    return server, client, manager


def test_list_dir_lists_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")
    out = _server(tmp_path).list_dir(".")
    assert "a.txt" in out
    assert "sub/b.txt" in out


def test_list_dir_truncates(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    out = _server(tmp_path).list_dir(".", max_files=2)
    assert "more files not shown" in out


def test_list_dir_confines_to_workspace(tmp_path):
    out = _server(tmp_path).list_dir("../../../etc")
    assert "must stay within the workspace root" in out


def test_list_dir_matches_filesystem_tool(tmp_path):
    """The inlined walk must agree with env.tools.filesystem.list_dir_tool (drift guard)."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.txt").write_text("x")
    (tmp_path / "c.txt").write_text("y")
    assert _server(tmp_path).list_dir(".") == list_dir_tool(str(tmp_path))


def test_health_response_carries_protocol_version():
    health = HealthResponse(status="OK", startup_token="t", kernel_ready=True)  # noqa: S106
    assert health.protocol_version == PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_list_dir_and_health_endpoints(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n")
    transport = httpx.ASGITransport(app=create_app(_server(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/list_dir", params={"directory": ".", "max_files": 20, "show_hidden": False})
        assert listing.status_code == 200
        assert "data.csv" in listing.json()["listing"]

        health = await client.get("/health")
        assert health.json()["protocol_version"] == PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_reset_endpoint_accepts_seed(tmp_path, monkeypatch):
    server = _server(tmp_path)
    captured: list[int | None] = []

    async def fake_reset(seed=None):  # noqa: RUF029
        captured.append(seed)
        return ResetResponse(success=True, seed=seed)

    monkeypatch.setattr(server, "reset", fake_reset)
    transport = httpx.ASGITransport(app=create_app(server))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/reset", json={"seed": 987})
        unseeded_response = await client.post("/reset")

    assert response.status_code == 200
    assert unseeded_response.status_code == 200
    assert captured == [987, None]


@pytest.mark.asyncio
async def test_timeout_interrupts_and_allows_next_execution(tmp_path):
    server, client, manager = _ready_server(tmp_path)

    timed_out = await server.execute(
        "while True: pass",
        timeout=0.001,
        timeout_recovery="interrupt",
        interrupt_grace_seconds=0.1,
    )
    next_result = await server.execute("print('ready')", timeout=0.1)

    assert timed_out.timed_out is True
    assert timed_out.timeout_recovery == "interrupted"
    assert timed_out.interrupt_seconds is not None
    assert manager.interrupt_count == 1
    assert next_result.error_occurred is False
    assert client.execute_count == 2
    assert server._execution_state == KernelExecutionState.IDLE


@pytest.mark.asyncio
async def test_failed_interrupt_marks_kernel_wedged(tmp_path, monkeypatch):
    server, client, _ = _ready_server(tmp_path)

    async def fail_recovery(msg_id, grace_seconds):
        await asyncio.sleep(0)
        del msg_id, grace_seconds
        return False

    monkeypatch.setattr(server, "_interrupt_and_drain", fail_recovery)
    timed_out = await server.execute(
        "while True: pass",
        timeout=0.001,
        timeout_recovery="interrupt",
        interrupt_grace_seconds=0.1,
    )
    rejected = await server.execute("print('must not queue')")

    assert timed_out.timeout_recovery == "wedged"
    assert rejected.notebook_outputs[0]["ename"] == "KernelUnresponsiveError"
    assert client.execute_count == 1


@pytest.mark.asyncio
async def test_active_execution_is_not_queued(tmp_path):
    server, client, _ = _ready_server(tmp_path)
    await server._set_execution_state(KernelExecutionState.EXECUTING)

    rejected = await server.execute("print('must not queue')")

    assert rejected.notebook_outputs[0]["ename"] == "KernelBusyError"
    assert client.execute_count == 0


@pytest.mark.asyncio
async def test_cancelled_request_interrupts_before_releasing_kernel(tmp_path):
    server, client, _ = _ready_server(tmp_path)
    task = asyncio.create_task(
        server.execute(
            "while True: pass",
            timeout=100,
            timeout_recovery="interrupt",
            interrupt_grace_seconds=0.1,
        )
    )
    await client.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert server._execution_state == KernelExecutionState.IDLE
    next_result = await server.execute("print('ready')", timeout=0.1)
    assert next_result.error_occurred is False


@pytest.mark.asyncio
async def test_timeout_interrupt_preserves_real_kernel_state(tmp_path):
    server = _server(tmp_path)
    await server.start()
    try:
        await server.execute("sentinel = 41", timeout=5)
        timed_out = await server.execute(
            "import time; time.sleep(10); sentinel = 99",
            timeout=0.1,
            timeout_recovery="interrupt",
            interrupt_grace_seconds=5,
        )
        after = await server.execute("print(sentinel + 1)", timeout=5)
    finally:
        await server.close()

    assert timed_out.timeout_recovery == "interrupted"
    assert after.error_occurred is False
    assert any("42" in output.get("text", "") for output in after.notebook_outputs)
