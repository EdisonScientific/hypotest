"""Unit tests for HttpKernelClient (the kernel-server wire protocol)."""

import httpx
import pytest

from hypotest.env.sandbox.http_client import HttpKernelClient, ProtocolVersionError

_STREAM_OUTPUT = {"output_type": "stream", "name": "stdout", "text": "hi\n"}


@pytest.mark.asyncio
async def test_execute_parses_outputs(stub_request):
    client = HttpKernelClient(
        stub_request(
            lambda m, e, **kw: httpx.Response(
                200, json={"notebook_outputs": [_STREAM_OUTPUT], "error_occurred": False, "execution_time": 0.1}
            )
        )
    )
    result = await client.execute("print('hi')")
    assert result.error_occurred is False
    assert "hi" in result.get_combined_text()
    assert result.execution_time == 0.1


@pytest.mark.asyncio
async def test_execute_timeout_returns_error_result(stub_request):
    def handler(method, endpoint, **kwargs):
        raise httpx.ReadTimeout("simulated read timeout")

    client = HttpKernelClient(stub_request(handler), execution_timeout=42)
    result = await client.execute("import time; time.sleep(99)")
    assert result.error_occurred is True
    assert "TimeoutError" in result.get_combined_text()
    assert result.execution_time == 42  # falls back to execution_timeout


@pytest.mark.asyncio
async def test_execute_forwards_req_uuid_header(stub_request):
    captured: dict[str, object] = {}

    def handler(method, endpoint, **kwargs):
        captured.update(kwargs)
        return httpx.Response(200, json={"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0})

    client = HttpKernelClient(stub_request(handler))
    await client.execute("print(1)", req_uuid="req-123")
    assert captured["headers"] == {"X-Req-UUID": "req-123"}


@pytest.mark.asyncio
async def test_reset_timeout_raises_runtimeerror(stub_request):
    def handler(method, endpoint, **kwargs):
        raise httpx.ReadTimeout("simulated")

    client = HttpKernelClient(stub_request(handler))
    with pytest.raises(RuntimeError, match="Kernel reset timed out"):
        await client.reset()


@pytest.mark.asyncio
async def test_list_dir_returns_listing(stub_request):
    client = HttpKernelClient(stub_request(lambda m, e, **kw: httpx.Response(200, json={"listing": "a.txt\nb.txt"})))
    assert await client.list_dir(".") == "a.txt\nb.txt"


@pytest.mark.asyncio
async def test_load_capsule_returns_object_count(stub_request):
    client = HttpKernelClient(stub_request(lambda m, e, **kw: httpx.Response(200, json={"objects": 7})))
    assert await client.load_capsule("uuid-1") == 7


@pytest.mark.asyncio
async def test_health_true_when_version_matches(stub_request):
    client = HttpKernelClient(stub_request(lambda m, e, **kw: httpx.Response(200, json={"protocol_version": 1})))
    assert await client.health() is True


@pytest.mark.asyncio
async def test_health_rejects_protocol_skew(stub_request):
    client = HttpKernelClient(stub_request(lambda m, e, **kw: httpx.Response(200, json={"protocol_version": 999})))
    with pytest.raises(ProtocolVersionError):
        await client.health()


@pytest.mark.asyncio
async def test_health_false_on_transport_error(stub_request):
    def handler(method, endpoint, **kwargs):
        raise httpx.ConnectError("down")

    assert await HttpKernelClient(stub_request(handler)).health() is False
