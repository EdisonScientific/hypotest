"""Unit tests for HttpKernelClient (the kernel-server wire protocol)."""

import httpx
import pytest

from hypotest.env.sandbox import http_client as httpclientmod
from hypotest.env.sandbox.http_client import (
    HttpKernelClient,
    ProtocolVersionError,
    execute_wire_timeout_seconds,
)

_STREAM_OUTPUT = {"output_type": "stream", "name": "stdout", "text": "hi\n"}


def _execute_handler(result, *, captured=None, pending_polls=0):
    polls = 0

    def handler(method, endpoint, **kwargs):
        nonlocal polls
        if captured is not None:
            captured.append((method, endpoint, kwargs))
        if method == "POST" and endpoint == "/execute":
            return httpx.Response(202, json={"execution_id": "exec-1", "status": "queued"})
        if method == "GET" and endpoint == "/execute/exec-1":
            polls += 1
            if polls <= pending_polls:
                return httpx.Response(200, json={"execution_id": "exec-1", "status": "running"})
            return httpx.Response(
                200,
                json={"execution_id": "exec-1", "status": "completed", "result": result},
            )
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_execute_parses_outputs(stub_request):
    result = {"notebook_outputs": [_STREAM_OUTPUT], "error_occurred": False, "execution_time": 0.1}
    client = HttpKernelClient(
        stub_request(_execute_handler(result, pending_polls=1)),
        execution_poll_interval_seconds=0,
    )
    result = await client.execute("print('hi')")
    assert result.error_occurred is False
    assert "hi" in result.get_combined_text()
    assert result.execution_time == 0.1


@pytest.mark.asyncio
async def test_execute_parses_timeout_recovery_metadata(stub_request):
    result = {
        "notebook_outputs": [],
        "error_occurred": True,
        "execution_time": 12.0,
        "timed_out": True,
        "timeout_recovery": "interrupted",
        "interrupt_seconds": 0.2,
        "kernel_restarted": True,
        "kernel_state_lost": True,
        "kernel_exit_code": 137,
    }
    client = HttpKernelClient(
        stub_request(_execute_handler(result)),
        execution_poll_interval_seconds=0,
    )

    result = await client.execute("slow()")

    assert result.timed_out is True
    assert result.timeout_recovery == "interrupted"
    assert result.interrupt_seconds == 0.2
    assert result.kernel_restarted is True
    assert result.kernel_state_lost is True
    assert result.kernel_exit_code == 137


@pytest.mark.asyncio
async def test_execute_timeout_returns_error_result(stub_request):
    def handler(method, endpoint, **kwargs):
        raise httpx.ReadTimeout("simulated read timeout")

    client = HttpKernelClient(stub_request(handler), execution_timeout=42)
    result = await client.execute("import time; time.sleep(99)")
    assert result.error_occurred is True
    assert "TimeoutError" in result.get_combined_text()
    assert result.execution_time is None  # transport wait is not kernel execution


@pytest.mark.asyncio
async def test_execute_wire_deadline_excludes_infrastructure_admission_wait(monkeypatch):
    infrastructure_wait = 0.0

    async def request(method, endpoint, **_kwargs):
        nonlocal infrastructure_wait
        if method == "POST" and endpoint == "/execute":
            await httpclientmod.asyncio.sleep(0.05)
            infrastructure_wait += 0.05
            response = httpx.Response(202, json={"execution_id": "exec-1", "status": "queued"})
        else:
            response = httpx.Response(
                200,
                json={
                    "execution_id": "exec-1",
                    "status": "completed",
                    "result": {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.25},
                },
            )
        response.request = httpx.Request(method, f"http://stub{endpoint}")
        return response

    monkeypatch.setattr(httpclientmod, "execute_wire_timeout_seconds", lambda *_args: 0.01)
    client = HttpKernelClient(
        request,
        execution_poll_interval_seconds=0,
        infrastructure_wait_seconds=lambda: infrastructure_wait,
    )

    result = await client.execute("print(1)")

    assert result.error_occurred is False
    assert result.execution_time == 0.25


@pytest.mark.asyncio
async def test_execute_forwards_req_uuid_header(stub_request):
    captured: list[tuple[str, str, dict]] = []
    result = {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0}

    client = HttpKernelClient(
        stub_request(_execute_handler(result, captured=captured)),
        execution_poll_interval_seconds=0,
    )
    await client.execute("print(1)", req_uuid="req-123")
    assert captured[0][2]["headers"] == {"X-Req-UUID": "req-123"}
    assert captured[1][2]["headers"] == {"X-Req-UUID": "req-123"}


@pytest.mark.asyncio
async def test_execute_uses_short_proxy_safe_control_requests(stub_request):
    """No individual submit/poll request should inherit the long cell budget."""
    captured: list[tuple[str, str, dict]] = []
    result = {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0}

    client = HttpKernelClient(
        stub_request(_execute_handler(result, captured=captured, pending_polls=1)),
        execution_timeout=600,
        execution_poll_interval_seconds=0,
        execution_poll_request_timeout_seconds=5,
    )
    await client.execute("print(1)", timeout=120)  # explicit per-call cell budget
    assert captured[0][2]["json"]["timeout"] == 120
    assert captured[0][2]["timeout"].read == 30
    assert all(call[2]["timeout"].read == 5 for call in captured[1:])


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_poll_request_timeout_must_be_finite_and_positive(stub_request, value):
    with pytest.raises(ValueError, match="execution_poll_request_timeout_seconds"):
        HttpKernelClient(
            stub_request(lambda *_args, **_kwargs: httpx.Response(200)), execution_poll_request_timeout_seconds=value
        )


@pytest.mark.asyncio
async def test_execute_retries_lost_submit_with_same_idempotency_key(stub_request):
    attempts = 0
    request_ids: list[str] = []
    result = {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0}
    terminal = _execute_handler(result)

    def handler(method, endpoint, **kwargs):
        nonlocal attempts
        if method == "POST" and endpoint == "/execute":
            attempts += 1
            request_ids.append(kwargs["headers"]["X-Req-UUID"])
            if attempts == 1:
                raise httpx.ReadTimeout("submit response was lost")
        return terminal(method, endpoint, **kwargs)

    client = HttpKernelClient(stub_request(handler), execution_poll_interval_seconds=0)
    result_value = await client.execute("print(1)", req_uuid="stable-request")

    assert result_value.error_occurred is False
    assert attempts == 2
    assert request_ids == ["stable-request", "stable-request"]


@pytest.mark.asyncio
async def test_execute_polling_backs_off_to_configured_cap(stub_request, monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:  # noqa: RUF029
        sleeps.append(delay)

    monkeypatch.setattr(httpclientmod.asyncio, "sleep", fake_sleep)
    result = {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0}
    client = HttpKernelClient(
        stub_request(_execute_handler(result, pending_polls=3)),
        execution_poll_interval_seconds=1,
        execution_poll_max_interval_seconds=4,
        execution_poll_backoff_multiplier=2,
        execution_poll_jitter_ratio=0,
    )

    result_value = await client.execute("print(1)", req_uuid="backoff-request")

    assert result_value.error_occurred is False
    assert sleeps == [1, 2, 4]


@pytest.mark.asyncio
async def test_execute_polling_retries_transient_transport_and_status_errors(stub_request, monkeypatch):
    poll_attempts = 0
    sleeps: list[float] = []
    result = {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0}

    async def fake_sleep(delay: float) -> None:  # noqa: RUF029
        sleeps.append(delay)

    def handler(method, endpoint, **kwargs):
        nonlocal poll_attempts
        if method == "POST" and endpoint == "/execute":
            return httpx.Response(202, json={"execution_id": "exec-1", "status": "queued"})
        if method == "GET" and endpoint == "/execute/exec-1":
            poll_attempts += 1
            if poll_attempts == 1:
                raise httpx.ReadError("poll response was lost")
            if poll_attempts == 2:
                return httpx.Response(503, headers={"Retry-After": "3"})
            return httpx.Response(
                200,
                json={"execution_id": "exec-1", "status": "completed", "result": result},
            )
        return httpx.Response(404)

    monkeypatch.setattr(httpclientmod.asyncio, "sleep", fake_sleep)
    client = HttpKernelClient(
        stub_request(handler),
        execution_poll_interval_seconds=1,
        execution_poll_max_interval_seconds=4,
        execution_poll_backoff_multiplier=2,
        execution_poll_jitter_ratio=0,
        execution_poll_max_retries=2,
    )

    result_value = await client.execute("print(1)")

    assert result_value.error_occurred is False
    assert poll_attempts == 3
    assert sleeps == [1, 3]


@pytest.mark.asyncio
async def test_execute_polling_retries_past_old_default_cap_until_success(stub_request, monkeypatch):
    poll_attempts = 0
    sleeps: list[float] = []
    result = {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0}

    async def fake_sleep(delay: float) -> None:  # noqa: RUF029
        sleeps.append(delay)

    def handler(method, endpoint, **kwargs):
        nonlocal poll_attempts
        if method == "POST" and endpoint == "/execute":
            return httpx.Response(202, json={"execution_id": "exec-1", "status": "queued"})
        if method == "GET" and endpoint == "/execute/exec-1":
            poll_attempts += 1
            if poll_attempts <= 10:
                return httpx.Response(502)
            return httpx.Response(
                200,
                json={"execution_id": "exec-1", "status": "completed", "result": result},
            )
        return httpx.Response(404)

    monkeypatch.setattr(httpclientmod.asyncio, "sleep", fake_sleep)
    client = HttpKernelClient(
        stub_request(handler),
        execution_poll_interval_seconds=1,
        execution_poll_max_interval_seconds=4,
        execution_poll_backoff_multiplier=2,
        execution_poll_jitter_ratio=0,
    )

    result_value = await client.execute("print(1)")

    assert result_value.error_occurred is False
    assert poll_attempts == 11
    assert sleeps == [1, 2, 4, 4, 4, 4, 4, 4, 4, 4]


@pytest.mark.asyncio
async def test_execute_polling_honors_explicit_transient_retry_cap(stub_request, monkeypatch):
    poll_attempts = 0
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:  # noqa: RUF029
        sleeps.append(delay)

    def handler(method, endpoint, **kwargs):
        nonlocal poll_attempts
        if method == "POST" and endpoint == "/execute":
            return httpx.Response(202, json={"execution_id": "exec-1", "status": "queued"})
        poll_attempts += 1
        return httpx.Response(503)

    monkeypatch.setattr(httpclientmod.asyncio, "sleep", fake_sleep)
    client = HttpKernelClient(
        stub_request(handler),
        execution_poll_interval_seconds=1,
        execution_poll_max_interval_seconds=4,
        execution_poll_backoff_multiplier=2,
        execution_poll_jitter_ratio=0,
        execution_poll_max_retries=2,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.execute("print(1)")

    assert exc_info.value.response.status_code == 503
    assert poll_attempts == 3
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_execute_polling_does_not_retry_nontransient_status(stub_request, monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:  # noqa: RUF029
        sleeps.append(delay)

    def handler(method, endpoint, **kwargs):
        if method == "POST" and endpoint == "/execute":
            return httpx.Response(202, json={"execution_id": "exec-1", "status": "queued"})
        return httpx.Response(401)

    monkeypatch.setattr(httpclientmod.asyncio, "sleep", fake_sleep)
    client = HttpKernelClient(
        stub_request(handler),
        execution_poll_interval_seconds=1,
        execution_poll_max_interval_seconds=4,
        execution_poll_backoff_multiplier=2,
        execution_poll_jitter_ratio=0,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.execute("print(1)")

    assert exc_info.value.response.status_code == 401
    assert sleeps == []


def test_execute_wire_timeout_contains_recovery_budget():
    assert execute_wire_timeout_seconds(120, "interrupt", 10) == 150
    assert execute_wire_timeout_seconds(120, "interrupt", 25) == 165
    assert execute_wire_timeout_seconds(120, "none", 25) == 150


@pytest.mark.asyncio
async def test_load_capsule_uses_generous_wire_timeout(stub_request):
    """Capsule pulls (large S3 objects) must not be capped at the connector's default 60s."""
    captured: dict[str, object] = {}

    def handler(method, endpoint, **kwargs):
        captured.update(kwargs)
        return httpx.Response(200, json={"objects": 3, "seed": 123})

    client = HttpKernelClient(stub_request(handler), execution_timeout=600)
    await client.load_capsule("uuid-1", seed=123)
    assert isinstance(captured["timeout"], httpx.Timeout)
    assert captured["timeout"].read == 600.0
    assert captured["json"] == {"capsule_uuid": "uuid-1", "seed": 123}


@pytest.mark.asyncio
async def test_reset_retries_transient_transport_and_status_errors(stub_request, monkeypatch):
    attempts = 0
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:  # noqa: RUF029
        sleeps.append(delay)

    def handler(method, endpoint, **kwargs):
        nonlocal attempts
        assert (method, endpoint) == ("POST", "/reset")
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("reset response was lost")
        if attempts == 2:
            return httpx.Response(503, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"seed": 123})

    monkeypatch.setattr(httpclientmod.asyncio, "sleep", fake_sleep)
    client = HttpKernelClient(
        stub_request(handler),
        execution_poll_interval_seconds=1,
        execution_poll_max_interval_seconds=4,
        execution_poll_backoff_multiplier=2,
        execution_poll_jitter_ratio=0,
    )

    await client.reset(seed=123)

    assert attempts == 3
    assert sleeps == [1, 3]


@pytest.mark.asyncio
async def test_list_dir_normalizes_invalid_max_files_like_local_kernel(stub_request):
    captured: dict[str, object] = {}

    def handler(method, endpoint, **kwargs):
        captured.update(kwargs)
        return httpx.Response(200, json={"listing": "Files in directory:"})

    client = HttpKernelClient(stub_request(handler))

    listing = await client.list_dir(max_files="file", show_hidden=1)  # type: ignore[arg-type]

    assert listing == "Files in directory:"
    assert captured["params"] == {
        "directory": ".",
        "max_files": 20,
        "show_hidden": True,
    }


@pytest.mark.asyncio
async def test_reset_forwards_seed(stub_request):
    captured: dict[str, object] = {}

    def handler(method, endpoint, **kwargs):
        captured.update(kwargs)
        return httpx.Response(200, json={"success": True, "seed": 456})

    await HttpKernelClient(stub_request(handler)).reset(seed=456)
    assert captured["json"] == {"seed": 456}


@pytest.mark.asyncio
async def test_seeded_calls_reject_server_that_does_not_echo_seed(stub_request):
    def handler(method, endpoint, **kwargs):
        if endpoint == "/reset":
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, json={"objects": 1})

    client = HttpKernelClient(stub_request(handler))
    with pytest.raises(ProtocolVersionError, match="did not confirm deterministic reset seed"):
        await client.reset(seed=1)
    with pytest.raises(ProtocolVersionError, match="did not confirm deterministic capsule seed"):
        await client.load_capsule("uuid-1", seed=1)


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
async def test_list_dir_retries_transient_disconnect(stub_request):
    attempts = 0

    def handler(method, endpoint, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError("peer disconnected")
        return httpx.Response(200, json={"listing": "a.txt"})

    client = HttpKernelClient(stub_request(handler), execution_poll_interval_seconds=0)
    assert await client.list_dir(".") == "a.txt"
    assert attempts == 2


@pytest.mark.asyncio
async def test_load_capsule_returns_object_count(stub_request):
    client = HttpKernelClient(stub_request(lambda m, e, **kw: httpx.Response(200, json={"objects": 7})))
    assert await client.load_capsule("uuid-1") == 7


@pytest.mark.asyncio
async def test_health_true_when_version_matches(stub_request):
    client = HttpKernelClient(stub_request(lambda m, e, **kw: httpx.Response(200, json={"protocol_version": 2})))
    assert await client.health() is True


@pytest.mark.asyncio
async def test_health_false_when_http_server_has_no_live_kernel(stub_request):
    client = HttpKernelClient(
        stub_request(
            lambda m, e, **kw: httpx.Response(
                200,
                json={"status": "OK", "kernel_ready": False, "protocol_version": 2},
            )
        )
    )
    assert await client.health() is False


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
