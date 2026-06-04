"""Shared fixtures for the sandbox backend tests."""

from collections.abc import Callable

import httpx
import pytest

from hypotest.env.sandbox.http_client import RequestFn


@pytest.fixture
def stub_request() -> Callable[[Callable[..., httpx.Response]], RequestFn]:
    """Build a `RequestFn` from a sync `handler(method, endpoint, **kwargs) -> httpx.Response`.

    Lets HttpKernelClient and the HTTP-backed sandboxes be unit-tested with no
    container/ray/k8s — the handler can return a canned response or raise to
    simulate transport errors.
    """

    def _make(handler: Callable[..., httpx.Response]) -> RequestFn:
        async def _request(method: str, endpoint: str, **kwargs: object) -> httpx.Response:  # noqa: RUF029
            response = handler(method, endpoint, **kwargs)
            # raise_for_status() needs a request bound to the response.
            response.request = httpx.Request(method, f"http://stub{endpoint}")
            return response

        return _request

    return _make


@pytest.fixture
def make_fake_sandbox() -> Callable[[Callable[..., httpx.Response]], object]:
    """Build a fake agent-sandbox `AsyncSandbox` whose `.connector.send_request` rides a handler.

    Lets `K8sSandbox` (and the scheduler) be unit-tested by overriding `_allocate` to return
    this — no real `k8s_agent_sandbox` SDK, no cluster. `handler` is the same
    `(method, endpoint, **kwargs) -> httpx.Response` shape as `stub_request`; `.connector.calls`
    records every request and `.terminate()` flips `.terminated` so leak/cleanup can be asserted.
    """

    def _make(handler: Callable[..., httpx.Response]) -> object:
        class _Connector:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            async def send_request(self, method: str, endpoint: str, **kwargs: object) -> httpx.Response:
                self.calls.append((method, endpoint))
                response = handler(method, endpoint, **kwargs)
                response.request = httpx.Request(method, f"http://stub{endpoint}")
                return response

        class _Sandbox:
            def __init__(self) -> None:
                self.sandbox_id = "sb-fake"
                self.connector = _Connector()
                self.terminated = False

            async def terminate(self) -> None:
                self.terminated = True

        return _Sandbox()

    return _make
