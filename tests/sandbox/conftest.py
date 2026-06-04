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
