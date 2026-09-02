"""Tests for cluster-local rubric provider diagnostics."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar

import litellm
import pytest

from hypotest.rubric_provider_debug import (
    RubricProviderDebugLogger,
    resolve_rubric_provider_debug_path,
)


class _OriginalProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__("original provider error")
        self.status_code = status_code
        self.body = {
            "message": "Service temporarily overloaded",
            "type": "Overloaded",
            "code": status_code,
            "messages": ["private prompt"],
            "api_key": "sk-private-key",
        }


class _MappedProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__("mapped provider error")
        self.status_code = status_code
        self.message = "mapped provider error"
        self.llm_provider = "openai"
        self.num_retries = 1
        self.max_retries = 3
        self.litellm_response_headers = {
            "x-request-id": f"request-{status_code}",
            "retry-after": "17",
            "x-envoy-upstream-service-time": "321",
            "authorization": "Bearer secret",
        }
        self.__context__ = _OriginalProviderError(status_code)


def _kwargs(metadata: dict, *, call_id: str, trace_id: str, previous_attempts: int = 0) -> dict:
    return {
        "model": "openai/nvidia/zai-org/glm-5.2",
        "litellm_call_id": call_id,
        "litellm_trace_id": trace_id,
        "litellm_params": {
            "metadata": metadata
            | {
                "previous_models": [{} for _ in range(previous_attempts)],
            }
        },
        "standard_logging_object": {
            "trace_id": trace_id,
            "api_base": "https://inference-api.nvidia.com/v1?secret=value",
            "model_parameters": {
                "temperature": 0.7,
                "reasoning_effort": "max",
            },
        },
    }


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_default_path_is_adjacent_to_ray_logs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    assert resolve_rubric_provider_debug_path(None) == tmp_path / "rubric-provider-debug.jsonl"
    assert resolve_rubric_provider_debug_path(tmp_path / "explicit.jsonl") == tmp_path / "explicit.jsonl"


def test_default_path_requires_ray_log_context(monkeypatch) -> None:
    monkeypatch.delenv("LOG_DIR", raising=False)
    with pytest.raises(ValueError, match="requires LOG_DIR"):
        resolve_rubric_provider_debug_path(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [500, 529])
async def test_failure_attempt_preserves_diagnostics_without_sensitive_payloads(tmp_path, status_code) -> None:
    path = tmp_path / "events.jsonl"
    debug_logger = RubricProviderDebugLogger(path, model_name="openai/nvidia/zai-org/glm-5.2")
    request = debug_logger.begin_request(
        prompt="private prompt",
        rubric_images=[{"data_url": "data:image/png;base64,PRIVATEIMAGE"}],
        env_idx=7,
        problem_id="problem-1",
    )
    kwargs = _kwargs(
        debug_logger.metadata_for(request),
        call_id="call-1",
        trace_id="trace-1",
        previous_attempts=1,
    )
    debug_logger.log_pre_api_call(kwargs["model"], [], kwargs)
    error = _MappedProviderError(status_code)
    kwargs["exception"] = error
    now = dt.datetime.now(dt.UTC)
    await debug_logger.async_log_failure_event(kwargs, None, now, now + dt.timedelta(milliseconds=250))
    debug_logger.finish_request(request, error)
    debug_logger.close()

    raw = path.read_text()
    assert "private prompt" not in raw
    assert "PRIVATEIMAGE" not in raw
    assert "sk-private-key" not in raw
    assert "Bearer secret" not in raw

    attempt = next(event for event in _events(path) if event["event"] == "provider_attempt_finished")
    assert attempt["status_code"] == status_code
    assert attempt["attempt"] == 2
    assert attempt["litellm_trace_id"] == "trace-1"
    assert attempt["litellm_call_id"] == "call-1"
    assert attempt["error_body"] == {
        "api_key": "<redacted>",
        "code": status_code,
        "message": "Service temporarily overloaded",
        "messages": "<redacted>",
        "type": "Overloaded",
    }
    assert attempt["response_headers"] == {
        "retry-after": "17",
        "x-envoy-upstream-service-time": "321",
        "x-request-id": f"request-{status_code}",
    }
    assert attempt["api_base"] == "https://inference-api.nvidia.com/v1"


@pytest.mark.asyncio
async def test_retry_attempts_share_request_and_trace_ids(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    debug_logger = RubricProviderDebugLogger(path, model_name="model")
    request = debug_logger.begin_request(prompt="p", rubric_images=[], env_idx=1, problem_id="problem")
    metadata = debug_logger.metadata_for(request)
    now = dt.datetime.now(dt.UTC)

    for index, status_code in enumerate((500, 529)):
        kwargs = _kwargs(metadata, call_id=f"call-{index}", trace_id="trace", previous_attempts=index)
        debug_logger.log_pre_api_call(kwargs["model"], [], kwargs)
        error = _MappedProviderError(status_code)
        kwargs["exception"] = error
        await debug_logger.async_log_failure_event(kwargs, None, now, now + dt.timedelta(milliseconds=10))

    success_kwargs = _kwargs(metadata, call_id="call-2", trace_id="trace", previous_attempts=2)
    debug_logger.log_pre_api_call(success_kwargs["model"], [], success_kwargs)
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop")],
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
        _hidden_params={"headers": {"x-request-id": "request-200", "authorization": "secret"}},
    )
    await debug_logger.async_log_success_event(success_kwargs, response, now, now + dt.timedelta(milliseconds=20))
    debug_logger.finish_request(request)
    debug_logger.close()

    attempts = [event for event in _events(path) if event["event"] == "provider_attempt_finished"]
    assert [event["attempt"] for event in attempts] == [1, 2, 3]
    assert [event["status_code"] for event in attempts] == [500, 529, 200]
    assert {event["request_id"] for event in attempts} == {request.request_id}
    assert {event["litellm_trace_id"] for event in attempts} == {"trace"}
    assert {event["litellm_call_id"] for event in attempts} == {"call-0", "call-1", "call-2"}
    assert attempts[-1]["response_headers"] == {"x-request-id": "request-200"}
    assert attempts[-1]["usage"]["total_tokens"] == 12


@pytest.mark.asyncio
async def test_registered_callback_observes_each_router_attempt(tmp_path) -> None:
    class Handler(BaseHTTPRequestHandler):
        calls = 0
        request_bodies: ClassVar[list[dict]] = []

        def do_POST(self) -> None:
            request_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            type(self).request_bodies.append(json.loads(request_body))
            type(self).calls += 1
            if type(self).calls == 1:
                body = b'{"message":"temporary failure","type":"Internal Server Error","code":500}'
                status = 500
            else:
                body = (
                    b'{"id":"response","object":"chat.completion","created":1,"model":"model",'
                    b'"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},'
                    b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
                )
                status = 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-request-id", f"request-{type(self).calls}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    path = tmp_path / "events.jsonl"
    debug_logger = RubricProviderDebugLogger(path, model_name="model")
    debug_logger.register()
    router = litellm.Router(
        model_list=[
            {
                "model_name": "model",
                "litellm_params": {
                    "model": "openai/model",
                    "api_key": "placeholder",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                },
            }
        ],
        num_retries=1,
        retry_after=0,
        disable_cooldowns=True,
    )
    request = debug_logger.begin_request(prompt="test", rubric_images=[], env_idx=0, problem_id="problem")

    try:
        response = await router.acompletion(
            model="model",
            messages=[{"role": "user", "content": "test"}],
            metadata=debug_logger.metadata_for(request),
        )
        assert response.choices[0].message.content == "ok"
        await asyncio.sleep(0.2)
        debug_logger.finish_request(request)
    finally:
        debug_logger.unregister()
        debug_logger.close()
        server.shutdown()
        server.server_close()

    attempts = [event for event in _events(path) if event["event"] == "provider_attempt_finished"]
    assert [event["status_code"] for event in attempts] == [500, 200]
    assert [event["attempt"] for event in attempts] == [1, 2]
    assert {event["request_id"] for event in attempts} == {request.request_id}
    assert len({event["litellm_trace_id"] for event in attempts}) == 1
    assert len({event["litellm_call_id"] for event in attempts}) == 2
    assert all("hypotest_rubric_debug" not in json.dumps(body) for body in Handler.request_bodies)
