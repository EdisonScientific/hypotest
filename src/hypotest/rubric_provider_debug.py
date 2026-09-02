"""Cluster-local LiteLLM attempt logging for rubric-model diagnostics."""

from __future__ import annotations

import atexit
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_METADATA_KEY = "hypotest_rubric_debug"
_DEFAULT_FILENAME = "rubric-provider-debug.jsonl"
_MAX_TEXT_CHARS = 2048
_MAX_COLLECTION_ITEMS = 64
_DATA_URL_RE = re.compile(r"data:[^;,]+(?:;[^,]+)*;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_REDACTED_BODY_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "data_url",
    "headers",
    "image_url",
    "input",
    "messages",
    "password",
    "prompt",
    "request",
    "secret",
    "set-cookie",
}
_NORMALIZED_REDACTED_BODY_KEYS = {item.replace("-", "_") for item in _REDACTED_BODY_KEYS}
_SAFE_HEADER_NAMES = {
    "cf-ray",
    "date",
    "request-id",
    "retry-after",
    "server",
    "server-timing",
    "traceparent",
    "via",
    "x-correlation-id",
    "x-envoy-upstream-service-time",
    "x-request-id",
}
_SAFE_HEADER_PREFIXES = ("ratelimit-", "x-ratelimit-", "x-nv-", "x-nvidia-")
_MODEL_PARAMETER_NAMES = {
    "max_completion_tokens",
    "max_tokens",
    "reasoning_effort",
    "stream",
    "temperature",
    "timeout",
}


@dataclass(frozen=True, slots=True)
class RubricDebugRequest:
    """Correlation state for one logical rubric request."""

    request_id: str
    started_monotonic: float
    metadata: dict[str, Any]


def resolve_rubric_provider_debug_path(configured_path: Path | None) -> Path:
    """Resolve an explicit path or default beside ray-driver.log."""
    ray_log_dir = os.getenv("LOG_DIR")
    if configured_path is None:
        if not ray_log_dir:
            raise ValueError("rubric_provider_debug_log requires LOG_DIR or rubric_provider_debug_log_path")
        return Path(ray_log_dir) / _DEFAULT_FILENAME

    if configured_path.is_absolute():
        return configured_path
    if not ray_log_dir:
        raise ValueError("relative rubric_provider_debug_log_path requires LOG_DIR")
    return Path(ray_log_dir) / configured_path


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_url(value: Any) -> str | None:
    if not value:
        return None
    try:
        parts = urlsplit(str(value))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except ValueError:
        return None


def _sanitize_text(value: Any) -> str:
    text = str(value)
    text = _DATA_URL_RE.sub("<redacted-data-url>", text)
    text = _API_KEY_RE.sub("<redacted-api-key>", text)
    if len(text) > _MAX_TEXT_CHARS:
        return f"{text[:_MAX_TEXT_CHARS]}...<truncated:{len(text) - _MAX_TEXT_CHARS}>"
    return text


def _is_sensitive_body_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _NORMALIZED_REDACTED_BODY_KEYS or normalized.endswith(("_api_key", "_password", "_secret"))


def _sanitize_json(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "<max-depth>"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:_MAX_COLLECTION_ITEMS]:
            key_str = str(key)
            output[key_str] = "<redacted>" if _is_sensitive_body_key(key_str) else _sanitize_json(item, depth=depth + 1)
        if len(items) > _MAX_COLLECTION_ITEMS:
            output["_truncated_items"] = len(items) - _MAX_COLLECTION_ITEMS
        return output
    if isinstance(value, list | tuple):
        output = [_sanitize_json(item, depth=depth + 1) for item in value[:_MAX_COLLECTION_ITEMS]]
        if len(value) > _MAX_COLLECTION_ITEMS:
            output.append(f"<truncated-items:{len(value) - _MAX_COLLECTION_ITEMS}>")
        return output
    return _sanitize_text(value)


def _exception_chain(exception: BaseException | None) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exception
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _error_body(exception: BaseException | None) -> Any:
    for item in _exception_chain(exception):
        body = getattr(item, "body", None)
        if body is not None:
            return _sanitize_json(body)
        response = getattr(item, "response", None)
        if response is None:
            continue
        with contextlib.suppress(Exception):
            return _sanitize_json(response.json())
    return None


def _raw_headers(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return dict(value or {})
    except (TypeError, ValueError):
        return {}


def _safe_headers(headers: Any) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in _raw_headers(headers).items():
        normalized = str(key).lower()
        if normalized in _SAFE_HEADER_NAMES or normalized.startswith(_SAFE_HEADER_PREFIXES):
            output[normalized] = _sanitize_text(value)
    return output


def _first_safe_headers(*candidates: Any) -> dict[str, str]:
    for headers in candidates:
        if safe := _safe_headers(headers):
            return safe
    return {}


def _failure_headers(exception: BaseException | None) -> dict[str, str]:
    for item in _exception_chain(exception):
        response = getattr(item, "response", None)
        safe = _first_safe_headers(
            getattr(item, "litellm_response_headers", None),
            getattr(response, "headers", None),
            getattr(item, "headers", None),
        )
        if safe:
            return safe
    return {}


def _metadata(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, Mapping):
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata
    metadata = kwargs.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _debug_metadata(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = _metadata(kwargs).get(_METADATA_KEY)
    return value if isinstance(value, Mapping) else None


def _call_id(kwargs: Mapping[str, Any]) -> str | None:
    value = kwargs.get("litellm_call_id")
    if value:
        return str(value)
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, Mapping) and litellm_params.get("litellm_call_id"):
        return str(litellm_params["litellm_call_id"])
    return None


def _trace_id(kwargs: Mapping[str, Any]) -> str | None:
    standard = kwargs.get("standard_logging_object")
    if isinstance(standard, Mapping) and standard.get("trace_id"):
        return str(standard["trace_id"])
    value = kwargs.get("litellm_trace_id")
    return str(value) if value else None


def _attempt_number(kwargs: Mapping[str, Any]) -> int:
    previous = _metadata(kwargs).get("previous_models")
    return len(previous) + 1 if isinstance(previous, list) else 1


def _callback_seconds(start_time: Any, end_time: Any) -> float | None:
    try:
        return max(0.0, float((end_time - start_time).total_seconds()))
    except (AttributeError, TypeError, ValueError):
        return None


def _model_parameters(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    standard = kwargs.get("standard_logging_object")
    params = standard.get("model_parameters") if isinstance(standard, Mapping) else None
    params = params if isinstance(params, Mapping) else {}
    output: dict[str, Any] = {}
    for name in _MODEL_PARAMETER_NAMES:
        value = params.get(name, kwargs.get(name))
        if value is not None:
            output[name] = _sanitize_json(value)
    return output


def _success_headers(kwargs: Mapping[str, Any], response_obj: Any) -> dict[str, str]:
    hidden = getattr(response_obj, "_hidden_params", None)
    response_hidden_headers = hidden.get("headers") if isinstance(hidden, Mapping) else None
    metadata = _metadata(kwargs)
    hidden = metadata.get("hidden_params")
    metadata_hidden_headers = hidden.get("headers") if isinstance(hidden, Mapping) else None
    response_headers = getattr(response_obj, "_response_headers", None)
    return _first_safe_headers(response_hidden_headers, metadata_hidden_headers, response_headers)


def _success_details(response_obj: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    choices = getattr(response_obj, "choices", None)
    if choices:
        output["finish_reason"] = getattr(choices[0], "finish_reason", None)
    usage = getattr(response_obj, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        output["usage"] = _sanitize_json(usage)
    return output


class RubricProviderDebugLogger(CustomLogger):
    """Write sanitized logical-request and physical-attempt events to JSONL."""

    def __init__(self, path: Path, *, model_name: str) -> None:
        super().__init__(turn_off_message_logging=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.model_name = model_name
        self._lock = threading.Lock()
        self._fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        self._closed = False
        self._active_logical = 0
        self._active_attempts = 0
        self._attempt_starts: dict[str, tuple[float, int]] = {}
        atexit.register(self.close)
        self._write({
            "event": "logger_started",
            "hostname": socket.gethostname(),
            "litellm_version": _package_version("litellm"),
            "fhlmi_version": _package_version("fhlmi"),
            "model": model_name,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_restart_count": os.getenv("SLURM_RESTART_COUNT"),
        })

    def register(self) -> None:
        """Register for every physical LiteLLM attempt in this process."""
        manager = litellm.logging_callback_manager
        manager.add_litellm_input_callback(self)
        manager.add_litellm_async_success_callback(self)
        manager.add_litellm_async_failure_callback(self)

    def unregister(self) -> None:
        """Remove callbacks; primarily useful for focused tests."""
        for callbacks in (
            litellm.input_callback,
            litellm._async_success_callback,
            litellm._async_failure_callback,
        ):
            while self in callbacks:
                callbacks.remove(self)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            os.close(self._fd)

    def metadata_for(self, request: RubricDebugRequest) -> dict[str, Any]:
        return {_METADATA_KEY: request.metadata}

    def begin_request(
        self,
        *,
        prompt: str,
        rubric_images: list[Mapping[str, Any]],
        env_idx: int,
        problem_id: str,
    ) -> RubricDebugRequest:
        prompt_bytes = prompt.encode("utf-8", errors="replace")
        payload_hasher = hashlib.sha256(prompt_bytes)
        image_chars = 0
        for image in rubric_images:
            data_url = str(image.get("data_url", ""))
            encoded = data_url.encode("utf-8", errors="replace")
            image_chars += len(data_url)
            payload_hasher.update(hashlib.sha256(encoded).digest())

        with self._lock:
            self._active_logical += 1
            active_logical = self._active_logical

        request_id = str(uuid.uuid4())
        request = RubricDebugRequest(
            request_id=request_id,
            started_monotonic=time.monotonic(),
            metadata={
                "request_id": request_id,
                "env_idx": env_idx,
                "problem_id": problem_id,
                "model": self.model_name,
                "prompt_chars": len(prompt),
                "prompt_bytes": len(prompt_bytes),
                "image_count": len(rubric_images),
                "image_data_url_chars": image_chars,
                "request_bytes": len(prompt_bytes) + image_chars,
                "payload_sha256": payload_hasher.hexdigest(),
                "active_logical_at_start": active_logical,
            },
        )
        self._write({"event": "logical_request_started", **request.metadata})
        return request

    def finish_request(self, request: RubricDebugRequest, error: BaseException | None = None) -> None:
        with self._lock:
            self._active_logical = max(0, self._active_logical - 1)
            active_logical = self._active_logical
        event: dict[str, Any] = {
            "event": "logical_request_finished",
            "request_id": request.request_id,
            "outcome": "failure" if error is not None else "success",
            "logical_seconds": max(0.0, time.monotonic() - request.started_monotonic),
            "active_logical_after": active_logical,
        }
        if error is not None:
            event.update({
                "error_class": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
                "num_retries": getattr(error, "num_retries", None),
                "max_retries": getattr(error, "max_retries", None),
                "error_message": _sanitize_text(getattr(error, "message", error)),
                "error_body": _error_body(error),
            })
        self._write(event)

    def log_pre_api_call(self, model: str, messages: Any, kwargs: dict[str, Any]) -> None:
        _ = model, messages
        if _debug_metadata(kwargs) is None:
            return
        call_id = _call_id(kwargs)
        if call_id is None:
            return
        with self._lock:
            self._active_attempts += 1
            self._attempt_starts[call_id] = (time.monotonic(), self._active_attempts)

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        self._complete_attempt(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            error=None,
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        self._complete_attempt(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            error=kwargs.get("exception"),
        )

    def _complete_attempt(
        self,
        *,
        kwargs: Mapping[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        error: BaseException | None,
    ) -> None:
        debug = _debug_metadata(kwargs)
        if debug is None:
            return
        call_id = _call_id(kwargs)
        with self._lock:
            start = self._attempt_starts.pop(call_id, None) if call_id is not None else None
            if start is not None:
                self._active_attempts = max(0, self._active_attempts - 1)
            active_attempts_after = self._active_attempts
            active_logical = self._active_logical

        standard = kwargs.get("standard_logging_object")
        api_base = standard.get("api_base") if isinstance(standard, Mapping) else None
        event: dict[str, Any] = {
            "event": "provider_attempt_finished",
            "request_id": debug.get("request_id"),
            "litellm_trace_id": _trace_id(kwargs),
            "litellm_call_id": call_id,
            "attempt": _attempt_number(kwargs),
            "outcome": "failure" if error is not None else "success",
            "status_code": getattr(error, "status_code", None) if error is not None else 200,
            "callback_seconds": _callback_seconds(start_time, end_time),
            "attempt_wall_seconds": max(0.0, time.monotonic() - start[0]) if start is not None else None,
            "active_attempts_at_start": start[1] if start is not None else None,
            "active_attempts_after": active_attempts_after,
            "active_logical_at_completion": active_logical,
            "model": kwargs.get("model"),
            "provider": getattr(error, "llm_provider", None) if error is not None else None,
            "api_base": _safe_url(api_base),
            "model_parameters": _model_parameters(kwargs),
        }
        if error is not None:
            event.update({
                "error_class": type(error).__name__,
                "error_message": _sanitize_text(getattr(error, "message", error)),
                "error_body": _error_body(error),
                "response_headers": _failure_headers(error),
            })
        else:
            event["response_headers"] = _success_headers(kwargs, response_obj)
            event.update(_success_details(response_obj))
        self._write(event)

    def _write(self, event: Mapping[str, Any]) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "timestamp": _utc_now(),
            "pid": os.getpid(),
            **event,
        }
        data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()
        try:
            with self._lock:
                if not self._closed:
                    os.write(self._fd, data)
        except OSError:
            logger.exception("Failed to write rubric provider debug event to %s", self.path)
