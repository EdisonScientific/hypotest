# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The kernel-server HTTP wire protocol, transport-agnostic.

`HttpKernelClient` speaks the `kernel_server.py` contract (`/execute`, `/reset`,
`/health`, `/list_dir`, `/load_capsule`) over any httpx-shaped request function:

- docker passes `httpx.AsyncClient(base_url=...).request`
- k8s passes the agent-sandbox `AsyncSandboxConnector.send_request`
- enroot wraps one of these inside its ray actor

so every HTTP-backed `Sandbox` shares one client + one response parser. See
docs/adr/0001-sandbox-backend-abstraction.md §2.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import nbformat

from hypotest.env.interpreter import ExecutionResult
from hypotest.env.sandbox.base import RequestFn

logger = logging.getLogger(__name__)

# Bumped in lockstep with kernel_server.PROTOCOL_VERSION (added in PR d).
EXPECTED_PROTOCOL_VERSION = 2

_HEALTH_TIMEOUT = httpx.Timeout(5.0, connect=3.0)

# The wire deadline must outlive both the cell and its bounded interrupt/drain.
_MIN_WIRE_TIMEOUT_HEADROOM_S = 30.0
_POST_RECOVERY_TRANSPORT_MARGIN_S = 20.0
_CONTROL_REQUEST_TIMEOUT_S = 30.0
_BOUNDED_REQUEST_MAX_RETRIES = 2
_RETRYABLE_REQUEST_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def execute_wire_timeout_seconds(
    execution_timeout: float,
    timeout_recovery: Literal["none", "interrupt"],
    interrupt_grace_seconds: float,
) -> float:
    recovery_budget = interrupt_grace_seconds if timeout_recovery == "interrupt" else 0.0
    headroom = max(_MIN_WIRE_TIMEOUT_HEADROOM_S, recovery_budget + _POST_RECOVERY_TRANSPORT_MARGIN_S)
    return execution_timeout + headroom


class ProtocolVersionError(RuntimeError):
    """The kernel server speaks an incompatible protocol version (deploy skew)."""


@dataclass
class _ExecutionWireDeadline:
    """Client deadline that explicitly excludes infrastructure admission wait."""

    expires_at: float
    infrastructure_wait_seconds: Callable[[], float] | None = None
    _last_infrastructure_wait_seconds: float = 0.0

    @classmethod
    def start(
        cls,
        duration: float,
        infrastructure_wait_seconds: Callable[[], float] | None,
    ) -> _ExecutionWireDeadline:
        observed = infrastructure_wait_seconds() if infrastructure_wait_seconds is not None else 0.0
        return cls(
            expires_at=asyncio.get_running_loop().time() + duration,
            infrastructure_wait_seconds=infrastructure_wait_seconds,
            _last_infrastructure_wait_seconds=observed,
        )

    def exclude_new_infrastructure_wait(self) -> None:
        if self.infrastructure_wait_seconds is None:
            return
        observed = self.infrastructure_wait_seconds()
        delta = max(0.0, observed - self._last_infrastructure_wait_seconds)
        self.expires_at += delta
        self._last_infrastructure_wait_seconds = observed


def _parse_execute_data(data: dict[str, Any]) -> ExecutionResult:
    """Deserialize a terminal execution payload into an ExecutionResult."""
    notebook_outputs = [nbformat.from_dict(o) for o in data["notebook_outputs"]]
    return ExecutionResult(
        notebook_outputs=notebook_outputs,
        error_occurred=data["error_occurred"],
        execution_time=data.get("execution_time"),
        timed_out=data.get("timed_out", False),
        timeout_recovery=data.get("timeout_recovery"),
        interrupt_seconds=data.get("interrupt_seconds"),
        kernel_restarted=data.get("kernel_restarted", False),
        kernel_state_lost=data.get("kernel_state_lost", False),
        kernel_exit_code=data.get("kernel_exit_code"),
    )


class HttpKernelClient:
    """Client for the kernel-server HTTP protocol over an arbitrary transport."""

    def __init__(
        self,
        request: RequestFn,
        *,
        execution_timeout: float = 600,
        timeout_recovery: Literal["none", "interrupt"] = "none",
        interrupt_grace_seconds: float = 10.0,
        label: str = "kernel",
        owns: httpx.AsyncClient | None = None,
        execution_poll_interval_seconds: float = 0.5,
        execution_poll_max_interval_seconds: float = 5.0,
        execution_poll_backoff_multiplier: float = 1.5,
        execution_poll_jitter_ratio: float = 0.2,
        execution_poll_max_retries: int | None = None,
        execution_poll_request_timeout_seconds: float = _CONTROL_REQUEST_TIMEOUT_S,
        infrastructure_wait_seconds: Callable[[], float] | None = None,
    ) -> None:
        if execution_poll_interval_seconds < 0:
            raise ValueError("execution_poll_interval_seconds cannot be negative")
        if execution_poll_max_interval_seconds < execution_poll_interval_seconds:
            raise ValueError("execution_poll_max_interval_seconds cannot be less than execution_poll_interval_seconds")
        if execution_poll_backoff_multiplier < 1:
            raise ValueError("execution_poll_backoff_multiplier must be at least 1")
        if not 0 <= execution_poll_jitter_ratio <= 1:
            raise ValueError("execution_poll_jitter_ratio must be between 0 and 1")
        if execution_poll_max_retries is not None and execution_poll_max_retries < 0:
            raise ValueError("execution_poll_max_retries cannot be negative")
        if execution_poll_request_timeout_seconds <= 0 or not math.isfinite(execution_poll_request_timeout_seconds):
            raise ValueError("execution_poll_request_timeout_seconds must be finite and positive")
        self._request = request
        self._execution_timeout = execution_timeout
        self._timeout_recovery = timeout_recovery
        self._interrupt_grace_seconds = interrupt_grace_seconds
        self._label = label
        self._execution_poll_interval_seconds = execution_poll_interval_seconds
        self._execution_poll_max_interval_seconds = execution_poll_max_interval_seconds
        self._execution_poll_backoff_multiplier = execution_poll_backoff_multiplier
        self._execution_poll_jitter_ratio = execution_poll_jitter_ratio
        self._execution_poll_max_retries = execution_poll_max_retries
        self._execution_poll_request_timeout_seconds = execution_poll_request_timeout_seconds
        self._infrastructure_wait_seconds = infrastructure_wait_seconds
        # An httpx client this wrapper owns and should aclose() on close (docker);
        # for the connector transport this is None (the SDK owns it).
        self._owns = owns

    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        """Submit code and hide the asynchronous polling protocol from callers.

        Every HTTP request stays short enough to traverse a remote lifecycle
        proxy. The overall deadline still outlives the cell and its bounded
        interrupt/drain. ``X-Req-UUID`` makes a repeated submit idempotent if the
        first response is lost.
        """
        effective_timeout = timeout if timeout is not None else self._execution_timeout
        logical_request_id = req_uuid or str(uuid.uuid4())
        headers = {"X-Req-UUID": logical_request_id}
        payload: dict[str, Any] = {
            "code": code,
            "timeout": timeout,
            "timeout_recovery": self._timeout_recovery,
            "interrupt_grace_seconds": self._interrupt_grace_seconds,
        }
        overall_timeout = execute_wire_timeout_seconds(
            effective_timeout,
            self._timeout_recovery,
            self._interrupt_grace_seconds,
        )
        deadline = _ExecutionWireDeadline.start(overall_timeout, self._infrastructure_wait_seconds)
        execution_id = ""

        try:
            execution_id = await self._submit_execution(payload, headers, deadline)
            return await self._poll_execution(
                execution_id,
                headers,
                deadline,
                overall_timeout,
                logical_request_id,
            )
        except httpx.TimeoutException as e:
            if execution_id:
                await self._cancel_after_client_timeout(execution_id, headers)
            logger.warning(
                "[%s] HTTP %s during submit/poll execution (requested kernel timeout=%.1fs): %s",
                self._label,
                type(e).__name__,
                effective_timeout,
                e,
            )
            timeout_output = nbformat.v4.new_output(
                output_type="error",
                ename="TimeoutError",
                evalue=f"Code execution timed out after {effective_timeout}s (HTTP layer)",
                traceback=[f"TimeoutError: Code execution timed out after {effective_timeout}s (HTTP layer)"],
            )
            return ExecutionResult(
                notebook_outputs=[timeout_output],
                error_occurred=True,
                # Only the kernel server can report execution duration. Never
                # convert proxy/network wait into chargeable model time.
                execution_time=None,
                timed_out=True,
                timeout_recovery="wedged",
            )

    async def _submit_execution(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        deadline: _ExecutionWireDeadline,
    ) -> str:
        async def request() -> httpx.Response:
            return await self._request_excluding_infrastructure_wait(
                deadline,
                "POST",
                "/execute",
                json=payload,
                headers=headers,
                timeout=self._control_timeout(deadline),
            )

        response, _ = await self._request_with_transient_retries(
            request,
            operation="execution submit",
            max_retries=_BOUNDED_REQUEST_MAX_RETRIES,
            deadline=deadline,
        )
        return str(response.json()["execution_id"])

    async def _request_with_transient_retries(
        self,
        request: Callable[[], Awaitable[httpx.Response]],
        *,
        operation: str,
        max_retries: int | None,
        initial_delay: float | None = None,
        deadline: _ExecutionWireDeadline | None = None,
        jitter_rng: random.Random | None = None,
    ) -> tuple[httpx.Response, float]:
        """Run one HTTP request under the shared transient-failure policy.

        ``max_retries`` counts retries after the initial request; ``None`` is
        valid only with a deadline and retries until that deadline. Callers are
        responsible for using this only for idempotent operations.
        """
        if max_retries is None and deadline is None:
            raise ValueError("unbounded request retries require a deadline")

        retry_count = 0
        delay = self._execution_poll_interval_seconds if initial_delay is None else initial_delay
        loop = asyncio.get_running_loop()
        while True:
            if deadline is not None and loop.time() >= deadline.expires_at:
                raise httpx.ReadTimeout(f"{operation} exceeded its wire deadline")
            try:
                response = await request()
                response.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if not self._is_retryable_request_error(exc):
                    raise
                if deadline is not None and loop.time() >= deadline.expires_at:
                    raise httpx.ReadTimeout(f"{operation} exceeded its wire deadline") from exc
                if max_retries is not None and retry_count >= max_retries:
                    raise
                retry_count += 1
                logger.debug(
                    "[%s] transient %s %s; retry %d/%s",
                    self._label,
                    operation,
                    self._request_error_label(exc),
                    retry_count,
                    max_retries if max_retries is not None else "deadline",
                )
                retry_after = (
                    self._retry_after_seconds(exc.response) if isinstance(exc, httpx.HTTPStatusError) else None
                )
                await self._sleep_with_optional_deadline(
                    delay,
                    deadline,
                    jitter_rng,
                    retry_after=retry_after,
                )
                delay = self._next_poll_delay(delay)
            else:
                return response, delay

    async def _poll_execution(
        self,
        execution_id: str,
        headers: dict[str, str],
        deadline: _ExecutionWireDeadline,
        overall_timeout: float,
        logical_request_id: str,
    ) -> ExecutionResult:
        poll_delay = self._execution_poll_interval_seconds
        jitter_rng = random.Random(logical_request_id)

        while True:
            if asyncio.get_running_loop().time() >= deadline.expires_at:
                raise httpx.ReadTimeout(f"execution polling exceeded {overall_timeout:g}s")
            poll, poll_delay = await self._request_execution_poll(
                execution_id,
                headers,
                deadline,
                poll_delay,
                jitter_rng,
            )
            data = poll.json()
            job_status = data["status"]
            if job_status == "completed":
                result = data.get("result")
                if result is None:
                    raise RuntimeError(f"Kernel execution {execution_id} completed without a result")
                return _parse_execute_data(result)
            if job_status in {"failed", "cancelled"}:
                raise RuntimeError(
                    f"Kernel execution {execution_id} {job_status}: {data.get('error') or 'unknown error'}"
                )
            await self._sleep_with_optional_deadline(poll_delay, deadline, jitter_rng)
            poll_delay = self._next_poll_delay(poll_delay)

    async def _request_execution_poll(
        self,
        execution_id: str,
        headers: dict[str, str],
        deadline: _ExecutionWireDeadline,
        poll_delay: float,
        jitter_rng: random.Random,
    ) -> tuple[httpx.Response, float]:
        async def request() -> httpx.Response:
            return await self._request_excluding_infrastructure_wait(
                deadline,
                "GET",
                f"/execute/{execution_id}",
                headers=headers,
                timeout=self._poll_timeout(deadline),
            )

        return await self._request_with_transient_retries(
            request,
            operation="execution poll",
            max_retries=self._execution_poll_max_retries,
            initial_delay=poll_delay,
            deadline=deadline,
            jitter_rng=jitter_rng,
        )

    @staticmethod
    def _is_retryable_request_error(exc: httpx.TransportError | httpx.HTTPStatusError) -> bool:
        return isinstance(exc, httpx.TransportError) or exc.response.status_code in _RETRYABLE_REQUEST_STATUS_CODES

    @staticmethod
    def _request_error_label(exc: httpx.TransportError | httpx.HTTPStatusError) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}"
        return type(exc).__name__

    async def _request_excluding_infrastructure_wait(
        self,
        deadline: _ExecutionWireDeadline,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            return await self._request(method, endpoint, **kwargs)
        finally:
            deadline.exclude_new_infrastructure_wait()

    @staticmethod
    def _control_timeout(deadline: _ExecutionWireDeadline) -> httpx.Timeout:
        remaining = max(0.001, deadline.expires_at - asyncio.get_running_loop().time())
        total = min(_CONTROL_REQUEST_TIMEOUT_S, remaining)
        return httpx.Timeout(total, connect=min(10.0, total))

    def _poll_timeout(self, deadline: _ExecutionWireDeadline) -> httpx.Timeout:
        """Keep one status request short without shortening the cell deadline."""
        remaining = max(0.001, deadline.expires_at - asyncio.get_running_loop().time())
        total = min(self._execution_poll_request_timeout_seconds, remaining)
        return httpx.Timeout(total, connect=min(3.0, total))

    def _next_poll_delay(self, current: float) -> float:
        return min(
            self._execution_poll_max_interval_seconds,
            current * self._execution_poll_backoff_multiplier,
        )

    async def _sleep_with_optional_deadline(
        self,
        delay: float,
        deadline: _ExecutionWireDeadline | None,
        jitter_rng: random.Random | None,
        *,
        retry_after: float | None = None,
    ) -> None:
        if delay > 0 and jitter_rng is not None and self._execution_poll_jitter_ratio > 0:
            jitter = jitter_rng.uniform(
                1 - self._execution_poll_jitter_ratio,
                1 + self._execution_poll_jitter_ratio,
            )
            delay *= jitter
        if retry_after is not None:
            delay = max(delay, retry_after)
        if deadline is not None:
            remaining = max(0.0, deadline.expires_at - asyncio.get_running_loop().time())
            delay = min(delay, remaining)
        await asyncio.sleep(delay)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            delay = float(value)
        except ValueError:
            return None
        return delay if delay >= 0 and math.isfinite(delay) else None

    async def _cancel_after_client_timeout(self, execution_id: str, headers: dict[str, str]) -> None:
        """Best-effort interrupt so an abandoned request does not occupy the kernel."""
        try:
            response = await self._request(
                "POST",
                f"/execute/{execution_id}/cancel",
                headers=headers,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
            response.raise_for_status()
        except Exception:
            logger.warning("[%s] failed to cancel timed-out execution %s", self._label, execution_id)

    async def reset(self, seed: int | None = None) -> None:
        """Reset the kernel via POST /reset."""
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["json"] = {"seed": seed}

        async def request() -> httpx.Response:
            return await self._request(
                "POST",
                "/reset",
                timeout=httpx.Timeout(_CONTROL_REQUEST_TIMEOUT_S, connect=10.0),
                **kwargs,
            )

        try:
            response, _ = await self._request_with_transient_retries(
                request,
                operation="reset",
                max_retries=_BOUNDED_REQUEST_MAX_RETRIES,
            )
        except httpx.TimeoutException as e:
            logger.warning("[%s] HTTP %s during /reset: %s", self._label, type(e).__name__, e)
            raise RuntimeError(f"Kernel reset timed out: {e}") from e
        if seed is not None and response.json().get("seed") != seed:
            raise ProtocolVersionError(f"kernel server did not confirm deterministic reset seed {seed} (deploy skew)")

    async def list_dir(
        self,
        directory: str = ".",
        max_files: int = 20,
        show_hidden: bool = False,
    ) -> str:
        """List the workspace via GET /list_dir (endpoint added in PR d)."""
        try:
            normalized_max_files = int(max_files)
        except (TypeError, ValueError):
            normalized_max_files = 20
        params = {
            "directory": directory,
            "max_files": normalized_max_files,
            "show_hidden": bool(show_hidden),
        }

        async def request() -> httpx.Response:
            return await self._request(
                "GET",
                "/list_dir",
                params=params,
                timeout=httpx.Timeout(_CONTROL_REQUEST_TIMEOUT_S, connect=10.0),
            )

        response, _ = await self._request_with_transient_retries(
            request,
            operation="list_dir",
            max_retries=_BOUNDED_REQUEST_MAX_RETRIES,
        )
        return response.json()["listing"]

    async def load_capsule(self, uuid: str, seed: int | None = None) -> int:
        """Pull a capsule by exact key (with legacy UUID fallback) via POST /load_capsule.

        Returns the number of objects placed.
        """
        # Capsule pulls (S3, can be >100MB) routinely exceed the connector's default 60s wire
        # timeout; bound them by the execution budget instead so large capsules don't fail to load.
        payload: dict[str, str | int] = {"capsule_uuid": uuid}
        if seed is not None:
            payload["seed"] = seed
        response = await self._request(
            "POST",
            "/load_capsule",
            json=payload,
            timeout=httpx.Timeout(self._execution_timeout, connect=10.0),
        )
        response.raise_for_status()
        data = response.json()
        if seed is not None and data.get("seed") != seed:
            raise ProtocolVersionError(f"kernel server did not confirm deterministic capsule seed {seed} (deploy skew)")
        return data["objects"]

    async def health(self, *, raise_for_status: bool = False) -> bool:
        """Return whether GET /health reports ready, rejecting protocol skew."""
        try:
            response = await self._request("GET", "/health", timeout=_HEALTH_TIMEOUT)
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
            if raise_for_status:
                response.raise_for_status()
            return False
        data = response.json()
        version = data.get("protocol_version")
        if version is not None and version != EXPECTED_PROTOCOL_VERSION:
            raise ProtocolVersionError(
                f"kernel server protocol_version={version} != expected {EXPECTED_PROTOCOL_VERSION} (deploy skew)"
            )
        # Older additive-only protocol-v2 servers omitted these fields in some
        # test doubles, so missing values remain compatible. A current server
        # must not advertise a dead/recovering kernel as ready merely because
        # its HTTP process still responds.
        return data.get("status", "OK") == "OK" and data.get("kernel_ready", True) is True

    async def aclose(self) -> None:
        """Close the owned httpx client, if any."""
        if self._owns is not None:
            await self._owns.aclose()
            self._owns = None
