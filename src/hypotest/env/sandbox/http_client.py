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
import uuid
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
_SUBMIT_ATTEMPTS = 3


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
    ) -> None:
        self._request = request
        self._execution_timeout = execution_timeout
        self._timeout_recovery = timeout_recovery
        self._interrupt_grace_seconds = interrupt_grace_seconds
        self._label = label
        self._execution_poll_interval_seconds = execution_poll_interval_seconds
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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + overall_timeout
        execution_id = ""

        try:
            response: httpx.Response | None = None
            for attempt in range(_SUBMIT_ATTEMPTS):
                try:
                    response = await self._request(
                        "POST",
                        "/execute",
                        json=payload,
                        headers=headers,
                        timeout=self._control_timeout(deadline),
                    )
                    response.raise_for_status()
                    break
                except httpx.TransportError:
                    if attempt + 1 >= _SUBMIT_ATTEMPTS or loop.time() >= deadline:
                        raise
                    await asyncio.sleep(min(self._execution_poll_interval_seconds, max(0.0, deadline - loop.time())))

            assert response is not None
            execution_id = str(response.json()["execution_id"])

            while True:
                if loop.time() >= deadline:
                    raise httpx.ReadTimeout(f"execution polling exceeded {overall_timeout:g}s")
                poll = await self._request(
                    "GET",
                    f"/execute/{execution_id}",
                    headers=headers,
                    timeout=self._control_timeout(deadline),
                )
                poll.raise_for_status()
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
                await asyncio.sleep(min(self._execution_poll_interval_seconds, max(0.0, deadline - loop.time())))
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
                execution_time=effective_timeout,
                timed_out=True,
                timeout_recovery="wedged",
            )

    @staticmethod
    def _control_timeout(deadline: float) -> httpx.Timeout:
        remaining = max(0.001, deadline - asyncio.get_running_loop().time())
        total = min(_CONTROL_REQUEST_TIMEOUT_S, remaining)
        return httpx.Timeout(total, connect=min(10.0, total))

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
        try:
            response = await self._request("POST", "/reset", **kwargs)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            logger.warning("[%s] HTTP %s during /reset: %s", self._label, type(e).__name__, e)
            raise RuntimeError(f"Kernel reset timed out: {e}") from e
        if seed is not None and response.json().get("seed") != seed:
            raise ProtocolVersionError(f"kernel server did not confirm deterministic reset seed {seed} (deploy skew)")

    async def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        """List the workspace via GET /list_dir (endpoint added in PR d)."""
        response = await self._request(
            "GET",
            "/list_dir",
            params={"directory": directory, "max_files": max_files, "show_hidden": show_hidden},
        )
        response.raise_for_status()
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
        version = response.json().get("protocol_version")
        if version is not None and version != EXPECTED_PROTOCOL_VERSION:
            raise ProtocolVersionError(
                f"kernel server protocol_version={version} != expected {EXPECTED_PROTOCOL_VERSION} (deploy skew)"
            )
        return True

    async def aclose(self) -> None:
        """Close the owned httpx client, if any."""
        if self._owns is not None:
            await self._owns.aclose()
            self._owns = None
