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

import logging
from typing import Any, Literal

import httpx
import nbformat

from hypotest.env.interpreter import ExecutionResult
from hypotest.env.sandbox.base import RequestFn

logger = logging.getLogger(__name__)

# Bumped in lockstep with kernel_server.PROTOCOL_VERSION (added in PR d).
EXPECTED_PROTOCOL_VERSION = 1

_HEALTH_TIMEOUT = httpx.Timeout(5.0, connect=3.0)

# The wire deadline must outlive both the cell and its bounded interrupt/drain.
_MIN_WIRE_TIMEOUT_HEADROOM_S = 30.0
_POST_RECOVERY_TRANSPORT_MARGIN_S = 20.0


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


def _parse_execute_response(response: httpx.Response) -> ExecutionResult:
    """Deserialize an /execute response body into an ExecutionResult."""
    data = response.json()
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
    ) -> None:
        self._request = request
        self._execution_timeout = execution_timeout
        self._timeout_recovery = timeout_recovery
        self._interrupt_grace_seconds = interrupt_grace_seconds
        self._label = label
        # An httpx client this wrapper owns and should aclose() on close (docker);
        # for the connector transport this is None (the SDK owns it).
        self._owns = owns

    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        """Execute code via POST /execute.

        Converts httpx timeouts into an error ExecutionResult, since the kernel
        server's internal asyncio.timeout may not cancel the ZMQ recv promptly.
        """
        effective_timeout = timeout if timeout is not None else self._execution_timeout
        kwargs: dict[str, Any] = {
            "json": {
                "code": code,
                "timeout": timeout,
                "timeout_recovery": self._timeout_recovery,
                "interrupt_grace_seconds": self._interrupt_grace_seconds,
            },
            "timeout": httpx.Timeout(
                execute_wire_timeout_seconds(
                    effective_timeout,
                    self._timeout_recovery,
                    self._interrupt_grace_seconds,
                ),
                connect=10.0,
            ),
        }
        if req_uuid:
            kwargs["headers"] = {"X-Req-UUID": req_uuid}
        try:
            response = await self._request("POST", "/execute", **kwargs)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            logger.warning(
                "[%s] HTTP %s during /execute (requested kernel timeout=%.1fs): %s",
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
        return _parse_execute_response(response)

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
            raise ProtocolVersionError(
                f"kernel server did not confirm deterministic reset seed {seed} (deploy skew)"
            )

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
        """Pull the most-recent capsule for `uuid` in-pod via POST /load_capsule.

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
            raise ProtocolVersionError(
                f"kernel server did not confirm deterministic capsule seed {seed} (deploy skew)"
            )
        return data["objects"]

    async def health(self) -> bool:
        """Return whether GET /health reports ready, rejecting protocol skew."""
        try:
            response = await self._request("GET", "/health", timeout=_HEALTH_TIMEOUT)
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
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
