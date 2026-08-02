#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Inspect a slow trace cell through the independent OpenSandbox command channel.

The probe replays one trajectory until an action exceeds a short threshold,
then compares proxy health with container-local health and captures sanitized
process/cgroup/log counters. It never persists credentials, endpoints, image
references, capsule names, source code, notebook outputs, or raw log lines.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx
from opensandbox.models.execd import RunCommandOpts

from hypotest.env.interpreter_env import InterpreterEnv, ProblemInstance
from hypotest.env.sandbox import OpenSandboxSandbox

_REPLAY_SCRIPT = Path(__file__).with_name("replay_opensandbox_trace.py")
_OPEN_SANDBOX_API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"
_RECOVERY_SENTINEL = "__hypotest_recovery_sentinel__"
_FAILURE_PATTERNS = {
    "container_exit": re.compile(r"container.{0,40}(?:exit|terminat|crash)", re.IGNORECASE | re.DOTALL),
    "eviction": re.compile(r"evict|memorypressure|diskpressure|pidpressure", re.IGNORECASE),
    "exit_137": re.compile(r"(?:exit(?:ed)?|code).{0,20}137|137.{0,20}(?:exit|code)", re.IGNORECASE),
    "liveness": re.compile(r"liveness|health.?check|unhealthy", re.IGNORECASE),
    "node_failure": re.compile(r"nodenotready|node.{0,20}(?:lost|unreachable|not ready)", re.IGNORECASE),
    "oom": re.compile(r"oomkilled|oom.?kill|out of memory|memory cgroup", re.IGNORECASE),
    "sigkill": re.compile(r"sigkill|signal.{0,10}(?:9|killed)", re.IGNORECASE),
}
_CANONICAL_TERMINATION_TOKENS = (
    "OOMKilled",
    "ContainerStatusUnknown",
    "DeadlineExceeded",
    "Error",
    "Evicted",
)
_EXIT_CODE_PATTERN = re.compile(
    r"(?:exit(?:ed)?(?:\s+with)?(?:\s+code)?|exit[_ -]?code)\s*[:=]?\s*(\d{1,3})",
    re.IGNORECASE,
)


def _endpoint_url(endpoint: str, protocol: str) -> str:
    endpoint = endpoint.strip()
    if endpoint.startswith(("http://", "https://")):
        return endpoint.rstrip("/") + "/"
    return f"{protocol}://{endpoint.rstrip('/')}/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capsule-map", type=Path, required=True)
    parser.add_argument("--rollout-ref", required=True)
    parser.add_argument("--trace-step", type=int, default=40)
    parser.add_argument("--probe-after-seconds", type=float, default=45)
    parser.add_argument("--action-limit", type=int)
    parser.add_argument("--baseline-before-index", type=int)
    parser.add_argument("--mounted-root", default="/mnt/s3-data/data/bbh/capsules/edison-20260725/")
    parser.add_argument("--image", default=os.getenv("GITLAB_IMAGE") or os.getenv("OPEN_SANDBOX_IMAGE"))
    parser.add_argument("--run-id", default="kernel-wedge-probe")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qualification/kernel-wedge-probe.json"))
    parser.add_argument("--cpu-request", type=float, default=0.25)
    parser.add_argument("--memory-request-mb", type=int, default=512)
    parser.add_argument("--cpu-limit", type=float, default=4)
    parser.add_argument("--memory-limit-mb", type=int, default=65536)
    parser.add_argument(
        "--kernel-memory-limit-mb",
        type=int,
        help="inner Jupyter RLIMIT_AS ceiling; must be below --memory-limit-mb",
    )
    parser.add_argument("--ephemeral-storage-gib", type=int, default=50)
    parser.add_argument("--cell-timeout-seconds", type=int, default=900)
    parser.add_argument("--job-timeout-seconds", type=int, default=10800)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--create-attempts",
        type=int,
        default=3,
        help="maximum OpenSandbox allocation attempts (default: 3)",
    )
    parser.add_argument("--ttl-seconds", type=int, default=14400)
    parser.add_argument(
        "--recover-preserving-state",
        action="store_true",
        help="after detecting a wedge, test fresh-proxy and pause/resume recovery without replacing the sandbox",
    )
    parser.add_argument("--recovery-timeout-seconds", type=float, default=90)
    parser.add_argument("--recovery-poll-seconds", type=float, default=30)
    parser.add_argument(
        "--expect-completion",
        action="store_true",
        help="pass when every selected action completes and the sandbox remains running",
    )
    parser.add_argument("--probe-diagnostics", action="store_true", help="call the optional stable diagnostics API")
    return parser.parse_args()


def _load_replay_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hypotest_opensandbox_replay_probe", _REPLAY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load replay harness from {_REPLAY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _container_probe_command() -> str:
    code = r"""
import json
import pathlib
import urllib.request

def read(path):
    item = pathlib.Path(path)
    try:
        return item.read_text(errors="replace").strip() if item.exists() else None
    except Exception:
        return None

health = {"reachable": False}
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
        payload = json.loads(response.read())
        health = {
            "reachable": True,
            "http_status": response.status,
            "status": payload.get("status"),
            "kernel_ready": payload.get("kernel_ready"),
            "protocol_version": payload.get("protocol_version"),
        }
except Exception as exc:
    health = {"reachable": False, "error_type": type(exc).__name__}

processes = []
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    name = read(proc / "comm")
    if not name or not any(token in name.lower() for token in ("python", "jupyter", "uvicorn")):
        continue
    fields = {}
    for line in (read(proc / "status") or "").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in {"State", "VmRSS", "VmPeak", "Threads"}:
                fields[key] = value.strip()
    processes.append({
        "pid": int(proc.name),
        "name": name,
        "state": fields.get("State"),
        "rss": fields.get("VmRSS"),
        "peak": fields.get("VmPeak"),
        "threads": fields.get("Threads"),
        "oom_score": read(proc / "oom_score"),
    })

log_path = pathlib.Path("/workspace/.container_logs/container.log")
log_text = read(log_path) or ""
log_tail = log_text[-200000:]
patterns = ("traceback", "error", "exception", "killed", "oom", "timeout", "execute")
workspace_sizes = []
for item in pathlib.Path("/workspace").rglob("*"):
    try:
        if item.is_file() and not any(part.startswith(".") for part in item.relative_to("/workspace").parts):
            workspace_sizes.append(item.stat().st_size)
    except Exception:
        pass

print(json.dumps({
    "local_health": health,
    "processes": sorted(processes, key=lambda item: item["pid"]),
    "cgroup": {
        path: read(path)
        for path in (
            "/sys/fs/cgroup/cpu.stat",
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.peak",
            "/sys/fs/cgroup/memory.events",
            "/sys/fs/cgroup/memory.pressure",
            "/sys/fs/cgroup/pids.current",
            "/sys/fs/cgroup/pids.max",
        )
    },
    "container_log": {
        "bytes": len(log_text.encode(errors="replace")),
        "tail_pattern_counts": {
            pattern: log_tail.lower().count(pattern)
            for pattern in patterns
        },
    },
    "workspace_files": {
        "count": len(workspace_sizes),
        "total_bytes": sum(workspace_sizes),
        "largest_bytes": sorted(workspace_sizes, reverse=True)[:10],
    },
}, sort_keys=True))
"""
    return f"/app/kernel_env/bin/python -c {shlex.quote(code)}"


async def _proxy_health(sandbox: OpenSandboxSandbox) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        healthy = await asyncio.wait_for(sandbox.health(), timeout=10)
        return {
            "completed": True,
            "healthy": healthy,
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }


async def _resolve_direct_target(
    sandbox: OpenSandboxSandbox,
) -> tuple[httpx.URL, dict[str, str]]:
    remote = sandbox._sandbox
    if remote is None:
        raise RuntimeError("remote sandbox is unavailable")
    endpoint = await remote._sandbox_service.get_sandbox_endpoint(
        remote.id,
        sandbox._spec.kernel_port,
        use_server_proxy=False,
    )
    protocol = getattr(remote.connection_config, "protocol", None) or "http"
    return (
        httpx.URL(_endpoint_url(str(endpoint.endpoint), protocol)),
        dict(getattr(endpoint, "headers", {}) or {}),
    )


async def _resolve_proxy_target(
    sandbox: OpenSandboxSandbox,
    *,
    refresh: bool,
) -> tuple[httpx.URL, dict[str, str]]:
    remote = sandbox._sandbox
    if remote is None:
        raise RuntimeError("remote sandbox is unavailable")
    if refresh:
        remote._sandbox_service.invalidate_endpoint_cache(remote.id)
    endpoint = await remote._sandbox_service.get_sandbox_endpoint(
        remote.id,
        sandbox._spec.kernel_port,
        use_server_proxy=True,
    )
    config = remote.connection_config
    protocol = getattr(config, "protocol", None) or "http"
    headers = dict(getattr(endpoint, "headers", {}) or {})
    get_api_key = getattr(config, "get_api_key", None)
    api_key = get_api_key() if callable(get_api_key) else None
    if api_key:
        headers.setdefault(_OPEN_SANDBOX_API_KEY_HEADER, api_key)
    return httpx.URL(_endpoint_url(str(endpoint.endpoint), protocol)), headers


async def _target_health(
    target: tuple[httpx.URL, dict[str, str]] | None,
    resolution_error_type: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if target is None:
        return {
            "completed": False,
            "error_type": resolution_error_type or "MissingDirectEndpoint",
            "seconds": 0.0,
        }
    base_url, headers = target
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
            response = await client.get(base_url.join("health"), headers=headers)
            response.raise_for_status()
            payload = response.json()
        return {
            "completed": True,
            "healthy": payload.get("status") == "OK" and payload.get("kernel_ready") is True,
            "http_status": response.status_code,
            "protocol_version": payload.get("protocol_version"),
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }


async def _fresh_proxy_health(sandbox: OpenSandboxSandbox) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        target = await _resolve_proxy_target(sandbox, refresh=True)
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }
    result = await _target_health(target, None)
    result["seconds"] += time.perf_counter() - started - result["seconds"]
    return result


def _await_chain(task: asyncio.Task[Any]) -> list[str]:
    names: list[str] = []
    current: Any = task.get_coro()
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(names) < 30:
        seen.add(id(current))
        code = (
            getattr(current, "cr_code", None) or getattr(current, "ag_code", None) or getattr(current, "gi_code", None)
        )
        names.append(code.co_name if code is not None else type(current).__name__)
        current = (
            getattr(current, "cr_await", None)
            or getattr(current, "ag_await", None)
            or getattr(current, "gi_yieldfrom", None)
        )
    return names


async def _command_channel_probe(sandbox: OpenSandboxSandbox) -> dict[str, Any]:
    started = time.perf_counter()
    remote = sandbox._sandbox
    if remote is None:
        return {"completed": False, "error_type": "MissingRemoteSandbox", "seconds": 0.0}
    try:
        execution = await asyncio.wait_for(
            remote.commands.run(
                _container_probe_command(),
                opts=RunCommandOpts(timeout=timedelta(seconds=30)),
            ),
            timeout=40,
        )
        stdout = "".join(message.text for message in execution.logs.stdout)
        payload = json.loads(stdout)
        return {
            "completed": True,
            "exit_code": execution.exit_code,
            "seconds": time.perf_counter() - started,
            "payload": payload,
        }
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }


async def _metrics_probe(sandbox: OpenSandboxSandbox) -> dict[str, Any]:
    started = time.perf_counter()
    remote = sandbox._sandbox
    if remote is None:
        return {"completed": False, "error_type": "MissingRemoteSandbox", "seconds": 0.0}
    try:
        metric = await asyncio.wait_for(remote.get_metrics(), timeout=10)
        return {
            "completed": True,
            "seconds": time.perf_counter() - started,
            "cpu_count": metric.cpu_count,
            "cpu_used_percentage": metric.cpu_used_percentage,
            "memory_total_mib": metric.memory_total_in_mib,
            "memory_used_mib": metric.memory_used_in_mib,
        }
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }


def _failure_signals(text: str | None) -> dict[str, bool]:
    value = text or ""
    return {name: pattern.search(value) is not None for name, pattern in _FAILURE_PATTERNS.items()}


def _structured_failure_evidence(text: str | None) -> dict[str, Any]:
    """Extract auditable allowlisted facts without retaining the source message."""
    value = text or ""
    canonical_tokens = [
        token for token in _CANONICAL_TERMINATION_TOKENS if re.search(rf"\b{re.escape(token)}\b", value)
    ]
    exit_codes = sorted({int(match.group(1)) for match in _EXIT_CODE_PATTERN.finditer(value)})
    summary_parts: list[str] = []
    if canonical_tokens:
        summary_parts.append("termination=" + ",".join(canonical_tokens))
    if exit_codes:
        summary_parts.append("exit_code=" + ",".join(str(code) for code in exit_codes))
    return {
        "canonical_termination_tokens": canonical_tokens,
        "exit_codes": exit_codes,
        "redacted_summary": "; ".join(summary_parts) if summary_parts else None,
        "source_message_length": len(value),
        "source_message_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None,
    }


async def _lifecycle_probe(sandbox: OpenSandboxSandbox) -> dict[str, Any]:
    """Read only stable status codes; omit free-form messages and identifiers."""
    started = time.perf_counter()
    remote = sandbox._sandbox
    if remote is None:
        return {"completed": False, "error_type": "MissingRemoteSandbox", "seconds": 0.0}
    try:
        info = await asyncio.wait_for(remote.get_info(), timeout=10)
        state = str(info.status.state)
        reason = info.status.reason
        safe_reason = reason if reason and re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", reason) else None
        return {
            "completed": True,
            "state": state,
            "reason": safe_reason,
            "message_signals": _failure_signals(info.status.message),
            "message_evidence": _structured_failure_evidence(info.status.message),
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }


async def _diagnostics_probe(sandbox: OpenSandboxSandbox) -> dict[str, Any]:
    """Classify diagnostics in memory without persisting their raw content or URLs."""
    started = time.perf_counter()
    remote = sandbox._sandbox
    if remote is None:
        return {"completed": False, "error_type": "MissingRemoteSandbox", "seconds": 0.0}

    async def read(kind: str) -> dict[str, Any]:
        try:
            diagnostic = (
                await remote.get_diagnostic_events("all")
                if kind == "events"
                else await remote.get_diagnostic_logs("all")
            )
            return {
                "completed": True,
                "delivery": diagnostic.delivery,
                "truncated": diagnostic.truncated,
                "content_length": diagnostic.content_length,
                "signals": _failure_signals(diagnostic.content),
            }
        except Exception as exc:
            return {"completed": False, "error_type": type(exc).__name__}

    try:
        events, logs = await asyncio.wait_for(asyncio.gather(read("events"), read("logs")), timeout=20)
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }
    return {
        "completed": True,
        "events": events,
        "logs": logs,
        "seconds": time.perf_counter() - started,
    }


async def _pause_resume_reconnect(
    sandbox: OpenSandboxSandbox,
    timeout_seconds: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    remote = sandbox._sandbox
    pool = sandbox._client_pool
    if remote is None or pool is None:
        return {"completed": False, "error_type": "MissingRemoteSandbox"}

    connection_config = sandbox._make_connection_config(transport=pool.lifecycle_transport)
    started = time.perf_counter()
    try:
        phase_started = time.perf_counter()
        await asyncio.wait_for(remote.pause(), timeout=timeout_seconds)
        report["pause_seconds"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        resumed = await asyncio.wait_for(
            type(remote).resume(
                remote.id,
                connection_config=connection_config,
                resume_timeout=timedelta(seconds=timeout_seconds),
                skip_health_check=True,
            ),
            timeout=timeout_seconds,
        )
        report["resume_seconds"] = time.perf_counter() - phase_started
        sandbox._sandbox = resumed
        sandbox._client = None
        with contextlib.suppress(Exception):
            await remote.close()

        phase_started = time.perf_counter()
        await asyncio.wait_for(sandbox._connect_kernel(connection_config), timeout=timeout_seconds)
        report["kernel_reconnect_seconds"] = time.perf_counter() - phase_started
    except Exception as exc:
        report.update({
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        })
    else:
        report.update({"completed": True, "seconds": time.perf_counter() - started})
    return report


async def _poll_retained_execution(
    sandbox: OpenSandboxSandbox,
    execution_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    if sandbox._client is None:
        return {"completed": False, "error_type": "MissingKernelClient", "polls": 0, "seconds": 0.0}
    client = sandbox._client
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    polls = 0
    last_status: str | None = None
    try:
        while asyncio.get_running_loop().time() < deadline:
            response = await client._request(
                "GET",
                f"/execute/{execution_id}",
                timeout=httpx.Timeout(5.0, connect=3.0),
            )
            response.raise_for_status()
            polls += 1
            last_status = str(response.json().get("status"))
            if last_status in {"completed", "failed", "cancelled"}:
                return {
                    "completed": True,
                    "polls": polls,
                    "status": last_status,
                    "seconds": time.perf_counter() - started,
                }
            await asyncio.sleep(min(2.0, max(0.0, deadline - asyncio.get_running_loop().time())))
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "polls": polls,
            "seconds": time.perf_counter() - started,
        }
    return {
        "completed": False,
        "error_type": "RecoveryPollTimeout",
        "polls": polls,
        "status": last_status,
        "seconds": time.perf_counter() - started,
    }


async def _cancel_retained_execution(
    sandbox: OpenSandboxSandbox,
    execution_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    if sandbox._client is None:
        return {"completed": False, "error_type": "MissingKernelClient", "seconds": 0.0}
    try:
        response = await sandbox._client._request(
            "POST",
            f"/execute/{execution_id}/cancel",
            timeout=httpx.Timeout(30.0, connect=3.0),
        )
        response.raise_for_status()
        status = str(response.json().get("status"))
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }
    return {"completed": True, "status": status, "seconds": time.perf_counter() - started}


async def _sentinel_check(sandbox: OpenSandboxSandbox) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await sandbox.execute(
            f"print({_RECOVERY_SENTINEL!r} in globals())",
            timeout=30,
            req_uuid="hypotest-recovery-sentinel-check",
        )
        preserved = not result.error_occurred and result.get_combined_text().strip().endswith("True")
    except Exception as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }
    return {
        "completed": True,
        "preserved": preserved,
        "seconds": time.perf_counter() - started,
    }


async def _recover_preserving_state(
    sandbox: OpenSandboxSandbox,
    execution_id: str | None,
    options: argparse.Namespace,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "execution_id_captured": execution_id is not None,
        "fresh_proxy": await _fresh_proxy_health(sandbox),
    }
    report["pause_resume"] = await _pause_resume_reconnect(sandbox, options.recovery_timeout_seconds)
    if not report["pause_resume"]["completed"]:
        return report
    if execution_id is not None:
        retained = await _poll_retained_execution(sandbox, execution_id, options.recovery_poll_seconds)
        report["retained_execution"] = retained
        if not retained["completed"]:
            report["retained_execution_cancel"] = await _cancel_retained_execution(sandbox, execution_id)
    report["sentinel"] = await _sentinel_check(sandbox)
    return report


async def _run(options: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0912, PLR0914, PLR0915
    if not options.image:
        raise ValueError("--image or GITLAB_IMAGE/OPEN_SANDBOX_IMAGE is required")
    if options.probe_after_seconds <= 0:
        raise ValueError("--probe-after-seconds must be positive")

    replay = _load_replay_module()
    batch = replay.load_trace_batch(options.trace, options.trace_step)
    mapping = replay._load_capsule_map(options.capsule_map, batch.task_files)
    try:
        rollout = next(item for item in batch.rollouts if item.ref == options.rollout_ref)
    except StopIteration as exc:
        raise ValueError(f"unknown rollout ref {options.rollout_ref!r}") from exc

    config = replay._make_config(options)
    if options.recover_preserving_state:
        config.execution_config.cell_timeout_recovery = "interrupt"
    work_root = Path(tempfile.mkdtemp(prefix="hypotest-kernel-wedge-"))
    work_dir = work_root / "workspace"
    work_dir.mkdir()
    problem = ProblemInstance(
        id=uuid5(NAMESPACE_URL, rollout.ref),
        hypothesis="OpenSandbox kernel wedge probe",
        protocol="Replay until the first slow sandbox action",
        answer=True,
        rubric="Infrastructure diagnostics only",
        max_points=1,
        input_data_path=mapping[rollout.task_idx],
    )
    env = InterpreterEnv(problem=problem, work_dir=work_dir, config=config)
    sandbox: OpenSandboxSandbox | None = None
    direct_target: tuple[httpx.URL, dict[str, str]] | None = None
    direct_resolution_error_type: str | None = None
    actions_report: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    probe: dict[str, Any] | None = None
    post_run: dict[str, Any] | None = None
    recovery: dict[str, Any] | None = None
    captured_execution_id: str | None = None
    cleanup: dict[str, Any] = {}
    actions = rollout.actions
    if options.action_limit is not None:
        actions = actions[: options.action_limit]
    try:
        await env.reset()
        if not isinstance(env.state.sandbox, OpenSandboxSandbox):
            raise TypeError(f"expected OpenSandboxSandbox, got {type(env.state.sandbox).__name__}")
        sandbox = env.state.sandbox
        if options.recover_preserving_state:
            sentinel = await sandbox.execute(
                f"{_RECOVERY_SENTINEL} = object()",
                timeout=30,
                req_uuid="hypotest-recovery-sentinel-set",
            )
            if sentinel.error_occurred:
                raise RuntimeError("could not install recovery sentinel")
            assert sandbox._client is not None
            original_request = sandbox._client._request

            async def capture_execution_id(method: str, path: str, **kwargs: Any) -> httpx.Response:
                nonlocal captured_execution_id
                response = await original_request(method, path, **kwargs)
                if method == "POST" and path == "/execute" and response.status_code == 202:
                    captured_execution_id = str(response.json().get("execution_id"))
                return response

            sandbox._client._request = capture_execution_id
        try:
            direct_target = await _resolve_direct_target(sandbox)
        except Exception as exc:
            direct_resolution_error_type = type(exc).__name__
        for index, action in enumerate(actions):
            if options.baseline_before_index == index:
                baseline_results = await asyncio.gather(
                    _proxy_health(sandbox),
                    _target_health(direct_target, direct_resolution_error_type),
                    _command_channel_probe(sandbox),
                    _metrics_probe(sandbox),
                    _lifecycle_probe(sandbox),
                    _diagnostics_probe(sandbox)
                    if options.probe_diagnostics
                    else asyncio.sleep(0, result={"completed": False, "error_type": "Disabled"}),
                )
                baseline = {
                    "before_index": index,
                    "proxy_health": baseline_results[0],
                    "direct_health": baseline_results[1],
                    "command_channel": baseline_results[2],
                    "metrics": baseline_results[3],
                    "lifecycle": baseline_results[4],
                    "diagnostics": baseline_results[5],
                }
            started = time.perf_counter()
            task = asyncio.create_task(replay._dispatch_action(env, action))
            done, _ = await asyncio.wait({task}, timeout=options.probe_after_seconds)
            if done:
                try:
                    _, observed_error = await task
                except Exception as exc:
                    actions_report.append({
                        "index": index,
                        "name": action.name,
                        "completed": False,
                        "seconds": time.perf_counter() - started,
                        "error_type": type(exc).__name__,
                    })
                    break
                actions_report.append({
                    "index": index,
                    "name": action.name,
                    "completed": True,
                    "seconds": time.perf_counter() - started,
                    "observed_code_error": observed_error,
                })
                continue

            probe_results = await asyncio.gather(
                _proxy_health(sandbox),
                _target_health(direct_target, direct_resolution_error_type),
                _command_channel_probe(sandbox),
                _metrics_probe(sandbox),
                _lifecycle_probe(sandbox),
                _diagnostics_probe(sandbox)
                if options.probe_diagnostics
                else asyncio.sleep(0, result={"completed": False, "error_type": "Disabled"}),
            )
            probe = {
                "trigger": {
                    "index": index,
                    "name": action.name,
                    "elapsed_seconds": time.perf_counter() - started,
                    "await_chain": _await_chain(task),
                },
                "proxy_health": probe_results[0],
                "direct_health": probe_results[1],
                "command_channel": probe_results[2],
                "metrics": probe_results[3],
                "lifecycle": probe_results[4],
                "diagnostics": probe_results[5],
            }
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=20)
            if options.recover_preserving_state:
                recovery = await _recover_preserving_state(sandbox, captured_execution_id, options)
            break

        if probe is None:
            post_results = await asyncio.gather(
                _proxy_health(sandbox),
                _command_channel_probe(sandbox),
                _metrics_probe(sandbox),
                _lifecycle_probe(sandbox),
            )
            post_run = {
                "proxy_health": post_results[0],
                "command_channel": post_results[1],
                "metrics": post_results[2],
                "lifecycle": post_results[3],
            }
    finally:
        cleanup_started = time.perf_counter()
        if hasattr(env, "state"):
            try:
                await asyncio.wait_for(env.close(), timeout=180)
                cleanup["env_close"] = "completed"
            except Exception as exc:
                cleanup["env_close"] = type(exc).__name__
        if sandbox is not None and sandbox._sandbox is not None:
            remote = sandbox._sandbox
            with contextlib.suppress(Exception):
                await remote.kill()
            with contextlib.suppress(Exception):
                await remote.close()
        cleanup["seconds"] = time.perf_counter() - cleanup_started
        shutil.rmtree(work_root, ignore_errors=True)

    passed = probe is not None
    if options.expect_completion:
        passed = bool(
            probe is None
            and len(actions_report) == len(actions)
            and all(action_report.get("completed") is True for action_report in actions_report)
            and post_run is not None
            and post_run.get("lifecycle", {}).get("state") == "Running"
        )
    if options.recover_preserving_state:
        passed = bool(
            passed
            and recovery is not None
            and recovery.get("pause_resume", {}).get("completed") is True
            and recovery.get("sentinel", {}).get("preserved") is True
        )
    return {
        "passed": passed,
        "rollout_ref": rollout.ref,
        "actions": actions_report,
        "resources": {
            "cpu_request": options.cpu_request,
            "cpu_limit": options.cpu_limit,
            "memory_request_mib": options.memory_request_mb,
            "memory_limit_mib": options.memory_limit_mb,
            "kernel_memory_limit_mib": options.kernel_memory_limit_mb,
        },
        "baseline": baseline,
        "probe": probe,
        "post_run": post_run,
        "recovery": recovery,
        "cleanup": cleanup,
    }


def main() -> int:
    options = _parse_args()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(_run(options))
    except Exception as exc:
        report = {
            "passed": False,
            "error_type": type(exc).__name__,
        }
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
