#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Qualify OpenSandbox resource requests, limits, and burst behavior.

The probe creates one short-lived sandbox with a low CPU/memory request and a
higher limit, inspects its cgroup state, then briefly consumes more than the
requested resources. Credentials and endpoints are read from environment
variables and are never included in the JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shlex
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.sandboxes import PlatformSpec, SandboxImageAuth, SandboxImageSpec

from hypotest.env.sandbox import ResourceSpec
from hypotest.env.sandbox.opensandbox import _resource_map, _resource_request_map

_CGROUP_FILES = (
    "/proc/self/cgroup",
    "/sys/fs/cgroup/cpu.max",
    "/sys/fs/cgroup/cpu.stat",
    "/sys/fs/cgroup/cpu.weight",
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory.high",
    "/sys/fs/cgroup/memory.current",
    "/sys/fs/cgroup/memory.peak",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=os.getenv("GITLAB_IMAGE") or os.getenv("OPEN_SANDBOX_IMAGE"))
    parser.add_argument("--cpu-request", type=float, default=0.25)
    parser.add_argument("--memory-request-mb", type=int, default=512)
    parser.add_argument("--cpu-limit", type=float, default=4)
    parser.add_argument("--memory-limit-mb", type=int, default=65536)
    parser.add_argument("--burst-seconds", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _connection_config() -> ConnectionConfig:
    kwargs: dict[str, Any] = {"request_timeout": timedelta(seconds=300), "use_server_proxy": True}
    if domain := os.getenv("OPEN_SANDBOX_DOMAIN"):
        kwargs["domain"] = domain
    if api_key := os.getenv("OPEN_SANDBOX_API_KEY"):
        kwargs["api_key"] = api_key
    if protocol := os.getenv("OPEN_SANDBOX_PROTOCOL"):
        kwargs["protocol"] = protocol
    return ConnectionConfig(**kwargs)


def _image_spec(image: str) -> str | SandboxImageSpec:
    username = os.getenv("REGISTRY_USERNAME")
    password = os.getenv("REGISTRY_PASSWORD")
    if bool(username) != bool(password):
        raise ValueError("REGISTRY_USERNAME and REGISTRY_PASSWORD must be set together")
    if username and password:
        return SandboxImageSpec(image=image, auth=SandboxImageAuth(username=username, password=password))
    return image


def _execution_stdout(execution: Any) -> str:
    return "".join(message.text for message in execution.logs.stdout)


def _cgroup_probe_command() -> str:
    code = (
        "import json,pathlib;"
        f"paths={_CGROUP_FILES!r};"
        "print(json.dumps({p:(pathlib.Path(p).read_text().strip() "
        "if pathlib.Path(p).exists() else None) for p in paths},sort_keys=True))"
    )
    return f"/app/kernel_env/bin/python -c {shlex.quote(code)}"


def _burst_command(memory_mib: int, cpu_workers: int, seconds: float) -> str:
    code = f"""
import os
import time

payload = bytearray({memory_mib} * 1024 * 1024)
deadline = time.monotonic() + {seconds!r}
children = []
for _ in range({cpu_workers}):
    pid = os.fork()
    if pid == 0:
        value = 1
        while time.monotonic() < deadline:
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        os._exit(0)
    children.append(pid)
while time.monotonic() < deadline:
    payload[0] = (payload[0] + 1) % 255
    time.sleep(0.05)
for pid in children:
    os.waitpid(pid, 0)
from pathlib import Path
import json
print(json.dumps({{
    "cpu_stat": Path("/sys/fs/cgroup/cpu.stat").read_text(),
    "memory_current": Path("/sys/fs/cgroup/memory.current").read_text().strip(),
    "memory_peak": (
        Path("/sys/fs/cgroup/memory.peak").read_text().strip()
        if Path("/sys/fs/cgroup/memory.peak").exists()
        else None
    ),
}}))
"""
    return f"/app/kernel_env/bin/python -c {shlex.quote(code)}"


def _cpu_limit_from_cgroup(value: str | None) -> float | None:
    if not value:
        return None
    quota, period, *_ = value.split()
    if quota == "max":
        return math.inf
    return int(quota) / int(period)


def _memory_limit_mib_from_cgroup(value: str | None) -> float | None:
    if not value or value == "max":
        return math.inf if value == "max" else None
    return int(value) / (1024 * 1024)


def _cpu_usage_seconds(value: str | None) -> float | None:
    if not value:
        return None
    fields = dict(line.split(maxsplit=1) for line in value.splitlines())
    usage_usec = fields.get("usage_usec")
    return int(usage_usec) / 1_000_000 if usage_usec is not None else None


async def _sample_metrics(
    remote: Sandbox,
    command: str,
    seconds: float,
) -> tuple[list[dict[str, float]], Any, float]:
    started = time.perf_counter()
    execution_task = asyncio.create_task(
        remote.commands.run(
            command,
            opts=RunCommandOpts(timeout=timedelta(seconds=seconds + 30)),
        )
    )
    samples: list[dict[str, float]] = []
    while not execution_task.done():
        try:
            metric = await remote.get_metrics()
            samples.append({
                "cpu_count": metric.cpu_count,
                "cpu_used_percentage": metric.cpu_used_percentage,
                "memory_total_mib": metric.memory_total_in_mib,
                "memory_used_mib": metric.memory_used_in_mib,
            })
        except Exception:
            # Metrics are optional on some OpenSandbox providers; cgroup
            # inspection still verifies the QoS class and hard ceilings.
            pass
        await asyncio.sleep(1)
    execution = await execution_task
    return samples, execution, time.perf_counter() - started


async def _run(options: argparse.Namespace) -> dict[str, Any]:
    if not options.image:
        raise ValueError("--image or GITLAB_IMAGE/OPEN_SANDBOX_IMAGE is required")

    resources = ResourceSpec(
        cpu=options.cpu_limit,
        cpu_request=options.cpu_request,
        mem_mb=options.memory_limit_mb,
        mem_request_mb=options.memory_request_mb,
    )
    limits = _resource_map(resources)
    requests = _resource_request_map(resources)
    connection = _connection_config()
    remote: Sandbox | None = None
    started = time.perf_counter()
    try:
        remote = await Sandbox.create(
            image=_image_spec(options.image),
            timeout=timedelta(minutes=15),
            ready_timeout=timedelta(minutes=5),
            metadata={"hypotest-purpose": "resource-qualification"},
            resource=limits,
            resource_requests=requests,
            platform=PlatformSpec(os="linux", arch="amd64"),
            extensions={
                "imagePullPolicy": "Always",
                "opensandbox.extensions.image-pull-policy": "Always",
            },
            entrypoint=["sh", "-lc", "sleep 900"],
            connection_config=connection,
        )
        allocation_seconds = time.perf_counter() - started

        probe = await remote.commands.run(
            _cgroup_probe_command(),
            opts=RunCommandOpts(timeout=timedelta(seconds=30)),
        )
        if probe.exit_code not in (None, 0):
            raise RuntimeError(f"cgroup probe failed with exit code {probe.exit_code}")
        cgroup = json.loads(_execution_stdout(probe))

        memory_burst_mib = min(
            max(options.memory_request_mb + 256, math.ceil(options.memory_request_mb * 1.5)),
            options.memory_limit_mb - 128,
        )
        if memory_burst_mib <= options.memory_request_mb:
            raise ValueError("memory limit needs at least 129 MiB of headroom above the request for the burst probe")
        cpu_workers = max(1, min(2, math.floor(options.cpu_limit)))
        samples, burst_execution, burst_wall_seconds = await _sample_metrics(
            remote,
            _burst_command(memory_burst_mib, cpu_workers, options.burst_seconds),
            options.burst_seconds,
        )
        burst_payload = json.loads(_execution_stdout(burst_execution))

        cgroup_path = cgroup.get("/proc/self/cgroup") or ""
        cpu_weight_raw = cgroup.get("/sys/fs/cgroup/cpu.weight")
        cpu_weight = int(cpu_weight_raw) if cpu_weight_raw and cpu_weight_raw.isdigit() else None
        observed_cpu_limit = _cpu_limit_from_cgroup(cgroup.get("/sys/fs/cgroup/cpu.max"))
        observed_memory_limit = _memory_limit_mib_from_cgroup(cgroup.get("/sys/fs/cgroup/memory.max"))
        peak_memory_mib = max((sample["memory_used_mib"] for sample in samples), default=None)
        peak_cpu_cores = max(
            (
                sample["cpu_count"] * sample["cpu_used_percentage"] / 100
                for sample in samples
            ),
            default=None,
        )
        cpu_before = _cpu_usage_seconds(cgroup.get("/sys/fs/cgroup/cpu.stat"))
        cpu_after = _cpu_usage_seconds(burst_payload.get("cpu_stat"))
        cgroup_cpu_cores = (
            (cpu_after - cpu_before) / burst_wall_seconds
            if cpu_before is not None and cpu_after is not None
            else None
        )
        cgroup_peak_memory_mib = _memory_limit_mib_from_cgroup(
            burst_payload.get("memory_peak") or burst_payload.get("memory_current")
        )
        observed_burst_memory_mib = (
            cgroup_peak_memory_mib if cgroup_peak_memory_mib is not None else peak_memory_mib
        )
        observed_burst_cpu_cores = cgroup_cpu_cores if cgroup_cpu_cores is not None else peak_cpu_cores
        # Some runtimes use a private cgroup namespace and expose the process
        # path as "/" instead of retaining "kubepods/burstable". A non-minimal
        # CPU weight is the equivalent in-container evidence that the request
        # was applied; unequal request/limit pairs then imply Burstable QoS.
        burstable_qos = "burstable" in cgroup_path.lower() or (
            cpu_weight is not None
            and cpu_weight > 1
            and (
                options.cpu_request < options.cpu_limit
                or options.memory_request_mb < options.memory_limit_mb
            )
        )
        checks = {
            "burstable_qos": burstable_qos,
            "cpu_limit_matches": observed_cpu_limit is not None
            and math.isclose(observed_cpu_limit, options.cpu_limit, rel_tol=0.01, abs_tol=0.01),
            "memory_limit_matches": observed_memory_limit is not None
            and math.isclose(observed_memory_limit, options.memory_limit_mb, rel_tol=0.01, abs_tol=1),
            "burst_command_succeeded": burst_execution.exit_code in (None, 0),
            "memory_burst_observed": observed_burst_memory_mib is not None
            and observed_burst_memory_mib > options.memory_request_mb,
            "cpu_burst_observed": observed_burst_cpu_cores is not None
            and observed_burst_cpu_cores > options.cpu_request,
        }
        return {
            "passed": all(checks.values()),
            "allocation_seconds": allocation_seconds,
            "requested": requests,
            "limits": limits,
            "observed": {
                "qos_class": "Burstable" if burstable_qos else "unknown",
                "cpu_limit": observed_cpu_limit,
                "memory_limit_mib": observed_memory_limit,
                "cpu_weight": cpu_weight,
                "cgroup_burst_cpu_cores": cgroup_cpu_cores,
                "cgroup_peak_memory_mib": cgroup_peak_memory_mib,
                "peak_cpu_cores": peak_cpu_cores,
                "peak_memory_mib": peak_memory_mib,
                "metric_samples": len(samples),
            },
            "checks": checks,
        }
    finally:
        if remote is not None:
            try:
                await remote.kill()
            finally:
                await remote.close()


def main() -> int:
    options = _parse_args()
    report = asyncio.run(_run(options))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
