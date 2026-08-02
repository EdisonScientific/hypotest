"""Tests for the OpenSandbox trace replay qualification harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import httpx
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "replay_opensandbox_trace.py"
SPEC = importlib.util.spec_from_file_location("hypotest_replay_opensandbox_trace", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def _rollout() -> replay.ReplayRollout:
    action = replay.ReplayAction(
        name="run_cell",
        arguments={"code": "1 + 1"},
        expected_error=False,
    )
    return replay.ReplayRollout(
        ref="s40/t7/0",
        task_idx=7,
        actions=(action,),
        initial_files=frozenset({"matrix.csv"}),
        code_file_literals=frozenset({"matrix.csv"}),
        skipped_submit_calls=1,
        skipped_non_sandbox_calls=0,
    )


def _options(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "output": tmp_path / "replay.json",
        "limit_rollouts": None,
        "rollout_ref": [],
        "concurrency": 1,
        "image": "registry.example/hypotest:test",
        "mounted_root": "/mnt/capsules",
        "run_id": "gbs1024-unit-test",
        "cpu_request": 0.25,
        "memory_request_mb": 512,
        "cpu_limit": 4.0,
        "memory_limit_mb": 16384,
        "kernel_memory_limit_mb": 14336,
        "ephemeral_storage_gib": 50,
        "job_timeout_seconds": 10800,
        "cell_timeout_seconds": 900,
        "ready_timeout_seconds": 900,
        "lifecycle_create_concurrency": 64,
        "kernel_request_concurrency": 128,
        "create_attempts": 2,
        "ttl_seconds": 14400,
        "max_wall_seconds": 45.0,
        "progress_seconds": 30.0,
        "trace_step": 40,
    }
    values.update(overrides)
    return Namespace(**values)


def test_metrics_persist_interval_throughput_latency_and_driver_stats(tmp_path):
    event_path = tmp_path / "events.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    metrics = replay.ReplayMetrics(
        total=1,
        event_path=event_path,
        metrics_path=metrics_path,
        run_id="gbs1024-unit-test",
    )
    rollout = _rollout()

    metrics.record_run_started({"concurrency": 1})
    metrics.rollout_started()
    metrics.rollout_ready({
        "allocation_seconds": 1.0,
        "kernel_connect_seconds": 0.5,
        "startup_seconds": 1.5,
        "create_attempts": 2,
    })
    metrics.action_finished(rollout.actions[0], latency=0.25, observed_error=False)
    metrics.rollout_finished(
        rollout,
        success=True,
        rollout_seconds=2.0,
        cleanup_seconds=0.1,
        diagnostic=replay.ReplayDiagnostic(completed_actions=1),
    )
    snapshot = metrics.progress(event_loop_lag_seconds=0.02, final=True)
    summary = metrics.summary(expected_actions=1, trace_step=40, config={"concurrency": 1})
    metrics.close()

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert [record["type"] for record in records] == ["run_started", "progress"]
    assert snapshot["interval"]["sandbox_actions_per_second"] > 0
    assert snapshot["latency_seconds"]["run_cell_seconds"]["p95"] == 0.25
    assert snapshot["driver"]["event_loop_lag_seconds"] == 0.02
    assert snapshot["driver"]["max_rss_mib"] > 0
    assert summary["passed"] is True
    assert summary["driver"]["event_loop_lag_seconds"]["p99"] == 0.02
    assert len(event_path.read_text(encoding="utf-8").splitlines()) == 1
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["completed_actions"] == 1
    assert event["http_status"] is None
    assert event["exception_chain"] == []


def test_failure_diagnostic_records_sanitized_full_exception_chain():
    request = httpx.Request("GET", "https://sandbox.invalid")
    response = httpx.Response(503, request=request)
    wire_error = httpx.HTTPStatusError("sensitive response", request=request, response=response)
    sdk_error = RuntimeError("sensitive cluster URL")
    sdk_error.__cause__ = wire_error
    wrapped = replay.ReplayError("sensitive image reference")
    wrapped.__cause__ = sdk_error

    diagnostic = replay.ReplayDiagnostic()
    diagnostic.capture_exception(wrapped)

    assert diagnostic.error_type == "ReplayError"
    assert diagnostic.cause_type == "RuntimeError"
    assert diagnostic.http_status == 503
    assert diagnostic.exception_chain == [
        {"type": "ReplayError", "http_status": None},
        {"type": "RuntimeError", "http_status": None},
        {"type": "HTTPStatusError", "http_status": 503},
    ]


def test_replay_can_select_exact_rollout_refs(monkeypatch, tmp_path):
    first = _rollout()
    second = replay.ReplayRollout(
        ref="s40/t7/1",
        task_idx=7,
        actions=first.actions,
        initial_files=first.initial_files,
        code_file_literals=first.code_file_literals,
        skipped_submit_calls=1,
        skipped_non_sandbox_calls=0,
    )
    batch = replay.TraceBatch(
        rollouts=(first, second),
        task_files={7: first.initial_files},
        task_code_file_literals={7: first.code_file_literals},
        action_counts=Counter({"run_cell": 2}),
    )
    options = _options(tmp_path, rollout_ref=[second.ref])
    replayed: list[str] = []

    monkeypatch.setattr(replay, "_make_config", lambda _options: object())
    monkeypatch.setattr(replay, "_raise_fd_limit", lambda _required: (4096, 4096))

    async def record_rollout(
        selected_rollout,
        *,
        capsule,
        config,
        work_root,
        metrics,
        action_limit=None,
    ):
        del capsule, config, work_root, action_limit
        replayed.append(selected_rollout.ref)
        metrics.rollout_started()
        metrics.rollout_ready({})
        metrics.action_finished(selected_rollout.actions[0], latency=0.01, observed_error=False)
        await replay.asyncio.sleep(0)
        metrics.rollout_finished(
            selected_rollout,
            success=True,
            rollout_seconds=0.01,
            cleanup_seconds=0.0,
            diagnostic=replay.ReplayDiagnostic(completed_actions=1),
        )

    monkeypatch.setattr(replay, "_run_one_rollout", record_rollout)

    summary = replay.asyncio.run(replay.run_replay(batch, {7: "private-capsule"}, options))

    assert summary["passed"] is True
    assert replayed == [second.ref]


def test_timeout_marks_summary_failed_and_cancels_active_worker(monkeypatch, tmp_path):
    rollout = _rollout()
    batch = replay.TraceBatch(
        rollouts=(rollout,),
        task_files={7: rollout.initial_files},
        task_code_file_literals={7: rollout.code_file_literals},
        action_counts=Counter({"run_cell": 1}),
    )
    options = _options(
        tmp_path,
        max_wall_seconds=0.02,
        progress_seconds=0.005,
    )

    monkeypatch.setattr(replay, "_make_config", lambda _options: object())
    monkeypatch.setattr(replay, "_raise_fd_limit", lambda _required: (4096, 4096))

    async def slow_rollout(
        selected_rollout,
        *,
        capsule,
        config,
        work_root,
        metrics,
        action_limit=None,
    ):
        del capsule, config, work_root, action_limit
        metrics.rollout_started()
        try:
            await replay.asyncio.sleep(60)
        finally:
            metrics.rollout_finished(
                selected_rollout,
                success=False,
                rollout_seconds=0.02,
                cleanup_seconds=0.0,
                diagnostic=replay.ReplayDiagnostic(
                    error_type="CancelledError",
                    failure_phase="action",
                ),
            )

    monkeypatch.setattr(replay, "_run_one_rollout", slow_rollout)

    summary = replay.asyncio.run(replay.run_replay(batch, {7: "private-capsule"}, options))

    assert summary["passed"] is False
    assert summary["timed_out"] is True
    assert summary["counts"]["failed"] == 1
    assert summary["failure_types"] == {"CancelledError": 1}
    assert summary["timing"]["timeout_observed_at_seconds"] == pytest.approx(0.02, abs=0.03)
    metric_records = [
        json.loads(line) for line in (tmp_path / "replay.metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metric_records[-1]["final"] is True
    assert metric_records[-1]["timed_out"] is True


def test_open_sandbox_config_tags_each_replay_and_keeps_fallback_disabled(tmp_path):
    config = replay._make_config(_options(tmp_path))

    spec = config.opensandbox_spec
    assert spec is not None
    assert spec.metadata == {
        "hypotest-purpose": "gbs1024-trace-replay",
        "hypotest-run": "gbs1024-unit-test",
    }
    assert spec.local_fallback_enabled is False
    assert spec.create_attempts == 2
    assert spec.lifecycle_create_concurrency == 64
    assert spec.kernel_request_concurrency == 128
    assert spec.kernel_memory_limit_mb == 14336
    assert config.execution_config.time_accounting.mode == "kernel_execution"
    assert config.execution_config.time_accounting.generation_latency.mode == "none"
    assert config.execution_config.sandbox_cpu_request == 0.25
    assert config.execution_config.sandbox_cpu == 4.0


@pytest.mark.asyncio
async def test_dispatch_normalizes_recorded_list_dir_directory_typo():
    captured = None

    class FakeEnv:
        async def list_dir(self, **arguments):
            nonlocal captured
            captured = arguments
            return "listing"

    action = replay.ReplayAction(
        name="list_dir",
        arguments={"directory:": "results", "max_files": 5, "show_hidden": False},
    )

    result, observed_error = await replay._dispatch_action(FakeEnv(), action)

    assert result == "listing"
    assert observed_error is None
    assert captured == {"directory": "results", "max_files": 5, "show_hidden": False}
