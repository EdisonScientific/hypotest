"""Tests for configurable episode-time accounting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aviary.core import ToolRequestMessage
from pydantic import ValidationError

from hypotest.env.config import ExecutionConfig
from hypotest.env.interpreter import ExecutionResult
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance


def _env(tmp_path, problem: ProblemInstance, time_accounting: dict) -> InterpreterEnv:
    return InterpreterEnv(
        problem=problem,
        work_dir=tmp_path,
        config=InterpreterEnvConfig(
            execution_config=ExecutionConfig(time_accounting=time_accounting),
            pull_capsule_in_pod=False,
        ),
    )


def _model_turn(response_id: str, turn_index: int, output_tokens: int | None) -> dict:
    usage = None
    if output_tokens is not None:
        usage = {
            "input_tokens": 100,
            "output_tokens": output_tokens,
            "total_tokens": 100 + output_tokens,
        }
    return {
        "response_id": response_id,
        "turn_index": turn_index,
        "usage": usage,
    }


def _action_with_model_turns(*turns: dict) -> ToolRequestMessage:
    return ToolRequestMessage(
        content="",
        tool_calls=[],
        info={
            "nemo_gym": {
                "step_context": {
                    "version": 1,
                    "model_turns": list(turns),
                }
            }
        },
    )


def test_wall_clock_is_default_and_rejects_generation_latency() -> None:
    assert ExecutionConfig().time_accounting.mode == "wall_clock"

    with pytest.raises(ValidationError, match="generation_latency"):
        ExecutionConfig.model_validate(
            {
                "time_accounting": {
                    "mode": "wall_clock",
                    "generation_latency": {"mode": "fixed", "seconds_per_generation": 1},
                },
            }
        )


@pytest.mark.parametrize("mode", ["fixed", "rolling_mean", "rolling_p95"])
def test_kernel_execution_accepts_positive_generation_estimates(mode: str) -> None:
    config = ExecutionConfig.model_validate(
        {
            "time_accounting": {
                "mode": "kernel_execution",
                "generation_latency": {"mode": mode, "seconds_per_generation": 2.5},
            },
        }
    )
    assert config.time_accounting.mode == "kernel_execution"
    assert config.time_accounting.generation_latency.mode == mode
    assert config.time_accounting.generation_latency.seconds_per_generation == 2.5


def test_kernel_execution_accepts_token_throughput() -> None:
    config = ExecutionConfig.model_validate(
        {
            "time_accounting": {
                "mode": "kernel_execution",
                "generation_latency": {
                    "mode": "token_throughput",
                    "output_tokens_per_second": 141,
                },
            },
        }
    )
    assert config.time_accounting.generation_latency.mode == "token_throughput"
    assert config.time_accounting.generation_latency.output_tokens_per_second == 141


@pytest.mark.parametrize(
    "time_accounting",
    [
        {"mode": "kernel_execution", "generation_latency": {"mode": "fixed"}},
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "fixed", "seconds_per_generation": 0},
        },
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "fixed", "seconds_per_generation": float("inf")},
        },
        {"mode": "kernel_execution", "generation_latency": {"mode": "token_throughput"}},
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "token_throughput", "output_tokens_per_second": 0},
        },
        {
            "mode": "kernel_execution",
            "generation_latency": {
                "mode": "token_throughput",
                "output_tokens_per_second": float("inf"),
            },
        },
        {"mode": "unknown"},
    ],
)
def test_invalid_time_accounting_configs_are_rejected(time_accounting: dict) -> None:
    with pytest.raises(ValidationError, match="time_accounting"):
        ExecutionConfig(time_accounting=time_accounting)


def test_wall_clock_mode_preserves_historical_elapsed_time(tmp_path, default_problem: ProblemInstance) -> None:
    env = _env(tmp_path, default_problem, {"mode": "wall_clock"})
    env.start_time = 100.0
    env._kernel_execution_seconds = 999.0

    with patch("hypotest.env.interpreter_env.time.perf_counter", return_value=112.25):
        assert env.get_elapsed_time() == 12.25
        assert env.get_remaining_time() == int(env.execution_config.job_timeout - 12.25)


def test_kernel_execution_ignores_wall_latency_and_adds_simulated_generation(
    tmp_path, default_problem: ProblemInstance
) -> None:
    env = _env(
        tmp_path,
        default_problem,
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "rolling_p95", "seconds_per_generation": 4.0},
        },
    )
    env.start_time = 1.0
    env._record_kernel_execution(7.5, observed_seconds=100.0, logical_execution_id="cell-1")
    env._record_policy_generation()
    env._record_policy_generation()

    with patch("hypotest.env.interpreter_env.time.perf_counter", return_value=10_000.0):
        assert env.get_elapsed_time() == 15.5
        metadata = env.get_time_accounting_metadata()

    assert metadata["mode"] == "kernel_execution"
    assert metadata["kernel_execution_seconds"] == 7.5
    assert metadata["simulated_generation_seconds"] == 8.0
    assert metadata["policy_generation_count"] == 2
    assert metadata["generation_latency"]["mode"] == "rolling_p95"


def test_token_throughput_charges_all_reported_model_turns(
    tmp_path, default_problem: ProblemInstance
) -> None:
    env = _env(
        tmp_path,
        default_problem,
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "token_throughput", "output_tokens_per_second": 141},
        },
    )
    action = _action_with_model_turns(
        _model_turn("resp-1", 1, 141),
        _model_turn("resp-2", 2, 282),
    )

    assert env._record_policy_generation(action) == 3.0
    assert env.get_elapsed_time() == 3.0
    metadata = env.get_time_accounting_metadata()
    assert metadata["policy_generation_count"] == 2
    assert metadata["policy_generation_output_tokens"] == 423


@pytest.mark.parametrize(
    "action",
    [
        ToolRequestMessage(content="", tool_calls=[]),
        _action_with_model_turns(_model_turn("resp-missing", 1, None)),
    ],
)
def test_token_throughput_requires_reported_output_tokens(
    tmp_path, default_problem: ProblemInstance, action: ToolRequestMessage
) -> None:
    env = _env(
        tmp_path,
        default_problem,
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "token_throughput", "output_tokens_per_second": 141},
        },
    )

    with pytest.raises(ValueError, match="token_throughput generation latency requires"):
        env._record_policy_generation(action)

    assert env.get_time_accounting_metadata()["policy_generation_count"] == 0


def test_generation_response_ids_are_only_charged_once(tmp_path, default_problem: ProblemInstance) -> None:
    env = _env(
        tmp_path,
        default_problem,
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "token_throughput", "output_tokens_per_second": 141},
        },
    )
    action = _action_with_model_turns(_model_turn("resp-1", 1, 141))

    assert env._record_policy_generation(action) == 1.0
    assert env._record_policy_generation(action) == 0.0
    metadata = env.get_time_accounting_metadata()
    assert metadata["policy_generation_count"] == 1
    assert metadata["policy_generation_output_tokens"] == 141
    assert metadata["duplicate_generation_accounting_suppressed"] == 1


@pytest.mark.asyncio
async def test_cell_accounting_prefers_backend_reported_time(tmp_path, default_problem: ProblemInstance) -> None:
    env = _env(tmp_path, default_problem, {"mode": "kernel_execution"})

    async def execute_and_add_cell(*args, **kwargs):  # noqa: RUF029
        return ExecutionResult(execution_time=6.25), 0

    env.state = SimpleNamespace(execute_and_add_cell=execute_and_add_cell)
    with patch("hypotest.env.interpreter_env.time.perf_counter", side_effect=[10.0, 999.0]):
        await env._execute_and_account_cell("1 + 1", cell_idx=None, timeout=30)

    assert env.get_elapsed_time() == 6.25
    assert env.get_time_accounting_metadata()["unreported_execution_time_count"] == 0


def test_missing_backend_time_does_not_charge_client_wait(tmp_path, default_problem: ProblemInstance) -> None:
    env = _env(tmp_path, default_problem, {"mode": "kernel_execution"})
    env._record_kernel_execution(None, observed_seconds=2.75, logical_execution_id="cell-1")

    assert env.get_elapsed_time() == 0.0
    metadata = env.get_time_accounting_metadata()
    assert metadata["unreported_execution_time_count"] == 1
    assert metadata["unreported_execution_observed_seconds"] == 2.75


def test_same_logical_execution_is_only_charged_once(tmp_path, default_problem: ProblemInstance) -> None:
    env = _env(tmp_path, default_problem, {"mode": "kernel_execution"})
    env._record_kernel_execution(4.5, observed_seconds=5.0, logical_execution_id="request-1")
    env._record_kernel_execution(4.5, observed_seconds=20.0, logical_execution_id="request-1")

    assert env.get_elapsed_time() == 4.5
    assert env.get_time_accounting_metadata()["duplicate_execution_accounting_suppressed"] == 1


@pytest.mark.asyncio
async def test_step_charges_exactly_one_policy_generation(tmp_path, default_problem: ProblemInstance) -> None:
    env = _env(
        tmp_path,
        default_problem,
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "fixed", "seconds_per_generation": 3.0},
        },
    )
    env.state = SimpleNamespace(actions=[], score=0.0, done=False)
    elapsed_at_tool_dispatch: list[float] = []

    async def exec_tool_calls(*args, **kwargs):  # noqa: RUF029
        elapsed_at_tool_dispatch.append(env.get_elapsed_time())
        return []

    env.exec_tool_calls = exec_tool_calls
    await env.step(ToolRequestMessage(content="", tool_calls=[]))

    assert env.get_elapsed_time() == 3.0
    assert elapsed_at_tool_dispatch == [3.0]
    assert env.get_time_accounting_metadata()["policy_generation_count"] == 1


@pytest.mark.asyncio
async def test_step_charges_token_throughput_before_tool_dispatch(
    tmp_path, default_problem: ProblemInstance
) -> None:
    env = _env(
        tmp_path,
        default_problem,
        {
            "mode": "kernel_execution",
            "generation_latency": {"mode": "token_throughput", "output_tokens_per_second": 141},
        },
    )
    env.state = SimpleNamespace(actions=[], score=0.0, done=False)
    elapsed_at_tool_dispatch: list[float] = []

    async def exec_tool_calls(*args, **kwargs):  # noqa: RUF029
        elapsed_at_tool_dispatch.append(env.get_elapsed_time())
        return []

    env.exec_tool_calls = exec_tool_calls
    await env.step(_action_with_model_turns(_model_turn("resp-1", 1, 282)))

    assert elapsed_at_tool_dispatch == [2.0]
    assert env.get_time_accounting_metadata()["policy_generation_output_tokens"] == 282
