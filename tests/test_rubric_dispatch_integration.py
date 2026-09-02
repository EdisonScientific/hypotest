"""Integration tests for rubric dispatch configuration and grading."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from hypotest.dataset_server import DatasetConfig
from hypotest.env.interpreter_env import (
    InterpreterEnv,
    InterpreterEnvConfig,
    InterpreterEnvState,
    ProblemInstance,
)
from hypotest.env.kernel_server import NBLanguage
from hypotest.rubric_dispatcher import RubricDispatchConfig, RubricDispatcher


def _dispatch_config(**overrides) -> RubricDispatchConfig:
    values = {
        "enabled": True,
        "max_concurrency": 1,
        "max_outstanding": 2,
        "first_attempt_reserved_slots": 1,
        "retry_reserved_slots": 0,
        "attempt_timeout_seconds": 1,
        "ready_queue_timeout_seconds": 1,
        "logical_timeout_seconds": 2,
        "max_attempts": 2,
        "retry_backoff_initial_seconds": 0,
        "retry_backoff_max_seconds": 0,
    }
    return RubricDispatchConfig(**(values | overrides))


def test_dataset_config_rejects_hidden_router_retries(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="num_retries=0"):
        DatasetConfig(
            problem_jsonl="tasks.jsonl",
            capsule_dir=str(tmp_path),
            rubric_dispatch=_dispatch_config(),
            rubric_model_config={"router_kwargs": {"num_retries": 3}},
        )


def test_dataset_config_rejects_unscheduled_faithfulness_calls(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="faithfulness_mode=off"):
        DatasetConfig(
            problem_jsonl="tasks.jsonl",
            capsule_dir=str(tmp_path),
            faithfulness_mode="hybrid",
            rubric_dispatch=_dispatch_config(),
            rubric_model_config={"router_kwargs": {"num_retries": 0}},
        )


def test_dataset_config_rejects_hidden_http_concurrency_queue(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="remove max_parallel_requests"):
        DatasetConfig(
            problem_jsonl="tasks.jsonl",
            capsule_dir=str(tmp_path),
            rubric_dispatch=_dispatch_config(),
            rubric_model_config={
                "router_kwargs": {"num_retries": 0},
                "model_list": [
                    {
                        "model_name": "model",
                        "litellm_params": {"model": "openai/model", "max_parallel_requests": 1},
                    }
                ],
            },
        )


@pytest.mark.asyncio
async def test_parse_failure_retries_through_dispatcher(tmp_path: pathlib.Path) -> None:
    class FakeRubricModel:
        def __init__(self) -> None:
            self.calls = 0
            self.timeouts: list[float] = []

        async def call_single(self, _messages: Any, **kwargs: Any) -> SimpleNamespace:
            self.calls += 1
            self.timeouts.append(kwargs["timeout"])
            text = "malformed" if self.calls == 1 else "<score>1</score>"
            return SimpleNamespace(text=text)

    problem = ProblemInstance(
        id=UUID("12345678-1234-5678-1234-567812345690"),
        hypothesis="The answer is true",
        protocol="Provide the answer",
        answer=True,
        rubric="* 1 point: correct answer",
        max_points=1,
    )
    model = FakeRubricModel()
    dispatcher = RubricDispatcher(_dispatch_config())
    env = InterpreterEnv(
        problem=problem,
        work_dir=tmp_path,
        rubric_model=model,  # type: ignore[arg-type]
        rubric_dispatcher=dispatcher,
        config=InterpreterEnvConfig(language=NBLanguage.PYTHON, normalize_reward=False),
    )
    env.state = InterpreterEnvState(
        work_dir=tmp_path,
        language=NBLanguage.PYTHON,
        use_docker=False,
        use_ray=False,
    )

    score = await env._evaluate_rubric("True", "notebook", [])

    assert score == 1
    assert model.calls == 2
    assert model.timeouts == [1, 1]
    assert env.state.rubric_model_failed is False
    metadata = env.get_result_metadata()
    assert metadata["rubric_dispatch_enabled"] is True
    assert metadata["rubric_dispatch_attempts"] == 2
    assert metadata["rubric_dispatch_retries"] == 1
    assert metadata["rubric_dispatch_final_reason"] == "success"


@pytest.mark.asyncio
async def test_terminal_provider_failure_preserves_rollout_mask(tmp_path: pathlib.Path) -> None:
    class ProviderError(Exception):
        status_code = 529

    class FailingRubricModel:
        def __init__(self) -> None:
            self.calls = 0

        async def call_single(self, _messages: Any, **_kwargs: Any) -> None:
            self.calls += 1
            raise ProviderError("overloaded")

    problem = ProblemInstance(
        id=UUID("12345678-1234-5678-1234-567812345691"),
        hypothesis="The answer is true",
        protocol="Provide the answer",
        answer=True,
        rubric="* 1 point: correct answer",
        max_points=1,
    )
    model = FailingRubricModel()
    env = InterpreterEnv(
        problem=problem,
        work_dir=tmp_path,
        rubric_model=model,  # type: ignore[arg-type]
        rubric_dispatcher=RubricDispatcher(_dispatch_config()),
        config=InterpreterEnvConfig(language=NBLanguage.PYTHON, normalize_reward=False),
    )
    env.state = InterpreterEnvState(
        work_dir=tmp_path,
        language=NBLanguage.PYTHON,
        use_docker=False,
        use_ray=False,
    )

    with pytest.raises(Exception, match="attempts_exhausted"):
        await env._evaluate_rubric("True", "notebook", [])

    metadata = env.get_result_metadata()
    assert model.calls == 2
    assert metadata["rubric_model_failed"] is True
    assert metadata["rubric_model_fail_request_error"] is True
    assert metadata["rubric_dispatch_attempts"] == 2
    assert metadata["rubric_dispatch_final_reason"] == "attempts_exhausted"
