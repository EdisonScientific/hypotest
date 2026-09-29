# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission feedback must not affect scoring or disclose correctness when disabled."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hypotest.dataset_server import Dataset, DatasetConfig
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, InterpreterEnvState, ProblemInstance
from hypotest.env.kernel_server import NBLanguage


@pytest.mark.asyncio
@pytest.mark.parametrize("include_feedback", [True, False])
@pytest.mark.parametrize("score", [0, 10])
async def test_submission_feedback_preserves_scoring(
    tmp_path: Path, default_problem: ProblemInstance, include_feedback: bool, score: int
) -> None:
    rubric = SimpleNamespace(call_single=AsyncMock(return_value=SimpleNamespace(text=f"<score>{score}</score>")))
    env = InterpreterEnv(
        problem=default_problem,
        work_dir=tmp_path,
        rubric_model=rubric,
        config=InterpreterEnvConfig(include_answer_feedback=include_feedback),
    )
    env.state = InterpreterEnvState(work_dir=tmp_path, language=NBLanguage.PYTHON, use_docker=False, use_ray=False)

    result = await env.submit_answer("My analysis")

    expected_feedback = "Correct answer!" if score else "Incorrect answer."
    assert result == (expected_feedback if include_feedback else "Answer submitted.")
    assert env.state.done
    assert env.state.answer == "My analysis"
    assert env.state.raw_score == score
    assert env.state.score == score / default_problem.max_score
    assert env.get_result_metadata()["rubric_model_parsed_score"] == score
    rubric.call_single.assert_awaited_once()
    # Duplicate submissions cannot change either the stored answer or its score.
    assert await env.submit_answer("Another analysis") == "Episode already finished."
    assert env.state.answer == "My analysis"
    rubric.call_single.assert_awaited_once()


@pytest.mark.parametrize("include_feedback", [True, False])
def test_dataset_propagates_feedback_setting(
    tmp_path: Path, default_problem: ProblemInstance, monkeypatch: pytest.MonkeyPatch, include_feedback: bool
) -> None:
    problem_file = tmp_path / "problems.jsonl"
    problem_file.write_text(
        json.dumps({
            "id": str(default_problem.id),
            "hypothesis": default_problem.hypothesis,
            "protocol": default_problem.protocol,
            "answer": True,
            "rubric": default_problem.rubric,
            "max_points": default_problem.max_score,
        })
        + "\n",
        encoding="utf-8",
    )
    capsules = tmp_path / "capsules"
    (capsules / f"CapsuleData-{default_problem.id}").mkdir(parents=True)
    monkeypatch.setattr("hypotest.dataset_server.LiteLLMModel", lambda **kwargs: None)
    dataset = Dataset(
        DatasetConfig(
            problem_jsonl=str(problem_file),
            capsule_dir=str(capsules),
            work_dir=tmp_path / "work",
            include_answer_feedback=include_feedback,
            use_ray=False,
            use_enroot=False,
        )
    )

    env = dataset.get_new_env_by_idx(0)

    assert env.config.include_answer_feedback is include_feedback
    assert DatasetConfig.model_fields["include_answer_feedback"].default is True
    assert InterpreterEnvConfig().include_answer_feedback is True
