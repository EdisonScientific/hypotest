"""Regrade trajectories from a benchmark run using a (potentially different) rubric model."""

import argparse
import asyncio
import json
import logging
import pickle
from pathlib import Path
from typing import Any, cast

import litellm
import tenacity
import yaml
from aviary.core import ToolRequestMessage
from ldp.data_structures import Trajectory
from ldp.graph import OpResult
from lmi import LiteLLMModel
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator
from tqdm.asyncio import tqdm

from hypotest.env.interpreter_env import ProblemInstance
from hypotest.env.prompts import RUBRIC_SCORE_PROMPT
from hypotest.env.utils.notebook_utils import limit_notebook_output

logger = logging.getLogger(__name__)


class RegradeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectories_pkl: FilePath
    problem_jsonl: FilePath
    output_file: Path

    rubric_model: str = "openai/gpt-5"
    rubric_model_config: dict[str, str | list[Any]] = Field(
        default_factory=lambda: cast(dict[str, str | list[Any]], {"reasoning_effort": "medium"})
    )

    normalize_reward: bool = True
    num_parallel: int = 8

    @field_validator("output_file", mode="after")
    @classmethod
    def ensure_parent_exists(cls, val: Path) -> Path:
        val.parent.mkdir(parents=True, exist_ok=True)
        return val


def reconstruct_notebook(trajectory: Trajectory) -> tuple[str, str | None]:
    """Reconstruct notebook content and submitted answer from a trajectory.

    Replays run_cell actions to build {cell_idx: (code, output)} and formats
    the result in the same style as view_notebook.

    Returns:
        Tuple of (notebook_markdown, answer). answer is None if submit_answer was never called.
    """
    cells: dict[int, tuple[str, str]] = {}
    next_idx = 0
    answer: str | None = None

    for step in trajectory.steps:
        action_raw = step.action.value if isinstance(step.action, OpResult) else step.action
        if action_raw is None:
            continue
        assert isinstance(action_raw, ToolRequestMessage)
        obs_list = step.next_observation

        for tc_idx, tc in enumerate(action_raw.tool_calls):
            name = tc.function.name
            args = tc.function.arguments

            if name == "run_cell":
                code = args["code"]
                idx = args.get("idx")
                if idx is not None:
                    idx = int(idx)

                if idx is None or idx >= next_idx:
                    cell_idx = next_idx
                    next_idx += 1
                else:
                    cell_idx = idx

                obs_content = obs_list[tc_idx].content if tc_idx < len(obs_list) else ""
                cells[cell_idx] = (code, limit_notebook_output(str(obs_content)))

            elif name == "submit_answer":
                answer = args["answer"]

    return _format_notebook(cells), answer


def _format_notebook(cells: dict[int, tuple[str, str]]) -> str:
    """Format reconstructed cells in the same style as view_notebook."""
    md: list[str] = []
    for idx in sorted(cells):
        code, output = cells[idx]
        md.append(f"### Cell {idx}:")
        md.extend(("```python", code, "```"))
        if output:
            md.extend((f"### Output {idx}:", "```", output, "```"))
    return "\n".join(md)


@tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(ValueError))
async def grade(
    rubric_model: LiteLLMModel,
    problem: ProblemInstance,
    notebook: str,
    answer: str,
    normalize: bool = True,
) -> tuple[float, int]:
    """Grade a notebook + answer against a problem's rubric.

    Returns:
        Tuple of (score, raw_score).
    """
    prompt = RUBRIC_SCORE_PROMPT.format(
        hypothesis=problem.hypothesis,
        accepted=problem.accepted,
        rubric=problem.rubric,
        notebook=notebook,
        proposed_solution=answer,
    )

    resp = await rubric_model.call_single(prompt, timeout=3 * 60)
    if not resp.text:
        raise ValueError("No response from rubric model")

    try:
        raw_score = int(resp.text.split("<score>")[1].split("</score>")[0])
    except Exception as e:
        raise ValueError("Failed to parse score from response") from e

    if normalize:
        score = raw_score / problem.max_score
        score = max(0.0, min(1.0, score))
    else:
        score = max(0.0, min(float(problem.max_score), float(raw_score)))

    return score, raw_score


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=FilePath)
    config_path = parser.parse_args().config
    config = RegradeConfig.model_validate(yaml.safe_load(config_path.read_text()))

    problems = [ProblemInstance.model_validate_json(line) for line in config.problem_jsonl.read_text().splitlines()]
    rubric_model = LiteLLMModel(name=config.rubric_model, config=config.rubric_model_config)

    trajectories: list[Trajectory] = pickle.loads(config.trajectories_pkl.read_bytes())  # noqa: S301

    semaphore = asyncio.Semaphore(config.num_parallel)

    async def regrade_one(trajectory: Trajectory) -> tuple[str, float, int]:
        assert trajectory.traj_id is not None
        traj_id = trajectory.traj_id
        idx = int(traj_id.split("_")[1])
        problem = problems[idx]
        notebook, answer = reconstruct_notebook(trajectory)

        if answer is None:
            logger.warning(f"{traj_id}: no submit_answer found, scoring as 0")
            return traj_id, 0.0, 0

        async with semaphore:
            try:
                score, raw_score = await grade(
                    rubric_model, problem, notebook, answer, normalize=config.normalize_reward
                )
            except (litellm.exceptions.BadRequestError, litellm.exceptions.ContextWindowExceededError) as e:
                logger.error(f"{traj_id}: skipping due to {type(e).__name__}: {e}")  # noqa: TRY400
                return traj_id, 0.0, 0
            logger.info(f"{traj_id}: {raw_score}/{problem.max_score} (score={score:.3f})")
            return traj_id, score, raw_score

    results = await tqdm.gather(
        *[regrade_one(t) for t in trajectories],
        ncols=0,
        desc="Regrading",  # codespell:ignore
    )

    rewards = {traj_id: score for traj_id, score, _ in results}
    config.output_file.write_text(json.dumps(rewards, indent=2))

    avg_reward = sum(rewards.values()) / len(rewards)
    frac_solved = sum(1 for r in rewards.values() if r == 1.0) / len(rewards)
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Fraction solved: {frac_solved:.3f}")
    print(f"Wrote {config.output_file}")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    asyncio.run(main())
