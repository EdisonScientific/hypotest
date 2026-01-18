import argparse
import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Literal, Self, cast

import yaml
from aviary.core import TaskDatasetClient
from ldp.agent import AgentConfig, SimpleAgent
from ldp.alg import RolloutManager
from ldp.data_structures import Trajectory
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator
from tqdm.asyncio import tqdm

from hypotest.dataset_server import DEFAULT_SERVER_PORT


class SimpleAgentConfig(AgentConfig):
    agent_type: Literal["SimpleAgent"] = "SimpleAgent"  # type: ignore[mutable-override]


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_url: str = f"http://localhost:{DEFAULT_SERVER_PORT}"
    api_key: str = Field(
        description="API key to access the server; passed either by value or as an environment variable."
    )
    agent_config: SimpleAgentConfig
    num_parallel: int = 16
    results_dir: Path

    @model_validator(mode="after")
    def make_dirs(self) -> Self:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self

    @field_validator("api_key")
    @classmethod
    def read_from_env(cls, val: str) -> str:
        return os.getenv(val, val)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=FilePath)
    config_path = parser.parse_args().config
    config = BenchmarkConfig.model_validate(yaml.safe_load(config_path.read_text()))

    client = TaskDatasetClient(config.server_url, api_key=config.api_key, request_timeout=600)
    agent = cast(SimpleAgent, config.agent_config.construct_agent())
    semaphore = asyncio.Semaphore(config.num_parallel)
    rm = RolloutManager(agent)

    async def rollout(idx: int) -> tuple[Trajectory, float]:
        async with semaphore:
            env = client.get_new_env_by_idx(idx)
            trajectory, *_ = await rm.sample_trajectories(environments=[env])
            trajectory.traj_id = f"task_{idx}"
            # assume only terminal reward
            return trajectory, trajectory.steps[-1].reward

    results = await tqdm.gather(*[rollout(i) for i in range(len(client))], ncols=0, desc="Rollouts")
    trajectories, rewards = zip(*results, strict=True)

    (config.results_dir / "rewards.json").write_text(
        json.dumps({t.traj_id: r for t, r in zip(trajectories, rewards, strict=True)}, indent=2)
    )
    (config.results_dir / "trajectories.pkl").write_bytes(pickle.dumps(trajectories))
    avg_reward = sum(rewards) / len(rewards)
    frac_solved = sum(1 for r in rewards if r == 1.0) / len(rewards)
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Fraction solved: {frac_solved:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
