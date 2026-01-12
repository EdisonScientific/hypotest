import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Literal, Self

import yaml
from aviary.core import TaskDatasetClient
from ldp.agent import AgentConfig
from ldp.alg import RolloutManager
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
    agent = config.agent_config.construct_agent()
    semaphore = asyncio.Semaphore(config.num_parallel)
    rm = RolloutManager(agent)

    async def rollout(idx: int) -> float:
        async with semaphore:
            env = client.get_new_env_by_idx(idx)
            trajectory, *_ = await rm.sample_trajectories(environments=[env])
            # assume only terminal reward
            return trajectory.steps[-1].reward

    results = await tqdm.gather(*[rollout(i) for i in range(len(client))], ncols=0, desc="Rollouts")

    (config.results_dir / "rewards.json").write_text(json.dumps(results, indent=2))
    avg_reward = sum(results) / len(results)
    frac_solved = sum(1 for r in results if r == 1.0) / len(results)
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Fraction solved: {frac_solved:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
