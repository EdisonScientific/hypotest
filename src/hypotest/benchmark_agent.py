import argparse
import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Self, cast

import yaml
from aviary.core import Message, TaskDatasetClient, ToolRequestMessage
from ldp.agent import AgentConfig, SimpleAgent
from ldp.agent.simple_agent import SimpleAgentState
from ldp.alg import RolloutManager
from ldp.data_structures import Trajectory
from ldp.graph import OpResult, compute_graph
from ldp.llms import prepend_sys
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator
from tqdm.asyncio import tqdm

from hypotest.dataset_server import DEFAULT_SERVER_PORT


def _strip_images(msg: Message) -> Message:
    if not msg.is_multimodal:
        return msg
    parsed = json.loads(msg.content)  # type: ignore[arg-type]
    text_parts = [item for item in parsed if item["type"] != "image_url"]
    if not text_parts:
        return msg.model_copy(update={"content": "[image removed]", "content_is_json_str": False})
    if len(text_parts) == 1 and text_parts[0]["type"] == "text":
        return msg.model_copy(update={"content": text_parts[0]["text"], "content_is_json_str": False})
    return msg.model_copy(update={"content": json.dumps(text_parts)})


class NoImagesAgent(SimpleAgent):
    """SimpleAgent that strips images from messages before LLM calls."""

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        next_state = agent_state.get_next_state(obs)

        messages = [_strip_images(m) for m in next_state.messages]
        messages = prepend_sys(messages, sys_content=self.sys_prompt) if self.sys_prompt is not None else messages
        result = cast(
            "OpResult[ToolRequestMessage]",
            await self._llm_call_op(await self._config_op(), msgs=messages, tools=next_state.tools),
        )
        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, 0.0


class SimpleAgentConfig(AgentConfig):
    agent_type: str = "SimpleAgent"


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
