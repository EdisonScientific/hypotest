import argparse
import asyncio
import os
import random
import shutil
import socket
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Self, cast
from uuid import UUID

import yaml
from aviary.core import TaskDataset, TaskDatasetServer
from lmi import LiteLLMModel
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, field_validator, model_validator

from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance
from hypotest.env.kernel_server import NBLanguage


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_jsonl: FilePath
    capsule_dir: DirectoryPath
    rubric_model: str = "openai/gpt-5"
    rubric_model_config: dict[str, str | list[Any]] = Field(
        default_factory=lambda: cast(dict[str, str | list[Any]], {"reasoning_effort": "medium"})
    )

    work_dir: Path | None = None
    use_docker: bool = True
    force_python: bool = True
    normalize_reward: bool = True
    save_dir: Path | None = None

    @model_validator(mode="after")
    def make_dirs(self) -> Self:
        for d in (self.work_dir, self.save_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)
        return self


class Dataset(TaskDataset[InterpreterEnv]):
    def __init__(self, config: DatasetConfig):
        self.config = config

        self.problems = [
            ProblemInstance.model_validate_json(line) for line in self.config.problem_jsonl.read_text().splitlines()
        ]

        self.rubric_model = LiteLLMModel(name=self.config.rubric_model, config=self.config.rubric_model_config)

        self.problem_counter: Counter[UUID] = Counter()

    def get_new_env_by_idx(self, idx: int) -> InterpreterEnv:
        problem = self.problems[idx]
        problem_count = self.problem_counter[problem.uuid]
        self.problem_counter[problem.uuid] += 1
        run_id = f"{problem.uuid}-iter{problem_count}"

        capsule_path = self.config.capsule_dir / f"CapsuleData-{problem.uuid}"
        problem_dir = Path(self.config.work_dir) / run_id if self.config.work_dir else Path(mkdtemp())
        problem_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(capsule_path, problem_dir, dirs_exist_ok=True)

        save_dir = Path(self.config.save_dir) / run_id if self.config.save_dir else None

        language = (
            NBLanguage.PYTHON
            if self.config.force_python
            else NBLanguage.from_string(cast(str, problem.metadata["nb_primary_language"]).upper())
        )

        return InterpreterEnv(
            problem=problem,
            rubric_model=self.rubric_model,
            work_dir=problem_dir,
            save_dir=save_dir,
            config=InterpreterEnvConfig(language=language, **self.config.model_dump()),
        )

    def __len__(self) -> int:
        return len(self.problems)


DEFAULT_SERVER_PORT = 8405


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    api_key: str = Field(
        description="API key to access the server; passed either by value or as an environment variable."
    )
    port: int = DEFAULT_SERVER_PORT

    @field_validator("api_key")
    @classmethod
    def read_from_env(cls, val: str) -> str:
        return os.getenv(val, val)

    def model_post_init(self, _):
        # Assign a random port when non-positive value.
        if self.port <= 0:
            self.port = random.randint(1024, 65535)


async def launch_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=FilePath)
    config_path = parser.parse_args().config
    config = ServerConfig.model_validate(yaml.safe_load(config_path.read_text()))

    dataset = Dataset(config.dataset)
    server = TaskDatasetServer(dataset, port=config.port, api_key=config.api_key)

    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"Starting dataset server: Node={socket.gethostname()} IPAddress={ip_address} Port={config.port}")

    await server.astart()


if __name__ == "__main__":
    asyncio.run(launch_server())
