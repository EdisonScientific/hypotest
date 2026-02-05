import json
import argparse
import asyncio
import os
import shutil
from collections import Counter
from pathlib import Path
import random
import socket
from tempfile import mkdtemp
from typing import Self, cast, Union
from uuid import UUID

import yaml
from aviary.core import TaskDataset, TaskDatasetServer
from lmi import LiteLLMModel
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, field_validator, model_validator

from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, InterpreterPool, ProblemInstance
from hypotest.env.kernel_server import NBLanguage


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to port 0 to let the OS choose a free port
        return s.getsockname()[1]


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_jsonl: FilePath
    capsule_dir: DirectoryPath
    rubric_model: str = "openai/gpt-5"
    rubric_model_config: dict[str, Union[str, list]] = Field(default_factory=lambda: {"reasoning_effort": "medium"})

    work_dir: Path | None = None
    use_docker: bool = True
    force_python: bool = True
    normalize_reward: bool = True
    save_dir: Path | None = None

    pool_size: int = 512
    warm_size: int = 32

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

    async def initialize(self):
        work_dir = Path(self.config.work_dir) if self.config.work_dir else Path(mkdtemp())

        # initialize python and R notebook pools
        python_pool = InterpreterPool(self.config.pool_size, NBLanguage.PYTHON, spares=self.config.warm_size)
        await python_pool.initialize(work_dir=work_dir, execution_timeout=600)

        r_lang_pool = InterpreterPool(self.config.pool_size, NBLanguage.R, spares=self.config.warm_size)
        await r_lang_pool.initialize(work_dir=work_dir, execution_timeout=600)

        self.interpreter_pool_dict = {NBLanguage.PYTHON: python_pool, NBLanguage.R: r_lang_pool}

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
            interpreter_pool_dict=self.interpreter_pool_dict
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
        ## Assign a random free port when non-positive value.
        if self.port <= 0:
            self.port = get_free_port()

async def launch_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=FilePath)
    parser.add_argument('-p', '--port', type=int, default=None)
    parser.add_argument('-d', '--registry-dir', type=str, default="/registry")
    parser.add_argument('-w', '--work-dir', type=str, default=None)
    args = parser.parse_args()
    config_path = args.config
    config = ServerConfig.model_validate(yaml.safe_load(config_path.read_text()))

    # if we get work dir in through args, override the config
    if args.work_dir is not None:
        config.dataset.work_dir = Path(args.work_dir)

    dataset = Dataset(config.dataset)
    await dataset.initialize()
    server = TaskDatasetServer(dataset, port=config.port if args.port is None else args.port, api_key=config.api_key)

    print(f"Starting dataset server: Node={socket.gethostname()} IPAddress={socket.gethostbyname(socket.gethostname())} Port={config.port}")

    hostname = socket.gethostname()
    with open(os.path.join(args.registry_dir, f"{hostname}.json"), 'w') as f:
        reg_entry = {'host': hostname, 'port': server.port}
        json.dump(reg_entry, f)

    await server.astart()

    

if __name__ == "__main__":
    asyncio.run(launch_server())
