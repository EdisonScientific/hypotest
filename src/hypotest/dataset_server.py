import argparse
import asyncio
import logging
import os
import random
import shutil
import socket
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal, Self, cast
from uuid import UUID

import yaml
from aviary.core import TaskDataset, TaskDatasetServer
from datasets import Dataset as HFDataset
from datasets import load_dataset
from lmi import LiteLLMModel
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator

from hypotest import s3_sync
from hypotest.env.config import ExecutionConfig
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance
from hypotest.env.kernel_server import NBLanguage

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Local path, or s3://bucket/prefix (pulled on start, or lazily per task — see capsule_pull).
    problem_jsonl: str | None = None
    capsule_dir: str
    hf_dataset: str | None = None
    # For an s3:// capsule_dir: "lazy" pulls each task's capsule on demand in
    # get_new_env_by_idx (default; avoids pulling the whole multi-100GB prefix per
    # pod); "eager" pulls the entire prefix on start. Ignored for local capsule_dir.
    capsule_pull: Literal["lazy", "eager"] = "lazy"

    rubric_model: str = "openai/gpt-5"
    rubric_model_config: dict[str, str | list[Any]] = Field(
        default_factory=lambda: cast(dict[str, str | list[Any]], {"reasoning_effort": "medium"})
    )

    work_dir: Path | None = None
    use_ray: bool = True
    use_docker: bool = False
    use_enroot: bool = True
    container_sqsh_path: str | None = None
    force_python: bool = True
    normalize_reward: bool = True
    enable_faithfulness_gate: bool = False
    faithfulness_mode: Literal["off", "binary", "shadow", "hybrid"] = "off"
    wager_mode: Literal["off", "shadow", "active"] = "off"
    wager_beta: float = 0.5
    wager_gamma: float = 0.3
    cell_timeout_override_mode: Literal["off", "on"] = "off"
    cell_timeout_min: float = 60.0
    cell_timeout_max: float = 1200.0
    save_dir: Path | None = None
    data_dir: Path | None = None  # local staging for s3:// sources (default: a temp dir)

    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def validate_dataset_source(self) -> Self:
        if not self.problem_jsonl and not self.hf_dataset:
            raise ValueError("Either problem_jsonl or hf_dataset must be provided")
        return self

    @model_validator(mode="after")
    def make_dirs(self) -> Self:
        for d in (self.work_dir, self.save_dir, self.data_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)
        return self

    def _load_from_hf(self) -> list[ProblemInstance]:
        ds: HFDataset = load_dataset(self.hf_dataset, split="train")
        return [ProblemInstance.model_validate(row) for row in ds]


class Dataset(TaskDataset[InterpreterEnv]):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._s3_client: Any = None
        self._capsule_s3: tuple[str, str] | None = None  # (bucket, prefix) when capsules are s3-backed
        self._staging_dir = config.data_dir or Path(mkdtemp(prefix="hypotest-s3-"))

        # Resolve data sources: the tasks jsonl is pulled now; s3 capsules are
        # pulled eagerly here or lazily per task (see DatasetConfig.capsule_pull).
        self.capsule_dir = self._resolve_capsule_dir()
        self.problems = self._load_problems()

        self.rubric_model = LiteLLMModel(name=self.config.rubric_model, config=self.config.rubric_model_config)

        self.problem_counter: Counter[UUID] = Counter()

    def _client(self) -> Any:
        if self._s3_client is None:
            self._s3_client = s3_sync.make_client()
        return self._s3_client

    def _resolve_capsule_dir(self) -> Path | None:
        """Resolve capsule_dir to a local dir, or None for lazy s3 capsules.

        None means each task's capsule is pulled in get_new_env_by_idx.
        """
        value = self.config.capsule_dir
        if s3_sync.is_s3_uri(value):
            self._capsule_s3 = s3_sync.parse_s3_uri(value)
            if self.config.capsule_pull == "eager":
                bucket, prefix = self._capsule_s3
                dest = self._staging_dir / "capsules"
                count = s3_sync.download_prefix(self._client(), bucket, prefix, dest)
                logger.warning("Eagerly pulled %d capsule objects from %s -> %s", count, value, dest)
                return dest
            logger.warning("Capsules will be pulled lazily per task from %s", value)
            return None
        path = Path(value)
        if not path.is_dir():
            raise ValueError(f"capsule_dir does not exist: {value}")
        return path

    def _load_problems(self) -> list[ProblemInstance]:
        if self.config.hf_dataset:
            return self.config._load_from_hf()
        jsonl = self.config.problem_jsonl
        assert jsonl is not None
        if s3_sync.is_s3_uri(jsonl):
            bucket, key = s3_sync.parse_s3_uri(jsonl)
            local = self._staging_dir / Path(key).name
            s3_sync.download_object(self._client(), bucket, key, local)
            logger.warning("Pulled tasks jsonl from %s -> %s", jsonl, local)
            text = local.read_text(encoding="utf-8")
        else:
            path = Path(jsonl)
            if not path.is_file():
                raise ValueError(f"problem_jsonl does not exist: {jsonl}")
            text = path.read_text(encoding="utf-8")
        return [ProblemInstance.model_validate_json(line) for line in text.splitlines() if line.strip()]

    def _max_existing_problem_iter(self, problem_id: UUID) -> int:
        max_existing_iter = -1
        prefix = f"{problem_id}-iter"
        for root in (self.config.work_dir, self.config.save_dir):
            if root is None or not root.exists():
                continue
            for candidate in root.glob(f"{problem_id}-iter*"):
                suffix = candidate.name.removeprefix(prefix)
                if suffix.isdigit():
                    max_existing_iter = max(max_existing_iter, int(suffix))
        return max_existing_iter

    def _reserve_run_id(self, problem: ProblemInstance) -> str:
        problem_count = self.problem_counter[problem.id]
        max_existing_iter = self._max_existing_problem_iter(problem.id)
        if max_existing_iter >= problem_count:
            logger.warning(
                "Found existing workspace/save data for problem %s up to iter%d; bumping counter from %d to %d",
                problem.id,
                max_existing_iter,
                problem_count,
                max_existing_iter + 1,
            )
            problem_count = max_existing_iter + 1

        while True:
            run_id = f"{problem.id}-iter{problem_count}"
            problem_dir = Path(self.config.work_dir) / run_id if self.config.work_dir else None
            save_dir = Path(self.config.save_dir) / run_id if self.config.save_dir else None
            if (problem_dir is None or not problem_dir.exists()) and (save_dir is None or not save_dir.exists()):
                self.problem_counter[problem.id] = problem_count + 1
                return run_id

            problem_count += 1
            logger.warning("run_id collision for problem %s; retrying with iter%d", problem.id, problem_count)

    def _pull_capsule_from_s3(self, problem: ProblemInstance, dest: Path) -> None:
        """Lazily download this problem's capsule from s3 directly into ``dest``."""
        assert self._capsule_s3 is not None
        bucket, base = self._capsule_s3
        base = base.rstrip("/")
        candidates = [c for c in (problem.input_data_path, f"CapsuleData-{problem.id}", f"capsule_{problem.id}") if c]
        for cand in candidates:
            prefix = f"{base}/{cand}" if base else cand
            count = s3_sync.download_prefix(self._client(), bucket, prefix, dest)
            if count > 0:
                logger.warning(
                    "Lazily pulled %d objects for problem %s from s3://%s/%s", count, problem.id, bucket, prefix
                )
                return
        raise FileNotFoundError(
            f"No capsule found for problem {problem.id} under s3://{bucket}/{base} (tried {candidates})"
        )

    def get_new_env_by_idx(self, idx: int) -> InterpreterEnv:
        problem = self.problems[idx]
        run_id = self._reserve_run_id(problem)

        if self.config.work_dir:
            problem_dir = Path(self.config.work_dir) / run_id
            problem_dir.mkdir(parents=True, exist_ok=False)
        else:
            problem_dir = Path(mkdtemp())

        if self.capsule_dir is None:  # lazy s3: pull this task's capsule straight into the work dir
            self._pull_capsule_from_s3(problem, problem_dir)
        else:
            capsule_path = self.capsule_dir / problem.input_data_path
            if not capsule_path.exists():
                capsule_path = self.capsule_dir / f"CapsuleData-{problem.id}"
                if not capsule_path.exists():
                    capsule_path = self.capsule_dir / f"capsule_{problem.id}"
            shutil.copytree(capsule_path, problem_dir, dirs_exist_ok=True)

        save_dir = Path(self.config.save_dir) / run_id if self.config.save_dir else None

        language = (
            NBLanguage.PYTHON if self.config.force_python else NBLanguage.from_string(problem.nb_primary_language)
        )
        language = language if language is not None else NBLanguage.PYTHON  # default auto language to python

        return InterpreterEnv(
            problem=problem,
            rubric_model=self.rubric_model,
            work_dir=problem_dir,
            save_dir=save_dir,
            config=InterpreterEnvConfig(language=language, **self.config.model_dump()),
        )

    def __len__(self) -> int:
        return len(self.problems)


HypotestDataset = Dataset
HypotestDatasetConfig = DatasetConfig


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
    parser.add_argument("config", type=FilePath, nargs="?")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--api-key", type=str, default=os.getenv("HYPOTEST_API_KEY"))
    parser.add_argument("--problem-jsonl", type=str)
    parser.add_argument("--hf-dataset", type=str)
    parser.add_argument("--capsule-dir", type=str)
    parser.add_argument("--rubric-model", type=str)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--rubric-model-api-base", type=str, default=os.getenv("HYPOTEST_RUBRIC_MODEL_API_BASE"))
    parser.add_argument("--rubric-model-api-key", type=str, default=os.getenv("HYPOTEST_RUBRIC_MODEL_API_KEY"))
    parser.add_argument("--use-docker", action="store_true")

    args = parser.parse_args()

    if args.config and args.config.exists():
        config = ServerConfig.model_validate(yaml.safe_load(args.config.read_text()))
    else:
        config = ServerConfig(
            dataset=DatasetConfig(
                problem_jsonl=args.problem_jsonl,
                hf_dataset=args.hf_dataset,
                capsule_dir=args.capsule_dir,
                rubric_model=args.rubric_model,
                rubric_model_config={
                    "model_list": [
                        {
                            "model_name": args.rubric_model,
                            "litellm_params": {
                                "model": args.rubric_model,
                                "api_base": args.rubric_model_api_base,
                                "api_key": args.rubric_model_api_key,
                                "reasoning_effort": args.reasoning_effort,
                                "drop_params": True,
                            },
                        },
                    ],
                },
                use_docker=args.use_docker,
                execution_config={"cell_execution_timeout": 600},
            ),
            port=args.port,
            api_key=args.api_key,
        )

    dataset = Dataset(config.dataset)
    server = TaskDatasetServer(dataset, port=config.port, api_key=config.api_key)

    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"Starting dataset server: IPAddress={ip_address} Port={config.port}")

    await server.astart()


if __name__ == "__main__":
    asyncio.run(launch_server())
