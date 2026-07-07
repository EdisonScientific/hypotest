"""Tests for deterministic per-environment seed derivation and kernel setup."""

from __future__ import annotations

import json
import random

import pytest

from hypotest.dataset_server import Dataset, DatasetConfig
from hypotest.env.determinism import EnvSeeds, derive_seed
from hypotest.env.interpreter import Interpreter
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, InterpreterEnvState, ProblemInstance
from hypotest.env.kernel_server import NBLanguage, deterministic_kernel_env, rng_bootstrap_code
from hypotest.env.sandbox import K8sSandboxSpec


def test_seed_streams_are_stable_independent_and_index_specific() -> None:
    seeds = EnvSeeds.derive(1234, 7)
    # Pin the derivation contract so seeds remain stable across processes and releases.
    assert seeds == EnvSeeds(kernel=1_393_248_706, scheduler=1_015_309_298, rubric=126_252_534)
    assert seeds == EnvSeeds.derive(1234, 7)
    assert len({seeds.kernel, seeds.scheduler, seeds.rubric}) == 3
    assert seeds != EnvSeeds.derive(1234, 8)

    # Consuming the kernel stream cannot perturb the stateless rubric stream.
    kernel_rng = random.Random(seeds.kernel)
    for _ in range(10_000):
        kernel_rng.random()
    assert derive_seed(1234, 7, "rubric") == seeds.rubric


def test_deterministic_kernel_configuration() -> None:
    assert deterministic_kernel_env(91) == {
        "PYTHONHASHSEED": "91",
        "HYPOTEST_SEED": "91",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    python_bootstrap = rng_bootstrap_code(NBLanguage.PYTHON, 91)
    assert "random.seed(91)" in python_bootstrap
    assert "numpy.random.seed(91)" in python_bootstrap
    assert "import torch" not in python_bootstrap
    assert "set.seed(91)" in rng_bootstrap_code(NBLanguage.R, 91)


def test_dataset_assigns_per_index_rubric_seed(tmp_path, monkeypatch) -> None:
    captured_configs: list[dict] = []

    class FakeModel:
        def __init__(self, *, name, config):
            self.name = name
            self.config = config
            captured_configs.append(config)

    monkeypatch.setattr("hypotest.dataset_server.LiteLLMModel", FakeModel)

    problem_id = "00000000-0000-0000-0000-000000000001"
    problem_jsonl = tmp_path / "problems.jsonl"
    problem_jsonl.write_text(
        json.dumps(
            {
                "id": problem_id,
                "hypothesis": "h",
                "protocol": "p",
                "answer": True,
                "rubric": "r",
                "max_points": 1,
                "input_data_path": "capsule",
            }
        )
        + "\n"
    )
    capsule_dir = tmp_path / "capsules"
    (capsule_dir / "capsule").mkdir(parents=True)
    (capsule_dir / "capsule" / "data.txt").write_text("x")

    dataset = Dataset(
        DatasetConfig(
            problem_jsonl=str(problem_jsonl),
            capsule_dir=str(capsule_dir),
            work_dir=tmp_path / "work",
            deterministic=True,
            seed=314,
            use_enroot=False,
            use_ray=False,
        )
    )
    env = dataset.get_new_env_by_idx(0)
    expected = EnvSeeds.derive(314, 0)

    assert env.config.deterministic is True
    assert env.config.env_idx == 0
    assert env.config.rubric_seed == expected.rubric
    assert env.rubric_model is dataset.rubric_model
    assert captured_configs == [{"reasoning_effort": "medium"}]


@pytest.mark.asyncio
async def test_rubric_seed_is_forwarded_per_request(tmp_path, default_problem: ProblemInstance) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    class FakeModel:
        async def call_single(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return object()

    env = InterpreterEnv(
        problem=default_problem,
        work_dir=tmp_path,
        rubric_model=FakeModel(),
        config=InterpreterEnvConfig(rubric_seed=17),
    )

    await env._call_rubric_model("grade this", [], timeout=12.0)

    assert calls == [("grade this", {"timeout": 12.0, "seed": 17})]


@pytest.mark.asyncio
async def test_environment_derives_kernel_and_scheduler_seeds(
    tmp_path, monkeypatch, default_problem: ProblemInstance
) -> None:
    async def skip_sandbox_start(self):  # noqa: RUF029
        self._started = True

    monkeypatch.setattr(InterpreterEnvState, "start", skip_sandbox_start)
    env = InterpreterEnv(
        problem=default_problem,
        work_dir=tmp_path,
        config=InterpreterEnvConfig(
            deterministic=True,
            seed=2718,
            env_idx=4,
            use_enroot=False,
            use_ray=False,
            pull_capsule_in_pod=False,
            k8s_sandbox_specs=[K8sSandboxSpec(template="test")],
        ),
    )
    await env.reset()

    expected = EnvSeeds.derive(2718, 4)
    assert env.kernel_seed == expected.kernel
    assert env.scheduler_seed == expected.scheduler
    assert env.state._sandbox_config.seed == expected.kernel
    assert env.state._scheduler is not None
    assert env.state._scheduler._rng.random() == random.Random(expected.scheduler).random()


@pytest.mark.asyncio
async def test_seeded_interpreter_reseeds_on_reset(tmp_path) -> None:
    interpreter = Interpreter(tmp_path, NBLanguage.PYTHON, seed=4242)
    code = """
import json
import os
import random
import numpy
print(json.dumps({
    "execution_count": get_ipython().execution_count,
    "hash_seed": os.environ["PYTHONHASHSEED"],
    "hash": hash("hypotest"),
    "python": random.random(),
    "numpy": float(numpy.random.random()),
}, sort_keys=True))
"""
    try:
        await interpreter.start()
        first = (await interpreter.execute_code(code)).get_combined_text()
        await interpreter.reset()
        second = (await interpreter.execute_code(code)).get_combined_text()
    finally:
        await interpreter.close()

    assert first == second
    assert '"execution_count": 1' in first
    assert '"hash_seed": "4242"' in first
