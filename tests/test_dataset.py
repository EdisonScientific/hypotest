import json
from tempfile import TemporaryDirectory
from uuid import uuid4

import hypotest.dataset_server as datasetmod
from hypotest.dataset_server import Dataset, DatasetConfig
from hypotest.env.interpreter_env import ProblemInstance
from hypotest.env.sandbox import OpenSandboxSpec


def test_load_from_hf():
    with TemporaryDirectory() as tmpdir:
        config = DatasetConfig(
            hf_dataset="EdisonScientific/bixbench_hypothesis",
            capsule_dir=tmpdir,
        )
        dataset = Dataset(config)

    assert len(dataset) > 0
    for problem in dataset.problems:
        assert isinstance(problem, ProblemInstance)
        assert problem.hypothesis
        assert problem.rubric
        assert problem.max_score > 0


def test_remote_capsule_is_still_staged_for_local_fallback(tmp_path, monkeypatch):
    problem_id = uuid4()
    capsule_dir = tmp_path / "capsules"
    capsule = capsule_dir / "capsule-a"
    capsule.mkdir(parents=True)
    (capsule / "matrix.tsv").write_text("gene\tvalue\nA\t1\n", encoding="utf-8")
    problem_jsonl = tmp_path / "problems.jsonl"
    problem_jsonl.write_text(
        json.dumps({
            "id": str(problem_id),
            "hypothesis": "A is enriched",
            "protocol": "Test enrichment",
            "answer": True,
            "rubric": "Show the test",
            "max_points": 1,
            "input_data_path": "capsule-a",
        })
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasetmod, "LiteLLMModel", lambda **kwargs: object())
    dataset = Dataset(
        DatasetConfig(
            problem_jsonl=str(problem_jsonl),
            capsule_dir=str(capsule_dir),
            work_dir=tmp_path / "work",
            use_ray=False,
            use_enroot=False,
            opensandbox_spec=OpenSandboxSpec(
                image="registry/kernel:latest",
                capsule_source="s3://remote-capsules/root",
                create_attempts=1,
            ),
        )
    )

    env = dataset.get_new_env_by_idx(0)

    assert (env.work_dir / "matrix.tsv").read_text(encoding="utf-8") == "gene\tvalue\nA\t1\n"
