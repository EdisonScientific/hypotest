from tempfile import TemporaryDirectory

from hypotest.dataset_server import Dataset, DatasetConfig
from hypotest.env.interpreter_env import ProblemInstance


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
