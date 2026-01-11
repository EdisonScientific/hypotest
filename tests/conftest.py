"""Pytest configuration and shared fixtures for hypotest tests."""

import importlib.util
import logging
import os
import shutil
import subprocess
from uuid import UUID

import pytest
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

from hypotest.env import config as cfg
from hypotest.env.interpreter_env import ProblemInstance

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
requires_matplotlib = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")

logger = logging.getLogger(__name__)

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


def docker_image_exists() -> bool:
    """Check if the Docker image exists."""
    try:
        docker_path = shutil.which("docker")
        if not docker_path:
            logger.info("Docker is not installed on this system")
            return False
        result = subprocess.run(  # noqa: S603
            [docker_path, "images", cfg.NB_ENVIRONMENT_DOCKER_IMAGE, "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            logger.info(f"Docker image {cfg.NB_ENVIRONMENT_DOCKER_IMAGE} found")
            return True
        logger.info(f"Docker image {cfg.NB_ENVIRONMENT_DOCKER_IMAGE} not found")
        return False  # noqa: TRY300
    except subprocess.CalledProcessError:
        logger.info("Docker is not available on this system")
        return False


def should_skip_docker_test(use_docker: bool) -> bool:
    """Determine if Docker tests should be skipped."""
    if use_docker and (IN_GITHUB_ACTIONS or not docker_image_exists()):
        logger.info(f"Skipping docker test in CI environment: {IN_GITHUB_ACTIONS} and use_docker={use_docker}")
        return True
    return False


def should_skip_r_test(language_str: str) -> bool:
    """Determine if R tests should be skipped."""
    if language_str.upper() != "R":
        return False

    try:
        kernel_spec_manager = KernelSpecManager()
        kernel_spec_manager.get_kernel_spec("ir")
    except NoSuchKernel:
        return True
    return False


@pytest.fixture
def skip_if_docker_unavailable(request):
    """Fixture to skip Docker tests when Docker is not available."""
    if hasattr(request, "param") and request.param and should_skip_docker_test(request.param):
        pytest.skip("Docker not available or image not found")


@pytest.fixture
def skip_if_r_unavailable(request):
    """Fixture to skip R tests when R kernel is not available."""
    if hasattr(request, "param") and request.param and should_skip_r_test(request.param):
        pytest.skip("R kernel is not available")


@pytest.fixture
def default_problem() -> ProblemInstance:
    """Default ProblemInstance for tests."""
    return ProblemInstance(
        uuid=UUID("00000000-0000-0000-0000-000000000000"),
        hypothesis="Test hypothesis",
        objective="Test objective",
        answer=True,
        rubric="Test rubric",
        max_points=10,
    )
