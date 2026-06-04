"""Unit tests for make_sandbox — the backend-selector mapping."""

from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import DockerSandbox, EnrootSandbox, LocalSandbox, SandboxConfig, make_sandbox


def _cfg(tmp_path, **kw):
    return SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON, **kw)


def test_make_sandbox_defaults_to_local(tmp_path):
    assert isinstance(make_sandbox(_cfg(tmp_path)), LocalSandbox)


def test_make_sandbox_docker(tmp_path):
    assert isinstance(make_sandbox(_cfg(tmp_path, use_docker=True)), DockerSandbox)


def test_make_sandbox_enroot(tmp_path):
    assert isinstance(make_sandbox(_cfg(tmp_path, use_enroot=True)), EnrootSandbox)


def test_make_sandbox_enroot_takes_precedence_over_docker(tmp_path):
    # Matches the dispatch order (enroot > docker > local).
    sandbox = make_sandbox(_cfg(tmp_path, use_docker=True, use_enroot=True))
    assert isinstance(sandbox, EnrootSandbox)
