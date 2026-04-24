# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the install-command shim scaffolding.

Exercise the _write_install_shims helper and verify each wrapper script's bash
syntax + end-to-end install/skip/passthrough decisions against a mocked real
installer. No kernel container required.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from hypotest.env.install_shim import (
    _APT_SHIM_BASH,
    _CONDA_SHIM_BASH,
    _PIP_SHIM_BASH,
    _R_SHIM_CODE,
    _write_install_shims,
)


# ---------- _write_install_shims scaffolding ----------

def test_write_install_shims_creates_hidden_dir_layout(tmp_path: Path) -> None:
    _write_install_shims(tmp_path)
    shim = tmp_path / ".install_shim"
    assert shim.is_dir()
    assert (shim / "bin").is_dir()
    for name in ("pip", "conda", "apt-get"):
        wrapper = shim / "bin" / name
        assert wrapper.is_file(), f"missing {name} wrapper"
        assert wrapper.stat().st_mode & stat.S_IXUSR, f"{name} wrapper not executable"
    assert (shim / "r_shim.R").is_file()


def test_write_install_shims_hidden_from_default_list(tmp_path: Path) -> None:
    """Dot-prefix so list_dir's show_hidden=False default hides the shim dir."""
    _write_install_shims(tmp_path)
    visible = [p.name for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert ".install_shim" not in visible
    hidden = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert ".install_shim" in hidden


# ---------- bash syntax ----------

@pytest.mark.parametrize("name,content", [
    ("pip", _PIP_SHIM_BASH),
    ("conda", _CONDA_SHIM_BASH),
    ("apt-get", _APT_SHIM_BASH),
])
def test_wrapper_bash_syntax_parses(name: str, content: str, tmp_path: Path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    wrapper = tmp_path / name
    wrapper.write_text(content)
    result = subprocess.run(["bash", "-n", str(wrapper)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n on {name}: {result.stderr}"


# ---------- end-to-end pip wrapper behavior ----------

def _setup_pip_harness(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the pip wrapper + a mock REAL_PIP; sed-substitute the hardcoded path."""
    _write_install_shims(tmp_path)
    wrapper = tmp_path / ".install_shim" / "bin" / "pip"
    mock_pip = tmp_path / "mock_real_pip"
    log_path = tmp_path / ".install_shim" / "log"

    mock_pip.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'INSTALLED="${PIP_INSTALLED_LIST:-}"\n'
        'case "$1" in\n'
        "    show)\n"
        '        pkg="$2"\n'
        '        if echo ",$INSTALLED," | grep -q ",$pkg,"; then\n'
        '            echo "Name: $pkg"\n'
        '            echo "Version: 1.2.3"\n'
        "            exit 0\n"
        "        fi\n"
        "        exit 1\n"
        "        ;;\n"
        "    install)\n"
        "        shift\n"
        '        echo "REAL_INSTALL_CALLED: $*"\n'
        "        exit 0\n"
        "        ;;\n"
        "    *)\n"
        '        echo "mock_real_pip: unhandled: $*" >&2\n'
        "        exit 2\n"
        "        ;;\n"
        "esac\n"
    )
    mock_pip.chmod(0o755)

    text = wrapper.read_text().replace(
        'REAL_PIP="/app/kernel_env/bin/pip"',
        f'REAL_PIP="{mock_pip}"',
    )
    wrapper.write_text(text)
    wrapper.chmod(0o755)
    return wrapper, mock_pip, log_path


def _run(cmd: list[str], *, installed: str, log_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PIP_INSTALLED_LIST"] = installed
    env["INSTALL_SHIM_LOG"] = str(log_path)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)


def _req_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


def test_pip_wrapper_skips_already_installed(tmp_path: Path) -> None:
    _req_bash()
    wrapper, _, log_path = _setup_pip_harness(tmp_path)

    res = _run([str(wrapper), "install", "numpy"], installed="numpy", log_path=log_path)
    assert res.returncode == 0, res.stderr
    assert "[pre-installed] numpy (1.2.3) already available, skipping" in res.stdout
    assert "REAL_INSTALL_CALLED" not in res.stdout
    entries = log_path.read_text().strip().splitlines()
    assert len(entries) == 1 and '"outcome":"skipped"' in entries[0]


def test_pip_wrapper_warns_on_version_mismatch_and_skips(tmp_path: Path) -> None:
    _req_bash()
    wrapper, _, log_path = _setup_pip_harness(tmp_path)

    res = _run(
        [str(wrapper), "install", "numpy==1.24.0"],
        installed="numpy",
        log_path=log_path,
    )
    assert res.returncode == 0, res.stderr
    assert "[pre-installed][version-mismatch]" in res.stdout
    assert "pinned to ==1.24.0" in res.stdout
    assert "but 1.2.3 is installed" in res.stdout
    assert "Pass --force-reinstall to override" in res.stdout
    assert "REAL_INSTALL_CALLED" not in res.stdout
    entries = log_path.read_text().strip().splitlines()
    assert '"outcome":"skipped_version_mismatch"' in entries[0]


def test_pip_wrapper_force_reinstall_passes_through(tmp_path: Path) -> None:
    _req_bash()
    wrapper, _, log_path = _setup_pip_harness(tmp_path)

    res = _run(
        [str(wrapper), "install", "--force-reinstall", "numpy==1.24.0"],
        installed="numpy",
        log_path=log_path,
    )
    assert res.returncode == 0, res.stderr
    assert "REAL_INSTALL_CALLED: --force-reinstall numpy==1.24.0" in res.stdout
    assert "[pre-installed]" not in res.stdout
    entries = log_path.read_text().strip().splitlines()
    assert '"outcome":"passthrough_force"' in entries[0]


def test_pip_wrapper_installs_when_package_missing(tmp_path: Path) -> None:
    _req_bash()
    wrapper, _, log_path = _setup_pip_harness(tmp_path)

    res = _run([str(wrapper), "install", "somelib"], installed="", log_path=log_path)
    assert res.returncode == 0, res.stderr
    assert "REAL_INSTALL_CALLED: somelib" in res.stdout
    entries = log_path.read_text().strip().splitlines()
    assert '"outcome":"installed"' in entries[0]


def test_pip_wrapper_git_url_passes_through(tmp_path: Path) -> None:
    _req_bash()
    wrapper, _, log_path = _setup_pip_harness(tmp_path)

    res = _run(
        [str(wrapper), "install", "git+https://github.com/pytest-dev/pytest"],
        installed="",
        log_path=log_path,
    )
    assert res.returncode == 0, res.stderr
    assert "REAL_INSTALL_CALLED:" in res.stdout
    assert "git+https://github.com/pytest-dev/pytest" in res.stdout
    entries = log_path.read_text().strip().splitlines()
    assert '"outcome":"passthrough"' in entries[0]


def test_pip_wrapper_requirements_file_passes_through(tmp_path: Path) -> None:
    _req_bash()
    wrapper, _, log_path = _setup_pip_harness(tmp_path)

    res = _run(
        [str(wrapper), "install", "-r", "reqs.txt"],
        installed="",
        log_path=log_path,
    )
    assert res.returncode == 0, res.stderr
    assert "REAL_INSTALL_CALLED: -r reqs.txt" in res.stdout
    entries = log_path.read_text().strip().splitlines()
    assert '"outcome":"passthrough"' in entries[0]


def test_pip_wrapper_non_install_subcommand_passes_through(tmp_path: Path) -> None:
    _req_bash()
    wrapper, _, log_path = _setup_pip_harness(tmp_path)

    res = _run([str(wrapper), "list"], installed="", log_path=log_path)
    assert res.returncode == 2  # mock's unhandled branch
    assert "mock_real_pip: unhandled: list" in res.stderr


# ---------- R shim smoke ----------

def test_r_shim_code_structure() -> None:
    assert "BiocManager" in _R_SHIM_CODE
    assert "install.packages" in _R_SHIM_CODE
    assert "assignInNamespace" in _R_SHIM_CODE
    assert "setTimeLimit(elapsed = 120" in _R_SHIM_CODE
    assert "shim_install_factory" in _R_SHIM_CODE
    assert "isTRUE(dots$force)" in _R_SHIM_CODE
    assert "[pre-installed]" in _R_SHIM_CODE
