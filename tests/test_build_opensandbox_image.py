"""Tests for the all-in-one generic OpenSandbox image builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_opensandbox_image.py"
SPEC = importlib.util.spec_from_file_location("hypotest_build_opensandbox_image", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _options(tmp_path: Path, **overrides):
    values = {
        "root": tmp_path,
        "image": "registry.example/hypotest:latest",
        "platform": "linux/amd64",
        "base_target": "full",
        "base_image": "hypotest-base:full-amd64",
        "build_base": True,
        "build_cutoff_date": "2026-07-17",
        "capsule_source": "s3://bucket/capsules",
        "capsule_key": None,
        "s3_endpoint_url": None,
        "s3_region": None,
        "registry_auth": False,
        "registry_username_env": "REGISTRY_USERNAME",
        "registry_password_env": "REGISTRY_PASSWORD",
        "image_pull_policy": "IfNotPresent",
        "push": False,
        "pull": False,
        "no_cache": False,
    }
    values.update(overrides)
    return builder.BuildOptions(**values)


def test_commands_build_generic_base_and_kernel_without_capsule_context(tmp_path):
    options = _options(tmp_path)

    base_command = builder.base_build_command(options)
    kernel_command = builder.kernel_build_command(options)

    assert base_command is not None
    assert "--target" in base_command
    assert base_command[base_command.index("--target") + 1] == "full"
    assert kernel_command[-1] == str(tmp_path)
    assert "CAPSULE_SOURCE=s3://bucket/capsules" in kernel_command
    assert "CAPSULE_KEY=" in kernel_command
    assert "--push" not in kernel_command
    assert all("capsule_dir" not in argument for argument in kernel_command)


def test_skip_base_build_reuses_requested_image(tmp_path):
    options = _options(tmp_path, build_base=False, base_image="registry.example/interpreter@sha256:" + "a" * 64)

    assert builder.base_build_command(options) is None
    command = builder.kernel_build_command(options)
    assert f"BASE_IMAGE={options.base_image}" in command


def test_digest_selection_stays_on_requested_registry_repository():
    target = "registry.example:5000/team/hypotest:v1"
    wrong = "other.example/team/hypotest@sha256:" + "a" * 64
    expected = "registry.example:5000/team/hypotest@sha256:" + "b" * 64

    assert builder._select_digest_reference(target, [wrong, expected]) == expected
    assert builder._select_digest_reference(target, [wrong]) is None


def test_dry_run_prints_push_and_runtime_configuration(monkeypatch, capsys):
    monkeypatch.setattr(builder.shutil, "which", lambda _executable: None)

    result = builder.main([
        "--image",
        "registry.example/hypotest:v1",
        "--capsule-source",
        "s3://bucket/capsules",
        "--capsule-key",
        "fixed-cap",
        "--s3-endpoint-url",
        "https://s3.example",
        "--registry-auth",
        "--registry-username-env",
        "HYPOTEST_REGISTRY_USER",
        "--registry-password-env",
        "HYPOTEST_REGISTRY_TOKEN",
        "--push",
        "--dry-run",
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "no capsule data in image layers" in output
    assert "docker build" in output
    assert "docker push registry.example/hypotest:v1" in output
    assert "capsule_key: null" in output
    assert "AWS_ENDPOINT_URL" in output
    assert 'username: "${HYPOTEST_REGISTRY_USER}"' in output
    assert 'password: "${HYPOTEST_REGISTRY_TOKEN}"' in output
    assert "image_pull_policy: IfNotPresent" in output
    assert "fake-registry-password" not in output.lower()


def test_push_requires_explicit_registry_image(capsys):
    with pytest.raises(SystemExit) as exc_info:
        builder.main(["--push", "--dry-run"])

    assert exc_info.value.code == 2
    assert "--push requires an explicit registry --image" in capsys.readouterr().err


def test_registry_auth_can_be_omitted_for_public_image(monkeypatch, capsys):
    monkeypatch.setattr(builder.shutil, "which", lambda _executable: None)

    assert builder.main(["--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "image_auth:" not in output
    assert "image_pull_policy: IfNotPresent" in output


def test_registry_auth_rejects_invalid_environment_name(capsys):
    with pytest.raises(SystemExit) as exc_info:
        builder.main(["--registry-auth", "--registry-password-env", "NOT-A-NAME", "--dry-run"])

    assert exc_info.value.code == 2
    assert "must be an environment variable name" in capsys.readouterr().err
