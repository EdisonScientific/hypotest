"""Tests for the standalone large-bundle image builder."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_large_bundle.py"
SPEC = importlib.util.spec_from_file_location("hypotest_build_large_bundle", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def _capsule(tmp_path: Path) -> Path:
    capsule = tmp_path / "Capsule Data 123"
    (capsule / "nested").mkdir(parents=True)
    (capsule / "matrix.tsv").write_text("gene\tvalue\nA\t1\n", encoding="utf-8")
    (capsule / "nested" / "notes.txt").write_text("hypothesis", encoding="utf-8")
    return capsule


def _collection(tmp_path: Path) -> Path:
    collection = tmp_path / "all capsules"
    for capsule_id, value in (("CapsuleData-123", "A\t1\n"), ("capsule_456", "B\t2\n")):
        capsule = collection / capsule_id
        capsule.mkdir(parents=True)
        (capsule / "matrix.tsv").write_text(value, encoding="utf-8")
    return collection


def test_inspect_capsule_counts_files_and_allows_internal_symlink(tmp_path):
    capsule = _capsule(tmp_path)
    (capsule / "matrix-link.tsv").symlink_to(capsule / "matrix.tsv")

    root, stats = bundle.inspect_capsule(capsule)

    assert root == capsule.resolve()
    assert stats.files == 2
    assert stats.symlinks == 1
    assert stats.logical_bytes > 0


def test_inspect_capsule_rejects_context_filter_and_escaping_symlink(tmp_path):
    capsule = _capsule(tmp_path)
    (capsule / ".dockerignore").write_text("*.tsv\n", encoding="utf-8")
    with pytest.raises(bundle.BundleBuildError, match=r"silently omit|control which"):
        bundle.inspect_capsule(capsule)

    (capsule / ".dockerignore").unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-fixture-data", encoding="utf-8")
    (capsule / "outside-link").symlink_to(outside)
    with pytest.raises(bundle.BundleBuildError, match="escapes the build context"):
        bundle.inspect_capsule(capsule)


def test_default_image_is_deterministic_and_docker_tag_safe():
    capsule_id = "Capsule Data/\N{GREEK SMALL LETTER ALPHA}:123"
    image = bundle.default_image(capsule_id)

    assert image == bundle.default_image(capsule_id)
    assert re.fullmatch(r"hypotest-capsule:[a-z0-9_.-]+", image)
    assert len(image.rsplit(":", maxsplit=1)[1]) <= 128


def test_inspect_collection_requires_nonempty_immediate_capsule_directories(tmp_path):
    collection = _collection(tmp_path)

    root, stats = bundle.inspect_collection(collection)

    assert root == collection.resolve()
    assert stats.capsules == 2
    assert stats.files == 2

    (collection / "README.txt").write_text("not a capsule", encoding="utf-8")
    with pytest.raises(bundle.BundleBuildError, match="only capsule directories"):
        bundle.inspect_collection(collection)

    (collection / "README.txt").unlink()
    (collection / "empty").mkdir()
    with pytest.raises(bundle.BundleBuildError, match="empty capsule directories"):
        bundle.inspect_collection(collection)


def test_build_command_emits_single_platform_standard_image(tmp_path):
    options = bundle.BuildOptions(
        capsule_dir=tmp_path,
        capsule_id="capsule-1",
        layout="single",
        image="registry.example/hypotest:capsule-1",
        base_image="registry.example/hypotest-kernel@sha256:" + "a" * 64,
        platform="linux/amd64",
        push=True,
        builder="remote-builder",
        progress="plain",
        no_cache=True,
    )

    command = bundle.build_command(options, tmp_path / "metadata.json")

    assert command[:3] == ["docker", "buildx", "build"]
    assert command[-2:] == ["--push", str(tmp_path)]
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert "--provenance=false" in command
    assert "--no-cache" in command
    assert "BUNDLE_LAYOUT=single" in command
    assert "COPY --link --chown=0:0 . /workspace/" in bundle.SINGLE_BUNDLE_DOCKERFILE
    assert "ENTRYPOINT" not in bundle.SINGLE_BUNDLE_DOCKERFILE
    assert "CMD" not in bundle.SINGLE_BUNDLE_DOCKERFILE


def test_collection_build_uses_shared_capsule_root(tmp_path):
    options = bundle.BuildOptions(
        capsule_dir=tmp_path,
        capsule_id=None,
        layout="collection",
        image="registry.example/hypotest:all-capsules",
        base_image="hypotest-kernel:latest",
        platform="linux/arm64",
        push=False,
    )

    command = bundle.build_command(options, tmp_path / "metadata.json")

    assert command[-2:] == ["--load", str(tmp_path)]
    assert "BUNDLE_LAYOUT=collection" in command
    assert "BUNDLE_ROOT=/opt/hypotest/capsules" in command
    assert "COPY --link --chown=0:0 . /opt/hypotest/capsules/" in bundle.dockerfile_for(options)
    assert "COPY --link --chown=0:0 . /workspace/" not in bundle.dockerfile_for(options)


def test_build_metadata_accepts_oci_manifest_and_builds_digest_reference(tmp_path):
    digest = "sha256:" + "b" * 64
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(
        json.dumps({
            "containerimage.descriptor": {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": digest,
            },
            "containerimage.digest": digest,
        }),
        encoding="utf-8",
    )

    media_type, actual_digest = bundle.verify_build_metadata(metadata_file)

    assert media_type == "application/vnd.oci.image.manifest.v1+json"
    assert actual_digest == digest
    assert (
        bundle.immutable_image_reference("registry.example:5000/team/capsule:mutable", digest)
        == f"registry.example:5000/team/capsule@{digest}"
    )


def test_main_pushes_and_prints_digest_pinned_opensandbox_mapping(tmp_path, monkeypatch, capsys):
    capsule = _capsule(tmp_path)
    digest = "sha256:" + "c" * 64
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["buildx", "build"]:
            assert kwargs["input"] == bundle.SINGLE_BUNDLE_DOCKERFILE
            metadata_file = Path(command[command.index("--metadata-file") + 1])
            metadata_file.write_text(
                json.dumps({
                    "containerimage.descriptor": {
                        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                        "digest": digest,
                    },
                    "containerimage.digest": digest,
                }),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bundle.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(bundle.subprocess, "run", fake_run)

    result = bundle.main([
        str(capsule),
        "--capsule-id",
        "problem/capsule-123",
        "--image",
        "registry.example/team/capsule:v1",
        "--push",
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert len(calls) == 2
    assert calls[1][-2:] == ["--push", str(capsule.resolve())]
    assert f'"problem/capsule-123": "registry.example/team/capsule@{digest}"' in output
    assert "platform_os: linux" in output
    assert "platform_arch: amd64" in output


def test_dry_run_does_not_require_docker(tmp_path, monkeypatch, capsys):
    capsule = _capsule(tmp_path)
    monkeypatch.setattr(bundle.shutil, "which", lambda executable: None)

    assert bundle.main([str(capsule), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "docker buildx build" in output
    assert "--load" in output
    assert "capsule_mode: large_bundle" in output


def test_all_capsules_push_prints_one_digest_pinned_shared_image(tmp_path, monkeypatch, capsys):
    collection = _collection(tmp_path)
    digest = "sha256:" + "d" * 64

    def fake_run(command, **kwargs):
        if command[1:3] == ["buildx", "build"]:
            assert kwargs["input"] == bundle.COLLECTION_BUNDLE_DOCKERFILE
            metadata_file = Path(command[command.index("--metadata-file") + 1])
            metadata_file.write_text(
                json.dumps({
                    "containerimage.descriptor": {
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": digest,
                    },
                    "containerimage.digest": digest,
                }),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bundle.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(bundle.subprocess, "run", fake_run)

    result = bundle.main([
        str(collection),
        "--all-capsules",
        "--image",
        "registry.example/team/capsules:v1",
        "--push",
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "Collection:" in output
    assert "2 capsules" in output
    assert f'large_bundle_image: "registry.example/team/capsules@{digest}"' in output
    assert "large_bundle_images:" not in output


def test_all_capsules_rejects_single_capsule_id(tmp_path, capsys):
    collection = _collection(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        bundle.main([str(collection), "--all-capsules", "--capsule-id", "cap-1", "--dry-run"])

    assert exc_info.value.code == 2
    assert "--capsule-id cannot be combined with --all-capsules" in capsys.readouterr().err


def test_push_requires_explicit_registry_image(tmp_path, capsys):
    capsule = _capsule(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        bundle.main([str(capsule), "--push", "--dry-run"])

    assert exc_info.value.code == 2
    assert "--push requires an explicit --image" in capsys.readouterr().err
