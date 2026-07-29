"""Tests for selecting one capsule from a large-bundle collection image."""

from pathlib import Path

import pytest

from hypotest import kernel_capsule_server as capsule_server
from hypotest.kernel_capsule_server import (
    copy_mounted_capsule,
    prepare_initial_workspace,
    project_collection_capsule,
    resolve_collection_capsule,
    resolve_mounted_capsule,
)


def _collection(tmp_path: Path) -> Path:
    root = tmp_path / "capsules"
    first = root / "cap-1"
    second = root / "CapsuleData-legacy-id"
    (first / "nested").mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "matrix.tsv").write_text("gene\tvalue\nA\t1\n", encoding="utf-8")
    (first / "nested" / "notes.txt").write_text("selected", encoding="utf-8")
    (second / "secret.txt").write_text("other capsule", encoding="utf-8")
    return root


def test_resolve_collection_capsule_supports_exact_nested_and_legacy_names(tmp_path):
    root = _collection(tmp_path)
    nested = root / "group" / "nested-cap"
    nested.mkdir(parents=True)

    assert resolve_collection_capsule(root, "cap-1") == (root / "cap-1").resolve()
    assert resolve_collection_capsule(root, "group/nested-cap") == nested.resolve()
    assert resolve_collection_capsule(root, "legacy-id") == (root / "CapsuleData-legacy-id").resolve()


@pytest.mark.parametrize("capsule_id", ["", "..", "../outside", "/absolute/path"])
def test_resolve_collection_capsule_rejects_unsafe_ids(tmp_path, capsule_id):
    root = _collection(tmp_path)

    with pytest.raises(ValueError, match="unsafe bundle capsule id"):
        resolve_collection_capsule(root, capsule_id)


def test_resolve_collection_capsule_rejects_symlink_escape(tmp_path):
    root = _collection(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="no capsule"):
        resolve_collection_capsule(root, "escape")


def test_project_collection_capsule_creates_zero_copy_workspace_view(tmp_path):
    root = _collection(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stale.txt").write_text("remove me", encoding="utf-8")

    selected = project_collection_capsule(root, "cap-1", workspace)

    assert selected == (root / "cap-1").resolve()
    assert not (workspace / "stale.txt").exists()
    assert (workspace / "matrix.tsv").is_symlink()
    assert (workspace / "matrix.tsv").read_text(encoding="utf-8").endswith("A\t1\n")
    assert (workspace / "nested").is_symlink()
    assert (workspace / "nested" / "notes.txt").read_text(encoding="utf-8") == "selected"
    assert not (workspace / "secret.txt").exists()


def test_project_collection_capsule_rejects_workspace_control_collisions(tmp_path):
    root = _collection(tmp_path)
    (root / "cap-1" / "pip.conf").write_text("capsule-owned", encoding="utf-8")

    with pytest.raises(ValueError, match=r"reserved workspace names: pip\.conf"):
        project_collection_capsule(root, "cap-1", tmp_path / "workspace")


def test_project_collection_capsule_rejects_overlapping_workspace(tmp_path):
    root = _collection(tmp_path)

    with pytest.raises(ValueError, match="must not overlap"):
        project_collection_capsule(root, "cap-1", root / "workspace")


def test_resolve_mounted_capsule_supports_cluster_directory_prefix(tmp_path):
    root = tmp_path / "mounted"
    capsule = root / "capsule_abc-123"
    capsule.mkdir(parents=True)

    assert resolve_mounted_capsule(root, "abc-123") == capsule.resolve()


def test_copy_mounted_capsule_creates_independent_writable_workspace(tmp_path):
    root = tmp_path / "mounted"
    capsule = root / "capsule_abc-123"
    nested = capsule / "nested"
    nested.mkdir(parents=True)
    source_file = capsule / "matrix.tsv"
    source_file.write_text("gene\tvalue\nA\t1\n", encoding="utf-8")
    source_file.chmod(0o444)
    (nested / "notes.txt").write_text("source", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stale.txt").write_text("remove me", encoding="utf-8")

    selected, count = copy_mounted_capsule(root, "abc-123", workspace)

    copied_file = workspace / "matrix.tsv"
    assert selected == capsule.resolve()
    assert count == 2
    assert not (workspace / "stale.txt").exists()
    assert not copied_file.is_symlink()
    assert copied_file.stat().st_mode & 0o200
    copied_file.write_text("model changed this", encoding="utf-8")
    (workspace / "model-output.txt").write_text("new output", encoding="utf-8")
    assert source_file.read_text(encoding="utf-8") == "gene\tvalue\nA\t1\n"
    assert not (capsule / "model-output.txt").exists()


def test_copy_mounted_capsule_rejects_symlinks_before_clearing_workspace(tmp_path):
    root = tmp_path / "mounted"
    capsule = root / "capsule_abc-123"
    capsule.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("not capsule data", encoding="utf-8")
    (capsule / "escape.txt").symlink_to(outside)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stale = workspace / "stale.txt"
    stale.write_text("keep on validation failure", encoding="utf-8")

    with pytest.raises(ValueError, match="contains a symlink"):
        copy_mounted_capsule(root, "abc-123", workspace)

    assert stale.exists()


@pytest.mark.asyncio
async def test_prepare_initial_workspace_pulls_before_kernel_start(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stale.txt").write_text("stale", encoding="utf-8")
    calls = []

    def fake_pull(source, key, dest):
        calls.append((source, key, dest))
        (dest / "matrix.tsv").write_text("ready", encoding="utf-8")
        return 1

    monkeypatch.setattr(capsule_server.s3_sync, "pull_capsule", fake_pull)

    selected, count = await prepare_initial_workspace(
        workspace,
        capsule_source="s3://bucket/capsules",
        capsule_key="cap-1",
        bundle_layout="none",
        bundle_root=None,
        bundle_capsule_id=None,
    )

    assert calls == [("s3://bucket/capsules", "cap-1", workspace)]
    assert selected == "s3://bucket/capsules/cap-1"
    assert count == 1
    assert not (workspace / "stale.txt").exists()
    assert (workspace / "matrix.tsv").read_text(encoding="utf-8") == "ready"


@pytest.mark.asyncio
async def test_prepare_initial_workspace_copies_mounted_capsule_before_kernel_start(tmp_path):
    root = tmp_path / "mounted"
    capsule = root / "capsule_cap-1"
    capsule.mkdir(parents=True)
    (capsule / "matrix.tsv").write_text("ready", encoding="utf-8")

    selected, count = await prepare_initial_workspace(
        tmp_path / "workspace",
        capsule_source=None,
        capsule_key=None,
        bundle_layout="none",
        bundle_root=None,
        bundle_capsule_id=None,
        mounted_capsule_root=root,
        mounted_capsule_id="cap-1",
    )

    assert selected == str(capsule.resolve())
    assert count == 1
    assert (tmp_path / "workspace" / "matrix.tsv").read_text(encoding="utf-8") == "ready"


@pytest.mark.asyncio
async def test_prepare_initial_workspace_requires_complete_mounted_volume_pair(tmp_path):
    with pytest.raises(ValueError, match="requires HYPOTEST_MOUNTED_CAPSULE_ROOT"):
        await prepare_initial_workspace(
            tmp_path / "workspace",
            capsule_source=None,
            capsule_key=None,
            bundle_layout="none",
            bundle_root=None,
            bundle_capsule_id=None,
            mounted_capsule_root=tmp_path / "mounted",
            mounted_capsule_id=None,
        )


@pytest.mark.asyncio
async def test_prepare_initial_workspace_requires_source_for_runtime_key(tmp_path):
    with pytest.raises(ValueError, match="CAPSULE_KEY requires CAPSULE_SOURCE"):
        await prepare_initial_workspace(
            tmp_path / "workspace",
            capsule_source=None,
            capsule_key="cap-1",
            bundle_layout="none",
            bundle_root=None,
            bundle_capsule_id=None,
        )


@pytest.mark.asyncio
async def test_single_bundle_ignores_baked_init_pull_defaults(tmp_path, monkeypatch):
    def unexpected_pull(*_args):
        raise AssertionError("single-capsule bundle must not pull from object storage")

    monkeypatch.setattr(capsule_server.s3_sync, "pull_capsule", unexpected_pull)

    selected, count = await prepare_initial_workspace(
        tmp_path / "workspace",
        capsule_source="s3://bucket/capsules",
        capsule_key="cap-1",
        bundle_layout="single",
        bundle_root=None,
        bundle_capsule_id=None,
    )

    assert selected is None
    assert count == 0
