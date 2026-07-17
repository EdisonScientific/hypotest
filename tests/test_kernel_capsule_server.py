"""Tests for selecting one capsule from a large-bundle collection image."""

from pathlib import Path

import pytest

from hypotest import kernel_capsule_server as capsule_server
from hypotest.kernel_capsule_server import (
    prepare_initial_workspace,
    project_collection_capsule,
    resolve_collection_capsule,
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
