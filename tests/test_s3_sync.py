"""Tests for exact-first capsule selection from object storage."""

from pathlib import Path

import pytest

from hypotest import s3_sync


def test_pull_capsule_prefers_exact_local_key(tmp_path):
    source = tmp_path / "capsules"
    exact = source / "group" / "cap-1"
    exact.mkdir(parents=True)
    (exact / "matrix.tsv").write_text("gene\tvalue\n", encoding="utf-8")
    destination = tmp_path / "workspace"

    count = s3_sync.pull_capsule(str(source), "group/cap-1", destination)

    assert count == 1
    assert (destination / "matrix.tsv").read_text(encoding="utf-8") == "gene\tvalue\n"


@pytest.mark.parametrize("key", ["", ".", "..", "../cap", "/absolute"])
def test_pull_capsule_rejects_unsafe_key(tmp_path, key):
    with pytest.raises(ValueError, match="capsule key"):
        s3_sync.pull_capsule(str(tmp_path), key, tmp_path / "workspace")


def test_pull_capsule_joins_s3_source_and_exact_key(tmp_path, monkeypatch):
    calls: list[tuple[object, str, str, Path]] = []
    client = object()

    def fake_download(actual_client, bucket, prefix, destination, max_workers=16):
        calls.append((actual_client, bucket, prefix, destination))
        return 3

    monkeypatch.setattr(s3_sync, "download_prefix", fake_download)

    count = s3_sync.pull_capsule("s3://bucket/base", "nested/cap-1", tmp_path / "workspace", client=client)

    assert count == 3
    assert calls == [(client, "bucket", "base/nested/cap-1", tmp_path / "workspace")]
