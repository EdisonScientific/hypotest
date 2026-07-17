"""Pull dataset sources from an S3-compatible bucket on dataset-server start.

The capsule data and the tasks JSONL may be given in the dataset config as
``s3://bucket/prefix`` paths instead of local paths; on start they are
downloaded to a local staging dir. The endpoint and credentials come from the
standard boto3 environment variables — never the config:

    AWS_ENDPOINT_URL   (e.g. https://s3.example.com for the S3-compatible store)
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION

``boto3`` is imported lazily so importing this module (and the dataset server)
costs nothing unless an ``s3://`` source is actually used.
"""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

S3_SCHEME = "s3://"


def is_s3_uri(value: object) -> bool:
    return isinstance(value, str) and value.startswith(S3_SCHEME)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/or/prefix`` into ``(bucket, key)``."""
    if not is_s3_uri(uri):
        raise ValueError(f"not an s3 uri: {uri!r}")
    bucket, _, key = uri[len(S3_SCHEME) :].partition("/")
    if not bucket:
        raise ValueError(f"s3 uri missing bucket: {uri!r}")
    return bucket, key


def make_client(endpoint_url: str | None = None) -> Any:
    """Build an S3 client.

    The endpoint defaults to the standard env vars (AWS_ENDPOINT_URL_S3 /
    AWS_ENDPOINT_URL); region and credentials are read from the standard AWS_*
    env vars by boto3.
    """
    import boto3  # noqa: PLC0415 (lazy: avoid importing boto3 unless S3 is used)

    endpoint = endpoint_url or os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("AWS_ENDPOINT_URL")
    return boto3.client("s3", endpoint_url=endpoint)


def download_object(client: Any, bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))


def download_prefix(client: Any, bucket: str, prefix: str, dest: Path, max_workers: int = 16) -> int:
    """Download every object under ``prefix`` into ``dest``, preserving structure.

    Returns the number of objects downloaded. The prefix is scoped to a "folder"
    (a trailing slash is enforced for the listing) so ``capsules`` does not also
    match ``capsules2/...``.
    """
    list_prefix = prefix.rstrip("/") + "/" if prefix else ""
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):  # skip directory-marker keys
                keys.append(key)

    if not keys:
        logger.warning("No objects under s3://%s/%s", bucket, list_prefix)
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    def _one(key: str) -> None:
        target = dest / key[len(list_prefix) :]
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(target))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # list() forces evaluation so any per-object exception propagates.
        list(ex.map(_one, keys))
    return len(keys)


def normalize_capsule_key(key: str) -> str:
    """Return a safe relative S3/local prefix for one capsule."""
    path = PurePosixPath(key)
    if not key or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"capsule key must be a non-empty relative prefix: {key!r}")
    normalized = str(path)
    if normalized in {"", "."}:
        raise ValueError(f"capsule key must identify a capsule prefix: {key!r}")
    return normalized


def pull_capsule(source: str, key: str, dest: Path, client: Any = None) -> int:
    """Pull one capsule by exact key first, then use legacy UUID discovery.

    ``source`` is the collection root (for example ``s3://bucket/capsules``)
    and ``key`` is a relative capsule directory/prefix. Exact lookup avoids the
    collection-wide listing and recency scan for the normal ``input_data_path``
    case. The older substring/latest lookup remains as a compatibility fallback
    when a problem exposes only its UUID.
    """
    normalized_key = normalize_capsule_key(key)
    if is_s3_uri(source):
        s3_client = client or make_client()
        bucket, base_prefix = parse_s3_uri(source)
        exact_prefix = "/".join(part for part in (base_prefix.strip("/"), normalized_key) if part)
        count = download_prefix(s3_client, bucket, exact_prefix, dest)
        if count:
            logger.warning("Pulled exact capsule s3://%s/%s (%d objects)", bucket, exact_prefix, count)
            return count
        return _pull_latest_capsule_s3(source, normalized_key, dest, s3_client)

    local_base = Path(source).resolve(strict=True)
    exact = (local_base / Path(*PurePosixPath(normalized_key).parts)).resolve(strict=False)
    if exact != local_base and local_base in exact.parents and exact.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(exact, dest, dirs_exist_ok=True)
        count = sum(1 for path in exact.rglob("*") if path.is_file())
        logger.warning("Pulled exact capsule %s (%d files)", exact, count)
        return count
    return _pull_latest_capsule_local(local_base, normalized_key, dest)


def pull_latest_capsule(source: str, uuid: str, dest: Path, client: Any = None) -> int:
    """Place the most-recent capsule for ``uuid`` into ``dest``.

    ``source`` is a local folder or an ``s3://bucket/prefix`` base. Among the
    immediate subdirectories whose name contains ``uuid`` (e.g. ``capsule_<uuid>``,
    or versioned siblings like ``capsule_<uuid>_<ts>``), the freshest is chosen by
    newest S3 ``LastModified`` (or file mtime for a folder) and its contents are
    copied into ``dest``. Returns the number of files placed. Raises
    ``FileNotFoundError`` if no capsule for ``uuid`` is found.
    """
    if is_s3_uri(source):
        return _pull_latest_capsule_s3(source, uuid, dest, client or make_client())
    return _pull_latest_capsule_local(Path(source), uuid, dest)


def _pull_latest_capsule_s3(source: str, uuid: str, dest: Path, client: Any) -> int:
    bucket, base = parse_s3_uri(source)
    list_prefix = base.rstrip("/") + "/" if base.strip("/") else ""
    paginator = client.get_paginator("list_objects_v2")

    candidate_prefixes = [
        cp["Prefix"]
        for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix, Delimiter="/")
        for cp in page.get("CommonPrefixes", [])
        if uuid in cp["Prefix"][len(list_prefix) :]
    ]
    if not candidate_prefixes:
        raise FileNotFoundError(f"no capsule for {uuid} under s3://{bucket}/{list_prefix}")

    best_prefix, best_lm = None, None
    for prefix in candidate_prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("/") and (best_lm is None or obj["LastModified"] > best_lm):
                    best_lm, best_prefix = obj["LastModified"], prefix
    if best_prefix is None:
        raise FileNotFoundError(f"no capsule objects for {uuid} under s3://{bucket}/{list_prefix}")

    logger.warning("Most-recent capsule for %s: s3://%s/%s (LastModified=%s)", uuid, bucket, best_prefix, best_lm)
    return download_prefix(client, bucket, best_prefix, dest)


def _pull_latest_capsule_local(base: Path, uuid: str, dest: Path) -> int:
    if not base.is_dir():
        raise FileNotFoundError(f"capsule source folder not found: {base}")
    candidates = [d for d in base.iterdir() if d.is_dir() and uuid in d.name]
    if not candidates:
        raise FileNotFoundError(f"no capsule for {uuid} under {base}")

    def recency(d: Path) -> float:
        return max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()), default=d.stat().st_mtime)

    best = max(candidates, key=recency)
    logger.warning("Most-recent capsule for %s: %s", uuid, best)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best, dest, dirs_exist_ok=True)
    return sum(1 for p in best.rglob("*") if p.is_file())
