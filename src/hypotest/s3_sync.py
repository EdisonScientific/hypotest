"""Pull dataset sources from an S3-compatible bucket on dataset-server start.

The capsule data and the tasks JSONL may be given in the dataset config as
``s3://bucket/prefix`` paths instead of local paths; on start they are
downloaded to a local staging dir. The endpoint and credentials come from the
standard boto3 environment variables — never the config:

    AWS_ENDPOINT_URL   (e.g. https://pdx.s8k.io for the S3-compatible store)
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION

``boto3`` is imported lazily so importing this module (and the dataset server)
costs nothing unless an ``s3://`` source is actually used.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
