#!/usr/bin/env python3
"""Audit or clean up run-tagged OpenSandbox allocations without printing IDs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from opensandbox.adapters.factory import AdapterFactory
from opensandbox.config import ConnectionConfig
from opensandbox.models.sandboxes import SandboxFilter

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")

_STATUS_MESSAGE_CATEGORIES = (
    ("registry_auth", ("unauthorized", "authentication required", "pull access denied")),
    ("image_pull", ("imagepull", "image pull", "pulling image", "failed to pull", "manifest unknown")),
    ("scheduling_capacity", ("failedscheduling", "unschedulable", "insufficient cpu", "insufficient memory", "quota")),
    ("volume_mount", ("failedmount", "failed attach", "volume", "mount")),
    ("container_startup", ("crashloop", "back-off restarting", "container", "entrypoint")),
    ("timeout", ("timed out", "timeout", "deadline exceeded")),
)


def _status_message_category(message: str | None) -> str:
    if not message:
        return "unspecified"
    normalized = message.lower()
    for category, markers in _STATUS_MESSAGE_CATEGORIES:
        if any(marker in normalized for marker in markers):
            return category
    return "other"


def _connection_config() -> ConnectionConfig:
    kwargs: dict[str, Any] = {"request_timeout": timedelta(seconds=300), "use_server_proxy": True}
    if domain := os.getenv("OPEN_SANDBOX_DOMAIN"):
        kwargs["domain"] = domain
    if api_key := os.getenv("OPEN_SANDBOX_API_KEY"):
        kwargs["api_key"] = api_key
    if protocol := os.getenv("OPEN_SANDBOX_PROTOCOL"):
        kwargs["protocol"] = protocol
    return ConnectionConfig(**kwargs).with_transport_if_missing()


async def _main(
    since_minutes: float,
    kill_run_ids: tuple[str, ...] = (),
    *,
    query_run_id: str | None = None,
    states: tuple[str, ...] = (),
    ids_output: Path | None = None,
) -> dict[str, Any]:
    service = AdapterFactory(_connection_config()).create_sandbox_service()
    cutoff = datetime.now(UTC) - timedelta(minutes=since_minutes)
    expected_image = os.getenv("GITLAB_IMAGE") or os.getenv("OPEN_SANDBOX_IMAGE")
    recent = []
    page = 1
    while True:
        response = await service.list_sandboxes(
            SandboxFilter(
                states=list(states) or None,
                metadata={"hypotest-run": query_run_id} if query_run_id is not None else None,
                page=page,
                page_size=200,
            )
        )
        for info in response.sandbox_infos:
            created_at = info.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            image = info.image.image if info.image is not None else None
            if created_at >= cutoff and (expected_image is None or image == expected_image):
                recent.append(info)
        if not response.pagination.has_next_page:
            break
        page += 1
    result: dict[str, Any] = {
        "since_minutes": since_minutes,
        "recent_matching_image": len(recent),
        "states": dict(sorted(Counter(info.status.state for info in recent).items())),
        "reasons": dict(
            sorted(Counter(info.status.reason or "unspecified" for info in recent).items())
        ),
        "message_categories": dict(
            sorted(Counter(_status_message_category(info.status.message) for info in recent).items())
        ),
        "purposes": dict(
            sorted(
                Counter((info.metadata or {}).get("hypotest-purpose", "unlabeled") for info in recent).items()
            )
        ),
        "runs": dict(
            sorted(
                Counter((info.metadata or {}).get("hypotest-run", "unlabeled") for info in recent).items()
            )
        ),
    }
    if kill_run_ids:
        requested = set(kill_run_ids)
        targets = [
            info
            for info in recent
            if (info.metadata or {}).get("hypotest-run") in requested
        ]

        async def kill(info: Any) -> str | None:
            try:
                await service.kill_sandbox(info.id)
            except Exception as exc:
                return type(exc).__name__
            return None

        errors = Counter(
            error
            for error in await asyncio.gather(*(kill(info) for info in targets))
            if error is not None
        )
        result["cleanup"] = {
            "requested_runs": sorted(requested),
            "matched": len(targets),
            "killed": len(targets) - sum(errors.values()),
            "failed": sum(errors.values()),
            "failure_types": dict(sorted(errors.items())),
        }
    if ids_output is not None:
        ids_output.parent.mkdir(parents=True, exist_ok=True)
        ids_output.write_text(
            json.dumps(
                {
                    "run_id": query_run_id,
                    "sandbox_ids": sorted(info.id for info in recent),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result["id_export"] = {"count": len(recent), "path": str(ids_output)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-minutes", type=float, default=15)
    parser.add_argument(
        "--kill-run-id",
        action="append",
        default=[],
        help="kill only sandboxes with this exact hypotest-run metadata value; repeatable",
    )
    parser.add_argument("--query-run-id", help="server-side metadata filter for one exact run ID")
    parser.add_argument("--state", action="append", default=[], help="server-side lifecycle-state filter")
    parser.add_argument("--ids-output", type=Path, help="explicitly export matching sandbox IDs to this JSON file")
    args = parser.parse_args()
    run_ids = [*args.kill_run_id, *([args.query_run_id] if args.query_run_id is not None else [])]
    invalid = [run_id for run_id in run_ids if _SAFE_RUN_ID.fullmatch(run_id) is None]
    if invalid:
        parser.error("--kill-run-id values must use 1-63 safe run-ID characters")
    print(
        json.dumps(
            asyncio.run(
                _main(
                    args.since_minutes,
                    tuple(args.kill_run_id),
                    query_run_id=args.query_run_id,
                    states=tuple(args.state),
                    ids_output=args.ids_output,
                )
            ),
            sort_keys=True,
        )
    )
