#!/usr/bin/env python3
"""Fetch redacted native OpenSandbox diagnostics for one sandbox ID."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import timedelta
from typing import Any

from opensandbox.adapters.factory import AdapterFactory
from opensandbox.config import ConnectionConfig


def _connection_config() -> ConnectionConfig:
    kwargs: dict[str, Any] = {
        "request_timeout": timedelta(seconds=60),
        "use_server_proxy": True,
    }
    if domain := os.getenv("OPEN_SANDBOX_DOMAIN"):
        kwargs["domain"] = domain
    if api_key := os.getenv("OPEN_SANDBOX_API_KEY"):
        kwargs["api_key"] = api_key
    if protocol := os.getenv("OPEN_SANDBOX_PROTOCOL"):
        kwargs["protocol"] = protocol
    return ConnectionConfig(**kwargs).with_transport_if_missing()


def _redact(value: str) -> str:
    secrets = (
        "OPEN_SANDBOX_API_KEY",
        "REGISTRY_PASSWORD",
        "GITLAB_PUSH_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "GITLAB_IMAGE",
        "OPEN_SANDBOX_DOMAIN",
        "AWS_ENDPOINT_URL",
    )
    for name in secrets:
        secret = os.getenv(name)
        if secret:
            value = value.replace(secret, f"<{name.lower()}>")
    return re.sub(r"https?://[^\s\"']+", "<url>", value)


async def _run(sandbox_id: str) -> dict[str, Any]:
    config = _connection_config()
    factory = AdapterFactory(config)
    service = factory.create_sandbox_service()
    diagnostics = factory.create_diagnostics_service()
    result: dict[str, Any] = {"sandbox_id": sandbox_id}
    try:
        info = await service.get_sandbox_info(sandbox_id)
        result["status"] = {
            "state": info.status.state,
            "reason": info.status.reason,
            "message": _redact(info.status.message or ""),
        }
        for kind, scope, getter in (
            ("logs", "container", diagnostics.get_logs),
            ("events", "all", diagnostics.get_events),
        ):
            try:
                content = await getter(sandbox_id, scope)
            except Exception as exc:
                result[kind] = {"error_type": type(exc).__name__}
                continue
            result[kind] = {
                "delivery": content.delivery,
                "scope": content.scope,
                "truncated": content.truncated,
                "content": _redact(content.content or "")[-12000:],
                "url_available": content.content_url is not None,
            }
    finally:
        await config.close_transport_if_owned()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sandbox_id")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    print(json.dumps(asyncio.run(_run(args.sandbox_id)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
