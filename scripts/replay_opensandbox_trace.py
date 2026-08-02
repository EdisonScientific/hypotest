#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Replay a Tracer rollout batch against mounted-volume OpenSandbox kernels.

The input is Tracer's compact JSONL table artifact. The harness extracts the
actual ordered Hypotest tool calls, discovers the cluster-mounted capsule
collection, matches each trace task by its initial file listing, and replays
every sandbox-facing action through ``InterpreterEnv``. ``submit_answer`` is
counted but intentionally skipped because it exercises the rubric model rather
than the sandbox.

Reports contain aggregate operational data only. Cluster URLs, API keys,
registry credentials, image references, code, outputs, and capsule names are
never written to the summary, event log, or performance-metrics stream.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import json
import logging
import math
import os
import re
import resource
import shlex
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from uuid import NAMESPACE_URL, uuid5

import httpx
from opensandbox import Sandbox as RawOpenSandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.sandboxes import PlatformSpec, SandboxImageAuth, SandboxImageSpec

from hypotest.env.config import ExecutionConfig
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance
from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import OpenSandboxImageAuth, OpenSandboxSandbox, OpenSandboxSpec

logger = logging.getLogger("hypotest.opensandbox_replay")

_SUPPORTED_ACTIONS = frozenset({"list_dir", "run_cell", "reset_kernel"})
_CONTROL_FILES = frozenset({"Rprofile", "notebook.ipynb", "pip.conf", "score_info.json"})
_ERROR_OUTPUT = re.compile(r"^\[Cell #\d+\] Error:", re.MULTILINE)
_SAFE_REF = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_PROGRESS_LATENCIES = (
    "create_queue_seconds",
    "allocation_seconds",
    "kernel_connect_seconds",
    "startup_seconds",
    "action_seconds",
    "run_cell_seconds",
    "list_dir_seconds",
    "reset_kernel_seconds",
    "rollout_seconds",
)

ReplayActionName = Literal["list_dir", "run_cell", "reset_kernel"]


@dataclass(frozen=True)
class ReplayAction:
    """One sandbox-facing tool call from a recorded rollout."""

    name: ReplayActionName
    arguments: dict[str, Any]
    expected_error: bool | None = None


@dataclass(frozen=True)
class ReplayRollout:
    """Ordered actions and matching hints for one recorded trajectory."""

    ref: str
    task_idx: int
    actions: tuple[ReplayAction, ...]
    initial_files: frozenset[str]
    code_file_literals: frozenset[str]
    skipped_submit_calls: int
    skipped_non_sandbox_calls: int


@dataclass(frozen=True)
class TraceBatch:
    """A complete replay batch extracted from a Tracer table."""

    rollouts: tuple[ReplayRollout, ...]
    task_files: dict[int, frozenset[str]]
    task_code_file_literals: dict[int, frozenset[str]]
    action_counts: Counter[str]


@dataclass
class ReplayDiagnostic:
    """Sanitized failure context safe to persist in qualification artifacts."""

    completed_actions: int = 0
    error_type: str | None = None
    failure_phase: str | None = None
    failed_action_index: int | None = None
    failed_action_name: ReplayActionName | None = None
    http_status: int | None = None
    cause_type: str | None = None
    # Exception types and statuses distinguish transport failures without
    # persisting messages that may contain credentials or private URLs.
    exception_chain: list[dict[str, Any]] = field(default_factory=list)
    _current_phase: str = "startup"
    _current_action_index: int | None = None
    _current_action_name: ReplayActionName | None = None

    def begin_action(self, index: int, action: ReplayAction) -> None:
        self._current_phase = "action"
        self._current_action_index = index
        self._current_action_name = action.name

    def action_completed(self) -> None:
        self.completed_actions += 1

    def capture_cancelled(self) -> None:
        self._capture("CancelledError")

    def capture_exception(self, exc: Exception) -> None:
        exception_chain: list[dict[str, Any]] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            status = _exception_http_status(current)
            exception_chain.append({"type": type(current).__name__, "http_status": status})
            explicit_cause = current.__cause__
            sdk_cause = getattr(current, "cause", None)
            context = current.__context__ if not current.__suppress_context__ else None
            current = next(
                (
                    candidate
                    for candidate in (explicit_cause, sdk_cause, context)
                    if isinstance(candidate, BaseException) and id(candidate) not in seen
                ),
                None,
            )
        http_status = next(
            (entry["http_status"] for entry in exception_chain if entry["http_status"] is not None),
            None,
        )
        cause_type = exception_chain[1]["type"] if len(exception_chain) > 1 else None
        self._capture(
            type(exc).__name__,
            http_status=http_status,
            cause_type=cause_type,
            exception_chain=exception_chain,
        )

    def _capture(
        self,
        error_type: str,
        *,
        http_status: int | None = None,
        cause_type: str | None = None,
        exception_chain: list[dict[str, Any]] | None = None,
    ) -> None:
        self.error_type = error_type
        self.failure_phase = self._current_phase
        self.failed_action_index = self._current_action_index
        self.failed_action_name = self._current_action_name
        self.http_status = http_status
        self.cause_type = cause_type
        self.exception_chain = exception_chain or [{"type": error_type, "http_status": http_status}]


def _exception_http_status(exc: BaseException) -> int | None:
    """Extract an HTTP status from an exception without retaining its message."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


class ReplayError(RuntimeError):
    """The qualification workload is incomplete, ambiguous, or failed."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Tracer compact full_result.table.json.compact.jsonl")
    parser.add_argument("--trace-step", type=int, default=40)
    parser.add_argument(
        "--mounted-root",
        default="/mnt/s3-data/data/bbh/capsules/edison-20260725/",
    )
    parser.add_argument("--image", default=os.getenv("GITLAB_IMAGE") or os.getenv("OPEN_SANDBOX_IMAGE"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qualification/gbs1024-replay.json"))
    parser.add_argument("--capsule-index", type=Path, help="reuse a prior private capsule-index JSON artifact")
    parser.add_argument("--capsule-map", type=Path, help="reuse a prior verified task-to-capsule JSON artifact")
    parser.add_argument(
        "--prepare-only", action="store_true", help="parse, inventory, and match capsules without replay"
    )
    parser.add_argument("--concurrency", type=int, default=1024)
    parser.add_argument("--limit-rollouts", type=int, help="bounded smoke run; omit for the complete batch")
    parser.add_argument(
        "--rollout-ref",
        action="append",
        default=[],
        help="replay one exact trace ref; repeat to select multiple refs",
    )
    parser.add_argument("--preflight-actions", type=int, default=3)
    parser.add_argument("--cpu-request", type=float, default=0.25)
    parser.add_argument("--memory-request-mb", type=int, default=512)
    parser.add_argument("--cpu-limit", type=float, default=4)
    parser.add_argument("--memory-limit-mb", type=int, default=65536)
    parser.add_argument(
        "--kernel-memory-limit-mb",
        type=int,
        help="inner Jupyter RLIMIT_AS ceiling; must be below --memory-limit-mb",
    )
    parser.add_argument("--ephemeral-storage-gib", type=int, default=50)
    parser.add_argument("--cell-timeout-seconds", type=int, default=900)
    parser.add_argument("--job-timeout-seconds", type=int, default=10800)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--lifecycle-create-concurrency",
        type=int,
        default=64,
        help="maximum OpenSandbox.create calls admitted into the lifecycle HTTP pool (default: 64)",
    )
    parser.add_argument(
        "--kernel-request-concurrency",
        type=int,
        default=128,
        help="maximum kernel-proxy HTTP requests admitted at once (default: 128)",
    )
    parser.add_argument(
        "--create-attempts",
        type=int,
        default=3,
        help="maximum OpenSandbox allocation attempts per rollout (default: 3)",
    )
    parser.add_argument("--ttl-seconds", type=int, default=14400)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=2700,
        help="stop launching/executing replay work after this many seconds (default: 45 minutes)",
    )
    parser.add_argument(
        "--run-id",
        default=f"gbs1024-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        help="safe identifier attached to cluster sandboxes and telemetry",
    )
    return parser.parse_args()


def _normalize_user_file(path: str) -> str | None:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized.endswith(":"):
        return None
    try:
        pure = PurePosixPath(normalized)
    except ValueError:
        return None
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        return None
    if pure.parts[0].startswith(".") or pure.name in _CONTROL_FILES:
        return None
    return pure.as_posix()


def _listing_files(output: str | None) -> frozenset[str]:
    if not output:
        return frozenset()
    files: set[str] = set()
    for line in output.splitlines():
        if not line.startswith("  "):
            continue
        value = line.strip()
        if value.startswith(("[", "Files in directory:")):
            continue
        if normalized := _normalize_user_file(value):
            files.add(normalized)
    return frozenset(files)


def _looks_like_file_literal(value: str) -> bool:
    if not value or len(value) > 300 or "\n" in value or "://" in value:
        return False
    name = PurePosixPath(value).name
    return "." in name and not name.startswith(".") and not any(character.isspace() for character in name)


def _code_file_literals(code: str) -> set[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return set()
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _looks_like_file_literal(node.value):
            normalized = _normalize_user_file(node.value)
            if normalized:
                values.add(normalized)
                values.add(PurePosixPath(normalized).name)
    return values


def _paired_outputs(messages: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(message["call_id"]): str(message.get("output") or "")
        for message in messages
        if message.get("type") == "function_call_output" and message.get("call_id")
    }


def load_trace_batch(path: Path, step: int) -> TraceBatch:  # noqa: PLR0912, PLR0914, PLR0915
    """Stream a compact Tracer table into the minimal replay representation."""
    if not path.is_file():
        raise ReplayError(f"trace artifact does not exist: {path}")

    rollouts: list[ReplayRollout] = []
    task_ordinals: Counter[int] = Counter()
    task_listing_candidates: defaultdict[int, list[frozenset[str]]] = defaultdict(list)
    task_literals: defaultdict[int, set[str]] = defaultdict(set)
    action_counts: Counter[str] = Counter()

    with path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_idx = int(row["task_idx"])
                messages = list(row["response"]["output"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ReplayError(f"invalid Tracer row {row_number}: {exc}") from exc

            ordinal = task_ordinals[task_idx]
            task_ordinals[task_idx] += 1
            ref = f"s{step}/t{task_idx}/{ordinal}"
            outputs = _paired_outputs(messages)
            actions: list[ReplayAction] = []
            initial_files: frozenset[str] = frozenset()
            code_literals: set[str] = set()
            skipped_submit_calls = 0
            skipped_non_sandbox_calls = 0

            for message in messages:
                if message.get("type") != "function_call":
                    continue
                name = str(message.get("name") or "")
                action_counts[name] += 1
                if name == "submit_answer":
                    skipped_submit_calls += 1
                    continue
                if name not in _SUPPORTED_ACTIONS:
                    # Hallucinated/unregistered tool calls were rejected by the
                    # original environment and generated no sandbox traffic.
                    skipped_non_sandbox_calls += 1
                    continue
                action_name = cast(ReplayActionName, name)
                try:
                    arguments = json.loads(message.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    raise ReplayError(f"{ref} has invalid arguments for {name}: {exc}") from exc
                if not isinstance(arguments, dict):
                    raise ReplayError(f"{ref} arguments for {name} are not an object")

                paired_output = outputs.get(str(message.get("call_id") or ""))
                expected_error = None
                if name == "run_cell":
                    code = arguments.get("code")
                    if not isinstance(code, str):
                        skipped_non_sandbox_calls += 1
                        continue
                    code_literals.update(_code_file_literals(code))
                    expected_error = _ERROR_OUTPUT.search(paired_output or "") is not None
                elif name == "list_dir" and not initial_files:
                    initial_files = _listing_files(paired_output)
                actions.append(
                    ReplayAction(
                        name=action_name,
                        arguments=arguments,
                        expected_error=expected_error,
                    )
                )

            if initial_files:
                task_listing_candidates[task_idx].append(initial_files)
            task_literals[task_idx].update(code_literals)
            rollouts.append(
                ReplayRollout(
                    ref=ref,
                    task_idx=task_idx,
                    actions=tuple(actions),
                    initial_files=initial_files,
                    code_file_literals=frozenset(code_literals),
                    skipped_submit_calls=skipped_submit_calls,
                    skipped_non_sandbox_calls=skipped_non_sandbox_calls,
                )
            )

    if not rollouts:
        raise ReplayError("trace artifact contains no rollouts")

    task_files: dict[int, frozenset[str]] = {}
    for task_idx in task_ordinals:
        candidates = task_listing_candidates.get(task_idx, [])
        if not candidates:
            raise ReplayError(f"task {task_idx} has no initial list_dir file signature")
        counts = Counter(candidates)
        task_files[task_idx] = counts.most_common(1)[0][0]
        if not task_files[task_idx]:
            raise ReplayError(f"task {task_idx} has an empty initial file signature")

    return TraceBatch(
        rollouts=tuple(rollouts),
        task_files=task_files,
        task_code_file_literals={task: frozenset(values) for task, values in task_literals.items()},
        action_counts=action_counts,
    )


def _connection_config() -> ConnectionConfig:
    kwargs: dict[str, Any] = {"request_timeout": timedelta(seconds=300), "use_server_proxy": True}
    if domain := os.getenv("OPEN_SANDBOX_DOMAIN"):
        kwargs["domain"] = domain
    if api_key := os.getenv("OPEN_SANDBOX_API_KEY"):
        kwargs["api_key"] = api_key
    if protocol := os.getenv("OPEN_SANDBOX_PROTOCOL"):
        kwargs["protocol"] = protocol
    return ConnectionConfig(**kwargs)


def _raw_image_spec(image: str) -> str | SandboxImageSpec:
    username = os.getenv("REGISTRY_USERNAME")
    password = os.getenv("REGISTRY_PASSWORD")
    if bool(username) != bool(password):
        raise ReplayError("REGISTRY_USERNAME and REGISTRY_PASSWORD must be set together")
    if username and password:
        return SandboxImageSpec(image=image, auth=SandboxImageAuth(username=username, password=password))
    return image


def _hypotest_image_auth() -> OpenSandboxImageAuth | None:
    username = os.getenv("REGISTRY_USERNAME")
    password = os.getenv("REGISTRY_PASSWORD")
    if bool(username) != bool(password):
        raise ReplayError("REGISTRY_USERNAME and REGISTRY_PASSWORD must be set together")
    if username and password:
        return OpenSandboxImageAuth(username=username, password=password)
    return None


def _index_command(root: str) -> str:
    code = f"""
import json
import os
from pathlib import Path

root = Path({root!r}).resolve(strict=True)
for capsule in sorted(root.iterdir(), key=lambda value: value.name):
    if capsule.is_symlink() or not capsule.is_dir():
        continue
    files = []
    for directory, dirnames, filenames in os.walk(capsule, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not (directory_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = directory_path / filename
            if path.is_symlink() or not path.is_file():
                continue
            files.append(path.relative_to(capsule).as_posix())
            if len(files) >= 5000:
                break
        if len(files) >= 5000:
            break
    print(json.dumps({{"capsule": capsule.name, "files": files}}, separators=(",", ":")))
"""
    return f"/app/kernel_env/bin/python -c {shlex.quote(code)}"


async def discover_capsule_index(image: str, mounted_root: str) -> dict[str, frozenset[str]]:
    """Inventory only relative filenames from the mounted collection."""
    remote: RawOpenSandbox | None = None
    try:
        remote = await RawOpenSandbox.create(
            image=_raw_image_spec(image),
            timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=5),
            metadata={"hypotest-purpose": "trace-capsule-discovery"},
            resource={"cpu": "1", "memory": "2048Mi"},
            resource_requests={"cpu": "0.25", "memory": "512Mi"},
            platform=PlatformSpec(os="linux", arch="amd64"),
            extensions={
                "imagePullPolicy": "IfNotPresent",
                "opensandbox.extensions.image-pull-policy": "IfNotPresent",
            },
            entrypoint=["sh", "-lc", "sleep 1200"],
            connection_config=_connection_config(),
        )
        execution = await remote.commands.run(
            _index_command(mounted_root),
            opts=RunCommandOpts(timeout=timedelta(minutes=10)),
        )
        if execution.exit_code not in {None, 0}:
            raise ReplayError(f"mounted capsule discovery failed with exit code {execution.exit_code}")
        stdout = "".join(message.text for message in execution.logs.stdout)
        index: dict[str, frozenset[str]] = {}
        decoder = json.JSONDecoder()
        position = 0
        while position < len(stdout):
            while position < len(stdout) and stdout[position].isspace():
                position += 1
            if position >= len(stdout):
                break
            record, position = decoder.raw_decode(stdout, position)
            capsule = str(record["capsule"])
            index[capsule] = frozenset(
                normalized for value in record["files"] if (normalized := _normalize_user_file(str(value)))
            )
        if not index:
            raise ReplayError("mounted capsule discovery returned no directories")
        return index
    finally:
        if remote is not None:
            with contextlib.suppress(Exception):
                await remote.kill()
            with contextlib.suppress(Exception):
                await remote.close()


def _fingerprint_command(root: str, requests: dict[str, tuple[str, ...]]) -> str:
    payload = json.dumps(requests, separators=(",", ":"))
    code = f"""
import hashlib
import json
from pathlib import Path

root = Path({root!r}).resolve(strict=True)
requests = json.loads({payload!r})
chunk_size = 1024 * 1024
for capsule_name, relative_paths in sorted(requests.items()):
    capsule = (root / capsule_name).resolve(strict=True)
    if capsule == root or root not in capsule.parents or capsule.is_symlink():
        raise ValueError("unsafe capsule path")
    fingerprints = {{}}
    for relative_path in relative_paths:
        path = (capsule / relative_path).resolve(strict=True)
        if path == capsule or capsule not in path.parents or path.is_symlink() or not path.is_file():
            raise ValueError("unsafe fingerprint path")
        size = path.stat().st_size
        digest = hashlib.sha256()
        digest.update(str(size).encode())
        with path.open("rb") as stream:
            if size <= chunk_size * 3:
                digest.update(stream.read())
            else:
                for offset in (0, max(0, size // 2 - chunk_size // 2), size - chunk_size):
                    stream.seek(offset)
                    digest.update(stream.read(chunk_size))
        fingerprints[relative_path] = f"{{size}}:{{digest.hexdigest()}}"
    print(json.dumps({{"capsule": capsule_name, "fingerprints": fingerprints}}, separators=(",", ":")))
"""
    return f"/app/kernel_env/bin/python -c {shlex.quote(code)}"


async def fingerprint_mounted_files(
    image: str,
    mounted_root: str,
    requests: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, str]]:
    """Compare sizes plus first/middle/last content samples for ambiguous files."""
    remote: RawOpenSandbox | None = None
    try:
        remote = await RawOpenSandbox.create(
            image=_raw_image_spec(image),
            timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=5),
            metadata={"hypotest-purpose": "trace-capsule-fingerprint"},
            resource={"cpu": "1", "memory": "2048Mi"},
            resource_requests={"cpu": "0.25", "memory": "512Mi"},
            platform=PlatformSpec(os="linux", arch="amd64"),
            extensions={
                "imagePullPolicy": "IfNotPresent",
                "opensandbox.extensions.image-pull-policy": "IfNotPresent",
            },
            entrypoint=["sh", "-lc", "sleep 1200"],
            connection_config=_connection_config(),
        )
        execution = await remote.commands.run(
            _fingerprint_command(mounted_root, requests),
            opts=RunCommandOpts(timeout=timedelta(minutes=10)),
        )
        if execution.exit_code not in {None, 0}:
            raise ReplayError(f"mounted capsule fingerprinting failed with exit code {execution.exit_code}")
        stdout = "".join(message.text for message in execution.logs.stdout)
        decoder = json.JSONDecoder()
        position = 0
        fingerprints: dict[str, dict[str, str]] = {}
        while position < len(stdout):
            while position < len(stdout) and stdout[position].isspace():
                position += 1
            if position >= len(stdout):
                break
            record, position = decoder.raw_decode(stdout, position)
            fingerprints[str(record["capsule"])] = {
                str(path): str(value) for path, value in record["fingerprints"].items()
            }
        if fingerprints.keys() != requests.keys():
            raise ReplayError("mounted capsule fingerprint response was incomplete")
        return fingerprints
    finally:
        if remote is not None:
            with contextlib.suppress(Exception):
                await remote.kill()
            with contextlib.suppress(Exception):
                await remote.close()


def _file_basename_set(files: Iterable[str]) -> frozenset[str]:
    return frozenset(PurePosixPath(value).name for value in files)


def _ranked_capsule_candidates(
    signature: frozenset[str],
    literals: frozenset[str],
    capsule_index: dict[str, frozenset[str]],
) -> list[tuple[int, int, str]]:
    signature_basenames = _file_basename_set(signature)
    candidates: list[tuple[int, int, str]] = []
    for capsule, files in capsule_index.items():
        file_basenames = _file_basename_set(files)
        relative_match = signature.issubset(files)
        basename_match = signature_basenames.issubset(file_basenames)
        if not (relative_match or basename_match):
            continue
        literal_hits = len({
            value for value in literals if value in files or PurePosixPath(value).name in file_basenames
        })
        # Prefer more code-referenced file hits, then the smallest capsule
        # that fully contains the initial listing.
        candidates.append((-literal_hits, len(files), capsule))
    candidates.sort()
    return candidates


def ambiguous_task_candidates(
    task_files: dict[int, frozenset[str]],
    task_literals: dict[int, frozenset[str]],
    capsule_index: dict[str, frozenset[str]],
) -> dict[int, list[str]]:
    """Return only equally ranked filename matches requiring content checks."""
    ambiguous: dict[int, list[str]] = {}
    for task_idx, signature in task_files.items():
        candidates = _ranked_capsule_candidates(
            signature,
            task_literals.get(task_idx, frozenset()),
            capsule_index,
        )
        if not candidates:
            continue
        best_rank = candidates[0][:2]
        best = [candidate[2] for candidate in candidates if candidate[:2] == best_rank]
        if len(best) > 1:
            ambiguous[task_idx] = best
    return ambiguous


def _resolve_signature_paths(signature: frozenset[str], files: frozenset[str]) -> tuple[str, ...]:
    resolved: list[str] = []
    by_basename: defaultdict[str, list[str]] = defaultdict(list)
    for value in files:
        by_basename[PurePosixPath(value).name].append(value)
    for value in sorted(signature):
        if value in files:
            resolved.append(value)
            continue
        basename_matches = by_basename[PurePosixPath(value).name]
        if len(basename_matches) != 1:
            raise ReplayError(f"cannot uniquely resolve signature file {value!r} inside a candidate capsule")
        resolved.append(basename_matches[0])
    return tuple(resolved)


def fingerprint_requests(
    task_files: dict[int, frozenset[str]],
    ambiguous: dict[int, list[str]],
    capsule_index: dict[str, frozenset[str]],
) -> dict[str, tuple[str, ...]]:
    """Build the minimal mounted-file fingerprint request for ambiguities."""
    requested: defaultdict[str, set[str]] = defaultdict(set)
    for task_idx, capsules in ambiguous.items():
        for capsule in capsules:
            requested[capsule].update(_resolve_signature_paths(task_files[task_idx], capsule_index[capsule]))
    return {capsule: tuple(sorted(files)) for capsule, files in requested.items()}


def match_tasks_to_capsules(
    task_files: dict[int, frozenset[str]],
    task_literals: dict[int, frozenset[str]],
    capsule_index: dict[str, frozenset[str]],
    fingerprints: dict[str, dict[str, str]] | None = None,
) -> dict[int, str]:
    """Resolve each task to exactly one mounted directory without using IDs."""
    mapping: dict[int, str] = {}
    for task_idx, signature in task_files.items():
        candidates = _ranked_capsule_candidates(
            signature,
            task_literals.get(task_idx, frozenset()),
            capsule_index,
        )
        if not candidates:
            raise ReplayError(f"task {task_idx} did not match any mounted capsule ({len(signature)} signature files)")
        best_rank = candidates[0][:2]
        best = [candidate for candidate in candidates if candidate[:2] == best_rank]
        if len(best) == 1:
            mapping[task_idx] = best[0][2]
            continue
        if fingerprints is None:
            raise ReplayError(f"task {task_idx} matched {len(best)} equally ranked mounted capsules")
        fingerprint_sets: list[tuple[tuple[str, str], ...]] = []
        for _, _, capsule in best:
            paths = _resolve_signature_paths(signature, capsule_index[capsule])
            try:
                fingerprint_sets.append(
                    tuple(sorted((PurePosixPath(path).name, fingerprints[capsule][path]) for path in paths))
                )
            except KeyError as exc:
                raise ReplayError(f"missing fingerprint for an ambiguous task {task_idx}") from exc
        if len(set(fingerprint_sets)) != 1:
            raise ReplayError(f"task {task_idx} has equally ranked capsules with different file contents")
        # The mounted payloads are content-equivalent for every file observed
        # by this task, so stable lexical selection preserves replay semantics.
        mapping[task_idx] = min(candidate[2] for candidate in best)
    return mapping


def _save_private_capsule_artifacts(
    output: Path,
    capsule_index: dict[str, frozenset[str]],
    mapping: dict[int, str] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = output.with_name(f"{output.stem}.capsule-index.json")
    mapping_path = output.with_name(f"{output.stem}.task-capsule-map.json")
    index_path.write_text(
        json.dumps({name: sorted(files) for name, files in capsule_index.items()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mapping is not None:
        mapping_path.write_text(
            json.dumps({str(task): capsule for task, capsule in mapping.items()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _load_capsule_index(path: Path) -> dict[str, frozenset[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ReplayError("capsule index must be a JSON object")
    return {
        str(capsule): frozenset(normalized for value in files if (normalized := _normalize_user_file(str(value))))
        for capsule, files in raw.items()
    }


def _load_capsule_map(path: Path, tasks: Iterable[int]) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ReplayError("capsule map must be a JSON object")
    mapping = {int(task): str(capsule) for task, capsule in raw.items()}
    expected = set(tasks)
    if mapping.keys() != expected:
        missing = len(expected - mapping.keys())
        extra = len(mapping.keys() - expected)
        raise ReplayError(f"capsule map task set mismatch: {missing} missing, {extra} extra")
    return mapping


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _raise_fd_limit(required: int) -> tuple[int, int]:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(required, hard), hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def _driver_max_rss_mib() -> float:
    """Return the replay driver's process high-water RSS in MiB."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # getrusage reports bytes on macOS and KiB on Linux/BSD.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return rss / divisor


def _progress_latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


class ReplayMetrics:
    """Aggregates plus sanitized, append-only event and time-series logs."""

    def __init__(
        self,
        *,
        total: int,
        event_path: Path,
        metrics_path: Path,
        run_id: str,
    ) -> None:
        self.total = total
        self.run_id = run_id
        self.started_monotonic = time.perf_counter()
        self.started_at = datetime.now(UTC)
        self.started = 0
        self.ready = 0
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self.active = 0
        self.peak_active = 0
        self.remote_backends = 0
        self.fallback_backends = 0
        self.actions_replayed = 0
        self.run_cells_replayed = 0
        self.list_dirs_replayed = 0
        self.kernel_resets_replayed = 0
        self.code_errors = 0
        self.unexpected_code_errors = 0
        self.resolved_code_errors = 0
        self.create_retries = 0
        self.error_types: Counter[str] = Counter()
        self.root_cause_types: Counter[str] = Counter()
        self.latencies: defaultdict[str, list[float]] = defaultdict(list)
        self.event_loop_lags: list[float] = []
        self.max_driver_rss_mib = _driver_max_rss_mib()
        self.first_ready_monotonic: float | None = None
        self.last_ready_monotonic: float | None = None
        self._last_progress_monotonic = self.started_monotonic
        self._last_progress_counts = self._progress_counts()
        event_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._events = event_path.open("w", encoding="utf-8")
        self._metrics = metrics_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._events.close()
        self._metrics.close()

    def _write_metric(self, record: dict[str, Any]) -> None:
        self._metrics.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        self._metrics.flush()

    def record_run_started(self, config: dict[str, Any]) -> None:
        self._write_metric({
            "type": "run_started",
            "timestamp": self.started_at.isoformat(),
            "run_id": self.run_id,
            "total_rollouts": self.total,
            "configuration": config,
            "driver": {"max_rss_mib": self.max_driver_rss_mib},
        })

    def _progress_counts(self) -> dict[str, int]:
        return {
            "started": self.started,
            "ready": self.ready,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "actions_replayed": self.actions_replayed,
            "run_cells_replayed": self.run_cells_replayed,
            "list_dirs_replayed": self.list_dirs_replayed,
            "kernel_resets_replayed": self.kernel_resets_replayed,
            "create_retries": self.create_retries,
            "unexpected_code_errors": self.unexpected_code_errors,
        }

    def rollout_started(self) -> None:
        self.started += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)

    def rollout_ready(self, timings: dict[str, float]) -> None:
        now = time.perf_counter()
        self.ready += 1
        self.remote_backends += 1
        self.first_ready_monotonic = self.first_ready_monotonic or now
        self.last_ready_monotonic = now
        for name, value in timings.items():
            if name == "create_attempts":
                self.create_retries += max(0, int(value) - 1)
            else:
                self.latencies[name].append(float(value))

    def action_finished(
        self,
        action: ReplayAction,
        latency: float,
        observed_error: bool | None,
    ) -> None:
        self.actions_replayed += 1
        self.latencies["action_seconds"].append(latency)
        self.latencies[f"{action.name}_seconds"].append(latency)
        if action.name == "run_cell":
            self.run_cells_replayed += 1
            if observed_error:
                self.code_errors += 1
            if action.expected_error is False and observed_error:
                self.unexpected_code_errors += 1
            elif action.expected_error is True and observed_error is False:
                self.resolved_code_errors += 1
        elif action.name == "list_dir":
            self.list_dirs_replayed += 1
        elif action.name == "reset_kernel":
            self.kernel_resets_replayed += 1

    def rollout_finished(
        self,
        rollout: ReplayRollout,
        *,
        success: bool,
        rollout_seconds: float,
        cleanup_seconds: float,
        diagnostic: ReplayDiagnostic,
    ) -> None:
        self.completed += 1
        self.active -= 1
        self.latencies["rollout_seconds"].append(rollout_seconds)
        self.latencies["cleanup_seconds"].append(cleanup_seconds)
        if success:
            self.succeeded += 1
        else:
            self.failed += 1
            if diagnostic.error_type:
                self.error_types[diagnostic.error_type] += 1
            if diagnostic.exception_chain:
                self.root_cause_types[diagnostic.exception_chain[-1]["type"]] += 1
        event = {
            "ref": rollout.ref,
            "task_idx": rollout.task_idx,
            "success": success,
            "actions": len(rollout.actions),
            "completed_actions": diagnostic.completed_actions,
            "rollout_seconds": rollout_seconds,
            "cleanup_seconds": cleanup_seconds,
            "error_type": diagnostic.error_type,
            "failure_phase": diagnostic.failure_phase,
            "failed_action_index": diagnostic.failed_action_index,
            "failed_action_name": diagnostic.failed_action_name,
            "http_status": diagnostic.http_status,
            "cause_type": diagnostic.cause_type,
            "exception_chain": diagnostic.exception_chain,
        }
        self._events.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._events.flush()

    def progress(
        self,
        *,
        event_loop_lag_seconds: float | None = None,
        final: bool = False,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        """Persist and return one cumulative and interval performance snapshot."""
        now = time.perf_counter()
        elapsed = now - self.started_monotonic
        interval_seconds = now - self._last_progress_monotonic
        counts = self._progress_counts()
        interval_counts = {name: value - self._last_progress_counts[name] for name, value in counts.items()}
        if event_loop_lag_seconds is not None:
            event_loop_lag_seconds = max(0.0, event_loop_lag_seconds)
            self.event_loop_lags.append(event_loop_lag_seconds)
        self.max_driver_rss_mib = max(self.max_driver_rss_mib, _driver_max_rss_mib())
        latency = {
            name: _progress_latency_summary(self.latencies[name])
            for name in _PROGRESS_LATENCIES
            if self.latencies[name]
        }
        record = {
            "type": "progress",
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "final": final,
            "timed_out": timed_out,
            "elapsed_seconds": round(elapsed, 3),
            "total_rollouts": self.total,
            "started": self.started,
            "queued": self.total - self.started,
            "ready": self.ready,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "active": self.active,
            "peak_active": self.peak_active,
            "actions_replayed": self.actions_replayed,
            "run_cells_replayed": self.run_cells_replayed,
            "list_dirs_replayed": self.list_dirs_replayed,
            "kernel_resets_replayed": self.kernel_resets_replayed,
            "fallback_backends": self.fallback_backends,
            "create_retries": self.create_retries,
            "code_errors": self.code_errors,
            "unexpected_code_errors": self.unexpected_code_errors,
            "resolved_code_errors": self.resolved_code_errors,
            "rollouts_per_second": self.completed / elapsed if elapsed else 0.0,
            "actions_per_second": self.actions_replayed / elapsed if elapsed else 0.0,
            "run_cells_per_second": self.run_cells_replayed / elapsed if elapsed else 0.0,
            "interval": {
                "seconds": interval_seconds,
                "rollouts_started_per_second": (
                    interval_counts["started"] / interval_seconds if interval_seconds else 0.0
                ),
                "sandboxes_ready_per_second": (
                    interval_counts["ready"] / interval_seconds if interval_seconds else 0.0
                ),
                "rollouts_completed_per_second": (
                    interval_counts["completed"] / interval_seconds if interval_seconds else 0.0
                ),
                "sandbox_actions_per_second": (
                    interval_counts["actions_replayed"] / interval_seconds if interval_seconds else 0.0
                ),
                "run_cells_per_second": (
                    interval_counts["run_cells_replayed"] / interval_seconds if interval_seconds else 0.0
                ),
                "failures": interval_counts["failed"],
            },
            "latency_seconds": latency,
            "driver": {
                "max_rss_mib": self.max_driver_rss_mib,
                "event_loop_lag_seconds": event_loop_lag_seconds,
            },
        }
        self._last_progress_monotonic = now
        self._last_progress_counts = counts
        self._write_metric(record)
        return record

    def summary(
        self,
        *,
        expected_actions: int,
        trace_step: int,
        config: dict[str, Any],
        timed_out: bool = False,
        timeout_observed_at_seconds: float | None = None,
    ) -> dict[str, Any]:
        ended_at = datetime.now(UTC)
        wall_seconds = time.perf_counter() - self.started_monotonic
        allocation_window = (
            self.last_ready_monotonic - self.started_monotonic if self.last_ready_monotonic is not None else None
        )
        passed = (
            not timed_out
            and self.completed == self.total
            and self.succeeded == self.total
            and self.failed == 0
            and self.fallback_backends == 0
            and self.remote_backends == self.total
            and self.actions_replayed == expected_actions
            and self.unexpected_code_errors == 0
        )
        return {
            "passed": passed,
            "timed_out": timed_out,
            "trace": {
                "step": trace_step,
                "rollouts": self.total,
                "expected_sandbox_actions": expected_actions,
            },
            "configuration": config,
            "timing": {
                "started_at": self.started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "wall_seconds": wall_seconds,
                "allocation_window_seconds": allocation_window,
                "timeout_observed_at_seconds": timeout_observed_at_seconds,
            },
            "throughput": {
                "rollouts_per_second": self.completed / wall_seconds if wall_seconds else 0.0,
                "sandbox_actions_per_second": self.actions_replayed / wall_seconds if wall_seconds else 0.0,
                "run_cells_per_second": self.run_cells_replayed / wall_seconds if wall_seconds else 0.0,
                "ready_sandboxes_per_second": (
                    self.ready / allocation_window if allocation_window and allocation_window > 0 else None
                ),
            },
            "counts": {
                "started": self.started,
                "ready": self.ready,
                "completed": self.completed,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "peak_active": self.peak_active,
                "remote_backends": self.remote_backends,
                "fallback_backends": self.fallback_backends,
                "create_retries": self.create_retries,
                "sandbox_actions_replayed": self.actions_replayed,
                "run_cells_replayed": self.run_cells_replayed,
                "list_dirs_replayed": self.list_dirs_replayed,
                "kernel_resets_replayed": self.kernel_resets_replayed,
                "code_errors": self.code_errors,
                "unexpected_code_errors": self.unexpected_code_errors,
                "resolved_code_errors": self.resolved_code_errors,
            },
            "latency_seconds": {name: _latency_summary(values) for name, values in sorted(self.latencies.items())},
            "driver": {
                "max_rss_mib": self.max_driver_rss_mib,
                "event_loop_lag_seconds": _latency_summary(self.event_loop_lags),
            },
            "failure_types": dict(sorted(self.error_types.items())),
            "root_cause_types": dict(sorted(self.root_cause_types.items())),
        }


def _make_config(options: argparse.Namespace) -> InterpreterEnvConfig:
    spec = OpenSandboxSpec(
        image=options.image,
        image_auth=_hypotest_image_auth(),
        capsule_mode="mounted_volume",
        mounted_capsule_root=options.mounted_root,
        capsule_key="{capsule_uuid}",
        local_fallback_enabled=False,
        install_shim_enabled=False,
        kernel_memory_limit_mb=options.kernel_memory_limit_mb,
        request_timeout_seconds=300,
        create_timeout_seconds=900,
        ready_timeout_seconds=options.ready_timeout_seconds,
        lifecycle_create_concurrency=options.lifecycle_create_concurrency,
        kernel_request_concurrency=options.kernel_request_concurrency,
        create_attempts=options.create_attempts,
        create_retry_delay_seconds=2,
        ttl_seconds=options.ttl_seconds,
        image_pull_policy="IfNotPresent",
        platform_os="linux",
        platform_arch="amd64",
        metadata={
            "hypotest-purpose": "gbs1024-trace-replay",
            "hypotest-run": options.run_id,
        },
    )
    execution = ExecutionConfig(
        job_timeout=options.job_timeout_seconds,
        warn_submit_threshold=1200,
        force_submit_threshold=600,
        cell_execution_timeout=options.cell_timeout_seconds,
        # Replayed traffic has no live policy generation. Charge only the
        # kernel-reported cell duration; proxy/admission wait remains telemetry.
        time_accounting={"mode": "kernel_execution", "generation_latency": {"mode": "none"}},
        safe_execute=False,
        sandbox_cpu_request=options.cpu_request,
        sandbox_memory_request_mb=options.memory_request_mb,
        sandbox_cpu=options.cpu_limit,
        sandbox_memory_limit_mb=options.memory_limit_mb,
        sandbox_ephemeral_storage_gib=options.ephemeral_storage_gib,
    )
    return InterpreterEnvConfig(
        language=NBLanguage.PYTHON,
        execution_config=execution,
        max_steps=10_000,
        use_ray=False,
        use_docker=False,
        use_enroot=False,
        enable_recovery=False,
        opensandbox_spec=spec,
        pull_capsule_in_pod=True,
        replace_image_payloads_with_placeholders=True,
    )


async def _dispatch_action(env: InterpreterEnv, action: ReplayAction) -> tuple[Any, bool | None]:
    if action.name == "run_cell":
        arguments = dict(action.arguments)
        code = arguments.pop("code")
        idx = arguments.pop("idx", None)
        timeout_seconds = arguments.pop("timeout_seconds", None)
        if arguments:
            raise ReplayError(f"run_cell has unsupported arguments: {sorted(arguments)}")
        if timeout_seconds is None:
            result = await env.run_cell(code, idx=idx)
        else:
            result = await env._run_cell_with_cap(code, idx=idx, timeout_cap=float(timeout_seconds))
        return result, _ERROR_OUTPUT.search(str(result)) is not None
    if action.name == "list_dir":
        arguments = dict(action.arguments)
        # One recorded GBS1024 call contains a trailing colon in this key. The
        # original intent and paired tool output are unambiguous; normalize it
        # so a trace-format typo does not abort the infrastructure replay.
        if "directory:" in arguments and "directory" not in arguments:
            arguments["directory"] = arguments.pop("directory:")
        return await env.list_dir(**arguments), None
    if action.name == "reset_kernel":
        if action.arguments:
            raise ReplayError(f"reset_kernel has unexpected arguments: {sorted(action.arguments)}")
        return await env.reset_kernel(), None
    raise ReplayError(f"unsupported action {action.name!r}")


def _require_remote_sandbox(
    env: InterpreterEnv,
    metrics: ReplayMetrics | None,
    rollout_ref: str,
) -> OpenSandboxSandbox:
    sandbox = env.state.sandbox
    if isinstance(sandbox, OpenSandboxSandbox):
        return sandbox
    if metrics is not None:
        metrics.fallback_backends += 1
    raise ReplayError(f"{rollout_ref} used {type(sandbox).__name__} instead of OpenSandboxSandbox")


async def _run_one_rollout(
    rollout: ReplayRollout,
    *,
    capsule: str,
    config: InterpreterEnvConfig,
    work_root: Path,
    metrics: ReplayMetrics | None,
    action_limit: int | None = None,
) -> None:
    started = time.perf_counter()
    if metrics is not None:
        metrics.rollout_started()
    safe_ref = _SAFE_REF.sub("_", rollout.ref)
    work_dir = work_root / safe_ref
    work_dir.mkdir(parents=True, exist_ok=False)
    problem = ProblemInstance(
        id=uuid5(NAMESPACE_URL, rollout.ref),
        hypothesis="OpenSandbox trace replay",
        protocol="Replay recorded notebook tool traffic",
        answer=True,
        rubric="Infrastructure qualification only",
        max_points=1,
        input_data_path=capsule,
    )
    env = InterpreterEnv(problem=problem, work_dir=work_dir, config=config)
    success = False
    diagnostic = ReplayDiagnostic()
    cleanup_seconds = 0.0
    try:
        await env.reset()
        sandbox = _require_remote_sandbox(env, metrics, rollout.ref)
        if metrics is not None:
            metrics.rollout_ready(sandbox.startup_timings)

        actions = rollout.actions if action_limit is None else rollout.actions[:action_limit]
        for action_index, action in enumerate(actions):
            diagnostic.begin_action(action_index, action)
            action_started = time.perf_counter()
            _, observed_error = await _dispatch_action(env, action)
            diagnostic.action_completed()
            if metrics is not None:
                metrics.action_finished(
                    action,
                    time.perf_counter() - action_started,
                    observed_error,
                )
        success = True
    except asyncio.CancelledError:
        diagnostic.capture_cancelled()
        raise
    except Exception as exc:
        diagnostic.capture_exception(exc)
        logger.warning(
            "%s failed with %s (phase=%s action=%s index=%s status=%s chain=%s)",
            rollout.ref,
            diagnostic.error_type,
            diagnostic.failure_phase,
            diagnostic.failed_action_name,
            diagnostic.failed_action_index,
            diagnostic.http_status,
            " -> ".join(entry["type"] for entry in diagnostic.exception_chain),
        )
    finally:
        cleanup_started = time.perf_counter()
        if hasattr(env, "state"):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(env.close(), timeout=180)
        cleanup_seconds = time.perf_counter() - cleanup_started
        if metrics is not None:
            metrics.rollout_finished(
                rollout,
                success=success,
                rollout_seconds=time.perf_counter() - started,
                cleanup_seconds=cleanup_seconds,
                diagnostic=diagnostic,
            )
    if not success:
        raise ReplayError(f"{rollout.ref} failed with {diagnostic.error_type}")


async def _progress_reporter(metrics: ReplayMetrics, interval: float) -> None:
    loop = asyncio.get_running_loop()
    next_tick = loop.time() + interval
    while metrics.completed < metrics.total:
        await asyncio.sleep(max(0.0, next_tick - loop.time()))
        woke_at = loop.time()
        if metrics.completed >= metrics.total:
            break
        snapshot = metrics.progress(
            event_loop_lag_seconds=max(0.0, woke_at - next_tick),
        )
        print(json.dumps(snapshot, sort_keys=True), flush=True)
        next_tick += interval
        if next_tick <= woke_at:
            next_tick = woke_at + interval


async def run_replay(  # noqa: PLR0914, PLR0915
    batch: TraceBatch,
    mapping: dict[int, str],
    options: argparse.Namespace,
) -> dict[str, Any]:
    selected = list(batch.rollouts)
    rollout_refs = set(options.rollout_ref)
    if rollout_refs:
        available_refs = {rollout.ref for rollout in selected}
        missing_refs = sorted(rollout_refs - available_refs)
        if missing_refs:
            raise ReplayError(f"unknown --rollout-ref values: {missing_refs}")
        selected = [rollout for rollout in selected if rollout.ref in rollout_refs]
    if options.limit_rollouts is not None:
        selected = selected[: options.limit_rollouts]
    if not selected:
        raise ReplayError("no rollouts selected")
    concurrency = min(options.concurrency, len(selected))
    if concurrency < 1:
        raise ReplayError("--concurrency must be positive")

    required_fds = max(1024, concurrency * 4 + 512)
    fd_soft, fd_hard = _raise_fd_limit(required_fds)
    if fd_soft < required_fds:
        raise ReplayError(f"open-file limit {fd_soft} is below the required {required_fds}; hard limit is {fd_hard}")

    config = _make_config(options)
    event_path = options.output.with_name(f"{options.output.stem}.events.jsonl")
    metrics_path = options.output.with_name(f"{options.output.stem}.metrics.jsonl")
    summary_config = {
        "run_id": options.run_id,
        "concurrency": concurrency,
        "cpu_request": options.cpu_request,
        "memory_request_mib": options.memory_request_mb,
        "cpu_limit": options.cpu_limit,
        "memory_limit_mib": options.memory_limit_mb,
        "kernel_memory_limit_mib": options.kernel_memory_limit_mb,
        "ephemeral_storage_gib": options.ephemeral_storage_gib,
        "mounted_capsule_tasks": len(mapping),
        "local_fallback_enabled": False,
        "fd_soft_limit": fd_soft,
        "max_wall_seconds": options.max_wall_seconds,
        "progress_interval_seconds": options.progress_seconds,
        "lifecycle_create_concurrency": options.lifecycle_create_concurrency,
        "kernel_request_concurrency": options.kernel_request_concurrency,
        "episode_time_accounting": "kernel_execution",
        "telemetry_files": {
            "events": event_path.name,
            "metrics": metrics_path.name,
        },
    }
    metrics = ReplayMetrics(
        total=len(selected),
        event_path=event_path,
        metrics_path=metrics_path,
        run_id=options.run_id,
    )
    metrics.record_run_started(summary_config)
    work_root = Path(tempfile.mkdtemp(prefix="hypotest-gbs1024-"))
    reporter = asyncio.create_task(_progress_reporter(metrics, options.progress_seconds))
    queue: asyncio.Queue[ReplayRollout | None] = asyncio.Queue()
    for rollout in selected:
        queue.put_nowait(rollout)
    for _ in range(concurrency):
        queue.put_nowait(None)

    async def worker() -> None:
        while (rollout := await queue.get()) is not None:
            try:
                await _run_one_rollout(
                    rollout,
                    capsule=mapping[rollout.task_idx],
                    config=config,
                    work_root=work_root,
                    metrics=metrics,
                )
            except ReplayError:
                pass
            finally:
                queue.task_done()
        queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    timed_out = False
    timeout_observed_at_seconds: float | None = None
    try:
        try:
            async with asyncio.timeout(options.max_wall_seconds):
                await queue.join()
                await asyncio.gather(*workers)
        except TimeoutError:
            timed_out = True
            timeout_observed_at_seconds = time.perf_counter() - metrics.started_monotonic
            snapshot = metrics.progress(timed_out=True)
            print(json.dumps(snapshot, sort_keys=True), flush=True)
    finally:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        reporter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reporter
        final_snapshot = metrics.progress(final=True, timed_out=timed_out)
        print(json.dumps(final_snapshot, sort_keys=True), flush=True)
        metrics.close()
        shutil.rmtree(work_root, ignore_errors=True)

    expected_actions = sum(len(rollout.actions) for rollout in selected)
    return metrics.summary(
        expected_actions=expected_actions,
        trace_step=options.trace_step,
        config=summary_config,
        timed_out=timed_out,
        timeout_observed_at_seconds=timeout_observed_at_seconds,
    )


async def _main_async(options: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0912
    if not options.image:
        raise ReplayError("--image or GITLAB_IMAGE/OPEN_SANDBOX_IMAGE is required")
    if options.limit_rollouts is not None and options.limit_rollouts < 1:
        raise ReplayError("--limit-rollouts must be positive")
    if options.limit_rollouts is not None and options.rollout_ref:
        raise ReplayError("--limit-rollouts and --rollout-ref cannot be combined")
    if options.progress_seconds <= 0:
        raise ReplayError("--progress-seconds must be positive")
    if options.max_wall_seconds <= 0:
        raise ReplayError("--max-wall-seconds must be positive")
    if _SAFE_RUN_ID.fullmatch(options.run_id) is None:
        raise ReplayError("--run-id must be 1-63 characters using only letters, digits, dot, underscore, or hyphen")

    batch = load_trace_batch(options.trace, options.trace_step)
    if options.limit_rollouts is None and not options.rollout_ref and len(batch.rollouts) != 1024:
        raise ReplayError(f"full GBS1024 qualification requires exactly 1024 rollouts, found {len(batch.rollouts)}")
    print(
        json.dumps({
            "type": "trace_loaded",
            "rollouts": len(batch.rollouts),
            "tasks": len(batch.task_files),
            "sandbox_actions": sum(len(rollout.actions) for rollout in batch.rollouts),
            "tool_calls": dict(sorted(batch.action_counts.items())),
        }),
        flush=True,
    )

    if options.capsule_map is not None:
        mapping = _load_capsule_map(options.capsule_map, batch.task_files)
        capsule_index_count = None
        ambiguous_count = None
    else:
        if options.capsule_index is not None:
            capsule_index = _load_capsule_index(options.capsule_index)
        else:
            capsule_index = await discover_capsule_index(options.image, options.mounted_root)
        _save_private_capsule_artifacts(options.output, capsule_index)
        ambiguous = ambiguous_task_candidates(
            batch.task_files,
            batch.task_code_file_literals,
            capsule_index,
        )
        fingerprints = None
        if ambiguous:
            requests = fingerprint_requests(batch.task_files, ambiguous, capsule_index)
            fingerprints = await fingerprint_mounted_files(
                options.image,
                options.mounted_root,
                requests,
            )
        mapping = match_tasks_to_capsules(
            batch.task_files,
            batch.task_code_file_literals,
            capsule_index,
            fingerprints,
        )
        _save_private_capsule_artifacts(options.output, capsule_index, mapping)
        capsule_index_count = len(capsule_index)
        ambiguous_count = len(ambiguous)
    print(
        json.dumps({
            "type": "capsules_matched",
            "mounted_capsules": capsule_index_count,
            "trace_tasks": len(mapping),
            "content_equivalent_disambiguations": ambiguous_count,
        }),
        flush=True,
    )

    if options.prepare_only:
        return {
            "passed": True,
            "prepared_only": True,
            "trace": {
                "step": options.trace_step,
                "rollouts": len(batch.rollouts),
                "tasks": len(batch.task_files),
                "sandbox_actions": sum(len(rollout.actions) for rollout in batch.rollouts),
            },
            "mounted_capsules": capsule_index_count,
            "matched_tasks": len(mapping),
        }

    if options.preflight_actions > 0:
        preflight_root = Path(tempfile.mkdtemp(prefix="hypotest-preflight-"))
        try:
            await _run_one_rollout(
                batch.rollouts[0],
                capsule=mapping[batch.rollouts[0].task_idx],
                config=_make_config(options),
                work_root=preflight_root,
                metrics=None,
                action_limit=options.preflight_actions,
            )
        finally:
            shutil.rmtree(preflight_root, ignore_errors=True)
        print(
            json.dumps({
                "type": "preflight_passed",
                "actions": options.preflight_actions,
                "backend": "OpenSandboxSandbox",
            }),
            flush=True,
        )

    return await run_replay(batch, mapping, options)


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # InterpreterEnv intentionally logs every reset/cleanup at WARNING in
    # production. At GBS1024 those routine lines obscure the progress metrics;
    # retain errors here while OpenSandbox retry/failure warnings remain visible.
    logging.getLogger("hypotest.env.interpreter_env").setLevel(logging.ERROR)
    # The SDK's warning messages interpolate private image references. Our
    # sanitized rollout diagnostics retain the exception type chain instead.
    logging.getLogger("opensandbox").setLevel(logging.ERROR)
    options = _parse_args()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = asyncio.run(_main_async(options))
    except KeyboardInterrupt:
        print(json.dumps({"passed": False, "error": "interrupted"}), flush=True)
        return 130
    except Exception as exc:
        failure = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        options.output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 1

    options.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"type": "summary", **summary}, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
