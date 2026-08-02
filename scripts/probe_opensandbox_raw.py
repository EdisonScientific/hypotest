#!/usr/bin/env python3
"""Minimal OpenSandbox SDK control probe with no Hypotest integration."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.sandboxes import PlatformSpec, SandboxImageAuth, SandboxImageSpec

_KERNEL_PROBE_SOURCE = """\
import asyncio
from pathlib import Path

from hypotest.env.kernel_server import KernelServer, NBLanguage


async def main():
    server = KernelServer(
        Path("/workspace"),
        NBLanguage.PYTHON,
        safe_execute=False,
    )
    try:
        await server.start()
        print("raw-opensandbox-ok")
    finally:
        await server.close()


asyncio.run(main())
"""

_KERNEL_HEALTH_PROBE_SOURCE = """\
import json
import time
import urllib.error
import urllib.request

deadline = time.monotonic() + 45
delay = 0.25
last_error = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            payload = json.load(response)
        if payload.get("status") == "OK" and payload.get("kernel_ready") is True:
            print("raw-opensandbox-ok")
            break
        last_error = RuntimeError("health endpoint reported an unready kernel")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        last_error = exc
    time.sleep(delay)
    delay = min(delay * 1.7, 4.0)
else:
    raise RuntimeError(f"kernel health did not become ready: {type(last_error).__name__}")
"""


def _connection_config(request_timeout_seconds: float = 300) -> ConnectionConfig:
    kwargs: dict[str, Any] = {
        "request_timeout": timedelta(seconds=request_timeout_seconds),
        "use_server_proxy": True,
    }
    if domain := os.getenv("OPEN_SANDBOX_DOMAIN"):
        kwargs["domain"] = domain
    if api_key := os.getenv("OPEN_SANDBOX_API_KEY"):
        kwargs["api_key"] = api_key
    if protocol := os.getenv("OPEN_SANDBOX_PROTOCOL"):
        kwargs["protocol"] = protocol
    return ConnectionConfig(**kwargs).with_transport_if_missing()


def _exception_chain(exc: BaseException) -> list[dict[str, str | int | None]]:
    chain: list[dict[str, str | int | None]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = None
        if isinstance(current, httpx.HTTPStatusError):
            status_code = current.response.status_code
        chain.append({"type": type(current).__name__, "http_status": status_code})
        current = current.__cause__ or current.__context__
    return chain


def _stdout(execution: Any) -> str:
    return "".join(message.text for message in execution.logs.stdout)


def _redact_diagnostic(value: str) -> str:
    for name in (
        "OPEN_SANDBOX_API_KEY",
        "REGISTRY_PASSWORD",
        "GITLAB_PUSH_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "GITLAB_IMAGE",
        "OPEN_SANDBOX_DOMAIN",
        "AWS_ENDPOINT_URL",
    ):
        secret = os.getenv(name)
        if secret:
            value = value.replace(secret, f"<{name.lower()}>")
    return value[-6000:]


def _execution_diagnostic(execution: Any) -> str:
    parts = [message.text for message in execution.logs.stderr]
    if execution.error is not None:
        parts.extend((execution.error.value, *execution.error.traceback))
    return _redact_diagnostic("\n".join(parts))


async def _run_remote_script_probe(
    remote: Sandbox,
    *,
    source: str,
    path: str,
    timeout_seconds: float,
    result_prefix: str,
) -> tuple[bool, dict[str, Any]]:
    await remote.files.write_file(path, source)
    started = time.perf_counter()
    execution = await remote.commands.run(
        f"/app/kernel_env/bin/python {path}",
        opts=RunCommandOpts(timeout=timedelta(seconds=timeout_seconds)),
    )
    stdout_verified = _stdout(execution).strip() == "raw-opensandbox-ok"
    details: dict[str, Any] = {
        f"{result_prefix}_seconds": time.perf_counter() - started,
        f"{result_prefix}_exit_code": execution.exit_code,
        f"{result_prefix}_stdout_verified": stdout_verified,
    }
    if execution.error is not None:
        details[f"{result_prefix}_error_type"] = execution.error.name
    if diagnostic := _execution_diagnostic(execution):
        details[f"{result_prefix}_diagnostic"] = diagnostic
    return execution.exit_code in {None, 0} and stdout_verified, details


def _image_spec(args: argparse.Namespace) -> tuple[str | SandboxImageSpec, str]:
    if args.image_env is None:
        return args.image, args.image
    image = os.environ[args.image_env]
    if not args.registry_auth:
        return image, f"env:{args.image_env}"
    auth = SandboxImageAuth(
        username=os.environ[args.registry_username_env],
        password=os.environ[args.registry_password_env],
    )
    return SandboxImageSpec(image=image, auth=auth), f"env:{args.image_env}+auth"


def _entrypoint(mode: str | None) -> list[str] | None:
    if mode == "simple":
        return ["sh", "-lc", "sleep 600"]
    if mode == "wrapper-sleep":
        return ["sh", "/opt/entrypoint.sh", "sh", "-lc", "sleep 600"]
    if mode is not None:
        command = [
            "sh",
            "/opt/entrypoint.sh",
            "/app/kernel_env/bin/python",
            "-m",
            "hypotest.kernel_capsule_server",
            "--port",
            "8000",
            "--language",
            "python",
        ]
        if mode in {"kernel-memory", "kernel-full"}:
            command.extend(("--kernel-memory-limit-mb", "57344"))
        if mode in {"kernel-no-shim", "kernel-full"}:
            command.append("--no-install-shim")
        return command
    return None


def _create_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.hypotest_metadata:
        options["metadata"] = {
            "hypotest-purpose": "raw-additive-probe",
            "hypotest-run": args.case,
        }
    if args.platform_amd64:
        options["platform"] = PlatformSpec(os="linux", arch="amd64")
    extensions: dict[str, str] = {}
    if args.pull_policy_primary is not None:
        extensions["imagePullPolicy"] = args.pull_policy_primary
    if args.pull_policy_compat is not None:
        extensions["opensandbox.extensions.image-pull-policy"] = args.pull_policy_compat
    if extensions:
        options["extensions"] = extensions
    if args.resource_limits:
        options["resource"] = {"cpu": "4", "memory": "65536Mi", "ephemeral-storage": "50Gi"}
    if args.resource_requests:
        options["resource_requests"] = {"cpu": "0.25", "memory": "512Mi"}
    if entrypoint := _entrypoint(args.entrypoint_mode):
        options["entrypoint"] = entrypoint
    env: dict[str, str] = {}
    if args.clear_object_store:
        env.update({"CAPSULE_SOURCE": "", "CAPSULE_KEY": ""})
    if args.mounted_root is not None:
        env.update({
            "HYPOTEST_MOUNTED_CAPSULE_ROOT": args.mounted_root,
            "HYPOTEST_MOUNTED_CAPSULE_ID": args.mounted_capsule_id,
        })
    if env:
        options["env"] = env
    return options


async def _obtain_sandbox(
    args: argparse.Namespace,
    image: str | SandboxImageSpec,
) -> tuple[Sandbox, str]:
    if args.connect_ids_file is None:
        remote = await Sandbox.create(
            image,
            timeout=timedelta(seconds=args.ttl_seconds),
            ready_timeout=timedelta(seconds=args.ready_timeout_seconds),
            connection_config=_connection_config(args.request_timeout_seconds),
            skip_health_check=args.skip_health_check,
            **_create_options(args),
        )
        return remote, "create"

    payload = json.loads(Path(args.connect_ids_file).read_text(encoding="utf-8"))
    sandbox_ids = payload.get("sandbox_ids", [])
    if not isinstance(sandbox_ids, list) or len(sandbox_ids) != 1 or not isinstance(sandbox_ids[0], str):
        raise ValueError("--connect-ids-file must contain exactly one sandbox ID")
    remote = await Sandbox.connect(
        sandbox_ids[0],
        connection_config=_connection_config(args.request_timeout_seconds),
        skip_health_check=True,
    )
    return remote, "connect"


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    image, image_label = _image_spec(args)
    remote: Sandbox | None = None
    allocation_started = time.perf_counter()
    result: dict[str, Any] = {"case": args.case, "image": image_label, "passed": False}
    try:
        remote, operation = await _obtain_sandbox(args, image)
        result.update({
            "allocation_seconds": time.perf_counter() - allocation_started,
            "operation": operation,
            "sandbox_id": remote.id,
            "state": (await remote.get_info()).status.state,
        })

        command_started = time.perf_counter()
        command = "printf raw-opensandbox-ok"
        if args.copy_mounted_capsule:
            python_code = (
                "import os; from pathlib import Path; "
                "from hypotest.kernel_capsule_server import copy_mounted_capsule; "
                'copy_mounted_capsule(Path(os.environ["HYPOTEST_MOUNTED_CAPSULE_ROOT"]), '
                'os.environ["HYPOTEST_MOUNTED_CAPSULE_ID"], Path("/workspace")); '
                'print("raw-opensandbox-ok")'
            )
            command = f"/app/kernel_env/bin/python -c '{python_code}'"
        elif args.verify_mounted_capsule:
            capsule_path = "$HYPOTEST_MOUNTED_CAPSULE_ROOT/$HYPOTEST_MOUNTED_CAPSULE_ID"
            checks = [
                'test -d "$HYPOTEST_MOUNTED_CAPSULE_ROOT" || exit 10',
                f'test -d "{capsule_path}" || exit 11',
                f'test -r "{capsule_path}" || exit 12',
                f'test -z "$(find "{capsule_path}" -type l -print -quit)" || exit 13',
            ]
            command = "sh -lc '" + "; ".join(checks) + "; printf raw-opensandbox-ok'"
        execution = await remote.commands.run(
            command,
            opts=RunCommandOpts(timeout=timedelta(seconds=30)),
        )
        result.update({
            "command_seconds": time.perf_counter() - command_started,
            "exit_code": execution.exit_code,
            "stdout_verified": _stdout(execution).strip() == "raw-opensandbox-ok",
        })
        if execution.error is not None:
            result["command_error_type"] = execution.error.name
        command_passed = execution.exit_code in {None, 0} and result["stdout_verified"] is True
        result["passed"] = command_passed
        if args.start_kernel_after_copy and command_passed:
            probe_passed, details = await _run_remote_script_probe(
                remote,
                source=_KERNEL_PROBE_SOURCE,
                path="/workspace/.raw_kernel_probe.py",
                timeout_seconds=120,
                result_prefix="kernel_probe",
            )
            result.update(details)
            result["passed"] = probe_passed
        if args.verify_kernel_health and command_passed:
            probe_passed, details = await _run_remote_script_probe(
                remote,
                source=_KERNEL_HEALTH_PROBE_SOURCE,
                path="/workspace/.raw_kernel_health_probe.py",
                timeout_seconds=60,
                result_prefix="health_probe",
            )
            result.update(details)
            result["passed"] = result["passed"] and probe_passed
    except BaseException as exc:
        result.update({
            "allocation_seconds": time.perf_counter() - allocation_started,
            "error_type": type(exc).__name__,
            "exception_chain": _exception_chain(exc),
        })
    finally:
        if remote is not None:
            try:
                await remote.kill()
            except Exception as exc:
                result["cleanup_error_type"] = type(exc).__name__
            finally:
                await remote.close()
            result["cleanup_attempted"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="public-defaults")
    parser.add_argument("--image", default="python:3.11")
    parser.add_argument("--image-env")
    parser.add_argument("--registry-auth", action="store_true")
    parser.add_argument("--registry-username-env", default="REGISTRY_USERNAME")
    parser.add_argument("--registry-password-env", default="REGISTRY_PASSWORD")
    parser.add_argument("--hypotest-metadata", action="store_true")
    parser.add_argument("--platform-amd64", action="store_true")
    parser.add_argument("--pull-policy-primary", choices=("Always", "IfNotPresent", "Never"))
    parser.add_argument("--pull-policy-compat", choices=("Always", "IfNotPresent", "Never"))
    parser.add_argument("--resource-limits", action="store_true")
    parser.add_argument("--resource-requests", action="store_true")
    parser.add_argument(
        "--entrypoint-mode",
        choices=("simple", "wrapper-sleep", "kernel-base", "kernel-no-shim", "kernel-memory", "kernel-full"),
    )
    parser.add_argument("--mounted-root")
    parser.add_argument("--mounted-capsule-id")
    parser.add_argument("--clear-object-store", action="store_true")
    parser.add_argument("--verify-mounted-capsule", action="store_true")
    parser.add_argument("--copy-mounted-capsule", action="store_true")
    parser.add_argument("--start-kernel-after-copy", action="store_true")
    parser.add_argument("--verify-kernel-health", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--connect-ids-file", type=Path)
    parser.add_argument("--ready-timeout-seconds", type=float, default=180)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--ttl-seconds", type=int, default=600)
    args = parser.parse_args()
    if (args.mounted_root is None) != (args.mounted_capsule_id is None):
        parser.error("--mounted-root and --mounted-capsule-id must be supplied together")

    logging.disable(logging.CRITICAL)
    result = asyncio.run(_run(args))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
