#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build and optionally push the generic Hypotest OpenSandbox image.

This is intentionally not a large-bundle builder: no capsule directory is a
build input and no capsule bytes enter an image layer. Each sandbox downloads
only its selected capsule into its ephemeral workspace during container init.
The script can bake non-secret S3 location defaults, while OpenSandbox
create-time environment variables retain precedence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_IMAGE = "hypotest-opensandbox:latest"
DEFAULT_PLATFORM = "linux/amd64"
SUPPORTED_PLATFORMS = ("linux/amd64", "linux/arm64")


class ImageBuildError(RuntimeError):
    """The OpenSandbox image could not be built or validated."""


@dataclass(frozen=True)
class BuildOptions:
    root: Path
    image: str
    platform: str
    base_target: str
    base_image: str
    build_base: bool
    build_cutoff_date: str
    capsule_source: str | None
    capsule_key: str | None
    s3_endpoint_url: str | None
    s3_region: str | None
    registry_auth: bool
    registry_username_env: str
    registry_password_env: str
    image_pull_policy: str | None
    kernel_memory_limit_mb: int | None
    push: bool
    pull: bool
    no_cache: bool
    docker: str = "docker"


def _validate_reference(value: str, option: str) -> str:
    if not value or value.startswith("-") or "://" in value or any(character.isspace() for character in value):
        raise ImageBuildError(f"{option} must be a container image reference, got {value!r}")
    return value


def _validate_non_secret_value(value: str | None, option: str) -> str | None:
    if value is None:
        return None
    if not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ImageBuildError(f"{option} must be non-empty and contain no control characters")
    return value


def _validate_environment_name(value: str, option: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ImageBuildError(f"{option} must be an environment variable name, got {value!r}")
    return value


def _default_base_image(target: str, platform: str, build_base: bool) -> str:
    if not build_base:
        return f"interpreter-env:{target}"
    architecture = platform.split("/", maxsplit=1)[1]
    return f"hypotest-opensandbox-base:{target}-{architecture}"


def base_build_command(options: BuildOptions) -> list[str] | None:
    if not options.build_base:
        return None
    command = [
        options.docker,
        "build",
        "--platform",
        options.platform,
        "--file",
        str(options.root / "Dockerfile"),
        "--target",
        options.base_target,
        "--build-arg",
        f"BUILD_CUTOFF_DATE={options.build_cutoff_date}",
        "--tag",
        options.base_image,
    ]
    if options.pull:
        command.append("--pull")
    if options.no_cache:
        command.append("--no-cache")
    command.append(str(options.root))
    return command


def kernel_build_command(options: BuildOptions) -> list[str]:
    command = [
        options.docker,
        "build",
        "--platform",
        options.platform,
        "--file",
        str(options.root / "Dockerfile.kernel"),
        "--build-arg",
        f"BASE_IMAGE={options.base_image}",
        "--build-arg",
        f"BUILD_CUTOFF_DATE={options.build_cutoff_date}",
        "--build-arg",
        f"CAPSULE_SOURCE={options.capsule_source or ''}",
        "--build-arg",
        f"CAPSULE_KEY={options.capsule_key or ''}",
        "--tag",
        options.image,
    ]
    if options.pull:
        command.append("--pull")
    if options.no_cache:
        command.append("--no-cache")
    command.append(str(options.root))
    return command


def _image_repository(reference: str) -> str:
    unpinned = reference.split("@", maxsplit=1)[0]
    last_slash = unpinned.rfind("/")
    last_colon = unpinned.rfind(":")
    return unpinned[:last_colon] if last_colon > last_slash else unpinned


def _select_digest_reference(image: str, repo_digests: Sequence[str]) -> str | None:
    repository = _image_repository(image)
    pattern = re.compile(rf"^{re.escape(repository)}@sha256:[0-9a-f]{{64}}$")
    return next((reference for reference in repo_digests if pattern.fullmatch(reference)), None)


def _inspect_image(options: BuildOptions) -> tuple[str | None, str | None]:
    platform_result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [options.docker, "image", "inspect", options.image, "--format", "{{.Os}}/{{.Architecture}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    platform = platform_result.stdout.strip() or None
    if platform is not None and platform != options.platform:
        raise ImageBuildError(f"built image platform is {platform}, expected {options.platform}")

    digest_result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [options.docker, "image", "inspect", options.image, "--format", "{{json .RepoDigests}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        repo_digests = json.loads(digest_result.stdout or "[]") or []
    except json.JSONDecodeError:
        repo_digests = []
    digest_reference = _select_digest_reference(options.image, repo_digests)
    return platform, digest_reference


def _print_opensandbox_config(options: BuildOptions, image: str) -> None:
    os_name, architecture = options.platform.split("/", maxsplit=1)
    print("\nOpenSandbox configuration:")
    print("  capsule_mode: object_store")
    print(f"  image: {json.dumps(image)}")
    print("  install_shim_enabled: false")
    if options.kernel_memory_limit_mb is not None:
        print(f"  kernel_memory_limit_mb: {options.kernel_memory_limit_mb}")
    if options.capsule_source is None:
        print('  capsule_source: "s3://bucket/base-prefix"')
    else:
        print(f"  # CAPSULE_SOURCE baked into the image: {json.dumps(options.capsule_source)}")
    if options.capsule_key is None:
        print('  capsule_key: "{capsule_uuid}"')
    else:
        print("  capsule_key: null  # preserve the CAPSULE_KEY baked into the image")
    print(f"  platform_os: {os_name}")
    print(f"  platform_arch: {architecture}")
    if options.registry_auth:
        print("  image_auth:")
        print(f'    username: "${{{options.registry_username_env}}}"')
        print(f'    password: "${{{options.registry_password_env}}}"')
    if options.image_pull_policy is not None:
        print(f"  image_pull_policy: {options.image_pull_policy}")
    if options.s3_endpoint_url or options.s3_region:
        print("  env:")
        if options.s3_endpoint_url:
            print(f"    AWS_ENDPOINT_URL: {json.dumps(options.s3_endpoint_url)}")
        if options.s3_region:
            print(f"    AWS_DEFAULT_REGION: {json.dumps(options.s3_region)}")
    print("  # Inject AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY at runtime, or use the sandbox's IAM identity.")
    if options.registry_auth:
        print("  # Registry auth is sent only in the OpenSandbox image spec; it is never baked into this image.")


def _print_build_summary(options: BuildOptions) -> None:
    print("Capsule delivery: init-time lazy pull (no capsule data in image layers)")
    print(f"Image:            {options.image}")
    print(f"Platform:         {options.platform}")
    print(f"Interpreter base: {options.base_image} ({'build' if options.build_base else 'reuse'})")
    if options.capsule_source:
        print(f"Baked source:     {options.capsule_source}")
    if options.capsule_key:
        print(f"Baked key:        {options.capsule_key}")
    if options.registry_auth:
        print(
            f"Registry pull auth: ${{{options.registry_username_env}}} / "
            f"${{{options.registry_password_env}}} (runtime only)"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a capsule-free Hypotest kernel image for OpenSandbox, optionally push it, "
            "and print the matching remote configuration."
        )
    )
    parser.add_argument("-t", "--tag", "--image", dest="image", help=f"output image (default: {DEFAULT_IMAGE})")
    parser.add_argument(
        "--platform",
        choices=SUPPORTED_PLATFORMS,
        default=os.getenv("HYPOTEST_SANDBOX_PLATFORM", DEFAULT_PLATFORM),
    )
    parser.add_argument("--base-target", choices=("full", "core"), default="full")
    parser.add_argument("--base-image", help="base tag/reference (default is derived from target and platform)")
    parser.add_argument(
        "--skip-base-build",
        action="store_true",
        help="use --base-image (or interpreter-env:<target>) instead of building the interpreter base",
    )
    parser.add_argument(
        "--build-cutoff-date",
        default=datetime.now(tz=UTC).date().isoformat(),
        help="latest allowed package publication date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capsule-source",
        help="optional baked s3://bucket/base-prefix default; OpenSandbox can override it per allocation",
    )
    parser.add_argument(
        "--capsule-key",
        help="optional baked relative capsule key/prefix for a fixed-key image",
    )
    parser.add_argument("--s3-endpoint-url", help="runtime AWS_ENDPOINT_URL to print in the OpenSandbox config")
    parser.add_argument("--s3-region", help="runtime AWS_DEFAULT_REGION to print in the OpenSandbox config")
    parser.add_argument(
        "--registry-auth",
        action="store_true",
        help="print private-registry image_auth using environment references (does not expose credentials to Docker)",
    )
    parser.add_argument(
        "--registry-username-env",
        default="REGISTRY_USERNAME",
        help="environment variable containing the registry username (default: REGISTRY_USERNAME)",
    )
    parser.add_argument(
        "--registry-password-env",
        default="REGISTRY_PASSWORD",
        help="environment variable containing the registry password/token (default: REGISTRY_PASSWORD)",
    )
    parser.add_argument(
        "--image-pull-policy",
        choices=("Always", "IfNotPresent", "Never"),
        default="IfNotPresent",
        help="OpenSandbox image pull policy printed in the runtime config (default: IfNotPresent)",
    )
    parser.add_argument(
        "--no-image-pull-policy",
        action="store_true",
        help="omit Hypotest's image-pull-policy extensions and use the OpenSandbox server default",
    )
    parser.add_argument(
        "--kernel-memory-limit-mb",
        type=int,
        help="inner Jupyter RLIMIT_AS value to print in the OpenSandbox runtime config",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--push", action="store_true", help="push the completed image to its registry")
    output.add_argument("--load", action="store_true", help="leave the completed image in the local daemon (default)")
    parser.add_argument("--pull", action="store_true", help="refresh upstream base layers")
    parser.add_argument(
        "--no-cache", action="store_true", help="disable Docker build cache (capsule data is never cached)"
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands without invoking Docker")
    parser.add_argument("--docker", default="docker", help=argparse.SUPPRESS)
    return parser


def _options_from_args(args: argparse.Namespace) -> BuildOptions:
    root = Path(__file__).resolve().parents[1]
    for required in (root / "Dockerfile", root / "Dockerfile.kernel", root / "pyproject.toml"):
        if not required.is_file():
            raise ImageBuildError(f"repository build input is missing: {required}")
    try:
        datetime.strptime(args.build_cutoff_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ImageBuildError("--build-cutoff-date must use YYYY-MM-DD") from exc
    if args.push and args.image is None:
        raise ImageBuildError("--push requires an explicit registry --image")
    if args.kernel_memory_limit_mb is not None and args.kernel_memory_limit_mb <= 0:
        raise ImageBuildError("--kernel-memory-limit-mb must be positive")

    build_base = not args.skip_base_build
    image = _validate_reference(args.image or DEFAULT_IMAGE, "--image")
    base_image = _validate_reference(
        args.base_image or _default_base_image(args.base_target, args.platform, build_base),
        "--base-image",
    )
    return BuildOptions(
        root=root,
        image=image,
        platform=args.platform,
        base_target=args.base_target,
        base_image=base_image,
        build_base=build_base,
        build_cutoff_date=args.build_cutoff_date,
        capsule_source=_validate_non_secret_value(args.capsule_source, "--capsule-source"),
        capsule_key=_validate_non_secret_value(args.capsule_key, "--capsule-key"),
        s3_endpoint_url=_validate_non_secret_value(args.s3_endpoint_url, "--s3-endpoint-url"),
        s3_region=_validate_non_secret_value(args.s3_region, "--s3-region"),
        registry_auth=args.registry_auth,
        registry_username_env=_validate_environment_name(args.registry_username_env, "--registry-username-env"),
        registry_password_env=_validate_environment_name(args.registry_password_env, "--registry-password-env"),
        image_pull_policy=None if args.no_image_pull_policy else args.image_pull_policy,
        kernel_memory_limit_mb=args.kernel_memory_limit_mb,
        push=args.push,
        pull=args.pull,
        no_cache=args.no_cache,
        docker=args.docker,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        options = _options_from_args(args)
    except ImageBuildError as exc:
        parser.error(str(exc))

    base_command = base_build_command(options)
    kernel_command = kernel_build_command(options)
    _print_build_summary(options)

    commands = [command for command in (base_command, kernel_command) if command is not None]
    if options.push:
        commands.append([options.docker, "push", options.image])
    if args.dry_run:
        for command in commands:
            print(f"\n$ {shlex.join(command)}")
        _print_opensandbox_config(options, options.image)
        return 0

    if shutil.which(options.docker) is None:
        raise ImageBuildError(f"Docker executable not found: {options.docker}")
    run_env = dict(os.environ)
    run_env.setdefault("DOCKER_BUILDKIT", "1")
    try:
        for command in commands:
            subprocess.run(command, check=True, env=run_env)  # noqa: S603 - fixed argv, no shell
        platform, digest_reference = _inspect_image(options)
    except subprocess.CalledProcessError as exc:
        raise ImageBuildError(f"Docker command failed with exit code {exc.returncode}") from exc

    selected_image = digest_reference or options.image
    print("\nOpenSandbox image is ready.")
    if platform:
        print(f"Platform: {platform}")
    if digest_reference:
        print(f"Digest:   {digest_reference}")
    elif not options.push:
        print(f"Push when ready: {options.docker} push {options.image}")
    _print_opensandbox_config(options, selected_image)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImageBuildError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
