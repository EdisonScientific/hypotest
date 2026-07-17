#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build capsule data into an OpenSandbox-compatible container image.

The generated image extends ``hypotest-kernel`` without changing its inherited
entrypoint or runtime. The capsule directory's *contents* become a dedicated
image layer rooted at ``/workspace``. With ``--all-capsules``, immediate child
directories instead share one layer under ``/opt/hypotest/capsules`` and the
requested task is projected into ``/workspace`` at startup. Docker Buildx
exports either layout to the local image store or to a registry.

No project package imports are used so this script can run directly from a
checkout with only Python 3 and Docker Buildx installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BUNDLE_FORMAT_VERSION = "1"
DEFAULT_BASE_IMAGE = "hypotest-kernel:latest"
DEFAULT_PLATFORM = "linux/amd64"
SUPPORTED_PLATFORMS = ("linux/amd64", "linux/arm64")
STANDARD_IMAGE_MEDIA_TYPES = {
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
}

# Kept inline so this file is the only build entrypoint/artifact Hypotest users
# need. The supplied directory itself is the Docker build context; no multi-GB
# staging copy is made. COPY --link keeps the data layer independent of the
# kernel base so it can be cached/rebased efficiently.
_DOCKERFILE_HEADER = f"""\
# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE={DEFAULT_BASE_IMAGE}
FROM ${{BASE_IMAGE}}

ARG BASE_IMAGE
ARG CAPSULE_ID
ARG BUNDLE_LAYOUT
ARG BUNDLE_ROOT

LABEL org.opencontainers.image.title="Hypotest large-bundle data" \\
      org.opencontainers.image.description="Hypotest kernel server with bundled task data" \\
      org.opencontainers.image.base.name="${{BASE_IMAGE}}" \\
      io.hypotest.bundle.format="{BUNDLE_FORMAT_VERSION}" \\
      io.hypotest.bundle.layout="${{BUNDLE_LAYOUT}}" \\
      io.hypotest.bundle.root="${{BUNDLE_ROOT}}" \\
      io.hypotest.bundle.workspace="/workspace" \\
      io.hypotest.capsule.id="${{CAPSULE_ID}}"

ENV HYPOTEST_BUNDLE_FORMAT={BUNDLE_FORMAT_VERSION} \\
    HYPOTEST_BUNDLE_LAYOUT="${{BUNDLE_LAYOUT}}" \\
    HYPOTEST_BUNDLE_ROOT="${{BUNDLE_ROOT}}"

WORKDIR /workspace
"""

SINGLE_BUNDLE_DOCKERFILE = _DOCKERFILE_HEADER + "COPY --link --chown=0:0 . /workspace/\n"
COLLECTION_BUNDLE_DOCKERFILE = _DOCKERFILE_HEADER + "COPY --link --chown=0:0 . /opt/hypotest/capsules/\n"


class BundleBuildError(RuntimeError):
    """The requested bundle cannot be built safely or compatibly."""


@dataclass(frozen=True)
class CapsuleStats:
    """Cheap metadata-only summary of a capsule build context."""

    files: int
    symlinks: int
    logical_bytes: int
    capsules: int = 1


@dataclass(frozen=True)
class BuildOptions:
    """Validated inputs used to construct the Docker Buildx command."""

    capsule_dir: Path
    capsule_id: str | None
    layout: Literal["single", "collection"]
    image: str
    base_image: str
    platform: str
    push: bool
    docker: str = "docker"
    builder: str | None = None
    progress: str = "auto"
    no_cache: bool = False


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def validate_capsule_id(value: str) -> str:
    """Validate the identity used by Hypotest's large_bundle_images mapping."""
    if not value or _contains_control_characters(value):
        raise BundleBuildError("capsule id must be non-empty and contain no control characters")
    return value


def default_image(capsule_id: str) -> str:
    """Return a deterministic, Docker-safe local tag for a capsule identity."""
    slug = re.sub(r"[^a-z0-9_.-]+", "-", capsule_id.lower()).strip(".-_")
    if not slug:
        slug = "capsule"
    # Docker tags are limited to 128 characters. Preserve readability while a
    # short identity hash prevents truncation/sanitization collisions.
    digest = hashlib.sha256(capsule_id.encode()).hexdigest()[:10]
    slug = slug[:105].rstrip(".-_") or "capsule"
    return f"hypotest-capsule:{slug}-{digest}"


def default_collection_image(directory_name: str) -> str:
    """Return a deterministic local tag for an all-capsules collection."""
    single_image = default_image(directory_name)
    return single_image.replace("hypotest-capsule:", "hypotest-capsules:", 1)


def validate_image_reference(value: str, *, option: str, allow_digest: bool = False) -> str:
    """Reject values that can never be Docker image references."""
    if not value or value.startswith("-") or "://" in value or ("@" in value and not allow_digest):
        requirement = "container image reference" if allow_digest else "taggable container image reference"
        raise BundleBuildError(f"{option} must be a {requirement}, got {value!r}")
    if any(character.isspace() for character in value) or _contains_control_characters(value):
        raise BundleBuildError(f"{option} must not contain whitespace or control characters")
    if "@" in value:
        repository, digest = value.rsplit("@", maxsplit=1)
        if not repository or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise BundleBuildError(f"{option} contains a malformed image digest")
    return value


def _inspect_capsule_entry(path: Path, root: Path) -> CapsuleStats:
    """Validate one context entry and return its contribution to the summary."""
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        try:
            target = path.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise BundleBuildError(f"capsule contains a broken or cyclic symlink: {path}") from exc
        if not target.is_relative_to(root):
            raise BundleBuildError(f"capsule symlink escapes the build context: {path} -> {target}")
        return CapsuleStats(files=0, symlinks=1, logical_bytes=0)
    if stat.S_ISREG(mode):
        return CapsuleStats(files=1, symlinks=0, logical_bytes=path.stat().st_size)
    if stat.S_ISDIR(mode):
        return CapsuleStats(files=0, symlinks=0, logical_bytes=0)
    raise BundleBuildError(f"capsule contains an unsupported special filesystem object: {path}")


def inspect_capsule(capsule_dir: Path) -> tuple[Path, CapsuleStats]:
    """Resolve and validate a self-contained Docker build context.

    Root-level ``.dockerignore`` is rejected because it could silently omit
    scientific inputs from the resulting bundle. Internal symlinks are allowed;
    broken or escaping symlinks and special filesystem objects are rejected.
    """
    try:
        root = capsule_dir.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise BundleBuildError(f"capsule directory does not exist: {capsule_dir}") from exc
    if not root.is_dir():
        raise BundleBuildError(f"capsule path is not a directory: {root}")
    if root == Path(root.anchor):
        raise BundleBuildError("refusing to use a filesystem root as the capsule build context")
    if (root / ".dockerignore").exists():
        raise BundleBuildError(
            f"{root / '.dockerignore'} would control which capsule files enter the image; "
            "remove or rename it before building"
        )

    files = symlinks = logical_bytes = 0
    try:
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            for name in [*dirnames, *filenames]:
                path = Path(directory, name)
                contribution = _inspect_capsule_entry(path, root)
                files += contribution.files
                symlinks += contribution.symlinks
                logical_bytes += contribution.logical_bytes
    except PermissionError as exc:
        raise BundleBuildError(f"capsule cannot be read: {exc.filename}") from exc

    if files == 0:
        raise BundleBuildError(f"capsule directory contains no regular files: {root}")
    return root, CapsuleStats(files=files, symlinks=symlinks, logical_bytes=logical_bytes)


def inspect_collection(capsule_dir: Path) -> tuple[Path, CapsuleStats]:
    """Validate a collection whose immediate child directories are capsules."""
    root, stats = inspect_capsule(capsule_dir)
    children = sorted(root.iterdir(), key=lambda path: path.name)
    capsule_dirs = [path for path in children if stat.S_ISDIR(path.lstat().st_mode)]
    non_directories = [path for path in children if not stat.S_ISDIR(path.lstat().st_mode)]
    if non_directories:
        examples = ", ".join(path.name for path in non_directories[:3])
        raise BundleBuildError(
            "--all-capsules expects only capsule directories at the collection root; "
            f"found non-directory entries: {examples}"
        )
    if not capsule_dirs:
        raise BundleBuildError(f"collection contains no capsule directories: {root}")
    empty_capsules = [path.name for path in capsule_dirs if not any(path.iterdir())]
    if empty_capsules:
        examples = ", ".join(empty_capsules[:3])
        raise BundleBuildError(f"collection contains empty capsule directories: {examples}")
    return root, CapsuleStats(
        files=stats.files,
        symlinks=stats.symlinks,
        logical_bytes=stats.logical_bytes,
        capsules=len(capsule_dirs),
    )


def dockerfile_for(options: BuildOptions) -> str:
    """Select the inline Dockerfile for one capsule or a shared collection."""
    return COLLECTION_BUNDLE_DOCKERFILE if options.layout == "collection" else SINGLE_BUNDLE_DOCKERFILE


def build_command(options: BuildOptions, metadata_file: Path) -> list[str]:
    """Construct a single-platform Docker/OCI image build command."""
    bundle_root = "/opt/hypotest/capsules" if options.layout == "collection" else "/workspace"
    command = [options.docker, "buildx", "build"]
    if options.builder:
        command.extend(["--builder", options.builder])
    command.extend([
        "--file",
        "-",
        "--platform",
        options.platform,
        "--build-arg",
        f"BASE_IMAGE={options.base_image}",
        "--build-arg",
        f"CAPSULE_ID={options.capsule_id or ''}",
        "--build-arg",
        f"BUNDLE_LAYOUT={options.layout}",
        "--build-arg",
        f"BUNDLE_ROOT={bundle_root}",
        "--tag",
        options.image,
        "--metadata-file",
        str(metadata_file),
        "--progress",
        options.progress,
        # OpenSandbox's Docker and Kubernetes runtimes consume ordinary image
        # manifests. Avoid wrapping this single-platform image in a provenance
        # attestation index, maximizing compatibility with private registries.
        "--provenance=false",
    ])
    if options.no_cache:
        command.append("--no-cache")
    command.extend(("--push" if options.push else "--load", str(options.capsule_dir)))
    return command


def verify_build_metadata(metadata_file: Path) -> tuple[str | None, str | None]:
    """Validate Buildx's exported descriptor as a standard Docker/OCI image."""
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise BundleBuildError(f"Docker did not produce readable build metadata: {metadata_file}") from exc

    descriptor = metadata.get("containerimage.descriptor") or {}
    media_type = descriptor.get("mediaType")
    if media_type is not None and media_type not in STANDARD_IMAGE_MEDIA_TYPES:
        raise BundleBuildError(f"Docker exported unsupported image media type {media_type!r}")
    digest = metadata.get("containerimage.digest") or descriptor.get("digest")
    if digest is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise BundleBuildError(f"Docker exported malformed image digest {digest!r}")
    return media_type, digest


def immutable_image_reference(image: str, digest: str | None) -> str | None:
    """Combine a pushed tag's repository with its content digest."""
    if digest is None:
        return None
    reference = image.split("@", maxsplit=1)[0]
    last_slash = reference.rfind("/")
    last_colon = reference.rfind(":")
    if last_colon > last_slash:
        reference = reference[:last_colon]
    return f"{reference}@{digest}"


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _print_opensandbox_config(options: BuildOptions, image: str) -> None:
    os_name, architecture = options.platform.split("/", maxsplit=1)
    print("\nOpenSandbox large-bundle configuration:")
    print("  capsule_mode: large_bundle")
    if options.layout == "collection":
        print(f"  large_bundle_image: {json.dumps(image)}")
    else:
        assert options.capsule_id is not None
        print("  large_bundle_images:")
        print(f"    {json.dumps(options.capsule_id)}: {json.dumps(image)}")
    print(f"  platform_os: {os_name}")
    print(f"  platform_arch: {architecture}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one capsule or an all-capsules collection into a standard Docker/OCI image "
            "that Hypotest can launch through OpenSandbox."
        ),
    )
    parser.add_argument(
        "capsule_dir",
        type=Path,
        help="one capsule directory, or a collection root when --all-capsules is set",
    )
    parser.add_argument(
        "-t",
        "--tag",
        "--image",
        dest="image",
        help="output image tag/registry URI (default: deterministic local tag derived from capsule id)",
    )
    parser.add_argument(
        "--capsule-id",
        help="Hypotest input_data_path/id used as the large_bundle_images key (default: directory name)",
    )
    parser.add_argument(
        "--all-capsules",
        action="store_true",
        help=(
            "treat each immediate child directory as a capsule and build one shared image; "
            "the selected task is projected into /workspace at runtime"
        ),
    )
    parser.add_argument(
        "--base-image",
        default=os.getenv("HYPOTEST_KERNEL_IMAGE", DEFAULT_BASE_IMAGE),
        help=f"kernel-server image to extend (default: {DEFAULT_BASE_IMAGE})",
    )
    parser.add_argument(
        "--platform",
        choices=SUPPORTED_PLATFORMS,
        default=os.getenv("HYPOTEST_BUNDLE_PLATFORM", DEFAULT_PLATFORM),
        help="target platform; it must match OpenSandboxSpec.platform_os/platform_arch",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--push",
        action="store_true",
        help="push to a registry visible to a remote OpenSandbox server",
    )
    output.add_argument(
        "--load",
        action="store_true",
        help="load into the local Docker daemon (default; same-daemon OpenSandbox only)",
    )
    parser.add_argument("--builder", help="Docker Buildx builder name")
    parser.add_argument("--progress", choices=("auto", "plain", "tty", "quiet", "rawjson"), default="auto")
    parser.add_argument("--no-cache", action="store_true", help="disable Docker build cache")
    parser.add_argument("--docker", default="docker", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the Docker command only")
    return parser


def _options_from_args(args: argparse.Namespace) -> tuple[BuildOptions, CapsuleStats]:
    layout: Literal["single", "collection"] = "collection" if args.all_capsules else "single"
    capsule_dir, stats = (
        inspect_collection(args.capsule_dir) if args.all_capsules else inspect_capsule(args.capsule_dir)
    )
    if args.all_capsules and args.capsule_id is not None:
        raise BundleBuildError("--capsule-id cannot be combined with --all-capsules")
    capsule_id = None if args.all_capsules else validate_capsule_id(args.capsule_id or capsule_dir.name)
    if args.push and args.image is None:
        raise BundleBuildError("--push requires an explicit --image registry URI")
    if args.platform not in SUPPORTED_PLATFORMS:
        raise BundleBuildError(f"--platform must be one of {', '.join(SUPPORTED_PLATFORMS)}; got {args.platform!r}")
    default_tag = default_collection_image(capsule_dir.name) if args.all_capsules else default_image(capsule_id or "")
    image = validate_image_reference(args.image or default_tag, option="--image")
    base_image = validate_image_reference(args.base_image, option="--base-image", allow_digest=True)
    return (
        BuildOptions(
            capsule_dir=capsule_dir,
            capsule_id=capsule_id,
            layout=layout,
            image=image,
            base_image=base_image,
            platform=args.platform,
            push=args.push,
            docker=args.docker,
            builder=args.builder,
            progress=args.progress,
            no_cache=args.no_cache,
        ),
        stats,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        options, stats = _options_from_args(args)
    except BundleBuildError as exc:
        parser.error(str(exc))

    subject = "Collection" if options.layout == "collection" else "Capsule"
    capsule_count = f", {stats.capsules} capsules" if options.layout == "collection" else ""
    print(
        f"{subject}: {options.capsule_dir} "
        f"({stats.files} files, {stats.symlinks} symlinks{capsule_count}, {_human_size(stats.logical_bytes)})"
    )
    print(f"Image:   {options.image}")
    print(f"Base:    {options.base_image}")
    print(f"Target:  {options.platform}")
    print("Output:  registry push" if options.push else "Output:  local Docker image store")
    if not options.push:
        print("Note: use --push with a registry tag when OpenSandbox does not share this Docker daemon.")

    with tempfile.TemporaryDirectory(prefix="hypotest-bundle-build-") as temp_dir:
        metadata_file = Path(temp_dir, "metadata.json")
        command = build_command(options, metadata_file)
        if args.dry_run:
            print(f"\n$ {shlex.join(command)}")
            _print_opensandbox_config(options, options.image)
            return 0

        if shutil.which(options.docker) is None:
            raise BundleBuildError(f"Docker executable not found: {options.docker}")
        try:
            subprocess.run(  # noqa: S603 - argv execution is intentional; no shell is involved
                [options.docker, "buildx", "version"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(  # noqa: S603 - validated argv is passed directly, never through a shell
                command,
                input=dockerfile_for(options),
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise BundleBuildError(f"Docker Buildx failed with exit code {exc.returncode}") from exc

        media_type, digest = verify_build_metadata(metadata_file)

    selected_image = immutable_image_reference(options.image, digest) if options.push else options.image
    selected_image = selected_image or options.image
    print("\nBuilt an OpenSandbox-compatible container image successfully.")
    if media_type:
        print(f"Media type: {media_type}")
    if digest:
        print(f"Digest:     {digest}")
    _print_opensandbox_config(options, selected_image)
    if options.push:
        print("Registry note: the OpenSandbox runtime needs independent pull credentials for private images.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleBuildError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
