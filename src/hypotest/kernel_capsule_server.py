"""Kernel server with init-time and on-request capsule loading.

Runs the standalone ``KernelServer`` (``env/kernel_server.py``) and adds a
``POST /load_capsule`` endpoint that pulls an exact capsule key first (with a
legacy most-recent/UUID fallback) from a local folder or
``s3://bucket/prefix`` into the kernel's work dir, then resets the kernel to
that workspace. When both
``CAPSULE_SOURCE`` and ``CAPSULE_KEY`` are present, the selected capsule is
pulled during process initialization *before* Jupyter and the HTTP health server
start. OpenSandbox uses that path so readiness means data and kernel are ready.

A capsule may instead come from a collection volume mounted by the sandbox
cluster. In that mode, the selected directory is copied into writable,
sandbox-local ``/workspace`` before startup; the model never works directly on
the shared mount.

A source is optional for large-bundle images. A single-capsule image already
has task data under ``/workspace``; a collection image stores every capsule
below ``/opt/hypotest/capsules`` and projects the requested one into
``/workspace`` before starting the kernel. Bundled layouts ignore init-pull
settings.

This lives in hypotest (not in ``kernel_server.py``) on purpose: ``kernel_server.py``
must stay import-free for the enroot/docker path, which has neither hypotest nor
boto3. This entrypoint only runs in the bundled image, where both exist. S3
endpoint + credentials come from the standard ``AWS_*`` env vars (see s3_sync).

    python -m hypotest.kernel_capsule_server \
        --capsule-source s3://train-data-05312026-agent-sandbox/capsules --port 8000

Then, per request:
    curl -XPOST localhost:8000/load_capsule -d '{"capsule_uuid": "<uuid>"}'
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import stat
from pathlib import Path

import uvicorn
from fastapi import HTTPException, status
from pydantic import BaseModel

from hypotest import s3_sync
from hypotest.env import install_shim
from hypotest.env.kernel_server import KernelServer, NBLanguage, create_app

logger = logging.getLogger(__name__)

_BUNDLE_LAYOUTS = {"none", "single", "collection"}
_WORKSPACE_CONTROL_NAMES = {".install_shim", "pip-cache", "pip.conf", "pydeps", "Rprofile", "r_libs"}


class LoadCapsuleRequest(BaseModel):
    capsule_uuid: str
    seed: int | None = None


class LoadCapsuleResponse(BaseModel):
    success: bool
    capsule_uuid: str
    objects: int
    seed: int | None = None


def _clear_dir(path: Path) -> None:
    """Remove the contents of ``path`` (keeping the dir itself)."""
    for child in list(path.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _resolve_capsule_directory(root_path: Path, capsule_id: str, *, source_kind: str) -> Path:
    """Resolve an exact or legacy-named capsule without allowing traversal."""
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"{source_kind} root is not a directory: {root}")
    try:
        relative = Path(capsule_id)
    except ValueError as exc:
        raise ValueError(f"{source_kind} capsule id is not a valid filesystem path") from exc
    if not capsule_id or relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {source_kind} capsule id: {capsule_id!r}")

    names = [relative]
    if len(relative.parts) == 1:
        names.extend((Path(f"CapsuleData-{capsule_id}"), Path(f"capsule_{capsule_id}")))
    for name in names:
        unresolved = root / name
        if unresolved.is_symlink():
            continue
        try:
            candidate = unresolved.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if candidate == root or root not in candidate.parents:
            continue
        if candidate.is_dir():
            return candidate
    attempted = ", ".join(str(name) for name in names)
    raise FileNotFoundError(f"no capsule for {capsule_id!r} under {root} (tried: {attempted})")


def resolve_collection_capsule(bundle_root: Path, capsule_id: str) -> Path:
    """Resolve one capsule baked into a collection image."""
    return _resolve_capsule_directory(bundle_root, capsule_id, source_kind="bundle")


def resolve_mounted_capsule(mounted_root: Path, capsule_id: str) -> Path:
    """Resolve one capsule below a cluster-mounted collection root."""
    return _resolve_capsule_directory(mounted_root, capsule_id, source_kind="mounted-volume")


def _validate_copy_source(source: Path) -> int:
    """Reject links and special files before copying a mounted capsule."""
    count = 0
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            child = directory_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"mounted capsule contains a symlink: {child.relative_to(source)}")
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError(f"mounted capsule contains a special filesystem object: {child.relative_to(source)}")
        for name in filenames:
            child = directory_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"mounted capsule contains a symlink: {child.relative_to(source)}")
            if not stat.S_ISREG(child_stat.st_mode):
                raise ValueError(f"mounted capsule contains a special filesystem object: {child.relative_to(source)}")
            count += 1
    return count


def _make_tree_owner_writable(root: Path) -> None:
    """Ensure copied data can be edited while preserving executable bits."""
    root.chmod(root.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            child = directory_path / name
            child.chmod(child.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        for name in filenames:
            child = directory_path / name
            child.chmod(child.stat().st_mode | stat.S_IWUSR)


def copy_mounted_capsule(mounted_root: Path, capsule_id: str, work_dir: Path) -> tuple[Path, int]:
    """Copy one mounted capsule into an independent writable workspace."""
    selected = resolve_mounted_capsule(mounted_root, capsule_id)
    workspace = work_dir.resolve(strict=False)
    if workspace == selected or workspace in selected.parents or selected in workspace.parents:
        raise ValueError(f"mounted capsule and workspace must not overlap: {selected}, {workspace}")

    entries = list(selected.iterdir())
    collisions = sorted(entry.name for entry in entries if entry.name in _WORKSPACE_CONTROL_NAMES)
    if collisions:
        raise ValueError(f"mounted capsule uses reserved workspace names: {', '.join(collisions)}")
    count = _validate_copy_source(selected)

    work_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.is_symlink():
        raise ValueError(f"mounted capsule workspace must be a real directory: {work_dir}")
    _clear_dir(work_dir)
    shutil.copytree(selected, work_dir, dirs_exist_ok=True, symlinks=False)
    _make_tree_owner_writable(work_dir)
    return selected, count


def project_collection_capsule(bundle_root: Path, capsule_id: str, work_dir: Path) -> Path:
    """Expose one collection member in the workspace without copying its data.

    Top-level entries become symlinks into the shared image layer. This keeps
    sandbox startup proportional to entry count rather than capsule size. The
    projection function is also the seam where a future mount namespace or
    bubblewrap launcher can replace symlinks with a confined bind mount.
    """
    root = bundle_root.resolve(strict=True)
    workspace = work_dir.resolve(strict=False)
    if workspace == root or workspace in root.parents or root in workspace.parents:
        raise ValueError(f"bundle root and workspace must not overlap: {root}, {workspace}")

    selected = resolve_collection_capsule(root, capsule_id)
    entries = list(selected.iterdir())
    if not entries:
        raise FileNotFoundError(f"selected bundle capsule is empty: {selected}")
    collisions = sorted(entry.name for entry in entries if entry.name in _WORKSPACE_CONTROL_NAMES)
    if collisions:
        raise ValueError(f"bundle capsule uses reserved workspace names: {', '.join(collisions)}")

    work_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.is_symlink():
        raise ValueError(f"bundle workspace must be a real directory: {work_dir}")
    _clear_dir(work_dir)
    for entry in entries:
        (work_dir / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    return selected


async def prepare_initial_workspace(
    work_dir: Path,
    *,
    capsule_source: str | None,
    capsule_key: str | None,
    bundle_layout: str,
    bundle_root: Path | None,
    bundle_capsule_id: str | None,
    mounted_capsule_root: Path | None = None,
    mounted_capsule_id: str | None = None,
) -> tuple[str | None, int]:
    """Populate the workspace before kernel/HTTP startup.

    Returns a human-readable selected source and its object count. A collection
    projection is metadata-only and therefore reports zero downloaded objects.
    """
    if bundle_layout not in _BUNDLE_LAYOUTS:
        raise ValueError(f"unsupported large-bundle layout: {bundle_layout!r}")
    work_dir.mkdir(parents=True, exist_ok=True)

    if bundle_layout == "collection":
        if bundle_root is None or bundle_capsule_id is None:
            raise ValueError("collection bundle requires HYPOTEST_BUNDLE_ROOT and HYPOTEST_BUNDLE_CAPSULE_ID")
        selected = project_collection_capsule(bundle_root, bundle_capsule_id, work_dir)
        return str(selected), 0
    if bundle_layout == "single":
        return None, 0
    if (mounted_capsule_root is None) != (mounted_capsule_id is None):
        raise ValueError(
            "mounted-volume delivery requires HYPOTEST_MOUNTED_CAPSULE_ROOT and HYPOTEST_MOUNTED_CAPSULE_ID"
        )
    if mounted_capsule_root is not None and mounted_capsule_id is not None:
        if capsule_source is not None or capsule_key is not None:
            raise ValueError("mounted-volume and object-store capsule delivery cannot be configured together")
        selected, count = await asyncio.to_thread(
            copy_mounted_capsule,
            mounted_capsule_root,
            mounted_capsule_id,
            work_dir,
        )
        return str(selected), count
    if capsule_key is None:
        return None, 0
    if capsule_source is None:
        raise ValueError("CAPSULE_KEY requires CAPSULE_SOURCE from OpenSandbox or the image")

    _clear_dir(work_dir)
    count = await asyncio.to_thread(s3_sync.pull_capsule, capsule_source, capsule_key, work_dir)
    return f"{capsule_source.rstrip('/')}/{capsule_key}", count


def _apply_workspace_env(work_dir: Path, *, install_shim_enabled: bool = True) -> None:
    """Put the workspace install model on the env the kernel will inherit.

    Uses the same ``install_shim.workspace_env`` source as the enroot path while
    allowing externally isolated backends to omit the interceptor directories.
    """
    for key, seg in install_shim.workspace_env(str(work_dir), install_shim_enabled=install_shim_enabled).items():
        if key in {"PYTHONPATH", "PATH"}:
            existing = os.environ.get(key, "")
            os.environ[key] = f"{seg}{os.pathsep}{existing}" if existing else seg
        else:
            os.environ[key] = seg


async def run_server(
    work_dir: Path,
    language: NBLanguage,
    capsule_source: str | None,
    port: int = 8000,
    startup_token: str = "",
    safe_execute: bool = True,
    pip_index_url: str | None = None,
    seed: int | None = None,
    bundle_layout: str = "none",
    bundle_root: Path | None = None,
    bundle_capsule_id: str | None = None,
    capsule_key: str | None = None,
    mounted_capsule_root: Path | None = None,
    mounted_capsule_id: str | None = None,
    install_shim_enabled: bool = True,
    kernel_memory_limit_mb: int | None = None,
) -> None:
    selected, object_count = await prepare_initial_workspace(
        work_dir,
        capsule_source=capsule_source,
        capsule_key=capsule_key,
        bundle_layout=bundle_layout,
        bundle_root=bundle_root,
        bundle_capsule_id=bundle_capsule_id,
        mounted_capsule_root=mounted_capsule_root,
        mounted_capsule_id=mounted_capsule_id,
    )
    if selected is not None:
        logger.warning(
            "Prepared capsule %s in %s before kernel startup (%d staged objects)",
            selected,
            work_dir,
            object_count,
        )
    # Lay down persistent pip/R workspace paths plus the optional compatibility
    # shim. Externally isolated OpenSandbox pods disable only the interceptors.
    # The workspace pip.conf overrides /etc/pip.conf, so re-state the cutoff index-url.
    install_shim.write_workspace_config(
        work_dir,
        str(work_dir),
        index_url=pip_index_url,
        install_shim_enabled=install_shim_enabled,
    )
    _apply_workspace_env(work_dir, install_shim_enabled=install_shim_enabled)
    server = KernelServer(
        work_dir,
        language,
        startup_token=startup_token,
        safe_execute=safe_execute,
        seed=seed,
        kernel_memory_limit_mb=kernel_memory_limit_mb,
    )
    await server.start()
    app = create_app(server)

    @app.post("/load_capsule")
    async def load_capsule(req: LoadCapsuleRequest) -> LoadCapsuleResponse:
        if capsule_source is None and mounted_capsule_root is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This kernel image has no capsule source; use a bundled workspace, "
                    "set CAPSULE_SOURCE, or set HYPOTEST_MOUNTED_CAPSULE_ROOT"
                ),
            )
        source_label = str(mounted_capsule_root) if mounted_capsule_root is not None else capsule_source
        logger.warning("Loading capsule %s from %s", req.capsule_uuid, source_label)
        # Stop the kernel (release its files in work_dir), swap in the capsule,
        # then restart the kernel in the repopulated workspace. The pull is
        # offloaded so a large download does not block the event loop.
        if req.seed is not None:
            server.seed = req.seed
        await server.close()
        if mounted_capsule_root is not None:
            _, count = await asyncio.to_thread(
                copy_mounted_capsule,
                mounted_capsule_root,
                req.capsule_uuid,
                work_dir,
            )
        else:
            assert capsule_source is not None
            _clear_dir(work_dir)
            count = await asyncio.to_thread(s3_sync.pull_capsule, capsule_source, req.capsule_uuid, work_dir)
        # Clearing wiped the scaffolding; re-lay it alongside the freshly pulled capsule.
        install_shim.write_workspace_config(
            work_dir,
            str(work_dir),
            index_url=pip_index_url,
            install_shim_enabled=install_shim_enabled,
        )
        await server.start()
        logger.warning("Loaded capsule %s (%d objects) into %s", req.capsule_uuid, count, work_dir)
        return LoadCapsuleResponse(
            success=True,
            capsule_uuid=req.capsule_uuid,
            objects=count,
            seed=server.seed,
        )

    config = uvicorn.Config(app, host="0.0.0.0", port=port, loop="asyncio")  # noqa: S104
    await uvicorn.Server(config).serve()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Kernel server with on-request capsule loading")
    parser.add_argument("--work_dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--language", type=str, default="python", choices=["python", "r"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--capsule-source",
        type=str,
        default=os.getenv("CAPSULE_SOURCE") or None,
        help="local folder or s3://bucket/prefix (defaults to the CAPSULE_SOURCE env var)",
    )
    parser.add_argument(
        "--capsule-key",
        default=os.getenv("CAPSULE_KEY") or None,
        help="relative capsule key/prefix to pull before startup (defaults to CAPSULE_KEY)",
    )
    parser.add_argument("--startup-token", type=str, default="")
    parser.add_argument("--safe-execute", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--kernel-memory-limit-mb",
        type=int,
        help="per-Jupyter-process RLIMIT_AS ceiling; leaves the HTTP server outside the limit",
    )
    parser.add_argument(
        "--bundle-layout",
        choices=sorted(_BUNDLE_LAYOUTS),
        default=os.getenv("HYPOTEST_BUNDLE_LAYOUT", "none"),
        help="large-bundle filesystem layout baked into the image",
    )
    bundle_root = os.getenv("HYPOTEST_BUNDLE_ROOT")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(bundle_root) if bundle_root else None,
        help="collection image root (defaults to HYPOTEST_BUNDLE_ROOT)",
    )
    parser.add_argument(
        "--bundle-capsule-id",
        default=os.getenv("HYPOTEST_BUNDLE_CAPSULE_ID"),
        help="collection member to project (defaults to HYPOTEST_BUNDLE_CAPSULE_ID)",
    )
    mounted_capsule_root = os.getenv("HYPOTEST_MOUNTED_CAPSULE_ROOT")
    parser.add_argument(
        "--mounted-capsule-root",
        type=Path,
        default=Path(mounted_capsule_root) if mounted_capsule_root else None,
        help="cluster-mounted capsule collection root (defaults to HYPOTEST_MOUNTED_CAPSULE_ROOT)",
    )
    parser.add_argument(
        "--mounted-capsule-id",
        default=os.getenv("HYPOTEST_MOUNTED_CAPSULE_ID"),
        help="mounted collection member to copy (defaults to HYPOTEST_MOUNTED_CAPSULE_ID)",
    )
    parser.add_argument(
        "--pip-index-url",
        type=str,
        default="http://127.0.0.1:8723/simple",
        help="pip index for agent installs (the runtime cutoff proxy); empty string disables it",
    )
    parser.add_argument(
        "--no-install-shim",
        action="store_false",
        dest="install_shim_enabled",
        default=True,
        help="run package managers directly while retaining workspace-scoped pip/R install paths",
    )
    args = parser.parse_args()
    language = NBLanguage.PYTHON if args.language == "python" else NBLanguage.R
    asyncio.run(
        run_server(
            args.work_dir,
            language,
            args.capsule_source,
            args.port,
            args.startup_token,
            safe_execute=args.safe_execute,
            pip_index_url=args.pip_index_url or None,
            seed=args.seed,
            bundle_layout=args.bundle_layout,
            bundle_root=args.bundle_root,
            bundle_capsule_id=args.bundle_capsule_id,
            capsule_key=args.capsule_key,
            mounted_capsule_root=args.mounted_capsule_root,
            mounted_capsule_id=args.mounted_capsule_id,
            install_shim_enabled=args.install_shim_enabled,
            kernel_memory_limit_mb=args.kernel_memory_limit_mb,
        )
    )


if __name__ == "__main__":
    main()
