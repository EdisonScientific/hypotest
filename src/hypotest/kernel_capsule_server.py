"""Kernel server with init-time and on-request capsule loading.

Runs the standalone ``KernelServer`` (``env/kernel_server.py``) and adds a
``POST /load_capsule`` endpoint that pulls an exact capsule key first (with a
legacy most-recent/UUID fallback) from a local folder or
``s3://bucket/prefix`` into the kernel's work dir, then resets the kernel to
that workspace. When both
``CAPSULE_SOURCE`` and ``CAPSULE_KEY`` are present, the selected capsule is
pulled during process initialization *before* Jupyter and the HTTP health server
start. OpenSandbox uses that path so readiness means data and kernel are ready.

A source is optional for large-bundle images. A single-capsule image already has
task data under ``/workspace``; a collection image stores every capsule below
``/opt/hypotest/capsules`` and projects the requested one into ``/workspace``
before starting the kernel. Bundled layouts ignore init-pull settings.

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


def resolve_collection_capsule(bundle_root: Path, capsule_id: str) -> Path:
    """Resolve an exact or legacy-named capsule without allowing traversal."""
    root = bundle_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"bundle root is not a directory: {root}")
    try:
        relative = Path(capsule_id)
    except ValueError as exc:
        raise ValueError("bundle capsule id is not a valid filesystem path") from exc
    if not capsule_id or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe bundle capsule id: {capsule_id!r}")

    names = [relative]
    if len(relative.parts) == 1:
        names.extend((Path(f"CapsuleData-{capsule_id}"), Path(f"capsule_{capsule_id}")))
    for name in names:
        try:
            candidate = (root / name).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if candidate == root or root not in candidate.parents:
            continue
        if candidate.is_dir():
            return candidate
    attempted = ", ".join(str(name) for name in names)
    raise FileNotFoundError(f"no capsule for {capsule_id!r} under {root} (tried: {attempted})")


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
    install_shim_enabled: bool = True,
) -> None:
    selected, object_count = await prepare_initial_workspace(
        work_dir,
        capsule_source=capsule_source,
        capsule_key=capsule_key,
        bundle_layout=bundle_layout,
        bundle_root=bundle_root,
        bundle_capsule_id=bundle_capsule_id,
    )
    if selected is not None:
        logger.warning(
            "Prepared capsule %s in %s before kernel startup (%d downloaded objects)",
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
    server = KernelServer(work_dir, language, startup_token=startup_token, safe_execute=safe_execute, seed=seed)
    await server.start()
    app = create_app(server)

    @app.post("/load_capsule")
    async def load_capsule(req: LoadCapsuleRequest) -> LoadCapsuleResponse:
        if capsule_source is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This kernel image has no capsule source; use a bundled workspace or set CAPSULE_SOURCE",
            )
        logger.warning("Loading capsule %s from %s", req.capsule_uuid, capsule_source)
        # Stop the kernel (release its files in work_dir), swap in the capsule,
        # then restart the kernel in the repopulated workspace. The pull is
        # offloaded so a large download does not block the event loop.
        if req.seed is not None:
            server.seed = req.seed
        await server.close()
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
            install_shim_enabled=args.install_shim_enabled,
        )
    )


if __name__ == "__main__":
    main()
