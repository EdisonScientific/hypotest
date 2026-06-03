"""Kernel server with on-request capsule loading (bundled-image entrypoint).

Runs the standalone ``KernelServer`` (``env/kernel_server.py``) and adds a
``POST /load_capsule`` endpoint that pulls the *most-recent* capsule for a given
UUID from a configured source — a local folder or ``s3://bucket/prefix`` — into
the kernel's work dir, then resets the kernel to that workspace.

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
from pydantic import BaseModel

from hypotest import s3_sync
from hypotest.env import install_shim
from hypotest.env.kernel_server import KernelServer, NBLanguage, create_app

logger = logging.getLogger(__name__)


class LoadCapsuleRequest(BaseModel):
    capsule_uuid: str


class LoadCapsuleResponse(BaseModel):
    success: bool
    capsule_uuid: str
    objects: int


def _clear_dir(path: Path) -> None:
    """Remove the contents of ``path`` (keeping the dir itself)."""
    for child in list(path.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _apply_workspace_env(work_dir: Path) -> None:
    """Put the workspace install model on the env the kernel will inherit.

    Mirrors the enroot bash's exports (same ``install_shim.workspace_env`` source),
    so an agent gets the same PATH / PYTHONPATH / pip / R behavior on either backend.
    """
    for key, seg in install_shim.workspace_env(str(work_dir)).items():
        if key in {"PYTHONPATH", "PATH"}:
            existing = os.environ.get(key, "")
            os.environ[key] = f"{seg}{os.pathsep}{existing}" if existing else seg
        else:
            os.environ[key] = seg


async def run_server(
    work_dir: Path,
    language: NBLanguage,
    capsule_source: str,
    port: int = 8000,
    startup_token: str = "",
    safe_execute: bool = True,
    pip_index_url: str | None = None,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    # Lay down the same workspace install model as the enroot path (shim + pydeps +
    # pip.conf + R) so the agent's install/import behavior matches on either backend.
    # The workspace pip.conf overrides /etc/pip.conf, so re-state the cutoff index-url.
    install_shim.write_workspace_config(work_dir, str(work_dir), index_url=pip_index_url)
    _apply_workspace_env(work_dir)
    server = KernelServer(work_dir, language, startup_token=startup_token, safe_execute=safe_execute)
    await server.start()
    app = create_app(server)

    @app.post("/load_capsule")
    async def load_capsule(req: LoadCapsuleRequest) -> LoadCapsuleResponse:
        logger.warning("Loading most-recent capsule %s from %s", req.capsule_uuid, capsule_source)
        # Stop the kernel (release its files in work_dir), swap in the capsule,
        # then restart the kernel in the repopulated workspace. The pull is
        # offloaded so a large download does not block the event loop.
        await server.close()
        _clear_dir(work_dir)
        count = await asyncio.to_thread(s3_sync.pull_latest_capsule, capsule_source, req.capsule_uuid, work_dir)
        # Clearing wiped the scaffolding; re-lay it alongside the freshly pulled capsule.
        install_shim.write_workspace_config(work_dir, str(work_dir), index_url=pip_index_url)
        await server.start()
        logger.warning("Loaded capsule %s (%d objects) into %s", req.capsule_uuid, count, work_dir)
        return LoadCapsuleResponse(success=True, capsule_uuid=req.capsule_uuid, objects=count)

    config = uvicorn.Config(app, host="0.0.0.0", port=port, loop="asyncio")  # noqa: S104
    await uvicorn.Server(config).serve()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Kernel server with on-request capsule loading")
    parser.add_argument("--work_dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--language", type=str, default="python", choices=["python", "r"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--capsule-source", type=str, required=True, help="local folder or s3://bucket/prefix")
    parser.add_argument("--startup-token", type=str, default="")
    parser.add_argument("--safe-execute", action="store_true")
    parser.add_argument(
        "--pip-index-url",
        type=str,
        default="http://127.0.0.1:8723/simple",
        help="pip index for agent installs (the runtime cutoff proxy); empty string disables it",
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
        )
    )


if __name__ == "__main__":
    main()
