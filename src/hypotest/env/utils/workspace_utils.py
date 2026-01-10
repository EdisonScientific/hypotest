import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir_exists(dest: Path, dir_name: str) -> Path:
    """Create the directory if it doesn't exist."""
    dir_path = dest / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_workspace_path(workspace_path: Path) -> None:
    """Validate that workspace path exists."""
    if not workspace_path.exists():
        raise ValueError(f"Workspace path does not exist: {workspace_path}")

    workspace_files = list(workspace_path.iterdir())
    if not workspace_files:
        logger.warning(f"No files found in workspace path: {workspace_path}")

    # TODO: check that config and packages directories exist?
