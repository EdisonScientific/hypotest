"""System tests for InterpreterEnv end-to-end execution.

These tests verify that code execution produces expected outputs, including
generated files and notebook state. Tests are parameterized for both local
and Docker execution modes.
"""

import hashlib
import json
import pathlib
import shutil
import tempfile

import pytest

from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance
from tests.conftest import should_skip_docker_test

# Set to True to regenerate expected assets in tests/assets/system_test/
# After regenerating, set back to False and commit the assets
SAVE_EXPECTED_OUTPUTS = False

ASSETS_DIR = pathlib.Path(__file__).parent / "assets" / "system_test"


def compute_md5(content: bytes) -> str:
    """Compute MD5 hash of bytes content."""
    return hashlib.md5(content).hexdigest()  # noqa: S324


def normalize_notebook(nb_path: pathlib.Path) -> bytes:
    """Extract deterministic content from notebook for comparison.

    Extracts only cell sources and text outputs, ignoring timestamps,
    execution counts, and other metadata that varies between runs.
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    normalized = []
    for cell in nb.get("cells", []):
        cell_data = {"source": cell.get("source", "")}
        text_outputs = []
        for output in cell.get("outputs", []):
            if "text" in output:
                text_outputs.append(output["text"])
            elif output.get("output_type") == "stream":
                text_outputs.append(output.get("text", ""))
        if text_outputs:
            cell_data["text_outputs"] = text_outputs
        normalized.append(cell_data)
    return json.dumps(normalized, sort_keys=True).encode()


def compare_files(actual_dir: pathlib.Path, expected_dir: pathlib.Path) -> None:
    """Compare actual outputs against expected files using MD5 checksums."""
    # Notebook: normalize both, compare MD5
    actual_nb = normalize_notebook(actual_dir / "notebook.ipynb")
    expected_nb = normalize_notebook(expected_dir / "notebook.ipynb")
    actual_nb_md5 = compute_md5(actual_nb)
    expected_nb_md5 = compute_md5(expected_nb)
    assert actual_nb_md5 == expected_nb_md5, (
        f"Notebook MD5 mismatch: actual={actual_nb_md5}, expected={expected_nb_md5}"
    )

    # Text file: direct comparison
    actual_txt = (actual_dir / "output.txt").read_bytes()
    expected_txt = (expected_dir / "output.txt").read_bytes()
    actual_txt_md5 = compute_md5(actual_txt)
    expected_txt_md5 = compute_md5(expected_txt)
    assert actual_txt_md5 == expected_txt_md5, (
        f"output.txt MD5 mismatch: actual={actual_txt_md5}, expected={expected_txt_md5}"
    )


class TestSystemExecution:
    """System tests for end-to-end code execution and file generation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_docker", [False, True])
    async def test_code_execution_and_file_creation(self, default_problem: ProblemInstance, use_docker: bool) -> None:
        """Test basic code execution and file creation.

        Executes simple arithmetic and creates a text file, then verifies
        the save_dir contains the expected notebook and output file.
        """
        if should_skip_docker_test(use_docker):
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as work_tmp, tempfile.TemporaryDirectory() as save_tmp:
            work_dir = pathlib.Path(work_tmp)
            save_dir = pathlib.Path(save_tmp) / "output"

            env = InterpreterEnv(
                problem=default_problem,
                work_dir=work_dir,
                save_dir=save_dir,
                config=InterpreterEnvConfig(use_docker=use_docker),
            )
            await env.reset()

            # Execute code: basic arithmetic with print
            await env.run_cell("x = 2 + 2\nprint(x)")

            # Execute code: create a text file
            await env.run_cell("with open('output.txt', 'w') as f:\n    f.write('hello world')")

            await env.close()

            # Verify files exist
            assert (save_dir / "notebook.ipynb").exists(), "notebook.ipynb not found in save_dir"
            assert (save_dir / "output.txt").exists(), "output.txt not found in save_dir"

            if SAVE_EXPECTED_OUTPUTS:
                ASSETS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(save_dir / "notebook.ipynb", ASSETS_DIR / "notebook.ipynb")
                shutil.copy(save_dir / "output.txt", ASSETS_DIR / "output.txt")
                pytest.skip(
                    "Saved expected outputs to tests/assets/system_test/ - rerun with SAVE_EXPECTED_OUTPUTS=False"
                )
            else:
                compare_files(save_dir, ASSETS_DIR)
