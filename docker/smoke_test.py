"""In-process smoke test for the bundled dataset-server image.

Exercises the production k8s path end-to-end without any inner container:
an InterpreterEnv with use_docker/use_enroot/use_ray all False launches a
Jupyter kernel in-process (from kernel_env's python3 kernelspec) and runs a
cell. Verifies that (a) hypotest is importable, (b) the kernelspec is
discoverable in the single-env layout, (c) the in-process kernel executes, and
(d) numpy imports.

Run inside the image:
    docker run --rm -v "$(pwd)/docker/smoke_test.py:/smoke.py" \
        hypotest-server:core /app/kernel_env/bin/python /smoke.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from hypotest.env.config import ExecutionConfig
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance
from hypotest.env.utils import NBLanguage


async def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="smoke-"))
    problem = ProblemInstance(
        id=uuid4(),
        hypothesis="smoke-test hypothesis",
        protocol="smoke-test protocol",
        answer=True,  # alias for `accepted`
        rubric="n/a",
        max_points=1,  # alias for `max_score`
    )
    config = InterpreterEnvConfig(
        language=NBLanguage.PYTHON,
        use_ray=False,
        use_docker=False,
        use_enroot=False,
        execution_config=ExecutionConfig(safe_execute=True, cell_execution_timeout=120),
    )
    env = InterpreterEnv(problem=problem, work_dir=work_dir, rubric_model=None, config=config)

    await env.reset()
    # Import the core scientific stack (not just numpy): conda-forge binaries
    # like scipy/sklearn are the ones that trip libstdc++/GLIBCXX issues when the
    # env isn't on the loader path, so they are the meaningful in-process check.
    cell = (
        "import numpy, scipy, pandas, sklearn\n"
        "print('IMPORTS_OK',\n"
        "      'numpy', numpy.__version__, 'scipy', scipy.__version__,\n"
        "      'pandas', pandas.__version__, 'sklearn', sklearn.__version__)"
    )
    try:
        out = str(await env.run_cell(cell))
    finally:
        await env.close()

    print("CELL OUTPUT:", out)
    if "IMPORTS_OK" not in out:
        print("SMOKE FAIL: scientific stack did not import cleanly in-process", file=sys.stderr)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
