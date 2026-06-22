# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

hypotest is a Python package that provides a Jupyter kernel-based code execution environment. It's part of the Edison Scientific platform ecosystem, designed for executing code with isolation and comprehensive notebook management.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
pytest tests/                             # all tests
pytest tests/test_foo.py::test_specific   # single test
pytest -n auto tests/                     # parallel execution

# Type checking
uv run mypy --scripts-are-modules

# Pre-commit checks (ruff, mypy, codespell, detect-secrets, prettier)
# prek only checks staged files, so `git add` first; re-stage after ruff auto-fixes
prek run --all-files

# Build Docker image
make image

# Run dataset server (for benchmarking)
make server CONFIG=server.yaml

# Run benchmark agent
uv run python src/hypotest/benchmark_agent.py benchmark.yaml
```

## Architecture

```
src/hypotest/
├── dataset_server.py   # Dataset(TaskDataset[InterpreterEnv]) + rubric grading; served via TaskDatasetServer
├── benchmark_agent.py  # Benchmark client using ldp RolloutManager
└── env/
    ├── config.py           # ExecutionConfig with profiles: standard, gpu, long_timeout
    ├── interpreter.py      # Interpreter class - Jupyter kernel lifecycle & code execution
    ├── interpreter_env.py  # InterpreterEnv (Environment subclass), ProblemInstance, InterpreterEnvConfig
    ├── notebook_env.py     # NotebookEnv(InterpreterEnv) - adds the run_cell tool (append/re-run by idx)
    ├── kernel_server.py    # FastAPI kernel server (KernelServer) + NBLanguage enum (PYTHON, R)
    ├── prompts.py          # System prompts, capability descriptions, RUBRIC_SCORE_PROMPT
    ├── tools/
    │   └── filesystem.py   # File I/O tools (read/write/edit) with format support
    └── utils/
        ├── core.py         # XML/markdown code extraction
        ├── img_utils.py    # Image encoding/compression
        ├── notebook_utils.py  # Cell execution helpers (re-exports NBLanguage from kernel_server)
        └── workspace_utils.py # Workspace management

tests/
├── conftest.py              # Shared fixtures (docker/R availability skips, default_problem)
├── test_interpreter.py      # Interpreter class tests
├── test_interpreter_env.py  # InterpreterEnv tests
├── test_dataset.py          # Dataset loading tests (hits the real HuggingFace Hub - needs network)
└── test_system.py           # End-to-end system tests
```

**Key patterns:**

- `ProblemInstance` (hypothesis, rubric, max_score) is the task unit; `Dataset.load_problems()` reads from either a HuggingFace dataset (`hf_dataset`) or a local `problem_jsonl`
- Grading is rubric-based: a `LiteLLMModel` (default `openai/gpt-5`) scores answers against the rubric via `RUBRIC_SCORE_PROMPT`; rewards are optionally normalized to `max_score`
- `ExecutionResult` stores notebook outputs in nbformat as single source of truth
- `ExecutionConfig` uses factory pattern with deployment profiles
- Tools use `fhaviary` (aviary.core) for Message/Tool abstractions
- Benchmarking uses `ldp` for agent rollouts and `aviary.core.TaskDatasetServer` for serving environments
- Async throughout - uses jupyter_client's async APIs

## Configuration

**Environment variables:**

- `DEPLOYMENT_PROFILE`: standard (default), gpu, or long_timeout
- `USE_DOCKER`: Enable Docker-based execution (default: false)
- `NB_ENVIRONMENT_DOCKER_IMAGE`: Docker image name (default: interpreter-env:latest)
- `AGENT_MAX_STEPS`: Max agent steps (default: 30)

**File limits:** 256KB text, 10MB PDF/PowerPoint, 3000 char notebook output

**Server/benchmark config:** `dataset_server.py` and `benchmark_agent.py` accept either a YAML config path (see README for schema) or equivalent CLI args. The dataset source is one of `hf_dataset` (e.g. `EdisonScientific/bixbench_hypothesis`) or a local `problem_jsonl`; both require a `capsule_dir` of task data. Note `DatasetConfig.use_docker` defaults to `True` (distinct from the `USE_DOCKER` env var above).

## CI

GitHub Actions workflow (`.github/workflows/tests.yml`) runs on PRs and pushes to main:

- Pre-commit checks (ruff, mypy, etc.)
- Pytest with parallel execution (`-n auto`)
- Matrix: Python 3.11 and 3.13

## Code Style

- Line length: 120 characters
- Docstrings: Google convention
- Type hints: Required, strict mypy checking with pydantic plugin
- Pre-commit hooks: ruff, mypy, codespell, detect-secrets, prettier
