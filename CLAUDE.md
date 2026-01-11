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

# Linting (auto-runs via pre-commit)
ruff check --fix src/
ruff format src/

# Build Docker image
make image
```

## Architecture

```
src/hypotest/env/
├── config.py           # ExecutionConfig with profiles: standard, gpu, long_timeout
├── interpreter.py      # Interpreter class - Jupyter kernel lifecycle & code execution
├── interpreter_env.py  # InterpreterEnv - lightweight env for standalone execution
├── kernel_server.py    # Kernel server management
├── prompts.py          # System prompts & capability descriptions
├── tools/
│   └── filesystem.py   # File I/O tools (read/write/edit) with format support
└── utils/
    ├── core.py         # XML/markdown code extraction
    ├── img_utils.py    # Image encoding/compression
    ├── notebook_utils.py  # Cell execution, NBLanguage enum (PYTHON, R)
    └── workspace_utils.py # Workspace management

tests/
├── conftest.py              # Shared fixtures
├── test_interpreter.py      # Interpreter class tests
└── test_interpreter_env.py  # InterpreterEnv tests
```

**Key patterns:**

- `ExecutionResult` stores notebook outputs in nbformat as single source of truth
- `ExecutionConfig` uses factory pattern with deployment profiles
- Tools use `fhaviary` (aviary.core) for Message/Tool abstractions
- Async throughout - uses jupyter_client's async APIs

## Configuration

**Environment variables:**

- `DEPLOYMENT_PROFILE`: standard (default), gpu, or long_timeout
- `USE_DOCKER`: Enable Docker-based execution (default: false)
- `NB_ENVIRONMENT_DOCKER_IMAGE`: Docker image name (default: interpreter-env:latest)
- `AGENT_MAX_STEPS`: Max agent steps (default: 30)

**File limits:** 256KB text, 10MB PDF/PowerPoint, 3000 char notebook output

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
