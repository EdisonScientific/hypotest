# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""LocalSandbox — the in-process backend.

Runs the Jupyter kernel directly in the host process via `Interpreter` (no
container, no HTTP); `list_dir` reads the host work_dir through `FilesystemTool`.
"""

from __future__ import annotations

from hypotest.env.interpreter import ExecutionResult, Interpreter
from hypotest.env.sandbox.base import Sandbox, SandboxConfig
from hypotest.env.tools.filesystem import FilesystemTool


class LocalSandbox(Sandbox):
    """In-process execution backend wrapping `Interpreter`."""

    def __init__(self, config: SandboxConfig) -> None:
        self.work_dir = config.work_dir
        self.language = config.language
        self._interpreter = Interpreter(
            work_dir=config.work_dir,
            language=config.language,
            execution_timeout=config.execution_timeout,
            use_host_env_vars=config.use_host_env_vars,
            extra_envs=config.extra_envs or None,
            seed=config.seed,
        )
        self._filesystem = FilesystemTool(config.work_dir)

    async def start(self) -> None:
        await self._interpreter.start()

    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109, ARG002
        return await self._interpreter.execute_code(code, timeout)

    async def reset(self) -> None:
        await self._interpreter.reset()

    async def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        return self._filesystem.list_dir(directory, max_files, show_hidden)

    async def close(self) -> None:
        await self._interpreter.close()

    async def health(self) -> bool:
        return self._interpreter.is_ready
