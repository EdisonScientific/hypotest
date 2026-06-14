# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""DockerSandbox — kernel server in an aiodocker container, reached over HTTP.

The workspace is bind-mounted (`work_dir:/data_workspace`), so `list_dir` reads
the host `work_dir` directly (the kernel-server `/list_dir` cutover waits for an
image rebuild — ADR §migration / PR d).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

import aiodocker
import httpx

from hypotest.env import config as cfg
from hypotest.env.interpreter import ExecutionResult
from hypotest.env.sandbox.base import (
    _CONTAINER_LOG_LEVEL,
    _USED_PORTS,
    Sandbox,
    SandboxConfig,
    _poll_kernel_health,
    get_free_port,
    used_ports_lock,
)
from hypotest.env.sandbox.http_client import HttpKernelClient
from hypotest.env.tools.filesystem import FilesystemTool

logger = logging.getLogger(__name__)


class DockerSandbox(Sandbox):
    """Docker-container execution backend."""

    def __init__(self, config: SandboxConfig) -> None:
        self.work_dir = config.work_dir
        self.language = config.language
        self._execution_timeout = config.execution_timeout
        self._safe_execute = config.safe_execute
        self._docker_client: aiodocker.Docker | None = None
        self._container: Any = None
        self._container_port: int | None = None
        self._client: HttpKernelClient | None = None
        self._filesystem = FilesystemTool(config.work_dir)

    def _label(self) -> str:
        return f"docker(port={self._container_port or '?'})"

    def _read_log_tail(self, max_chars: int = 2000) -> str:  # noqa: ARG002
        # Docker logs are retrieved via the API, not a host file; the startup
        # poller only uses this for diagnostics, so an empty tail is acceptable.
        return ""

    async def start(self) -> None:
        self._docker_client = aiodocker.Docker()
        self._container_port = await get_free_port()
        startup_token = str(uuid.uuid4())

        cmd_list = [
            "/app/kernel_env/bin/python",
            "/envs/kernel_server.py",
            "--work_dir",
            "/data_workspace",
            "--language",
            self.language.value,
            "--startup-token",
            startup_token,
        ]
        if self._safe_execute:
            cmd_list += ["--safe-execute"]

        docker_config = {
            "Image": cfg.NB_ENVIRONMENT_DOCKER_IMAGE,
            "Cmd": cmd_list,
            "HostConfig": {
                "Binds": [f"{self.work_dir}:/data_workspace"],
                "PortBindings": {f"{cfg.KERNEL_SERVER_PORT}/tcp": [{"HostPort": str(self._container_port)}]},
            },
            "WorkingDir": "/data_workspace",
            "Tty": True,
            "ExposedPorts": {f"{cfg.KERNEL_SERVER_PORT}/tcp": {}},
        }

        self._container = await self._docker_client.containers.run(config=cast(dict[str, Any], docker_config))
        logger.log(_CONTAINER_LOG_LEVEL, "Started docker container on port %s", self._container_port)

        http = httpx.AsyncClient(
            base_url=f"http://localhost:{self._container_port}",
            timeout=httpx.Timeout(self._execution_timeout + 10, connect=30.0),
        )
        self._client = HttpKernelClient(
            http.request, execution_timeout=self._execution_timeout, label=self._label(), owns=http
        )
        await _poll_kernel_health(
            request=http.request,
            enroot_proc=None,
            container_port=self._container_port,
            expected_startup_token=startup_token,
            read_log_tail=self._read_log_tail,
            label=self._label(),
        )

    async def execute(self, code: str, timeout: float | None = None, req_uuid: str = "") -> ExecutionResult:  # noqa: ASYNC109
        assert self._client is not None
        return await self._client.execute(code, timeout, req_uuid)

    async def reset(self) -> None:
        assert self._client is not None
        await self._client.reset()

    async def list_dir(self, directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
        return self._filesystem.list_dir(directory, max_files, show_hidden)

    async def health(self) -> bool:
        return await self._client.health() if self._client is not None else False

    async def close(self) -> None:
        if self._container_port is not None:
            async with used_ports_lock:
                _USED_PORTS.discard(self._container_port)
            self._container_port = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._container is not None:
            try:
                await self._container.stop()
                await self._container.delete()
            except Exception as e:
                logger.warning("Failed to stop/delete container: %s", e)
            self._container = None
        if self._docker_client is not None:
            await self._docker_client.close()
            self._docker_client = None
