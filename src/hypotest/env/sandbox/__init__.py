# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The sandbox backend abstraction.

A uniform `Sandbox` interface with local/docker/enroot/k8s implementations behind
a `make_sandbox` factory and a `SandboxScheduler`. See
docs/adr/0001-sandbox-backend-abstraction.md.

`K8sSandbox` is safe to import without the optional `k8s_agent_sandbox` SDK — it
imports the SDK lazily inside `_allocate()`, so importing this package never
requires the `k8s` extra to be installed.
"""

from hypotest.env.sandbox.base import (
    CapsuleRef,
    CgroupsV2Limiter,
    NoopLimiter,
    PrlimitLimiter,
    ResourceLimiter,
    ResourceSpec,
    Sandbox,
    SandboxConfig,
)
from hypotest.env.sandbox.docker import DockerSandbox
from hypotest.env.sandbox.enroot import EnrootSandbox
from hypotest.env.sandbox.factory import make_sandbox
from hypotest.env.sandbox.http_client import HttpKernelClient, ProtocolVersionError, RequestFn
from hypotest.env.sandbox.k8s import K8sSandbox, K8sSandboxSpec, NoCapacityError
from hypotest.env.sandbox.local import LocalSandbox
from hypotest.env.sandbox.scheduler import K8sFallbackScheduler, SandboxScheduler, StaticSandboxScheduler

__all__ = [
    "CapsuleRef",
    "CgroupsV2Limiter",
    "DockerSandbox",
    "EnrootSandbox",
    "HttpKernelClient",
    "K8sFallbackScheduler",
    "K8sSandbox",
    "K8sSandboxSpec",
    "LocalSandbox",
    "NoCapacityError",
    "NoopLimiter",
    "PrlimitLimiter",
    "ProtocolVersionError",
    "RequestFn",
    "ResourceLimiter",
    "ResourceSpec",
    "Sandbox",
    "SandboxConfig",
    "SandboxScheduler",
    "StaticSandboxScheduler",
    "make_sandbox",
]
