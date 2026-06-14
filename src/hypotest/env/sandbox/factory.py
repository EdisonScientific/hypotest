# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""make_sandbox — the single place the legacy backend booleans pick an impl.

This is the only code post-flip (PR b) that interprets `use_docker`/`use_enroot`/
`use_ray`. K8s is selected by the `SandboxScheduler` (PR e), not here.
"""

from __future__ import annotations

from hypotest.env.sandbox.base import Sandbox, SandboxConfig
from hypotest.env.sandbox.docker import DockerSandbox
from hypotest.env.sandbox.enroot import EnrootSandbox
from hypotest.env.sandbox.local import LocalSandbox


def make_sandbox(config: SandboxConfig) -> Sandbox:
    """Build the Sandbox for a config's backend selectors (enroot > docker > local)."""
    if config.use_enroot:
        return EnrootSandbox(config)
    if config.use_docker:
        return DockerSandbox(config)
    return LocalSandbox(config)
