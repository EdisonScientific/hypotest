# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SandboxScheduler — placement + fallback, above the Sandbox interface.

The scheduler is where load-balancing across sandbox clusters and the
`[k8s…, enroot]` fallback chain live, so the `Sandbox` impls stay
backend/cluster-agnostic. `acquire()` returns a STARTED sandbox (it must start
each candidate to detect a `NoCapacityError` and fall through to the next).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import replace

from hypotest.env.sandbox.base import CapsuleRef, ResourceSpec, Sandbox, SandboxConfig
from hypotest.env.sandbox.factory import make_sandbox
from hypotest.env.sandbox.k8s import K8sSandbox, K8sSandboxSpec, NoCapacityError, warmpool_ready_replicas
from hypotest.env.sandbox.opensandbox import (
    OpenSandboxSandbox,
    OpenSandboxSpec,
    OpenSandboxUnavailableError,
)

logger = logging.getLogger(__name__)


class SandboxScheduler(ABC):
    @abstractmethod
    async def acquire(self, ref: CapsuleRef, resources: ResourceSpec) -> Sandbox:
        """Select a placement, START a sandbox for the capsule ref, and return it ready."""


class StaticSandboxScheduler(SandboxScheduler):
    """Single fixed backend (local/docker/enroot) — preserves today's behavior."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    async def acquire(self, ref: CapsuleRef, resources: ResourceSpec) -> Sandbox:
        sandbox = make_sandbox(replace(self._config, ref=ref, resources=resources))
        await sandbox.start()
        return sandbox


class K8sFallbackScheduler(SandboxScheduler):
    """Place across k8s clusters by stateless power-of-two-choices; fall back to colocated enroot.

    Holds NO placement state (no cursor, no cached load), so it is correct at any replica count: each
    `acquire` samples two specs at random, reads each cluster's free warm-pod count (readyReplicas)
    live, and prefers the freer — the k8s API is the only shared substrate. P2C's randomization
    spreads load across independent envs/replicas and tolerates the signal's reconciliation lag,
    where a strict least-loaded read would herd. The chosen order is still tried in full on
    `NoCapacityError`, then the `enroot_config` backend as a last resort.
    """

    def __init__(
        self,
        k8s_config: SandboxConfig,
        specs: list[K8sSandboxSpec],
        enroot_config: SandboxConfig,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._k8s_config = k8s_config
        self._specs = specs
        self._enroot_config = enroot_config
        # Per-instance RNG; a fresh Random() is OS-seeded, so per-env schedulers draw independently
        # (good for spread). Tests inject a seeded Random for determinism.
        self._rng = rng or random.Random()

    async def acquire(self, ref: CapsuleRef, resources: ResourceSpec) -> Sandbox:
        k8s_cfg = replace(self._k8s_config, ref=ref, resources=resources)
        for spec in await self._placement_order():
            sandbox = K8sSandbox(k8s_cfg, spec)
            try:
                await sandbox.start()
            except NoCapacityError as e:
                logger.warning("k8s warmpool %s unavailable (%s); trying next placement", spec.warmpool, e)
                with contextlib.suppress(Exception):
                    await sandbox.close()
            else:
                return sandbox
        logger.warning("all %d k8s placement(s) exhausted; falling back to colocated enroot", len(self._specs))
        fallback = make_sandbox(replace(self._enroot_config, ref=ref, resources=resources))
        await fallback.start()
        return fallback

    async def _placement_order(self) -> list[K8sSandboxSpec]:
        """Stateless power-of-two-choices: sample 2 specs, prefer the one with more free warm pods.

        Remaining specs (for >2 clusters) follow in random order so the NoCapacity fallthrough still
        covers every placement. With <=1 spec there is no choice to make (and no load read).
        """
        specs = self._specs
        if len(specs) <= 1:
            return list(specs)
        a, b = self._rng.sample(specs, 2)
        ra, rb = await asyncio.gather(self._ready_replicas(a), self._ready_replicas(b))
        primary, secondary = (a, b) if ra >= rb else (b, a)
        rest = [s for s in specs if s is not a and s is not b]
        self._rng.shuffle(rest)
        return [primary, secondary, *rest]

    async def _ready_replicas(self, spec: K8sSandboxSpec) -> int:
        """Free warm-pod count for `spec`'s cluster (the P2C signal). Test seam — overridden in unit tests."""
        return await warmpool_ready_replicas(spec)


class OpenSandboxFallbackScheduler(SandboxScheduler):
    """Prefer a remote OpenSandbox server, then use the locally staged backend.

    Only placement/reachability failures trigger fallback. Protocol skew and
    capsule-loading errors remain visible because falling back would otherwise
    hide a broken image or data configuration.
    """

    def __init__(
        self,
        remote_config: SandboxConfig,
        spec: OpenSandboxSpec,
        fallback_config: SandboxConfig,
    ) -> None:
        self._remote_config = remote_config
        self._spec = spec
        self._fallback_config = fallback_config

    async def acquire(self, ref: CapsuleRef, resources: ResourceSpec) -> Sandbox:
        remote = OpenSandboxSandbox(
            replace(self._remote_config, ref=ref, resources=resources),
            self._spec,
        )
        try:
            await remote.start()
        except OpenSandboxUnavailableError as exc:
            logger.warning("OpenSandbox unavailable (%s); falling back to the locally staged backend", exc)
            with contextlib.suppress(Exception):
                await remote.close()
        else:
            return remote

        fallback = make_sandbox(replace(self._fallback_config, ref=ref, resources=resources))
        await fallback.start()
        return fallback
