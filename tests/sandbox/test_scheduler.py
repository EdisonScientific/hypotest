"""Unit tests for the sandbox schedulers (placement + k8s->enroot fallback)."""

import httpx
import pytest

from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import (
    CapsuleRef,
    K8sFallbackScheduler,
    K8sSandbox,
    K8sSandboxSpec,
    LocalSandbox,
    ResourceSpec,
    SandboxConfig,
    StaticSandboxScheduler,
)
from hypotest.env.sandbox.k8s import NoCapacityError


def _local_config(tmp_path):
    return SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON, execution_timeout=30)


def _healthy(method, endpoint, **kwargs):
    return httpx.Response(200, json={"protocol_version": 1} if endpoint == "/health" else {})


@pytest.mark.asyncio
async def test_static_scheduler_starts_single_backend(tmp_path):
    scheduler = StaticSandboxScheduler(_local_config(tmp_path))
    sandbox = await scheduler.acquire(CapsuleRef(), ResourceSpec())
    try:
        assert isinstance(sandbox, LocalSandbox)
        assert await sandbox.health() is True
    finally:
        await sandbox.close()


@pytest.mark.asyncio
async def test_fallback_to_enroot_when_all_k8s_at_capacity(tmp_path, monkeypatch):
    # Every k8s placement reports no capacity; the scheduler must try each, then fall back.
    attempts = {"n": 0}

    async def _no_capacity(self):  # noqa: RUF029
        attempts["n"] += 1
        raise NoCapacityError("warmpool empty")

    monkeypatch.setattr(K8sSandbox, "_allocate", _no_capacity)

    specs = [K8sSandboxSpec(template="py-sandbox", warmpool="a"), K8sSandboxSpec(template="py-sandbox", warmpool="b")]
    # The fallback config is local here so it starts in-process; in production it is a real enroot config.
    scheduler = K8sFallbackScheduler(_local_config(tmp_path), specs, _local_config(tmp_path))
    sandbox = await scheduler.acquire(CapsuleRef(), ResourceSpec())
    try:
        assert attempts["n"] == len(specs)  # every k8s placement was tried before falling back
        assert isinstance(sandbox, LocalSandbox)  # fell back to the (enroot) config backend
        assert await sandbox.health() is True
    finally:
        await sandbox.close()


@pytest.mark.asyncio
async def test_returns_k8s_sandbox_when_capacity_available(tmp_path, monkeypatch, make_fake_sandbox):
    # First placement has capacity -> a started K8sSandbox is returned (no fallback).
    fake = make_fake_sandbox(_healthy)

    async def _alloc(self):  # noqa: RUF029
        return fake

    monkeypatch.setattr(K8sSandbox, "_allocate", _alloc)
    scheduler = K8sFallbackScheduler(
        _local_config(tmp_path), [K8sSandboxSpec(template="py-sandbox", warmpool="a")], _local_config(tmp_path)
    )
    sandbox = await scheduler.acquire(CapsuleRef(), ResourceSpec())
    try:
        assert isinstance(sandbox, K8sSandbox)
        assert await sandbox.health() is True
    finally:
        await sandbox.close()
    assert fake.terminated  # close() terminated the claimed pod
