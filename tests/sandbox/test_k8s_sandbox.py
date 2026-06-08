"""Unit tests for K8sSandbox.

Most tests bypass `_allocate` and ride a stub connector (no SDK, no cluster). The exception is
`test_allocate_passes_per_spec_kubeconfig_context`, which drives the real `_allocate` with the SDK's
`AsyncSandboxClient` faked at its import site to assert the multi-cluster wiring.
"""

from importlib.util import find_spec

import httpx
import pytest

import hypotest.env.config as cfgmod
from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import CapsuleRef, SandboxConfig
from hypotest.env.sandbox.k8s import K8sSandbox, K8sSandboxSpec, NoCapacityError


def _ok_handler(method, endpoint, **kwargs):
    if endpoint == "/health":
        return httpx.Response(200, json={"protocol_version": 1})
    if endpoint == "/load_capsule":
        return httpx.Response(200, json={"objects": 3})
    if endpoint == "/execute":
        return httpx.Response(200, json={"notebook_outputs": [], "error_occurred": False, "execution_time": 0.0})
    if endpoint == "/reset":
        return httpx.Response(200, json={"success": True})
    if endpoint == "/list_dir":
        return httpx.Response(200, json={"listing": "a.txt"})
    return httpx.Response(404)


def _k8s(tmp_path, fake, ref=None):
    config = SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON, ref=ref or CapsuleRef())
    sandbox = K8sSandbox(config, K8sSandboxSpec(template="py-sandbox", warmpool="wp"))

    # bypass the real SDK; return the injected fake AsyncSandbox
    async def _alloc():  # noqa: RUF029
        return fake

    sandbox._allocate = _alloc  # type: ignore[method-assign]
    return sandbox


@pytest.mark.asyncio
async def test_start_claims_waits_health_loads_capsule(tmp_path, make_fake_sandbox):
    fake = make_fake_sandbox(_ok_handler)
    sandbox = _k8s(tmp_path, fake, ref=CapsuleRef(uuid="cap-1"))
    await sandbox.start()
    endpoints = [e for _, e in fake.connector.calls]
    assert "/health" in endpoints  # waited for the kernel
    assert "/load_capsule" in endpoints  # pulled the capsule (ref.uuid set)
    await sandbox.close()
    assert fake.terminated


@pytest.mark.asyncio
async def test_start_skips_load_capsule_without_uuid(tmp_path, make_fake_sandbox):
    fake = make_fake_sandbox(_ok_handler)
    sandbox = _k8s(tmp_path, fake, ref=CapsuleRef())
    await sandbox.start()
    assert "/load_capsule" not in [e for _, e in fake.connector.calls]
    await sandbox.close()


@pytest.mark.asyncio
async def test_execute_reset_list_dir_delegate(tmp_path, make_fake_sandbox):
    fake = make_fake_sandbox(_ok_handler)
    sandbox = _k8s(tmp_path, fake)
    await sandbox.start()
    result = await sandbox.execute("print(1)", req_uuid="r1")
    assert result.error_occurred is False
    await sandbox.reset()
    assert await sandbox.list_dir(".") == "a.txt"
    await sandbox.close()


@pytest.mark.asyncio
async def test_start_failure_terminates_pod(tmp_path, monkeypatch, make_fake_sandbox):
    # Kernel never becomes ready -> NoCapacityError, and the claimed pod is terminated (no leak).
    monkeypatch.setattr(cfgmod, "KERNEL_SERVER_STARTUP_TIMEOUT", 0.3)

    def never_ready(method, endpoint, **kwargs):
        return httpx.Response(503) if endpoint == "/health" else httpx.Response(200, json={})

    fake = make_fake_sandbox(never_ready)
    sandbox = _k8s(tmp_path, fake)
    with pytest.raises(NoCapacityError):
        await sandbox.start()
    assert fake.terminated


@pytest.mark.skipif(find_spec("k8s_agent_sandbox") is None, reason="agent-sandbox SDK not installed")
@pytest.mark.asyncio
async def test_allocate_passes_per_spec_kubeconfig_context(tmp_path, monkeypatch):
    """_allocate must thread the spec's kubeconfig/context into AsyncSandboxClient.

    That is the whole multi-cluster switch: each placement targets its OWN cluster's control plane.
    Without it, every claim would be created against the ambient / in-cluster kubeconfig.
    """
    import k8s_agent_sandbox  # noqa: PLC0415

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def create_sandbox(self, template, **kwargs):
            captured["template"] = template
            captured["create_kwargs"] = kwargs
            return object()

    monkeypatch.setattr(k8s_agent_sandbox, "AsyncSandboxClient", _FakeClient)

    config = SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON, ref=CapsuleRef())
    spec = K8sSandboxSpec(
        template="py-sandbox",
        warmpool="wp",
        connection="direct",
        api_url="http://cluster-a:31050",
        kubeconfig="/kube/cluster-a.yaml",
        context="cluster-a",
    )
    await K8sSandbox(config, spec)._allocate()

    assert captured["kubeconfig"] == "/kube/cluster-a.yaml"
    assert captured["context"] == "cluster-a"
    assert captured["template"] == "py-sandbox"
    assert captured["create_kwargs"]["warmpool"] == "wp"  # placement args flow through too
    assert captured["create_kwargs"]["namespace"] == "default"
    assert captured["connection_config"].api_url == "http://cluster-a:31050"  # data plane = same cluster
