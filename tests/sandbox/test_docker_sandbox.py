"""Unit tests for DockerSandbox (delegation + host-FS list_dir, no container)."""

import httpx
import pytest

from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import DockerSandbox, HttpKernelClient, SandboxConfig

_EXEC_OK = {
    "notebook_outputs": [{"output_type": "stream", "name": "stdout", "text": "hi\n"}],
    "error_occurred": False,
    "execution_time": 0.1,
}


def _docker(tmp_path):
    return DockerSandbox(SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON))


@pytest.mark.asyncio
async def test_execute_delegates_to_kernel_client(tmp_path, stub_request):
    calls: list[tuple[str, str]] = []

    def handler(method, endpoint, **kwargs):
        calls.append((method, endpoint))
        return httpx.Response(200, json=_EXEC_OK)

    sb = _docker(tmp_path)
    sb._client = HttpKernelClient(stub_request(handler))  # inject; bypass aiodocker start()
    result = await sb.execute("print('hi')", req_uuid="u1")
    assert "hi" in result.get_combined_text()
    assert ("POST", "/execute") in calls


@pytest.mark.asyncio
async def test_health_false_without_client_then_true(tmp_path, stub_request):
    sb = _docker(tmp_path)
    assert await sb.health() is False  # not started yet
    sb._client = HttpKernelClient(stub_request(lambda m, e, **kw: httpx.Response(200, json={"protocol_version": 1})))
    assert await sb.health() is True


@pytest.mark.asyncio
async def test_list_dir_reads_host_workspace(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    listing = await _docker(tmp_path).list_dir(".")
    assert "data.csv" in listing
