"""Tests for kernel_server: the /list_dir endpoint + protocol_version (no kernel needed)."""

import httpx
import pytest

from hypotest.env.kernel_server import (
    PROTOCOL_VERSION,
    HealthResponse,
    KernelServer,
    NBLanguage,
    ResetResponse,
    create_app,
)
from hypotest.env.tools.filesystem import list_dir_tool


def _server(work_dir):
    return KernelServer(work_dir, NBLanguage.PYTHON)


def test_list_dir_lists_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")
    out = _server(tmp_path).list_dir(".")
    assert "a.txt" in out
    assert "sub/b.txt" in out


def test_list_dir_truncates(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    out = _server(tmp_path).list_dir(".", max_files=2)
    assert "more files not shown" in out


def test_list_dir_confines_to_workspace(tmp_path):
    out = _server(tmp_path).list_dir("../../../etc")
    assert "must stay within the workspace root" in out


def test_list_dir_matches_filesystem_tool(tmp_path):
    """The inlined walk must agree with env.tools.filesystem.list_dir_tool (drift guard)."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.txt").write_text("x")
    (tmp_path / "c.txt").write_text("y")
    assert _server(tmp_path).list_dir(".") == list_dir_tool(str(tmp_path))


def test_health_response_carries_protocol_version():
    health = HealthResponse(status="OK", startup_token="t", kernel_ready=True)  # noqa: S106
    assert health.protocol_version == PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_list_dir_and_health_endpoints(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n")
    transport = httpx.ASGITransport(app=create_app(_server(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/list_dir", params={"directory": ".", "max_files": 20, "show_hidden": False})
        assert listing.status_code == 200
        assert "data.csv" in listing.json()["listing"]

        health = await client.get("/health")
        assert health.json()["protocol_version"] == PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_reset_endpoint_accepts_seed(tmp_path, monkeypatch):
    server = _server(tmp_path)
    captured: list[int | None] = []

    async def fake_reset(seed=None):  # noqa: RUF029
        captured.append(seed)
        return ResetResponse(success=True, seed=seed)

    monkeypatch.setattr(server, "reset", fake_reset)
    transport = httpx.ASGITransport(app=create_app(server))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/reset", json={"seed": 987})
        unseeded_response = await client.post("/reset")

    assert response.status_code == 200
    assert unseeded_response.status_code == 200
    assert captured == [987, None]
