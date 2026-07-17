"""Unit tests for the raw OpenSandbox lifecycle backend."""

import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest

from hypotest.env.kernel_server import NBLanguage
from hypotest.env.sandbox import CapsuleRef, ResourceSpec, SandboxConfig
from hypotest.env.sandbox import opensandbox as opensandboxmod
from hypotest.env.sandbox.opensandbox import (
    OpenSandboxSandbox,
    OpenSandboxSpec,
    OpenSandboxUnavailableError,
)


class _FakeConnectionConfig:
    created: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs):
        type(self).created.append(kwargs)
        self.protocol = kwargs.get("protocol", "http")
        self._api_key = kwargs.get("api_key", "fake-env-api-key")

    def get_api_key(self):
        return self._api_key


class _FakePlatformSpec:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeImageAuth:
    def __init__(self, **kwargs):
        self.username = kwargs["username"]
        self.password = kwargs["password"]


class _FakeImageSpec:
    def __init__(self, *, image, auth):
        self.image = image
        self.auth = auth


class _FakeRemote:
    def __init__(self, endpoint: str):
        self.id = "sb-123"
        self.endpoint = SimpleNamespace(endpoint=endpoint, headers={"X-OpenSandbox-Route": "route-1"})
        self.killed = False
        self.closed = False

    async def get_endpoint(self, port):
        assert port == 8000
        return self.endpoint

    async def kill(self):
        self.killed = True

    async def close(self):
        self.closed = True


def _install_fakes(
    monkeypatch,
    endpoint="proxy.example/sandboxes/sb-123/proxy/8000",
    *,
    health_status=200,
):
    remote = _FakeRemote(endpoint)
    create_kwargs: dict = {}

    class _FakeSDK:
        @classmethod
        async def create(cls, **kwargs):
            create_kwargs.update(kwargs)
            return remote

    monkeypatch.setattr(
        opensandboxmod,
        "_require_opensandbox_sdk",
        lambda: (_FakeSDK, _FakeConnectionConfig, _FakePlatformSpec, _FakeImageAuth, _FakeImageSpec),
    )

    requests: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        requests.append(request)
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(health_status, json={"protocol_version": 2})
        if path.endswith("/load_capsule"):
            body = json.loads(request.content)
            return httpx.Response(200, json={"objects": 7, "seed": body.get("seed")})
        if path.endswith("/reset"):
            body = json.loads(request.content) if request.content else {}
            return httpx.Response(200, json={"success": True, "seed": body.get("seed")})
        if path.endswith("/execute") and request.method == "POST":
            return httpx.Response(202, json={"execution_id": "exec-1", "status": "queued"})
        if path.endswith("/execute/exec-1"):
            return httpx.Response(
                200,
                json={
                    "execution_id": "exec-1",
                    "status": "completed",
                    "result": {"notebook_outputs": [], "error_occurred": False, "execution_time": 0.1},
                },
            )
        if path.endswith("/list_dir"):
            return httpx.Response(200, json={"listing": "data.csv"})
        return httpx.Response(404)

    def client_factory(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(opensandboxmod.httpx, "AsyncClient", client_factory)
    return remote, create_kwargs, requests


@pytest.mark.asyncio
async def test_object_store_lifecycle_preserves_proxy_path_headers_and_resources(tmp_path, monkeypatch):
    _FakeConnectionConfig.created.clear()
    remote, create_kwargs, requests = _install_fakes(monkeypatch)
    config = SandboxConfig(
        work_dir=tmp_path,
        language=NBLanguage.PYTHON,
        execution_timeout=60,
        safe_execute=True,
        extra_envs={"EXTRA": "value"},
        resources=ResourceSpec(mem_mb=8192, cpu=1.5, disk_gib=20, gpu=1, gpu_type="L40S"),
        ref=CapsuleRef(source="s3://capsules/root", uuid="cap-1", delivery="object_store"),
        seed=123,
        job_id="run-9",
    )
    spec = OpenSandboxSpec(
        image="registry/kernel:latest",
        domain="sandbox.example:8080",
        api_key="fake-lifecycle-api-key",
        protocol="https",
        capsule_source="s3://unused/default",
        create_attempts=1,
        health_poll_interval_seconds=0.001,
        execution_poll_interval_seconds=0.001,
        extensions={"poolRef": "hypotest-pool"},
        platform_os="linux",
        platform_arch="amd64",
    )
    sandbox = OpenSandboxSandbox(config, spec)

    await sandbox.start()
    result = await sandbox.execute("print(1)", req_uuid="request-1")

    assert result.error_occurred is False
    assert create_kwargs["image"] == "registry/kernel:latest"
    assert create_kwargs["timeout"] == timedelta(seconds=5400)
    assert create_kwargs["skip_health_check"] is True
    assert create_kwargs["env"]["CAPSULE_SOURCE"] == "s3://capsules/root"
    assert create_kwargs["env"]["CAPSULE_KEY"] == "cap-1"
    assert create_kwargs["env"]["EXTRA"] == "value"
    assert create_kwargs["metadata"]["hypotest-job"] == "run-9"
    assert create_kwargs["resource"] == {
        "cpu": "1.5",
        "memory": "8192Mi",
        "ephemeral-storage": "20Gi",
        "gpu": "1",
        "gpu_type": "L40S",
    }
    assert create_kwargs["extensions"] == {
        "poolRef": "hypotest-pool",
        "imagePullPolicy": "IfNotPresent",
        "opensandbox.extensions.image-pull-policy": "IfNotPresent",
    }
    assert create_kwargs["entrypoint"] == [
        "sh",
        "/opt/entrypoint.sh",
        "/app/kernel_env/bin/python",
        "-m",
        "hypotest.kernel_capsule_server",
        "--port",
        "8000",
        "--language",
        "python",
        "--safe-execute",
    ]
    assert create_kwargs["platform"].kwargs == {"os": "linux", "arch": "amd64"}
    assert _FakeConnectionConfig.created[-1]["domain"] == "sandbox.example:8080"
    assert _FakeConnectionConfig.created[-1]["api_key"] == "fake-lifecycle-api-key"

    # Relative request paths preserve the server-proxy prefix instead of
    # accidentally resolving /health at the lifecycle server's root.
    assert requests
    assert all(request.url.path.startswith("/sandboxes/sb-123/proxy/8000/") for request in requests)
    assert all(request.headers["X-OpenSandbox-Route"] == "route-1" for request in requests)
    assert all(request.headers["OPEN-SANDBOX-API-KEY"] == "fake-lifecycle-api-key" for request in requests)
    assert not any(request.url.path.endswith("/load_capsule") for request in requests)
    assert any(request.url.path.endswith("/reset") for request in requests)

    await sandbox.close()
    assert remote.killed is True
    assert remote.closed is True


@pytest.mark.asyncio
async def test_large_bundle_selects_image_skips_load_and_resets_seed(tmp_path, monkeypatch):
    remote, create_kwargs, requests = _install_fakes(monkeypatch, endpoint="10.0.0.8:8000")
    config = SandboxConfig(
        work_dir=tmp_path,
        language=NBLanguage.R,
        safe_execute=False,
        ref=CapsuleRef(uuid="cap-2", delivery="bundled"),
        seed=456,
    )
    spec = OpenSandboxSpec(
        image="registry/kernel:latest",
        use_server_proxy=False,
        capsule_mode="large_bundle",
        large_bundle_image_template="registry/capsule:{capsule_uuid}",
        create_attempts=1,
        health_poll_interval_seconds=0.001,
    )
    sandbox = OpenSandboxSandbox(config, spec)

    await sandbox.start()

    assert create_kwargs["image"] == "registry/capsule:cap-2"
    assert create_kwargs["env"]["HYPOTEST_BUNDLE_CAPSULE_ID"] == "cap-2"
    assert create_kwargs["entrypoint"][-2:] == ["--language", "r"]
    assert "--safe-execute" not in create_kwargs["entrypoint"]
    paths = [request.url.path for request in requests]
    assert not any(path.endswith("/load_capsule") for path in paths)
    assert any(path.endswith("/reset") for path in paths)
    assert all("OPEN-SANDBOX-API-KEY" not in request.headers for request in requests)

    await sandbox.close()
    assert remote.killed
    assert remote.closed


@pytest.mark.asyncio
async def test_private_registry_auth_is_attached_to_image_spec(tmp_path, monkeypatch):
    remote, create_kwargs, _ = _install_fakes(monkeypatch)
    sandbox = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON),
        OpenSandboxSpec(
            image="private.registry.example/hypotest:latest",
            image_auth={"username": "registry-user", "password": "fake-registry-password"},
            create_attempts=1,
            health_poll_interval_seconds=0.001,
        ),
    )

    await sandbox.start()

    image = create_kwargs["image"]
    assert isinstance(image, _FakeImageSpec)
    assert image.image == "private.registry.example/hypotest:latest"
    assert image.auth.username == "registry-user"
    assert image.auth.password == "fake-registry-password"
    assert "fake-registry-password" not in repr(sandbox._spec.image_auth)

    await sandbox.close()
    assert remote.killed
    assert remote.closed


def test_private_registry_auth_resolves_explicit_environment_references(monkeypatch):
    monkeypatch.setenv("HYPOTEST_REGISTRY_USER", "registry-user")
    monkeypatch.setenv("HYPOTEST_REGISTRY_TOKEN", "fake-registry-password")

    spec = OpenSandboxSpec(
        image="private.registry.example/hypotest:latest",
        image_auth={
            "username": "${HYPOTEST_REGISTRY_USER}",
            "password": "${HYPOTEST_REGISTRY_TOKEN}",
        },
    )

    assert spec.image_auth is not None
    assert spec.image_auth.username == "registry-user"
    assert spec.image_auth.password.get_secret_value() == "fake-registry-password"


def test_private_registry_auth_rejects_missing_environment_reference(monkeypatch):
    monkeypatch.delenv("HYPOTEST_MISSING_REGISTRY_TOKEN", raising=False)
    with pytest.raises(ValueError, match=r"HYPOTEST_MISSING_REGISTRY_TOKEN.*not set"):
        OpenSandboxSpec(
            image="private.registry.example/hypotest:latest",
            image_auth={"username": "registry-user", "password": "${HYPOTEST_MISSING_REGISTRY_TOKEN}"},
        )


@pytest.mark.parametrize(
    ("username", "password", "message"),
    [("", "token", "username"), ("user", "  ", "password")],
)
def test_private_registry_auth_rejects_blank_values(username, password, message):
    with pytest.raises(ValueError, match=message):
        OpenSandboxSpec(
            image="private.registry.example/hypotest:latest",
            image_auth={"username": username, "password": password},
        )


def test_image_pull_policy_populates_both_extension_spellings_and_honors_override():
    defaulted = OpenSandboxSpec(image="kernel:latest")
    assert defaulted.resolve_extensions() == {
        "imagePullPolicy": "IfNotPresent",
        "opensandbox.extensions.image-pull-policy": "IfNotPresent",
    }

    overridden = OpenSandboxSpec(
        image="kernel:latest",
        image_pull_policy="IfNotPresent",
        extensions={"imagePullPolicy": "Always"},
    )
    assert overridden.resolve_extensions() == {
        "imagePullPolicy": "Always",
        "opensandbox.extensions.image-pull-policy": "Always",
    }

    untouched = OpenSandboxSpec(
        image="kernel:latest",
        image_pull_policy=None,
        extensions={"runtime.example/option": "value"},
    )
    assert untouched.resolve_extensions() == {"runtime.example/option": "value"}


def test_large_bundle_shared_image_and_specific_override_precedence():
    shared = OpenSandboxSpec(
        image="kernel:latest",
        capsule_mode="large_bundle",
        large_bundle_image="registry/capsules:all",
    )
    assert shared.resolve_large_bundle_image("any-capsule") == "registry/capsules:all"

    overridden = OpenSandboxSpec(
        image="kernel:latest",
        capsule_mode="large_bundle",
        large_bundle_image="registry/capsules:all",
        large_bundle_image_template="registry/capsules:{capsule_uuid}",
        large_bundle_images={"special": "registry/capsules:special"},
    )
    assert overridden.resolve_large_bundle_image("special") == "registry/capsules:special"
    assert overridden.resolve_large_bundle_image("ordinary") == "registry/capsules:ordinary"


def test_init_pull_key_can_be_overridden_or_left_to_image():
    runtime = OpenSandboxSpec(
        image="kernel:latest",
        capsule_source="s3://bucket/capsules",
        capsule_key="tasks/{capsule_uuid}",
    )
    assert runtime.resolve_capsule_key("cap-1") == "tasks/cap-1"

    baked = OpenSandboxSpec(image="kernel:latest", capsule_key=None)
    assert baked.resolve_capsule_key("cap-1") is None

    invalid = OpenSandboxSpec(image="kernel:latest", capsule_key="{unknown}")
    with pytest.raises(ValueError, match="may only contain"):
        invalid.resolve_capsule_key("cap-1")


@pytest.mark.asyncio
async def test_init_pull_can_use_source_and_key_baked_into_image(tmp_path, monkeypatch):
    remote, create_kwargs, requests = _install_fakes(monkeypatch)
    sandbox = OpenSandboxSandbox(
        SandboxConfig(
            work_dir=tmp_path,
            language=NBLanguage.PYTHON,
            ref=CapsuleRef(uuid="cap-1", delivery="object_store"),
        ),
        OpenSandboxSpec(image="kernel-with-baked-capsule-config:latest", capsule_key=None, create_attempts=1),
    )

    await sandbox.start()

    assert "CAPSULE_SOURCE" not in create_kwargs["env"]
    assert "CAPSULE_KEY" not in create_kwargs["env"]
    assert not any(request.url.path.endswith("/load_capsule") for request in requests)

    await sandbox.close()
    assert remote.killed
    assert remote.closed


def test_connection_uses_official_sdk_environment_fallback(monkeypatch, tmp_path):
    _FakeConnectionConfig.created.clear()
    monkeypatch.setattr(
        opensandboxmod,
        "_require_opensandbox_sdk",
        lambda: (object, _FakeConnectionConfig, _FakePlatformSpec, _FakeImageAuth, _FakeImageSpec),
    )
    sandbox = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON),
        OpenSandboxSpec(image="kernel:latest"),
    )

    sandbox._make_connection_config()

    kwargs = _FakeConnectionConfig.created[-1]
    assert "domain" not in kwargs
    assert "api_key" not in kwargs
    assert "protocol" not in kwargs


class _FakeSDKError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"SDK request failed with status {status_code}")
        self.status_code = status_code


@pytest.mark.asyncio
async def test_create_authentication_error_is_not_treated_as_remote_unavailable(tmp_path, monkeypatch):
    error = _FakeSDKError(401)

    class _RejectingSDK:
        @classmethod
        async def create(cls, **_kwargs):
            raise error

    monkeypatch.setattr(
        opensandboxmod,
        "_require_opensandbox_sdk",
        lambda: (_RejectingSDK, _FakeConnectionConfig, _FakePlatformSpec, _FakeImageAuth, _FakeImageSpec),
    )
    sandbox = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON),
        OpenSandboxSpec(image="kernel:latest", create_attempts=3),
    )

    with pytest.raises(_FakeSDKError) as exc_info:
        await sandbox.start()

    assert exc_info.value is error


@pytest.mark.asyncio
async def test_create_service_failure_becomes_remote_unavailable_after_retries(tmp_path, monkeypatch):
    attempts = 0

    class _UnavailableSDK:
        @classmethod
        async def create(cls, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise _FakeSDKError(503)

    monkeypatch.setattr(
        opensandboxmod,
        "_require_opensandbox_sdk",
        lambda: (_UnavailableSDK, _FakeConnectionConfig, _FakePlatformSpec, _FakeImageAuth, _FakeImageSpec),
    )
    sandbox = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON),
        OpenSandboxSpec(
            image="kernel:latest",
            create_attempts=2,
            create_retry_delay_seconds=0,
        ),
    )

    with pytest.raises(OpenSandboxUnavailableError, match="after 2 attempt"):
        await sandbox.start()

    assert attempts == 2


@pytest.mark.asyncio
async def test_kernel_proxy_authentication_error_is_surfaced_and_remote_is_closed(tmp_path, monkeypatch):
    remote, _, _ = _install_fakes(monkeypatch, health_status=401)
    sandbox = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path, language=NBLanguage.PYTHON),
        OpenSandboxSpec(
            image="kernel:latest",
            create_attempts=1,
            health_poll_interval_seconds=0.001,
        ),
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await sandbox.start()

    assert exc_info.value.response.status_code == 401
    assert remote.killed is True
    assert remote.closed is True
