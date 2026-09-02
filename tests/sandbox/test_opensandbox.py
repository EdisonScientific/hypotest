"""Unit tests for the raw OpenSandbox lifecycle backend."""

import asyncio
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
        self.domain = kwargs.get("domain", "sandbox.example")
        self.transport = kwargs.get("transport")
        self.use_server_proxy = kwargs.get("use_server_proxy", False)
        self._api_key = kwargs.get("api_key", "fake-env-api-key")

    def get_api_key(self):
        return self._api_key

    def get_base_url(self):
        return f"{self.protocol}://{self.domain}/v1"


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
    def __init__(self, endpoint: str, *, sandbox_id: str = "sb-123", route: str = "route-1"):
        self.id = sandbox_id
        self.endpoint = SimpleNamespace(endpoint=endpoint, headers={"X-OpenSandbox-Route": route})
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
    monkeypatch.delenv("REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("REGISTRY_PASSWORD", raising=False)
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
        # The production client owns the shared kernel transport. Tests replace
        # only its wire behavior with a MockTransport.
        kwargs.pop("transport", None)
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(opensandboxmod.httpx, "AsyncClient", client_factory)
    return remote, create_kwargs, requests


@pytest.mark.parametrize(
    ("resources", "message"),
    [
        ({"cpu": 1, "cpu_request": 2}, "cpu_request"),
        ({"mem_mb": 512, "mem_request_mb": 1024}, "mem_request_mb"),
    ],
)
def test_resource_request_cannot_exceed_limit(resources, message):
    with pytest.raises(ValueError, match=message):
        ResourceSpec(**resources)


def test_runtime_credentials_are_not_required_in_serialized_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "runtime-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "runtime-secret-key")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://object-store.example")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-test-1")
    monkeypatch.setenv("REGISTRY_USERNAME", "runtime-registry-user")
    monkeypatch.setenv("REGISTRY_PASSWORD", "runtime-registry-password")
    config = SandboxConfig(
        work_dir=tmp_path,
        language=NBLanguage.PYTHON,
        execution_timeout=60,
        safe_execute=True,
        resources=ResourceSpec(),
        ref=CapsuleRef(source="s3://capsules/root", uuid="cap-1", delivery="object_store"),
    )
    spec = OpenSandboxSpec(image="registry/kernel:latest")
    sandbox = OpenSandboxSandbox(config, spec)

    allocation_env = sandbox._allocation_env()
    image_auth = sandbox._image_auth()

    assert allocation_env["AWS_ACCESS_KEY_ID"] == "runtime-access-key"
    assert allocation_env["AWS_SECRET_ACCESS_KEY"] == "runtime-secret-key"
    assert allocation_env["AWS_ENDPOINT_URL"] == "https://object-store.example"
    assert allocation_env["AWS_DEFAULT_REGION"] == "us-test-1"
    assert image_auth is not None
    assert image_auth.username == "runtime-registry-user"
    assert image_auth.password.get_secret_value() == "runtime-registry-password"
    assert "runtime-secret-key" not in repr(spec)
    assert "runtime-registry-password" not in repr(spec)


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
        resources=ResourceSpec(
            mem_mb=8192,
            mem_request_mb=512,
            cpu=1.5,
            cpu_request=0.25,
            disk_gib=20,
            gpu=1,
            gpu_type="L40S",
        ),
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
        kernel_memory_limit_mb=7168,
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
    assert create_kwargs["resource_requests"] == {
        "cpu": "0.25",
        "memory": "512Mi",
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
        "--kernel-memory-limit-mb",
        "7168",
        "--safe-execute",
        "--no-install-shim",
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
    assert sandbox.startup_timings.keys() == {
        "allocation_seconds",
        "create_attempts",
        "create_queue_seconds",
        "kernel_connect_seconds",
        "seed_reset_seconds",
        "startup_seconds",
    }

    await sandbox.close()
    assert remote.killed is True
    assert remote.closed is True


@pytest.mark.asyncio
async def test_concurrent_sandboxes_share_bounded_clients_until_last_close(tmp_path, monkeypatch):
    _FakeConnectionConfig.created.clear()
    _install_fakes(monkeypatch)
    spec = OpenSandboxSpec(
        image="registry/kernel:latest",
        create_attempts=1,
        health_poll_interval_seconds=0.001,
        lifecycle_max_connections=7,
        lifecycle_max_keepalive_connections=3,
        lifecycle_create_concurrency=5,
        kernel_max_connections=11,
        kernel_max_keepalive_connections=5,
        kernel_request_concurrency=4,
    )
    first = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path / "first", language=NBLanguage.PYTHON),
        spec,
    )
    second = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path / "second", language=NBLanguage.PYTHON),
        spec,
    )

    try:
        await asyncio.gather(first.start(), second.start())

        assert first._client_pool is not None
        assert first._client_pool is second._client_pool
        pool = first._client_pool
        assert pool.references == 2
        assert pool.lifecycle_create_semaphore._value == 5
        assert pool.kernel_request_semaphore._value == 4
        shared_lifecycle_transports = [
            kwargs["transport"] for kwargs in _FakeConnectionConfig.created if kwargs.get("transport") is not None
        ]
        assert shared_lifecycle_transports == [pool.lifecycle_transport, pool.lifecycle_transport]

        await first.close()
        assert pool.references == 1
        assert await second.health() is True

        await second.close()
        assert pool.references == 0
        assert asyncio.get_running_loop() not in opensandboxmod._OPEN_SANDBOX_CLIENT_POOLS
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_create_admission_waits_outside_sdk_request_pool(tmp_path, monkeypatch):
    remote, _, _ = _install_fakes(monkeypatch)
    release_creates = asyncio.Event()
    admission_full = asyncio.Event()
    active_creates = 0
    peak_creates = 0

    class _GatedSDK:
        @classmethod
        async def create(cls, **_kwargs):
            nonlocal active_creates, peak_creates
            active_creates += 1
            peak_creates = max(peak_creates, active_creates)
            if active_creates == 2:
                admission_full.set()
            try:
                await release_creates.wait()
                return remote
            finally:
                active_creates -= 1

    monkeypatch.setattr(
        opensandboxmod,
        "_require_opensandbox_sdk",
        lambda: (_GatedSDK, _FakeConnectionConfig, _FakePlatformSpec, _FakeImageAuth, _FakeImageSpec),
    )
    spec = OpenSandboxSpec(
        image="registry/kernel:latest",
        create_attempts=1,
        health_poll_interval_seconds=0.001,
        lifecycle_max_connections=2,
        lifecycle_max_keepalive_connections=2,
        lifecycle_create_concurrency=2,
    )
    sandboxes = [
        OpenSandboxSandbox(
            SandboxConfig(work_dir=tmp_path / f"sandbox-{index}", language=NBLanguage.PYTHON),
            spec,
        )
        for index in range(3)
    ]
    starts = [asyncio.create_task(sandbox.start()) for sandbox in sandboxes]

    try:
        await asyncio.wait_for(admission_full.wait(), timeout=1)
        await asyncio.sleep(0)
        assert active_creates == 2
        assert peak_creates == 2

        release_creates.set()
        await asyncio.gather(*starts)
        assert peak_creates == 2
        assert sandboxes[2].startup_timings["create_queue_seconds"] > 0
    finally:
        release_creates.set()
        await asyncio.gather(*starts, return_exceptions=True)
        await asyncio.gather(*(sandbox.close() for sandbox in sandboxes))


@pytest.mark.asyncio
async def test_kernel_request_admission_waits_outside_httpx_timeout():
    release_requests = asyncio.Event()
    admission_full = asyncio.Event()
    active_requests = 0
    peak_requests = 0

    class _BlockingClient:
        async def request(self, method, url, **_kwargs):
            nonlocal active_requests, peak_requests
            assert method == "GET"
            active_requests += 1
            peak_requests = max(peak_requests, active_requests)
            if active_requests == 2:
                admission_full.set()
            try:
                await release_requests.wait()
                return httpx.Response(200, request=httpx.Request(method, url))
            finally:
                active_requests -= 1

    pool = SimpleNamespace(
        kernel_request_semaphore=asyncio.Semaphore(2),
        kernel_client=_BlockingClient(),
    )
    requests = [
        asyncio.create_task(
            opensandboxmod._request_with_kernel_admission(pool, "GET", httpx.URL("https://proxy.invalid/health"))
        )
        for _ in range(3)
    ]

    try:
        await asyncio.wait_for(admission_full.wait(), timeout=1)
        await asyncio.sleep(0)
        assert active_requests == 2
        assert peak_requests == 2

        release_requests.set()
        responses = await asyncio.gather(*requests)
        assert all(response.status_code == 200 for response in responses)
        assert peak_requests == 2
    finally:
        release_requests.set()
        await asyncio.gather(*requests, return_exceptions=True)


@pytest.mark.asyncio
async def test_shared_kernel_client_keeps_endpoint_headers_sandbox_local(tmp_path, monkeypatch):
    remotes = iter([
        _FakeRemote(
            "proxy.example/sandboxes/sb-a/proxy/8000",
            sandbox_id="sb-a",
            route="route-a",
        ),
        _FakeRemote(
            "proxy.example/sandboxes/sb-b/proxy/8000",
            sandbox_id="sb-b",
            route="route-b",
        ),
    ])

    class _FakeSDK:
        @classmethod
        async def create(cls, **_kwargs):
            return next(remotes)

    monkeypatch.setattr(
        opensandboxmod,
        "_require_opensandbox_sdk",
        lambda: (_FakeSDK, _FakeConnectionConfig, _FakePlatformSpec, _FakeImageAuth, _FakeImageSpec),
    )
    requests: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"protocol_version": 2})

    def client_factory(**kwargs):
        kwargs.pop("transport", None)
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(opensandboxmod.httpx, "AsyncClient", client_factory)
    spec = OpenSandboxSpec(
        image="registry/kernel:latest",
        domain="sandbox.example",
        api_key="lifecycle-key",
        create_attempts=1,
        health_poll_interval_seconds=0.001,
    )
    first = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path / "first", language=NBLanguage.PYTHON),
        spec,
    )
    second = OpenSandboxSandbox(
        SandboxConfig(work_dir=tmp_path / "second", language=NBLanguage.PYTHON),
        spec,
    )

    try:
        await asyncio.gather(first.start(), second.start())

        assert first._client_pool is second._client_pool
        routes_by_sandbox = {
            "sb-a": {request.headers["X-OpenSandbox-Route"] for request in requests if "/sb-a/" in request.url.path},
            "sb-b": {request.headers["X-OpenSandbox-Route"] for request in requests if "/sb-b/" in request.url.path},
        }
        assert routes_by_sandbox == {"sb-a": {"route-a"}, "sb-b": {"route-b"}}
        assert all(request.headers["OPEN-SANDBOX-API-KEY"] == "lifecycle-key" for request in requests)
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_mounted_volume_lifecycle_stages_before_health_with_generic_image(tmp_path, monkeypatch):
    remote, create_kwargs, requests = _install_fakes(monkeypatch)
    sandbox = OpenSandboxSandbox(
        SandboxConfig(
            work_dir=tmp_path,
            language=NBLanguage.PYTHON,
            ref=CapsuleRef(
                source="/mnt/shared/capsules",
                uuid="cap-2",
                delivery="mounted_volume",
            ),
        ),
        OpenSandboxSpec(
            image="registry/kernel:latest",
            capsule_mode="mounted_volume",
            mounted_capsule_root="/mnt/unused",
            capsule_key="capsules/{capsule_uuid}",
            create_attempts=1,
            health_poll_interval_seconds=0.001,
        ),
    )

    await sandbox.start()

    assert create_kwargs["image"] == "registry/kernel:latest"
    assert not create_kwargs["env"]["CAPSULE_SOURCE"]
    assert not create_kwargs["env"]["CAPSULE_KEY"]
    assert create_kwargs["env"]["HYPOTEST_MOUNTED_CAPSULE_ROOT"] == "/mnt/shared/capsules"
    assert create_kwargs["env"]["HYPOTEST_MOUNTED_CAPSULE_ID"] == "capsules/cap-2"
    assert not any(request.url.path.endswith("/load_capsule") for request in requests)

    await sandbox.close()
    assert remote.killed
    assert remote.closed


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
        install_shim_enabled=True,
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
    assert defaulted.install_shim_enabled is False
    assert defaulted.execution_poll_interval_seconds == 0.5
    assert defaulted.execution_poll_max_interval_seconds == 10
    assert defaulted.execution_poll_backoff_multiplier == 2
    assert defaulted.execution_poll_request_timeout_seconds == 5
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "execution_poll_interval_seconds": 2,
                "execution_poll_max_interval_seconds": 1,
            },
            "execution_poll_max_interval_seconds",
        ),
        (
            {
                "lifecycle_max_connections": 2,
                "lifecycle_max_keepalive_connections": 3,
            },
            "lifecycle_max_keepalive_connections",
        ),
        (
            {
                "lifecycle_max_connections": 2,
                "lifecycle_max_keepalive_connections": 2,
                "lifecycle_create_concurrency": 3,
            },
            "lifecycle_create_concurrency",
        ),
        (
            {
                "kernel_max_connections": 2,
                "kernel_max_keepalive_connections": 3,
            },
            "kernel_max_keepalive_connections",
        ),
        (
            {
                "kernel_max_connections": 2,
                "kernel_max_keepalive_connections": 2,
                "kernel_request_concurrency": 3,
            },
            "kernel_request_concurrency",
        ),
    ],
)
def test_polling_and_shared_client_limits_are_consistent(overrides, message):
    with pytest.raises(ValueError, match=message):
        OpenSandboxSpec(image="kernel:latest", **overrides)


def test_kernel_memory_limit_must_leave_outer_container_headroom(tmp_path):
    config = SandboxConfig(
        work_dir=tmp_path,
        language=NBLanguage.PYTHON,
        resources=ResourceSpec(mem_mb=8192),
    )

    with pytest.raises(ValueError, match="must be less than"):
        OpenSandboxSandbox(
            config,
            OpenSandboxSpec(image="kernel:latest", kernel_memory_limit_mb=8192),
        )


def test_kernel_memory_limit_rejects_custom_entrypoint():
    with pytest.raises(ValueError, match="custom entrypoint"):
        OpenSandboxSpec(
            image="kernel:latest",
            kernel_memory_limit_mb=7168,
            entrypoint=["custom-server"],
        )


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


def test_mounted_capsule_root_must_be_an_absolute_nonblank_container_path():
    with pytest.raises(ValueError, match="requires mounted_capsule_root"):
        OpenSandboxSpec(image="kernel:latest", capsule_mode="mounted_volume")
    with pytest.raises(ValueError, match="cannot be blank"):
        OpenSandboxSpec(image="kernel:latest", capsule_mode="mounted_volume", mounted_capsule_root=" ")
    with pytest.raises(ValueError, match="absolute container path"):
        OpenSandboxSpec(image="kernel:latest", capsule_mode="mounted_volume", mounted_capsule_root="mnt/capsules")
    with pytest.raises(ValueError, match="requires a capsule_key"):
        OpenSandboxSpec(
            image="kernel:latest",
            capsule_mode="mounted_volume",
            mounted_capsule_root="/mnt/capsules",
            capsule_key=None,
        )


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
    assert asyncio.get_running_loop() not in opensandboxmod._OPEN_SANDBOX_CLIENT_POOLS


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
    assert asyncio.get_running_loop() not in opensandboxmod._OPEN_SANDBOX_CLIENT_POOLS


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
