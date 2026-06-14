# ADR 0001 — Sandbox Backend Abstraction

- **Status:** Proposed (2026-06-03)
- **Supersedes:** the ad-hoc backend branching in `src/hypotest/env/interpreter_env.py`
- **Deciders:** sandbox / RL infra

## 1. Context

`InterpreterEnvState` runs agent code in one of several execution backends. Today it
selects and drives them with `if use_ray and use_enroot / elif use_docker or use_enroot /
else` branches **duplicated across every lifecycle method** (`start`, `execute_and_add_cell`,
`reset_kernel`, `list_dir`, `close`). The backend implementations (`EnrootKernelServer` ray
actor, the enroot bash/cmd builders, the docker launch, `prlimit` prefixing, the HTTP client,
`_await_ray_ref`) all live in one ~2300-line file alongside the notebook and scoring logic.

Consequences we want to end:

- New backends (the upcoming **k8s / agent-sandbox** path) mean editing five branch
  sites instead of adding one class.
- `close()` carries a 20-branch teardown; the dispatch is copy-pasted; backend specifics
  (ray refs, aiodocker handles, `max_concurrency`) leak into the env layer.
- The file is a dumping ground — the original mistake we're correcting.

We also need, in the near term:

- A **k8s sandbox** built on [`kubernetes-sigs/agent-sandbox`](https://github.com/kubernetes-sigs/agent-sandbox)
  as the primary fleet, with the **colocated enroot** sandbox as fallback when k8s is
  busy/unavailable.
- That fallback to be **transparent at the execute layer**.

**Goal: build the abstraction once, cleanly, so backends are add-a-class and their specifics
never leak past the interface.**

## 2. Decision — two seams

The variation is along two independent axes that today's code conflates:

- **Transport** — how the orchestrator reaches the kernel: in-process calls, HTTP, HTTP
  relayed through a ray actor, or HTTP through the agent-sandbox connector.
- **Provisioning / placement / lifecycle** — how the kernel process is born, placed on a
  node, and torn down: nothing (local), aiodocker, enroot+ray (SPREAD), k8s pod.

We split on exactly those two axes.

### Seam 1 — `KernelClient` (the wire protocol)

The single in-sandbox contract is the existing `kernel_server.py` HTTP API
(`/execute`, `/reset`, `/health`, `/close`) **plus** a new `/list_dir` and `/load_capsule`.
docker, enroot, and k8s pods all run this server; only **local** talks to a `jupyter_client`
kernel directly.

`AsyncSandboxConnector.send_request(method, endpoint, **kwargs) -> httpx.Response` is
httpx-shaped, so k8s uses the **same client** as docker — only the request function differs:

```python
RequestFn = Callable[..., Awaitable[httpx.Response]]   # (method, endpoint, **kwargs)

class KernelClient(Protocol):
    async def execute(self, code: str, timeout: float | None, req_uuid: str = "") -> ExecutionResult: ...
    async def reset(self) -> None: ...
    async def list_dir(self, directory: str, max_files: int, show_hidden: bool) -> str: ...
    async def health(self) -> bool: ...
    async def load_capsule(self, uuid: str) -> CapsuleDigest: ...   # see §5

class HttpKernelClient:
    """Speaks the kernel_server HTTP protocol over any httpx-shaped request fn."""
    def __init__(self, request: RequestFn): self._request = request
    async def execute(self, code, timeout, req_uuid=""):
        r = await self._request("POST", "/execute", json={"code": code, "timeout": timeout},
                                headers={"X-Req-UUID": req_uuid}, timeout=...)
        return _parse_execute_response(r)        # identical for docker + k8s
    # reset / list_dir / health / load_capsule similar

# docker: HttpKernelClient(httpx.AsyncClient(base_url=f"http://localhost:{port}").request)
# k8s:    HttpKernelClient(sandbox.connector.send_request)
# enroot: the HttpKernelClient lives INSIDE the ray actor; the env reaches it via .remote()
```

### Seam 2 — `Sandbox` (placement + provisioning + lifecycle)

What `InterpreterEnvState` holds. **It never branches on backend type.**

```python
class Sandbox(ABC):
    work_dir: Path
    language: NBLanguage

    @abstractmethod
    async def start(self) -> None: ...        # provision + place + make data ready (§4)
    @abstractmethod
    async def execute(self, code, timeout=None, req_uuid="") -> ExecutionResult: ...
    @abstractmethod
    async def reset(self) -> None: ...
    @abstractmethod
    async def list_dir(self, directory=".", max_files=20, show_hidden=False) -> str: ...
    @abstractmethod
    async def close(self) -> None: ...
    @abstractmethod
    async def health(self) -> bool: ...
```

Implementations: `LocalSandbox` (wraps `Interpreter`, no client), `DockerSandbox`,
`EnrootSandbox` (ray placement), `K8sSandbox` (agent-sandbox). Each owns its provisioning and
a `KernelClient`; `execute/reset/list_dir/health` delegate to the client (or to `Interpreter`
for local).

```python
result = await self.sandbox.execute(code, timeout, req_uuid)   # the whole dispatch
```

`close()`'s 20-branch teardown disappears: it becomes `await self.sandbox.close()`, and each
teardown lives in its own impl.

## 3. Backend matrix

| | provisioning | transport | placement | data delivery (§4) | resource limit (§6) | `list_dir` |
|---|---|---|---|---|---|---|
| **local** | none (in-proc kernel) | direct `Interpreter` calls | host process | work_dir on host | none | `FilesystemTool` on work_dir |
| **docker** | aiodocker container | `HttpKernelClient` → `localhost:port` | single host | bind-mount, dataset-server pre-pull | none (unsupported) | `/list_dir` via client |
| **enroot** | enroot squashfs subprocess | ray actor `.remote()` → in-actor `HttpKernelClient` | **ray SPREAD** (required) | bind-mount, dataset-server pre-pull | `prlimit` (default) | `/list_dir` via client |
| **k8s** | agent-sandbox `Sandbox` claim, fresh per task | `connector.send_request` → `HttpKernelClient` | k8s scheduler + `SandboxScheduler` LB across clusters | in-pod `/load_capsule` pull | pod `resources.limits` (cgroup, kubelet-set) | `/list_dir` via client |

Backend specifics — `warmpool`, `terminate`, `X-Sandbox-*`, ray refs, aiodocker handles,
`max_concurrency` — **stay inside their impl**. The `Sandbox` ABC sees none of them.
**This no-leakage rule is the invariant that prevents the dumping ground from re-forming.**

Relocation note: moving `EnrootKernelServer` + the bash builders + `_await_ray_ref` into
`env/sandbox/enroot.py` also clears the bucket-4 lint debt (the `close` complexity and the
`@ray.remote(max_concurrency=…)` mypy error leave `interpreter_env.py`).

## 4. Data provisioning — a uniform ref, per-impl delivery

The capsule is identified by a **ref** `(s3 source, uuid)` — the uniform input. *Delivering*
it differs by backend and is each impl's job inside `start()`:

- **k8s**: `/load_capsule(uuid)` runs **inside the pod** → data lands in the pod FS.
- **enroot / docker**: the kernel reads a **host dir** bind-mounted in; the dataset server
  pulls the capsule onto that host dir first (the existing lazy pull).
- **local**: capsule populated into `work_dir`.

**Fallback implication:** the data a failed k8s `start()` pulled dies with the terminated
pod, so the enroot fallback re-provisions the same ref **its own way** (host-pull + mount) —
it does not inherit the pod's copy. The ref is shared; "data ready" is per-impl.

**Future — integrity verification.** For now we trust the two delivery paths produce identical
bytes. Long-run, `load_capsule` / the host-pull should return a `CapsuleDigest` (per-file or
manifest hash) so the orchestrator can assert k8s-delivered and enroot-delivered data match
before trusting a fallback or comparing runs.

## 5. Failure & recovery — swap + replay

Durable session state is the **cell history**, which lives in `InterpreterEnvState`, *above*
the seam. The sandbox is ephemeral and swappable, so one recovery path covers ray-actor death,
pod death, and kernel-server crash:

```python
async def recover(self):
    await self.sandbox.close()                 # best-effort
    self.sandbox = scheduler.acquire(self.ref) # may select the fallback backend (§7)
    await self.sandbox.start()                 # re-provisions the capsule
    replayed = 0
    for cell in self.history:                  # replay, capped
        if replayed >= REPLAY_BUDGET: break
        await self.sandbox.execute(cell.code); replayed += 1
    log(f"recovered: replayed {replayed}/{len(self.history)} cells")
```

Accepted limits (documented, not solved):

- **Fidelity** — replay reconstructs deterministic state only; wall-clock/RNG/GPU/external
  side-effects are not restored, and **non-idempotent cells run twice**. We bias toward
  fidelity by prompting the model for reproducible code (fixed seeds); correctness is not
  guaranteed. This is still preferable to dropping the rollout.
- **Cost** — capped by `REPLAY_BUDGET`; recovery emits a "replayed N/M" signal rather than
  silently diverging.

Recovery is backend-agnostic *because* history is above the seam.

## 6. Resource limits — pluggable, prlimit default

cgroups do **not** work on the current cluster (no cgroup delegation for the non-root enroot
user), so we do **not** mandate them.

- `ResourceSpec(mem_mb, mem_high_mb?, max_pids)` is the uniform spec.
- A **pluggable, config-selected limiter** maps it per backend:
  - **enroot (default): `prlimit`** — `RLIMIT_AS` + `RLIMIT_NPROC` (works today).
  - **cgroups v2 (opt-in):** `memory.max` (+ optional `memory.high`), `pids.max`, where the
    node supports delegation (e.g. via `systemd-run --scope`). Off by default.
  - **k8s:** pod `resources.limits` — the kubelet sets the cgroup; no delegation issue.
  - **local / docker:** unsupported (no-op).

**Known, accepted gap:** `RLIMIT_AS` limits *virtual address space* (over-reserved by
CUDA/BLAS/threaded runtimes → can spuriously trip), while k8s pod limits cap *RSS* and
OOM-kill. So the same `ResourceSpec` yields different failure behavior on the enroot fallback
vs. k8s. Documented; revisit if cgroup delegation becomes available cluster-wide.

## 7. Topology, placement & connection

The training job runs on the training cluster; agent-sandbox sandboxes run on **workstations**,
and there will be **multiple sandbox clusters (≈ one per workstation) load-balanced across**.
So placement is a first-class concern and lives **above** the `Sandbox` interface:

```python
class SandboxScheduler:
    """Selects a target and builds a Sandbox. Owns LB + the fallback chain."""
    async def acquire(self, ref, resources) -> Sandbox:
        for target in self.k8s_targets_by_load():        # LB across sandbox clusters
            try:
                return K8sSandbox(ref, resources, conn=self._conn_for(target))
            except NoCapacity:
                continue
        return EnrootSandbox(ref, resources)             # colocated fallback (ray)
```

Connection config is per target, and **both branches are implemented**:

- **Same cluster / network (workstation = node pool):** `SandboxInClusterConnectionConfig`
  — connector hits `{id}.{ns}.svc.cluster.local:{port}` / pod IP directly. No router, lowest
  latency.
- **Separate cluster / network (the likely case under multi-cluster LB):**
  `SandboxGatewayConnectionConfig` — router exposed via ingress + `X-Sandbox-*` headers +
  mTLS/token, and the orchestrator holds **each sandbox cluster's API creds** to CRUD claims.
  Every `/execute` is a cross-network round-trip; the capsule pull crosses the boundary.

`SandboxScheduler` chooses target + connection config; `K8sSandbox` stays cluster-agnostic.

Requirements either way:

- Orchestrator ServiceAccount needs **RBAC** to CRUD `Sandbox` CRDs in each sandbox namespace.
- Use the **async** SDK (`async_sandbox` / `async_connector`); the local-tunnel mode is
  sync/dev-only.
- Router `PROXY_TIMEOUT_SECONDS` (default **180s**) must be raised ≥ max cell timeout, or
  long executions are silently cut off. (Handled cluster-side.)

**`K8sSandbox.start()`** (fresh pod per task — chosen for isolation; no warm pool):
create claim → wait **`SandboxReady`** (CRD condition) → wait our **`/health`** (kernel up;
two-level readiness) → `/load_capsule(uuid)`. `close()` → `sandbox.terminate()`. Set a
**TTL / owner-ref** so the controller GCs orphaned sandboxes if the orchestrator dies.

Latency lever (fresh pod pays full cold-start + capsule pull, which governs the fallback
timeout): bake `CAPSULE_UUID` into the pod spec so the entrypoint pulls during boot, overlapping
kernel start, instead of a serial post-ready `/load_capsule`.

## 8. Protocol versioning

The `kernel_server` HTTP shapes are now spoken by three independently-built images (docker,
enroot `.sqsh`, k8s bundled) **and** the orchestrator client, deployed on different schedules
(and across clusters). Treat the endpoints as a public API:

- **Additive / backward-compatible changes only** (no field renames/removals; new fields
  optional).
- Expose **`protocol_version`** in `/health`; the client checks it and fails clearly on skew
  rather than mid-run with a parse error.

## 9. Cross-cutting (unchanged or already done)

- **Code safety**: AST `check_code_safety` runs in the env layer above the sandbox (all
  backends); the kernel-server regex (`_kernel_check_code_safety`) runs in-pod for HTTP
  backends. Unchanged — the `Sandbox` interface needs no safety method.
- **Workspace install model**: `install_shim.write_workspace_config` / `workspace_env` /
  `bash_export_block` are already the shared single source; each backend lays it down in
  `start()`. (Done in the prior parity refactor.)
- **`/list_dir`**: served by our kernel server (sees the kernel's cwd + files executed code
  wrote), not agent-sandbox's `files/filesystem` — one workspace view across backends.

## 10. Invariants & non-goals

- **Invariant:** no backend type crosses the `Sandbox` interface.
- **Invariant:** the in-sandbox contract is the `kernel_server` HTTP protocol; agent-sandbox
  `commands.run` (one-shot) is **not** used for execution (it can't hold the persistent kernel).
- **Non-goals:** changing the aviary `InterpreterEnv` tool surface; refactoring `_score_solution`
  / scoring logic (separate concern); a unified resource-limit *semantic* (gap accepted, §6).

## 11. Decisions log

| # | Decision |
|---|---|
| 1 | Router `PROXY_TIMEOUT_SECONDS` raised ≥ max cell timeout — handled cluster-side |
| 2 | **Fresh pod per task** (no warm pool) — bad-reset risk |
| 3 | `start()` folds capsule-load; k8s via `/load_capsule`; two-level readiness |
| 4 | Capsule **ref** uniform; per-impl delivery; **hash verification** is a future hook |
| 5 | Recovery = swap + capped replay; fidelity not guaranteed (mitigated by reproducible-code prompting) |
| 6 | Pluggable limiter; **prlimit default**, cgroups opt-in; prlimit↔cgroup semantic gap accepted |
| 7 | **Both** InCluster + Gateway connection branches; placement/LB across clusters in `SandboxScheduler`; fallback chain `[k8s…, enroot]` |
| 8 | `commands.run` not used; persistent kernel server reached over HTTP via the connector |
| 9 | `/list_dir` via our endpoint; ray kept as enroot placement |
| 10 | Protocol is a versioned contract: additive-only + `protocol_version` in `/health` |

## 12. Migration (outline — detailed impl plan to follow)

Each step is behavior-preserving until the last; no env behavior changes until the flip.

1. Introduce `Sandbox` ABC + `KernelClient` + `make_sandbox` / `SandboxScheduler`; wrap
   today's local/docker/enroot/ray paths as adapters (no behavior change).
2. Flip `InterpreterEnvState` to hold one `self.sandbox` and delegate; delete the per-method
   branches. (`close` complexity + dispatch dup gone.)
3. Extract `HttpKernelClient` (dedup docker + enroot-actor-internal client).
4. Add `/list_dir` + `protocol_version` to the kernel server; route `list_dir` through it.
5. Land `K8sSandbox` against the async agent-sandbox client + `SandboxScheduler` (both
   connection branches); wire the k8s→enroot fallback + `recover()`.
6. Move `EnrootKernelServer` + builders into `env/sandbox/enroot.py` (final relocation).
```
