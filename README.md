# hypotest

## Installation

```bash
uv venv
uv sync
```

The dataset is available on HuggingFace: [`EdisonScientific/bixbench_hypothesis`](https://huggingface.co/datasets/EdisonScientific/bixbench_hypothesis).

You'll also need the capsule data directory accessible on your filesystem.

## Downloading Capsule Data

The task capsule data is hosted on a public HuggingFace bucket:

```bash
hf sync hf://buckets/EdisonScientific/bixbench-hypothesis-capsules /path/to/capsules/
```

## Building the OpenSandbox Image

The primary OpenSandbox path uses one generic Docker/OCI image with no capsule
data in its layers. On each allocation, Hypotest passes the selected capsule's
source and exact relative key as `CAPSULE_SOURCE` and `CAPSULE_KEY`. The
container downloads that capsule before starting Jupyter or its HTTP server, so
a successful `/health` means both data and kernel are ready. `InterpreterEnv`
starts episode time accounting only after that initialization completes.

The all-in-one builder creates the scientific base and kernel-server image,
optionally pushes it, and prints the matching `opensandbox_spec` configuration:

```bash
docker login registry.example

./scripts/build_opensandbox_image.py \
  --image registry.example/hypotest-kernel:latest \
  --capsule-source s3://example-bucket/capsules \
  --s3-region us-west-2 \
  --kernel-memory-limit-mb 57344 \
  --registry-auth \
  --push
```

`--registry-auth` never reads or bakes registry credentials. It prints
`${REGISTRY_USERNAME}` and `${REGISTRY_PASSWORD}` references for the dataset
server configuration; export those variables in the dataset-server process.
The separate `docker push` uses the builder machine's normal Docker credential
store. OpenSandbox sends the credentials as `SandboxImageSpec.auth`, matching
NeMo Gym's private-registry integration. The remote OpenSandbox workload
provider must support per-request image auth; current OpenSandbox Docker and
Kubernetes BatchSandbox providers do, while an unsupported provider rejects the
request rather than silently dropping credentials.

The S3 source and capsule-key policy can be supplied either way:

- Runtime (recommended): set `capsule_source` and `capsule_key` in
  `opensandbox_spec`. The default key template is `{capsule_uuid}`.
- Image default: pass `--capsule-source` and/or `--capsule-key` while building,
  then set the corresponding runtime field to `null` to preserve the image
  value. Only non-secret locations belong in image layers.

S3 credentials use the standard runtime `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, and IAM identity chain.
`AWS_ENDPOINT_URL`/`AWS_ENDPOINT_URL_S3` selects an S3-compatible endpoint.
The builder accepts no capsule directory and the Dockerfiles copy only source
and runtime files, so capsule bytes do not enter the resulting image.

The allocation dataflow is:

```text
Sandbox.create(image[/auth], CAPSULE_SOURCE, CAPSULE_KEY)
  -> remote image pull and container start
  -> exact capsule prefix pulled into /workspace
  -> persistent Jupyter kernel and HTTP server start
  -> /health succeeds
  -> episode timer starts
```

See [`deploy/server.opensandbox.example.yaml`](deploy/server.opensandbox.example.yaml)
for a complete server configuration.

To keep an accidental large allocation from OOM-killing the complete sandbox,
configure an inner Jupyter limit below the outer OpenSandbox/Kubernetes limit:

```yaml
opensandbox_spec:
  kernel_memory_limit_mb: 57344
execution_config:
  sandbox_cpu_request: 0.25
  sandbox_memory_request_mb: 512
  sandbox_cpu: 4
  sandbox_memory_limit_mb: 65536
```

The outer limit remains the sandbox capacity ceiling. The inner limit is a
Linux `RLIMIT_AS` on only the Jupyter child, leaving the HTTP server outside it.
Allocations that honor `ENOMEM` surface as normal notebook `MemoryError`s. If a
native failure still terminates Jupyter, the server returns `KernelDiedError`,
restarts only the kernel, and preserves workspace files; in-memory variables
are explicitly reported as lost.

## Cluster-Mounted Capsule Collections

When the OpenSandbox cluster already mounts the capsule collection into every
sandbox, use the same generic kernel image with mounted-volume delivery:

```yaml
capsule_mode: mounted_volume
mounted_capsule_root: /mnt/capsules
capsule_key: "{capsule_uuid}"
```

For each allocation, Hypotest passes the selected identity as
`HYPOTEST_MOUNTED_CAPSULE_ID`. The kernel bootstrap resolves an exact directory
below `mounted_capsule_root`, also accepting the legacy `CapsuleData-<id>` and
`capsule_<id>` directory conventions. It rejects traversal, symlinked capsules,
symlinks or special files inside a capsule, and overlap with `/workspace`.

The shared mount is only a bootstrap source. Before Jupyter or `/health`
starts, the selected capsule is copied into the sandbox's local `/workspace`
and owner-write permission is added to the copied files and directories. The
model therefore sees a normal writable workspace: edits and newly generated
files remain sandbox-local and do not mutate the cluster mount. A successful
`/health` means the copy and kernel startup are complete, and episode time
accounting begins afterward.

Size ephemeral storage for the largest selected capsule plus package installs
and model outputs. The OpenSandbox workload provider is responsible for making
the mount visible at the configured container path; no volume declaration is
encoded in the OCI image or lifecycle request.

## Optional Large-Bundle Images for OpenSandbox

Large-bundle mode can package either one capsule or an entire capsule
collection into a standard Docker/OCI image. For the high-throughput collection
layout, the supplied directory's immediate child directories are capsules:

```bash
./scripts/build_large_bundle.py /path/to/capsules \
  --all-capsules \
  --image registry.example/hypotest-capsules:all \
  --push
```

The collection is stored at `/opt/hypotest/capsules`. Every OpenSandbox create
call uses the same image and passes the task's `input_data_path` (or problem ID)
as `HYPOTEST_BUNDLE_CAPSULE_ID`. Before the kernel starts, Hypotest resolves that
member without allowing path escape and creates a zero-copy, top-level symlink
projection in `/workspace`. Configure the digest-pinned value printed by the
script as one shared image:

```yaml
capsule_mode: large_bundle
large_bundle_image: registry.example/hypotest-capsules@sha256:<digest>
platform_os: linux
platform_arch: amd64
```

This trades some isolation for launch and cache performance: only the selected
capsule is presented in `/workspace`, but all bundled capsules remain readable
through their absolute paths inside the container. The projection function is
the intended seam for a future bubblewrap or mount-namespace policy; collection
mode does not currently claim to prevent a model from deliberately inspecting
siblings.

The original one-capsule layout remains available. Its contents are copied
directly to `/workspace`, and the script prints a `large_bundle_images` mapping:

```bash
./scripts/build_large_bundle.py \
  /path/to/capsules/CapsuleData-123 \
  --capsule-id CapsuleData-123 \
  --image registry.example/hypotest-capsule:capsule-data-123 \
  --push
```

Both layouts default to `linux/amd64` and extend `hypotest-kernel:latest`.
Without `--push`, the script derives a safe local tag and loads the image into
the current Docker daemon; that works only when OpenSandbox shares the daemon.
For a remote Docker or Kubernetes OpenSandbox server, use `--push` and a
registry-visible `--image`. The OpenSandbox runtime—not merely the build
machine—must have pull access to a private registry. The same `image_auth`
setting described above applies to generic, single-capsule, and collection
images.

The kernel base must already be available to the selected Buildx builder. Use
`make kernel-image` for the local production base or `make kernel-image-core`
for the native lightweight base. Pass `--platform linux/arm64` when the base and
OpenSandbox runtime are arm64. The builder rejects a root-level `.dockerignore`,
escaping/broken symlinks, special filesystem objects, and malformed collection
roots so capsule inputs cannot be silently omitted or read outside the supplied
directory.

## Running the Dataset Server

Create a `server.yaml` config file:

```yaml
dataset:
  hf_dataset: EdisonScientific/bixbench_hypothesis
  capsule_dir: /path/to/capsules/
  save_dir: /path/to/outputs/ # optional, for saving rollout artifacts

  # Optional best-effort reproducibility mode. Independent kernel, scheduler,
  # and rubric-model seeds are derived from this base seed and the env index.
  deterministic: true
  seed: 1234

api_key: YOUR_API_KEY # or env var name like HYPOTEST_SERVER_API_KEY
```

In deterministic mode, each environment index receives stable, domain-separated
seeds. Python's hash seed is fixed before kernel startup, and Python's global RNG
and NumPy's legacy global RNG are initialized by a hidden bootstrap on start,
reset, and recovery. Explicit generator objects and other libraries still need
their own seed. The rubric model receives its own per-index seed, derived without
consuming the environment RNG. Remote model providers, live cluster capacity,
external I/O, and some parallel/GPU operations can still be nondeterministic, so
this mode is best-effort rather than a bit-for-bit guarantee.

Rebuild and roll the kernel image with the dataset server before enabling this
mode. Seeded requests validate the seed echoed by the kernel, so stale warm pods
fail fast instead of silently ignoring it.

### Episode time accounting

Wall-clock accounting remains the default. To exclude policy-generation queue
latency and count only kernel execution, configure:

```yaml
dataset:
  execution_config:
    time_accounting:
      mode: kernel_execution
      generation_latency:
        mode: rolling_p95
        seconds_per_generation: 12.5
```

`generation_latency.mode` may be `none` (the default), `reported`, `fixed`,
`rolling_mean`, `rolling_p95`, or `token_throughput`. `reported` consumes the
measured `generation_seconds` attached to every model turn by the NeMo Gym
resources server and is the preferred non-simulated mode. The rolling modes
identify the source of a metrics snapshot; the supplied value is charged once
per policy generation/environment step. The
environment deliberately does not infer a random latency distribution from only
mean and p95. “Kernel execution” is the backend-reported elapsed duration of a
cell, not summed OS CPU-core utilization. Kernel execution, measured
generation, simulated generation, and observed wall-clock totals are all
written to rollout metadata. Invalid combinations—such as adding generation
latency to `wall_clock` mode or supplying a non-positive estimate—are rejected
during configuration validation.

Execution accounting is keyed by the logical cell request ID, so transport or
Ray wait retries cannot charge the same cell more than once. Client-observed
retry/wait duration is never substituted for a missing backend execution time;
it is recorded only as diagnostic metadata.

For measured generation accounting with no latency model:

```yaml
generation_latency:
  mode: reported
```

This mode requires each `nemo_gym.step_context.model_turns` entry to carry a
finite, non-negative `generation_seconds` value. Queue time is intentionally
excluded by the producer of that measurement.

#### NeMo Gym integration contract

To use `generation_latency.mode: reported`, the NeMo Gym resources server (or
the adapter immediately in front of Hypotest) must attach the following
versioned envelope to the `ToolRequestMessage.info` passed to `env.step()`:

```json
{
  "nemo_gym": {
    "step_context": {
      "version": 1,
      "model_turns": [
        {
          "response_id": "stable-model-response-id",
          "turn_index": 1,
          "generation_seconds": 2.75,
          "usage": {
            "input_tokens": 1200,
            "output_tokens": 388,
            "total_tokens": 1588
          }
        }
      ]
    }
  }
}
```

`generation_seconds` is measured model-service work for that response (for
example, prefill plus decode), from admission to generation until the completed
response is available. It must exclude upstream request queueing and rollout
scheduling. Gym must forward every newly completed model turn since the prior
environment step; `response_id` must remain stable when the same action is
retried and must be new when the policy is genuinely regenerated. Hypotest uses
that ID to suppress duplicate charges. The adapter must preserve `info` through
batching, serialization, and `ToolRequestMessage` construction rather than
placing the measurement in tool arguments or sleeping to simulate it.

Gym may retain a generous wall-clock watchdog for stuck rollouts, but its model
budget or normal truncation decision must use the environment's accounted
remaining time, not elapsed wall time since allocation or reset. Otherwise
OpenSandbox allocation, capsule setup, submit-and-poll transport waits, proxy
backoff, and cleanup would still shorten the episode outside Hypotest even
though they are excluded by its clock. The final `time_accounting` object from
the environment result should be retained in rollout artifacts; Gym should not
recompute it from end-to-end duration.

With this contract, the training configuration is simply:

```yaml
time_accounting:
  mode: kernel_execution
  generation_latency:
    mode: reported
```

No simulated latency is required. Hypotest charges backend-reported kernel
execution plus the supplied measured model-generation durations, while keeping
wall time as diagnostic metadata.

For the cluster-side GBS1024 reproduction, exact cleanup workflow, pass/fail
gates, and NeMo Gym/NRL test-train handoff, see
[`docs/opensandbox-stress-test-handoff.md`](docs/opensandbox-stress-test-handoff.md).

For deterministic latency proportional to generated output length, use:

```yaml
generation_latency:
  mode: token_throughput
  output_tokens_per_second: 141
```

This mode consumes the versioned model-turn metadata attached by the NeMo Gym
Aviary resources server. It charges `output_tokens / output_tokens_per_second`
before tool execution and fails explicitly if token usage is absent. Input and
total token counts are carried for future prefill models but are not charged.

Alternatively, you can point to a local JSONL file instead of the HuggingFace dataset:

```yaml
dataset:
  problem_jsonl: /path/to/tasks.jsonl
  capsule_dir: /path/to/capsules/

api_key: YOUR_API_KEY
```

Start the server:

```bash
make server CONFIG=server.yaml
```

## Running Benchmarks

Create a `benchmark.yaml` config file:

```yaml
results_dir: benchmark_results/

api_key: YOUR_API_KEY # must match server api_key

agent_config:
  agent_kwargs:
    llm_model:
      name: openai/gpt-5
      temperature: 1.0
      timeout: 600
      config:
        model_list:
          - model_name: openai/gpt-5
            litellm_params:
              model: openai/gpt-5
              timeout: 600
              temperature: 1.0
              reasoning_effort: medium
```

Run the benchmark:

```bash
uv run python src/hypotest/benchmark_agent.py benchmark.yaml
```
