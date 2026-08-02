# OpenSandbox GBS1024 Qualification Handoff

This document is the operator handoff for reproducing the Hypotest
OpenSandbox qualification run from a cluster-side driver, cleaning up the
resulting allocations, and deciding whether the path is ready for a NeMo RL
(NRL) test train. It covers the mounted-capsule path only; the stress image is
the generic bioinformatics image and contains no capsule data.

## Scope and current result

The stress harness replays recorded notebook tool traffic through the real
`InterpreterEnv` and `OpenSandboxSandbox` path. It allocates one remote sandbox
per rollout, copies the selected cluster-mounted capsule into the sandbox's
writable workspace before readiness, then replays `run_cell`, `list_dir`, and
`reset_kernel` actions. Recorded `submit_answer` calls are deliberately skipped,
so this test qualifies sandbox lifecycle and notebook execution, not rubric
scoring or reward parity.

The reference run was `stress-c1024-fresh-20260801a`:

| Measurement | Reference |
| --- | ---: |
| Rollouts started / ready / completed / infrastructure-succeeded | 1024 / 1024 / 1024 / 1024 |
| Remote / fallback backends | 1024 / 0 |
| Sandbox actions | 22,040 / 22,040 expected |
| Cells / directory listings / kernel resets | 20,599 / 1,401 / 40 |
| OpenSandbox or transport failures | 0 |
| Allocation p50 / p95 / p99 | 183.30 s / 245.63 s / 248.27 s |
| Cell p50 / p95 / p99 | 28.97 s / 60.82 s / 95.42 s |
| Actions per second | 7.91 |
| Ready sandboxes per second | 3.45 |
| Driver wall time | 2,785.51 s (46m 25.5s) |
| Driver maximum RSS | 9,690.8 MiB |

That run was an **infrastructure pass**: every rollout and action completed on
OpenSandbox with no fallback or orchestration failure. Its summary nevertheless
has `passed: false` because the strict semantic gate found 396 cells that
errored where the historical trace did not, plus 321 cells that succeeded where
the trace had errored. The harness did not retain cell output or error class, so
those 396 are action-outcome mismatches rather than proven model mistakes or
proven resource faults. Do not describe the reference as strict replay parity,
and do not use the stress result alone to claim reward parity.

## Files in this handoff

- `scripts/run_opensandbox_stress_test.sh`: exact GBS1024 profile, active-run
  guard, progress logging, and exact-run cleanup trap.
- `scripts/replay_opensandbox_trace.py`: trace parsing, capsule matching, real
  Hypotest replay, sanitized performance telemetry, and summary generation.
- `scripts/audit_opensandbox_sandboxes.py`: run-tagged state audit and cleanup;
  it does not print sandbox IDs unless explicitly asked to export them.
- `scripts/probe_opensandbox_resources.py`: one-sandbox cgroup and burstability
  qualification.
- `scripts/probe_opensandbox_raw.py`: additive raw-SDK probe used to isolate
  lifecycle, image, mount, entrypoint, and kernel layers.
- `scripts/get_opensandbox_diagnostics.py`: redacted native logs/events for one
  explicitly supplied sandbox ID.
- `scripts/probe_opensandbox_kernel_wedge.py`: targeted kernel/server recovery
  probe for a slow or memory-heavy cell.
- `scripts/build_opensandbox_image.py`: capsule-free `linux/amd64` image build
  and registry push.

All generated qualification files live under `artifacts/`, which is ignored by
Git and by the Docker build context.

At checkout, `git status --short` should be empty. `.env`, `.env.*` (except the
placeholder-only `.env.example`), `server.yaml`, and `artifacts/` are ignored.
Before building, record `git rev-parse HEAD` alongside the unique image tag so
the cluster result can be tied back to source. Do not run the qualification from
a dirty checkout.

## Inputs that are intentionally not committed

The exact replay requires two private local artifacts:

1. A compact Tracer table JSONL containing exactly 1,024 rollouts from one
   already-selected training step. The reference filename was
   `gbs1024-step40-replay-manifest.jsonl`.
2. The verified task-to-mounted-capsule mapping. The reference filename was
   `gbs1024-replay.task-capsule-map.json`.

Copy these to a protected path on the cluster driver. Do not add them to Git:
the trace contains model actions, and the mapping contains private capsule
identities. The replay reports intentionally contain aggregate operational data
only and omit code, outputs, capsule names, image references, endpoints, and
credentials.

If the capsule mapping is unavailable or the mounted collection changed, create
a new one before the stress run:

```bash
.venv/bin/python scripts/replay_opensandbox_trace.py \
  /secure/path/gbs1024-step40-replay-manifest.jsonl \
  --trace-step 40 \
  --mounted-root /mnt/s3-data/data/bbh/capsules/edison-20260725/ \
  --prepare-only \
  --output artifacts/qualification/gbs1024-prepare.json \
  --run-id gbs1024-prepare
```

This inventories relative filenames on the mounted collection, resolves the 64
trace tasks, fingerprints ambiguous candidates, and writes ignored
`.capsule-index.json` and `.task-capsule-map.json` files next to the output.
Review the match count before reusing the map.

## Cluster-driver prerequisites

Use a stable Linux driver with at least 16 GiB RAM. The reference driver peaked
near 9.7 GiB. It also needs an open-file hard limit of at least 4,608; the wrapper
requests 8,192 and the replay driver fails before allocation if the effective
limit is too low. Run inside `tmux`, a batch allocation, or another session that
will survive an SSH disconnect.

Initialize the repository and its vendored dependency:

```bash
git switch akomaragiri/bbh-rl-v2.1-opensandbox
git submodule update --init --recursive
uv venv --python 3.12
uv sync --extra opensandbox
```

Supply secrets through the cluster's secret manager or an untracked,
permission-restricted environment file. The scripts never source `.env`
automatically. Required variables are:

```text
OPEN_SANDBOX_DOMAIN       remote OpenSandbox API host
OPEN_SANDBOX_API_KEY      remote API credential
OPEN_SANDBOX_PROTOCOL     optional protocol override
GITLAB_IMAGE              immutable or unique target image reference
REGISTRY_USERNAME         runtime private-registry pull identity
REGISTRY_PASSWORD         runtime private-registry password/token
```

The mounted-volume qualification does not need S3 credentials: the cluster
provides `/mnt/s3-data/data/bbh/capsules/edison-20260725/`, and each selected
capsule is copied into a writable sandbox-local workspace before `/health`
succeeds. Do not put registry, OpenSandbox, or object-store secrets in YAML,
command history, an OCI build argument, or a tracked file.

## Build and push the image

Build a fresh, unique `linux/amd64` image from the exact commit being qualified.
The image must use the full bioinformatics base and must not bake or cache
capsules:

```bash
./scripts/build_opensandbox_image.py \
  --image "${GITLAB_IMAGE}" \
  --platform linux/amd64 \
  --base-target full \
  --kernel-memory-limit-mb 57344 \
  --registry-auth \
  --push
```

Use a unique tag or digest so a stale node cache cannot masquerade as the new
build. Keep `GITLAB_IMAGE` set to that exact reference for the resource probe,
stress run, audit, and eventual dataset-server deployment. The build may be run
on a separate Docker builder; the cluster driver only needs registry pull
credentials.

## Preflight

First verify the request/limit split independently of Hypotest:

```bash
.venv/bin/python scripts/probe_opensandbox_resources.py \
  --cpu-request 0.25 \
  --memory-request-mb 512 \
  --cpu-limit 4 \
  --memory-limit-mb 65536 \
  --output artifacts/qualification/resource-probe.json
```

The probe must report the low requests, the high cgroup ceilings, and successful
CPU/memory use above the request. This confirms a Burstable allocation; it does
not prove the cluster has 1,024-sandbox capacity.

Audit the cluster before launch:

```bash
.venv/bin/python scripts/audit_opensandbox_sandboxes.py \
  --since-minutes 720 \
  --state Pending \
  --state Running \
  --state Paused
```

Do not queue a new GBS1024 run while the `gbs1024-trace-replay` purpose count is
nonzero. The wrapper enforces the same guard and never kills an unknown run.

## Run the exact stress profile

Choose a unique run ID and launch the wrapper in the foreground:

```bash
scripts/run_opensandbox_stress_test.sh \
  /secure/path/gbs1024-step40-replay-manifest.jsonl \
  /secure/path/gbs1024-replay.task-capsule-map.json \
  stress-c1024-YYYYMMDDa
```

The fixed qualification profile is:

- 1,024 rollouts at concurrency 1,024;
- lifecycle create concurrency 64 and kernel request concurrency 128;
- request 0.25 CPU / 512 MiB;
- limit 4 CPU / 65,536 MiB;
- inner Jupyter `RLIMIT_AS` 57,344 MiB, preserving server headroom;
- 50 GiB ephemeral storage;
- 900-second cell and readiness limits;
- three allocation attempts, a 14,400-second sandbox TTL, and no local fallback;
- 30-second progress samples and a 3,600-second driver wall watchdog;
- three real actions in a one-sandbox preflight before the 1,024 launch.

The wrapper records:

```text
artifacts/qualification/<run-id>.json
artifacts/qualification/<run-id>.log
artifacts/qualification/<run-id>.events.jsonl
artifacts/qualification/<run-id>.metrics.jsonl
```

The log and telemetry are safe for operational sharing by design, but still
review them before moving them outside the project. The events file contains
stable trace references and task indices, not code or output.

To monitor from another terminal:

```bash
tail -f artifacts/qualification/<run-id>.log

.venv/bin/python scripts/audit_opensandbox_sandboxes.py \
  --since-minutes 720 \
  --query-run-id <run-id>
```

Expect roughly five minutes for the complete allocation window and about 45--60
minutes for the reference workload. A slower cluster is not automatically a
failure as long as the run stays inside the explicit watchdog and makes forward
progress.

## Cancellation and cleanup

Use `Ctrl-C` or send `SIGTERM` to the wrapper. Its exit trap kills only
sandboxes whose `hypotest-run` metadata exactly equals its run ID, then polls
until the audit reaches zero. It does not sweep by age, image prefix, or broad
namespace.

If the driver is killed without running its trap, reconcile the exact tag:

```bash
.venv/bin/python scripts/audit_opensandbox_sandboxes.py \
  --since-minutes 720 \
  --query-run-id <run-id> \
  --kill-run-id <run-id>

.venv/bin/python scripts/audit_opensandbox_sandboxes.py \
  --since-minutes 720 \
  --query-run-id <run-id>
```

The final audit must report `recent_matching_image: 0`. Never kill an unlabeled
or differently tagged sandbox merely because it uses the same image.

## Pass/fail decision

Evaluate the generated summary in two layers.

### Infrastructure gate

All of the following must hold:

- `timed_out` is false;
- `started`, `ready`, `completed`, `succeeded`, `remote_backends`, and
  `peak_active` are all 1,024;
- `failed` and `fallback_backends` are zero;
- `sandbox_actions_replayed` equals `trace.expected_sandbox_actions`;
- `failure_types` and `root_cause_types` are empty;
- the exact-run cleanup audit reaches zero.

HTTP errors, allocation failures, kernel connection failures, and thrown
execution exceptions fail this gate. A notebook cell returning an ordinary
Python/R error does not, by itself, mean the infrastructure failed.

### Semantic replay gate

The harness's top-level `passed` field additionally requires
`unexpected_code_errors == 0`. A nonzero value means a recorded-success cell
returned an error marker in the new environment. `resolved_code_errors` is the
opposite drift and is reported for context.

The reference run passed the infrastructure gate but not this strict gate. If
the cluster reproduction still has unexpected cell errors, retain the summary
and compare the mismatch rate with the reference. Before treating them as model
behavior for reward decisions, run a privacy-safe targeted replay that records
only trace reference, action index, and normalized error class; do not capture
cell code or output.

## Failure isolation

Use additive probes rather than starting with the full Hypotest stack and
removing pieces:

1. `probe_opensandbox_raw.py`: raw create/close.
2. Add private image auth, platform, pull policy, and resource limits/requests.
3. Add the production entrypoint and mounted-capsule copy.
4. Add kernel startup and direct `/health`.
5. Replay one rollout through `InterpreterEnv`.
6. Return to the complete stress driver.

For one failed sandbox, explicitly export its ID to an ignored file with
`audit_opensandbox_sandboxes.py --ids-output`, then pass the selected ID to
`get_opensandbox_diagnostics.py`. The diagnostic tool redacts configured
credentials, URLs, and image references. Delete the exported ID file after the
incident is resolved.

## NeMo Gym and NRL handoff

The stress replay contains no live policy generation, so it deliberately uses
kernel-only accounting with generation mode `none`. The live NRL integration
must switch the dataset server to measured generation accounting:

```yaml
execution_config:
  time_accounting:
    mode: kernel_execution
    generation_latency:
      mode: reported
```

No simulated latency is required. The integration responsibilities are:

1. **Policy/model server:** measure `generation_seconds` for each model response
   from inference admission through completed response. Include prefill and
   decode, but exclude upstream request queueing and rollout scheduling.
2. **NeMo Gym agent/resources adapter:** attach every newly completed model turn
   to the `ToolRequestMessage.info` sent to Hypotest:

   ```json
   {
     "nemo_gym": {
       "step_context": {
         "version": 1,
         "model_turns": [{
           "response_id": "stable-response-id",
           "turn_index": 1,
           "generation_seconds": 2.75,
           "usage": {
             "input_tokens": 1200,
             "output_tokens": 388,
             "total_tokens": 1588
           }
         }]
       }
     }
   }
   ```

   Preserve the `info` envelope through batching, serialization, and message
   conversion. Re-delivery of the same response must retain its ID so Hypotest
   suppresses a duplicate charge; a genuine regeneration receives a new ID.
3. **Hypotest:** charge each logical cell once using the kernel server's
   `execution_time`, charge each unique reported model turn once, and retain
   allocation, proxy polling/backoff, and cleanup only as wall-clock telemetry.
4. **NRL rollout controller:** let the environment's accounted clock drive
   normal budget warnings, cell caps, and force-submit behavior. A separate,
   generous wall-clock watchdog may terminate a genuinely stuck rollout, but it
   must not be the model's ordinary time budget. Otherwise allocation and proxy
   queueing still shorten episodes outside Hypotest.
5. **Artifact writer:** retain the final `time_accounting` object from Hypotest;
   do not reconstruct it from end-to-end rollout duration.

Enable `reported` only after Gym supplies complete metadata. Missing or invalid
`generation_seconds` fails explicitly instead of silently falling back to wall
time or a simulated estimate.

### Time-accounting acceptance tests

Before a train job, verify these cases through the actual Gym adapter:

- 60 seconds of allocation plus 2 seconds of generation plus 3 seconds of cell
  execution charges 5 seconds, while wall telemetry remains at least 65 seconds;
- a delayed or retried `/execute` poll does not add proxy wait to the budget;
- resending one action with the same response and cell request IDs charges both
  components once;
- a genuine new model response gets a new ID and a new generation charge;
- `simulated_generation_seconds == 0` and
  `unreported_execution_time_count == 0` in the final metadata;
- Gym/NRL does not truncate at the model budget's wall-clock anniversary while
  accounted time remains available.

## Test-train reward validation

Only proceed to the NRL test train after the infrastructure gate passes and the
semantic mismatch decision is documented. Use the same model checkpoint,
generation parameters, dataset slice, capsule revision, rubric configuration,
and seeds as the comparison evaluation. The stress replay itself provides no
reward baseline because it skips `submit_answer`.

For the test train, retain and compare at minimum:

- rollout/task counts and task distribution;
- mean `score`, `is_pass_rollout`, `is_pass90_rollout`, and `zero_reward` rate;
- score histogram or per-task paired scores;
- rubric request, parse, and empty-response failure rates;
- average tool steps and cell-error rate;
- kernel execution, reported generation, accounted total, and wall-clock
  distributions;
- remote/fallback backend counts and OpenSandbox failure types.

For deterministic paired runs, investigate any per-task reward difference. For
stochastic generation, compare confidence intervals over a sufficiently large
matched sample rather than requiring identical individual rollouts. A reward
shift accompanied by fewer available accounted seconds, missing generation
metadata, rubric failures, or fallback use is an integration failure—not a
training result.

Roll out in this order: Gym metadata producer and propagation tests, Hypotest
dataset-server image/config, a one-rollout live smoke, a small matched eval,
then the NRL test train. Roll back by disabling the new training deployment;
do not silently replace measured generation with simulated latency.
