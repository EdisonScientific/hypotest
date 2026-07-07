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

`generation_latency.mode` may be `none` (the default), `fixed`, `rolling_mean`,
`rolling_p95`, or `token_throughput`. The rolling modes identify the source of a
metrics snapshot; the supplied value is charged once per policy
generation/environment step. The
environment deliberately does not infer a random latency distribution from only
mean and p95. “Kernel execution” is the backend-reported elapsed duration of a
cell, not summed OS CPU-core utilization. Kernel execution, simulated generation,
and observed wall-clock totals are all written to rollout metadata. Invalid
combinations—such as adding generation latency to `wall_clock` mode or supplying
a non-positive estimate—are rejected during configuration validation.

Execution accounting is keyed by the logical cell request ID, so transport or
Ray wait retries cannot charge the same cell more than once. Client-observed
retry/wait duration is never substituted for a missing backend execution time;
it is recorded only as diagnostic metadata.

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
