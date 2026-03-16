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

api_key: YOUR_API_KEY # or env var name like HYPOTEST_SERVER_API_KEY
```

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
