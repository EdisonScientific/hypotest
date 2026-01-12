# hypotest

Jupyter kernel-based code execution environment for benchmarking agents.

## Installation

```bash
uv venv
uv sync
```

Place your problem JSONL file and capsule data directory somewhere accessible on your filesystem. You'll reference these paths in the server config.

## Running the Dataset Server

Create a `server.yaml` config file:

```yaml
dataset:
  problem_jsonl: /path/to/tasks.jsonl
  capsule_dir: /path/to/capsules/
  save_dir: /path/to/outputs/ # optional, for saving rollout artifacts

api_key: YOUR_API_KEY # or env var name like HYPOTEST_SERVER_API_KEY
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
