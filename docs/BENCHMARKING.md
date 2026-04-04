# Benchmarking guide

## Outputs

Runs append **JSON Lines** under `results/raw/` (e.g. `baseline.jsonl`, `speculative_greedy_fixed.jsonl`).  
That format is the project default: one JSON object per line, easy to concatenate many trials.

## Model choices (team decision)

Until you lock IDs in writing, use two **profiles**:

| Profile | Target | Draft | Typical use |
|--------|--------|-------|-------------|
| **Smoke** | `distilgpt2` | `distilgpt2` | Fast sanity checks on CPU; no gating. |
| **Report (proposed)** | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` | Closer to README examples; needs **HF access**, **`HF_TOKEN`**, and usually **GPU** for reasonable latency. |

Agree with your teammate on: exact Hub IDs, **device** (`cuda` vs `mps` vs `cpu`), and **one** environment for “canonical” tables in the final writeup.

## Config without typing long flags

1. Copy `.env.example` → `.env` and set `BENCHMARK_TARGET` / `BENCHMARK_DRAFT` (and `HF_TOKEN` for Llama).
2. Run the smoke suite **without** `--target` / `--draft` if those env vars are set:

```sh
uv run python scripts/run_smoke_suite.py --device cpu
```

CLI arguments still override env when provided.

## Commands

Single prompt (manual):

```sh
uv run python scripts/benchmark.py --target distilgpt2 --prompt "Hello" --method baseline --device cpu
```

Multi-prompt smoke suite:

```sh
uv run python scripts/run_smoke_suite.py --target distilgpt2 --draft distilgpt2 --device cpu
```

Dry-run (print subprocess commands only):

```sh
uv run python scripts/run_smoke_suite.py --target distilgpt2 --draft distilgpt2 --dry-run
```

## Datasets

Benchmark text can come from `prompts/smoke.txt` or from Hugging Face datasets (see `scripts/download.py --dataset ...` and `nvidia/SPEED-Bench`). Wiring SPEED-Bench prompts into the runner is a follow-up (“full experiment harness”).

## CSV export (optional)

JSONL remains the canonical log. To build tables for slides or sheets, flatten all `results/raw/*.jsonl` into one CSV:

```sh
uv run python scripts/export_runs_csv.py
```

Default output: `results/processed/runs.csv` (gitignored — regenerate anytime).

## Draft sweep (multiple drafts, one target)

```sh
uv run python scripts/run_draft_sweep.py \
  --target openai-community/gpt2 \
  --drafts distilgpt2 openai-community/gpt2 \
  --device cpu
```

Uses `prompts/smoke.txt` by default. Filter `runs.csv` by `draft_model` to compare acceptance and throughput across drafts.
