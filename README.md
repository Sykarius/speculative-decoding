# Speculative Decoding

Speculative Decoding is an inference optimization technique that accelerates Large Language Model (LLM) text generation by breaking the sequential "one-token-at-a-time" bottleneck

## Setup

Requires Python 3.13

### Installation
```sh
pip install uv
uv sync
```

Copy `.env.example` to `.env` when you use Hugging Face tokens or default benchmark model IDs (see [docs/BENCHMARKING.md](docs/BENCHMARKING.md)).

## Downloading Models

### Authentication

To access models on the Hugging Face Hub (especially private or gated models), you must provide a valid access token. This tool looks for a variable named HF_TOKEN. You can either set it in the respective shell environment or add it `.env` file in the project root as shown:

```.env
HF_TOKEN=hf_your_token_here
```

### Download

#### Model

```sh
python scripts/download.py --model <model_id> [OPTIONS]
```

#### Dataset
```sh
python scripts/download.py --dataset <dataset_path> [OPTIONS]
```

| Argument | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | `str` | No | `None` | The Hugging Face Hub model ID to download (e.g., 'distilgpt2'). |
| `--dataset` | `str` | No| `None` | The Hugging Face Hub dataset to download for benchmarking |
| `--revision` | `str` | No | `None` | The specific revision of the model/dataset to download (e.g., a branch name, tag, or commit hash). Default is 'main'. |
| `--path` | `str` | No | `None` | The local path to save the downloaded files in. |

**Note: Either `--model` or `--dataset` must be passed**

### Examples

#### 1. Standard Download

Downloads the latest version of a model to the default Hugging Face cache:
```sh
python scripts/download.py --model distilgpt2
```

#### 2. Download Specific Version
Download a model from a specific branch or using a unique commit hash for reproducibility:

```sh
python scripts/download.py --model openai-community/gpt2 --revision v1.1
```
#### 3. Save to Custom Directory
Useful for keeping model weights within a specific project folder (e.g., for containerization or offline use):

```sh
python scripts/download.py --model distilgpt2 --path ./models/
```


## Running the speculative decoding benchmark

```sh 
python scripts/benchmark.py --target <model_name> --prompt "<your_prompt>" [OPTIONS]
```

| Argument | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--target` | `str` | **Yes** | N/A | The Hugging Face Hub model ID to use as target (e.g., 'distilgpt2'). |
| `--prompt` | `str` | **Yes** | N/A | The input prompt to use for generation. |
| `--draft` | `str` | No | `None` | The Hugging Face Hub model ID of draft model to use for speculative decoding (not used in baseline). |
| `--max_new_tokens` | `int` | No | `32` | The maximum number of new tokens to generate. |
| `--method` | `str` | No | `baseline` | The decoding method to use. |
| `--device` | `str` | No | `cpu` | The device to run the benchmark on (e.g., 'cpu' or 'cuda' or 'mps'). |
| `--gamma` | `int` | No | `4` | Fixed lookahead for speculative methods. |
| `--temperature` | `float` | No | `1.0` | Temperature for stochastic speculative decoding.
| `--adaptive` | `str` | No | `None` | Type of adaptive strategy to use (eg. 'aimd') |
| `--gamma_range` | `str` | No | `None` | Bounds of the lookahead `gamma` when adaptive strategy is used |

### Supported Methods:

- `baseline`: Standard autoregressive language modeling. It generates one token at a time by selecting the highest-probability token from the target model's output distribution. This method does not utilize a draft model or speculative decoding techniques. 
- `speculative_greedy`: Linear speculative decoding with fixed lookahead `gamma`. Requires `--draft` to specify the draft model. It verifies the draft tokens greedily
- `speculative`: Linear speculative decoding with fixed lookahead `gamma`. Requires `--draft` to specify the draft model. It verifies the draft tokens stochastically

### Example

#### Baseline
```sh
python scripts/benchmark.py \
    --target 'meta-llama/Llama-3.1-8B' \
    --prompt "The future of AI is" \
    --max_new_tokens 50 \
    --device cpu
```

#### Fixed-Window Speculative Decoding Greedy Approach
```sh
python scripts/benchmark.py \
    --target 'meta-llama/Llama-3.1-8B' \
    --draft 'meta-llama/Llama-3.2-1B' \
    --prompt "The future of AI is" \
    --max_new_tokens 50 \
    --method speculative_greedy \
    --gamma 4 \
    --device cpu
```

#### Fixed-Window Speculative Decoding Stochastic Approach

```sh
python scripts/benchmark.py \
    --target 'meta-llama/Llama-3.1-8B' \
    --draft 'meta-llama/Llama-3.2-1B' \
    --prompt "The future of AI is" \
    --max_new_tokens 50 \
    --method speculative \
    --gamma 4 \
    --temperature 1.0 \
    --device cpu
```

#### Adaptive Speculative Decoding

```sh
python scripts/benchmark.py \
    --target 'meta-llama/Llama-3.1-8B' \
    --draft 'meta-llama/Llama-3.2-1B' \
    --prompt "The future of AI is" \
    --max_new_tokens 50 \
    --method speculative \
    --gamma 4 \
    --temperature 1.0 \
    --adaptive aimd \
    --gamma_range 1 16 \
    --device cpu
```

## Smoke suite (multi-prompt JSONL runs)

`prompts/smoke.txt` holds one prompt per line (`#` starts a comment; empty lines are skipped).  
`scripts/run_smoke_suite.py` calls `scripts/benchmark.py` once per prompt and method; each run **appends** a JSON line to the same files as manual runs (e.g. `results/raw/baseline.jsonl`, `results/raw/speculative_greedy_fixed.jsonl`).

```sh
uv run python scripts/run_smoke_suite.py \
  --target distilgpt2 \
  --draft distilgpt2 \
  --device cpu
```

Optional: `--methods baseline speculative_greedy speculative`, `--prompts-file path/to/prompts.txt`, `--dry-run` to print commands only.

Model IDs can come from **`BENCHMARK_TARGET` / `BENCHMARK_DRAFT`** in `.env` instead of `--target` / `--draft` (see `.env.example` and [docs/BENCHMARKING.md](docs/BENCHMARKING.md)).

## Export JSONL → CSV (optional)

For spreadsheet / report tables, flatten all `results/raw/*.jsonl` into `results/processed/runs.csv`:

```sh
uv run python scripts/export_runs_csv.py
```

JSONL remains the source of truth; CSV is regenerated on demand (see [docs/BENCHMARKING.md](docs/BENCHMARKING.md)).