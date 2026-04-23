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


## Running the Speculative Decoding Benchmark

The benchmarking script is now configured using a YAML file. 

```sh 
python scripts/benchmark.py --config <path_to_config.yaml>
```

### Configuration Structure (YAML)

Below are the supported fields for your configuration file, mapped directly to the `BenchmarkConfig` schema.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `method` | `str` | **Required** | The decoding method to use. Must be `baseline`, `speculative_greedy`, or `speculative`. |
| `target_model` | `str` | **Required** | The Hugging Face Hub model ID to use as target (e.g., 'meta-llama/Llama-3.1-8B'). |
| `prompt` / `data` | `str` | **Required** | Provide exactly one of these. Use `prompt` for a single string input, or `data` for the path to a dataset. |
| `output` | `str` | `"output.jsonl"` | Path to the output metrics file. Must end with `.jsonl`. |
| `max_new_tokens` | `int` | `32` | The maximum number of new tokens to generate. Must be > 0. |
| `device` | `str` | `"cpu"` | Hardware to run the benchmark on (`cpu`, `cuda`, `mps`). |
| `dtype` | `str` | `"bfloat16"` | Precision for model weights (`float16`, `bfloat16`, `float32`, `auto`). |
| `seed` | `float` | `690` | Random seed to lock stochastic processes for reproducibility. |
| `warmup_steps` | `int` | `10` | Number of dummy runs to execute before recording benchmark metrics. |

#### Speculative Decoding Fields
If your `method` is `speculative` or `speculative_greedy`, the following fields are **required**:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `draft_model` | `str` | **Required** | The Hugging Face Hub model ID of the draft model. |
| `gamma` | `int` | **Required** | Fixed lookahead window size for speculative generation. Must be > 0. |
| `temperature` | `float` | `1.0` | Sampling temperature. Only applicable to the `speculative` method. |

#### Adaptive Fields
You can optionally define an `adaptive` block in your YAML to dynamically scale `gamma`. All adaptive blocks require `strategy`, `gamma_min`, and `gamma_max`.

**Shared Adaptive Fields:**
* `strategy`: (`"aimd"`, `"entropy"`, or `"jsd"`)
* `gamma_min` (int): Minimum lookahead window.
* `gamma_max` (int): Maximum lookahead window.
* `step_size` (int, default: 1): How much to increase the window.
* `decrease_factor` (float, default: 0.5): Multiplier to shrink window on failure.

**Strategy-Specific Fields:**
* **Entropy (`strategy: entropy`)**:
  * `low_entropy_threshold` (float, default: 5.0)
  * `high_entropy_threshold` (float, default: 7.0)
  * `smoothing_factor` (float, default: 0.9)
  * `warmup_steps` (int, default: 10)
* **JSD (`strategy: jsd`)**:
  * `low_jsd_threshold` (float, default: 0.1)
  * `high_jsd_threshold` (float, default: 0.3)
  * `high_entropy_threshold` (float, default: 7.0)
  * `smoothing_factor` (float, default: 0.9)
  * `warmup_steps` (int, default: 10)


### Example Configurations

#### 1. Baseline Autoregressive
```yaml
method: baseline
target_model: meta-llama/Llama-3.1-8B
prompt: "The future of AI is"
max_new_tokens: 50
device: mps
```

#### 2. Fixed-Window Speculative Greedy
```yaml
method: speculative_greedy
target_model: meta-llama/Llama-3.1-8B
draft_model: meta-llama/Llama-3.2-1B
prompt: "The future of AI is"
gamma: 4
max_new_tokens: 50
device: cuda
```

#### 3. Adaptive AIMD Speculative
```yaml
method: speculative
target_model: meta-llama/Llama-3.1-8B
draft_model: meta-llama/Llama-3.2-1B
data: "path/to/dataset.json"
gamma: 4
temperature: 1.0
max_new_tokens: 128
adaptive:
  strategy: aimd
  gamma_min: 1
  gamma_max: 16
  step_size: 2
  decrease_factor: 0.5
```