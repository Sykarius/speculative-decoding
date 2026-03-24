# Speculative Decoding

Speculative Decoding is an inference optimization technique that accelerates Large Language Model (LLM) text generation by breaking the sequential "one-token-at-a-time" bottleneck

## Setup

Requires Python 3.13

### Installation
```sh
pip install uv
uv sync
```

## Downloading Models

### Authentication

To access models on the Hugging Face Hub (especially private or gated models), you must provide a valid access token. This tool looks for a variable named HF_TOKEN. You can either set it in the respective shell environment or add it `.env` file in the project root as shown:

```.env
HF_TOKEN=hf_your_token_here
```

### Download

```sh
python scripts/download.py --model <model_id> [OPTIONS]
```

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
python scripts/download.py --model distilgpt2 --local_dir ./models/
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
| `--method` | `str` | No | `baseline_greedy` | The decoding method to use. |
| `--device` | `str` | No | `cpu` | The device to run the benchmark on (e.g., 'cpu' or 'cuda' or 'mps'). |
| `--k` | `int` | No | `4` | Fixed lookahead for `speculative_fixed_k`. |

### Supported Methods:

- `baseline_greedy`: Standard autoregressive language modeling. It generates one token at a time by selecting the highest-probability token from the target model's output distribution. This method does not utilize a draft model or speculative decoding techniques. 
- `speculative_fixed_k`: Linear speculative decoding with fixed lookahead `k`. Requires `--draft` to specify the draft model.


### Example

#### Basic Greedy
```sh
python scripts/benchmark.py --target distilgpt2 --prompt "The future of AI is" --max_new_tokens 50
```

#### Fixed-k Speculative
```sh
python scripts/benchmark.py --target distilgpt2 --draft distilgpt2 --prompt "The future of AI is" --method speculative_fixed_k --k 4 --max_new_tokens 50 --device cpu
```