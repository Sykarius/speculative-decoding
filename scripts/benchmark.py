import argparse
from pathlib import Path
import yaml
import os
from datasets import load_dataset

from baseline import run as run_baseline 
from speculative import run as run_speculative
from config import MethodType, ModelInput, ModelPair, BenchmarkConfig
from common import load_models, set_global_seed

BASE_DATA_PATH = "./data/nvidia___speed-bench/"

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run speculative decoding experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML file containing all benchmark arguments.")

    return parser

def load_yaml_config(path: str) -> BenchmarkConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML config is empty: {config_path}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict at top-level. Got: {type(data).__name__}")

    benchmark_config = BenchmarkConfig(**data)
    return benchmark_config


def run_benchmark_prompt(model_pair: ModelPair, benchmark_config: BenchmarkConfig, model_input: ModelInput) -> str:
    if benchmark_config.method == "baseline":
        return run_baseline(model_pair, benchmark_config, model_input)
    elif benchmark_config.method == "speculative":
        return run_speculative(model_pair, benchmark_config, model_input)
    else:
        raise ValueError(f"Unsupported method: {benchmark_config.method}. Supported methods: {MethodType.__args__}.")
    

def run_benchmark_data(model_pair: ModelPair, benchmark_config: BenchmarkConfig, data_path: str) -> str:
    full_path = os.path.join(BASE_DATA_PATH, data_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file not found: {full_path}")
    data_files = {
        "test": full_path
    }
    dataset = load_dataset("arrow", data_files=data_files)["test"]
    for item in dataset:
        num_turns = len(item["turns"])
        context = ""
        for turn_idx in range(num_turns):
            context += item["turns"][turn_idx]
            model_input = ModelInput(
                prompt=context,
                category=item["category"],
                sub_category=item["sub_category"],
                question_id=item["question_id"],
                multiturn=item["multiturn"],
                difficulty=item["difficulty"]
            )
            output_txt = run_benchmark_prompt(model_pair, benchmark_config, model_input)
            context += "\n" + output_txt + "\n"


def run_warmup(model_pair: ModelPair, benchmark_config: BenchmarkConfig):
    if benchmark_config.warmup_steps > 0:
        set_global_seed(benchmark_config.seed)
        print(f"Warming up for {benchmark_config.warmup_steps} steps...")

        warmup_config = benchmark_config.model_copy(
            update= {
                "max_new_tokens": benchmark_config.gamma + 1,
                "output": "warmup_output.jsonl"
            }
        )

        for i in range(benchmark_config.warmup_steps):
            dummy_input = ModelInput(prompt=f"Hello, this is a short test prompt")
            run_benchmark_prompt(model_pair, warmup_config, dummy_input)


def run_benchmark(config_path: str):
    benchmark_config = load_yaml_config(config_path)
    target_model, draft_model, tokenizer = load_models(benchmark_config)
    model_pair = ModelPair(
        tokenizer=tokenizer,
        target=target_model,
        target_name=benchmark_config.target_model,
        draft=draft_model,
        draft_name=benchmark_config.draft_model
    )

    run_warmup(model_pair, benchmark_config)
    set_global_seed(benchmark_config.seed)

    if benchmark_config.prompt:
        model_input = ModelInput(prompt=benchmark_config.prompt)
        run_benchmark_prompt(model_pair, benchmark_config, model_input)
    else:
        run_benchmark_data(model_pair, benchmark_config, benchmark_config.data)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run_benchmark(args.config)
