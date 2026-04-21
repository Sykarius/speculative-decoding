import argparse
from marshal import version
from pathlib import Path
import yaml
from typing import Tuple
import os
from datasets import load_dataset

from baseline import run as run_baseline 
from speculative import run as run_speculative
from config import MethodType, ModelInput, ModelPair, BenchmarkConfig, InputConfig

BASE_DATA_PATH = "./data/nvidia___speed-bench/"

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run speculative decoding experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML file containing all benchmark arguments.")

    return parser

def load_yaml_config(path: str) -> Tuple[ModelPair, BenchmarkConfig, InputConfig]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML config is empty: {config_path}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict at top-level. Got: {type(data).__name__}")
    if "models" not in data or "config" not in data:
        raise ValueError(f"YAML config must contain 'models' and 'config' sections. Got keys: {list(data.keys())}")
    if "prompt" not in data and "data" not in data:
        raise ValueError(f"YAML config 'config' section must contain either 'prompt' or 'data'. Got keys: {list(data['config'].keys())}")

    model_pair = ModelPair(**data['models'])
    benchmark_config = BenchmarkConfig(**data['config'])
    input_config = InputConfig(prompt=data.get('prompt'), data=data.get('data'))

    return model_pair, benchmark_config, input_config


def run_benchmark_prompt(model_pair: ModelPair, benchmark_config: BenchmarkConfig, model_input: ModelInput) -> str:
    if benchmark_config.method == "baseline":
        return run_baseline(model_pair, benchmark_config, model_input)
    elif benchmark_config.method in ("speculative_greedy", "speculative"):
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

def run_benchmark(model_pair: ModelPair, benchmark_config: BenchmarkConfig, input_config: InputConfig):
    if input_config.prompt:
        model_input = ModelInput(prompt=input_config.prompt)
        run_benchmark_prompt(model_pair, benchmark_config, model_input)
    else:
        run_benchmark_data(model_pair, benchmark_config, input_config.data)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    model_pair, benchmark_config, input_config = load_yaml_config(args.config)
    run_benchmark(model_pair, benchmark_config, input_config)
