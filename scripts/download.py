from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os
import argparse
from datasets import load_dataset

DATASET_PATH = "nvidia/SPEED-Bench"

parser = argparse.ArgumentParser(description="Download a model/dataset from Hugging Face Hub.")
parser.add_argument("--model", type=str, default=None, help="The Hugging Face Hub model ID to download (e.g., 'distilgpt2').")
parser.add_argument("--dataset", type=str, default=None, help="The Hugging Face Hub dataset to download for benchmarking")
parser.add_argument("--revision", type=str, default="main", help="The specific revision of the model to download (e.g., a branch name, tag, or commit hash). Default is 'main'.")
parser.add_argument("--path", type=str, default=None, help="The local directory to save the downloaded model. Default is 'None'.")

allow_patterns = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model"
]

def download_model(model, revision, local_dir, hf_token):
    path = snapshot_download(
        repo_id=model,
        token=hf_token,
        local_dir=local_dir,
        revision=revision,
        local_files_only=False,
        allow_patterns=allow_patterns
    )

    print(f"Model '{args.model}' downloaded to: {path}")

def download_dataset(dataset, revision, local_dir):
    ds = load_dataset(DATASET_PATH, dataset, cache_dir=local_dir, revision=revision)
    print(f'Completed download of {DATASET_PATH}/{dataset}')
    print(ds.column_names)
    

if __name__ == '__main__':
    load_dotenv()
    args = parser.parse_args()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        print('Warning: HF_TOKEN environment variable is not set')

    if args.model:
        download_model(args.model, args.revision, args.path, hf_token)
    elif args.dataset:
        download_dataset(args.dataset, args.revision, args.path)
    else:
        raise ValueError('Either dataset/model must be passed')