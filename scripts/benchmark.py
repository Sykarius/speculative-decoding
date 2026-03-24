import argparse
from baseline import run as run_baseline 


supported_methods = ["baseline_greedy"]

parser = argparse.ArgumentParser(description="Run speculative decoding experiments.")
parser.add_argument("--target", type=str, required=True, help="The Hugging Face Hub model ID to use as target (e.g., 'distilgpt2').")
parser.add_argument("--draft", type=str, default=None, help="The The Hugging Face Hub model ID of draft model to use for speculative decoding (not used in baseline).")
parser.add_argument("--prompt", type=str, required=True, help="The input prompt to use for generation.")
parser.add_argument("--max_new_tokens", type=int, default=32, help="The maximum number of new tokens to generate.")
parser.add_argument("--method", type=str, default="baseline_greedy", help=f"The decoding method to use (e.g., '{supported_methods}').")
parser.add_argument("--device", type=str, default="cpu", help="The device to run the benchmark on (e.g., 'cpu' or 'cuda' or 'mps').")

if __name__ == '__main__':
    args = parser.parse_args()

    inputs = {
        "target_model": args.target,
        "draft_model": args.draft, 
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device
    }

    if args.method == "baseline_greedy":
        run_baseline(**inputs)
    else:
        raise ValueError(f"Unsupported method: {args.method}. Currently only 'baseline_greedy' is implemented.")
