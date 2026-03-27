import argparse
from baseline import run as run_baseline 
from speculative import run as run_speculative
from common import load_models
from config import ModelPair, BenchmarkConfig


supported_methods = ["baseline", "speculative_greedy", "speculative"]

parser = argparse.ArgumentParser(description="Run speculative decoding experiments.")
parser.add_argument("--target", type=str, required=True, help="The Hugging Face Hub model ID to use as target (e.g., 'distilgpt2').")
parser.add_argument("--draft", type=str, default=None, help="The The Hugging Face Hub model ID of draft model to use for speculative decoding (not used in baseline).")
parser.add_argument("--prompt", type=str, required=True, help="The input prompt to use for generation.")
parser.add_argument("--max_new_tokens", type=int, default=32, help="The maximum number of new tokens to generate.")
parser.add_argument("--method", type=str, default="baseline", help=f"The decoding method to use (e.g., '{supported_methods}').")
parser.add_argument("--device", type=str, default="cpu", help="The device to run the benchmark on (e.g., 'cpu' or 'cuda' or 'mps').")
parser.add_argument("--gamma", type=int, default=4, help="Fixed lookahead gamma for speculative methods.")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for stochastic speculative decoding.")
parser.add_argument("--adaptive", type=str, default=None, help="Adaptive strategy for adjusting gamma (e.g., 'aimd').")
parser.add_argument("--gamma_range", type=int, nargs=2, default=[1, 16], help="Range of gamma values for adaptive strategies (e.g., --gamma_range 1 16).")

if __name__ == '__main__':
    args = parser.parse_args()

    target, draft, tokenizer = load_models(args.target, args.draft, args.device)

    model_pair = ModelPair(
        target=target,
        target_name=args.target,
        draft=draft,
        draft_name=args.draft,
        tokenizer=tokenizer
    )

    benchmark_config = BenchmarkConfig(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        device=args.device,
        method=args.method,
        temperature=args.temperature,
        apdaptive=args.adaptive,
        gamma_range=tuple(args.gamma_range)
    )

    if args.method == "baseline":
        run_baseline(model_pair, benchmark_config)
    elif args.method == "speculative_greedy" or args.method == "speculative":
        run_speculative(model_pair, benchmark_config)
    else:
        raise ValueError(f"Unsupported method: {args.method}. Supported methods: {supported_methods}.")
