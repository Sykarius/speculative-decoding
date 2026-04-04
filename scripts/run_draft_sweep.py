"""
Draft model sweep: one fixed --target, several --drafts, same prompts.

For each prompt:
  - optionally runs baseline once (target only, no draft)
  - runs speculative_greedy once per draft model

Each run invokes scripts/benchmark.py (JSONL append behavior unchanged).

Example (open models, CPU-friendly):
  uv run python scripts/run_draft_sweep.py \\
    --target openai-community/gpt2 \\
    --drafts distilgpt2 openai-community/gpt2 \\
    --device cpu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse command builder + prompt loader from smoke suite
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_smoke_suite import build_cmd, load_prompts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep multiple draft models against one target (same prompts).",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=REPO_ROOT / "prompts" / "smoke.txt",
        help="Newline-delimited prompts (# and empty lines skipped).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target model id (or BENCHMARK_TARGET in .env).",
    )
    parser.add_argument(
        "--drafts",
        nargs="+",
        required=True,
        help="One or more draft model ids to compare (e.g. distilgpt2 openai-community/gpt2).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda | mps (or BENCHMARK_DEVICE; default cpu).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adaptive", type=str, default=None)
    parser.add_argument("--gamma-range", type=int, nargs=2, default=[1, 16])
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline runs (only speculative_greedy per draft).",
    )
    parser.add_argument("--dry-run", action="store_true")
    load_dotenv(REPO_ROOT / ".env")
    args = parser.parse_args()

    target = args.target or os.environ.get("BENCHMARK_TARGET")
    device = args.device or os.environ.get("BENCHMARK_DEVICE") or "cpu"

    if not target:
        print(
            "Missing --target or BENCHMARK_TARGET in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts_path = args.prompts_file
    if not prompts_path.is_file():
        print(f"Prompt file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_prompts(prompts_path)
    if not entries:
        print(f"No prompts in {prompts_path}", file=sys.stderr)
        sys.exit(1)

    gamma_range = (args.gamma_range[0], args.gamma_range[1])

    for line_no, prompt in entries:
        if not args.no_baseline:
            cmd = build_cmd(
                method="baseline",
                target=target,
                draft=None,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                device=device,
                temperature=args.temperature,
                adaptive=args.adaptive,
                gamma_range=gamma_range,
            )
            label = f"{prompts_path.name}:{line_no} [baseline]"
            if args.dry_run:
                print(label)
                print(" ", " ".join(cmd))
            else:
                print(f"Running {label} ...", flush=True)
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        for draft in args.drafts:
            cmd = build_cmd(
                method="speculative_greedy",
                target=target,
                draft=draft,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                device=device,
                temperature=args.temperature,
                adaptive=args.adaptive,
                gamma_range=gamma_range,
            )
            label = f"{prompts_path.name}:{line_no} [speculative_greedy draft={draft}]"
            if args.dry_run:
                print(label)
                print(" ", " ".join(cmd))
            else:
                print(f"Running {label} ...", flush=True)
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
