"""
Run baseline and/or speculative benchmarks over a newline-delimited prompt file.

Each invocation calls scripts/benchmark.py as a subprocess. Results append to the
same JSONL files as a manual run (e.g. results/raw/baseline.jsonl).

Usage (from repo root):
  uv run python scripts/run_smoke_suite.py --target distilgpt2 --draft distilgpt2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_prompts(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append((i + 1, s))
    return lines


def build_cmd(
    method: str,
    target: str,
    draft: str | None,
    prompt: str,
    max_new_tokens: int,
    gamma: int,
    device: str,
    temperature: float,
    adaptive: str | None,
    gamma_range: tuple[int, int],
) -> list[str]:
    script = REPO_ROOT / "scripts" / "benchmark.py"
    cmd: list[str] = [sys.executable, str(script)]
    cmd += ["--target", target, "--prompt", prompt, "--max_new_tokens", str(max_new_tokens)]
    cmd += ["--method", method, "--device", device]
    if method != "baseline" and draft:
        cmd += ["--draft", draft]
    if method in ("speculative_greedy", "speculative"):
        cmd += ["--gamma", str(gamma)]
    if method == "speculative":
        cmd += ["--temperature", str(temperature)]
    if adaptive:
        cmd += ["--adaptive", adaptive, "--gamma_range", str(gamma_range[0]), str(gamma_range[1])]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark smoke suite over a prompt file.")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=REPO_ROOT / "prompts" / "smoke.txt",
        help="Path to newline-delimited prompts (# and empty lines skipped).",
    )
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--draft", type=str, default=None, help="Required for speculative methods.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "speculative_greedy"],
        choices=["baseline", "speculative_greedy", "speculative"],
        help="Methods to run per prompt (default: baseline speculative_greedy).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adaptive", type=str, default=None)
    parser.add_argument("--gamma-range", type=int, nargs=2, default=[1, 16])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running.",
    )
    args = parser.parse_args()

    prompts_path = args.prompts_file
    if not prompts_path.is_file():
        print(f"Prompt file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_prompts(prompts_path)
    if not entries:
        print(f"No prompts loaded from {prompts_path}", file=sys.stderr)
        sys.exit(1)

    for method in args.methods:
        if method != "baseline" and not args.draft:
            print(f"--draft is required for method {method}", file=sys.stderr)
            sys.exit(1)

    gamma_range = (args.gamma_range[0], args.gamma_range[1])

    for line_no, prompt in entries:
        for method in args.methods:
            cmd = build_cmd(
                method=method,
                target=args.target,
                draft=args.draft,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                device=args.device,
                temperature=args.temperature,
                adaptive=args.adaptive,
                gamma_range=gamma_range,
            )
            label = f"{prompts_path.name}:{line_no} [{method}]"
            if args.dry_run:
                print(label)
                print(" ", " ".join(cmd))
                continue
            print(f"Running {label} ...", flush=True)
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
