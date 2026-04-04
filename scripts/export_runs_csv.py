"""
Flatten JSONL benchmark logs under results/raw/ into a single CSV for tables / spreadsheets.

Does not replace JSONL — it is an export layer for reporting. Run after benchmarks:

  uv run python scripts/export_runs_csv.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FIELDNAMES = [
    "jsonl_file",
    "line",
    "timestamp",
    "target_model",
    "draft_model",
    "method",
    "device",
    "dtype",
    "prompt",
    "prompt_tokens",
    "max_new_tokens",
    "gamma",
    "adaptive",
    "generated_tokens",
    "time_to_first_token",
    "time_per_output_token",
    "tokens_per_sec",
    "total_elapsed",
    "acceptance_rate",
    "verification_rounds",
    "drafted_tokens_total",
    "accepted_tokens_total",
]


def _prompt_cell(text: str | None, max_len: int = 500) -> str:
    if text is None:
        return ""
    s = " ".join(text.split())
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def flatten_line(obj: dict, jsonl_name: str, line_no: int) -> dict[str, object]:
    md = obj.get("metadata") or {}
    sm = obj.get("speculative_metrics") or {}
    if not isinstance(sm, dict):
        sm = {}
    row: dict[str, object] = {
        "jsonl_file": jsonl_name,
        "line": line_no,
        "timestamp": md.get("timestamp"),
        "target_model": md.get("target_model"),
        "draft_model": md.get("draft_model"),
        "method": md.get("method"),
        "device": md.get("device"),
        "dtype": md.get("dtype"),
        "prompt": _prompt_cell(md.get("prompt")),
        "prompt_tokens": md.get("prompt_tokens"),
        "max_new_tokens": md.get("max_new_tokens"),
        "gamma": md.get("gamma"),
        "adaptive": md.get("adaptive"),
        "generated_tokens": obj.get("generated_tokens"),
        "time_to_first_token": obj.get("time_to_first_token"),
        "time_per_output_token": obj.get("time_per_output_token"),
        "tokens_per_sec": obj.get("tokens_per_sec"),
        "total_elapsed": obj.get("total_elapsed"),
        "acceptance_rate": sm.get("acceptance_rate"),
        "verification_rounds": sm.get("verification_rounds"),
        "drafted_tokens_total": sm.get("drafted_tokens_total"),
        "accepted_tokens_total": sm.get("accepted_tokens_total"),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results/raw/*.jsonl to a CSV summary.")
    parser.add_argument(
        "--glob",
        dest="pattern",
        default=str(REPO_ROOT / "results" / "raw" / "*.jsonl"),
        help="Glob pattern for JSONL files (default: results/raw/*.jsonl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "processed" / "runs.csv",
        help="Output CSV path (default: results/processed/runs.csv).",
    )
    args = parser.parse_args()

    paths = sorted(glob(args.pattern))
    if not paths:
        print(f"No files matched: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, object]] = []
    for path_str in paths:
        path = Path(path_str)
        name = path.name
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skip {name}:{i}: invalid JSON ({e})", file=sys.stderr)
                    continue
                rows.append(flatten_line(obj, name, i))

    if not rows:
        print("No JSON lines parsed; nothing to write.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
