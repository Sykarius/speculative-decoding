# Project TODO (Shared)

## Done (merged on `master`)

- [x] **Baseline greedy decoding** — `scripts/baseline.py`, TTFT / TPOT / throughput, JSONL logs under `results/raw/`.
- [x] **Benchmark entrypoint** — `scripts/benchmark.py` with `ModelPair` / `BenchmarkConfig` (`common.py`, `config.py`).
- [x] **Fixed lookahead speculative (greedy verify)** — `--method speculative_greedy`, `--gamma`, `--draft`.
- [x] **Stochastic speculative** — `--method speculative`, `--temperature`, draft + target distributions.
- [x] **Adaptive lookahead (AIMD on γ)** — `--adaptive aimd`, `--gamma_range`; increases / shrinks γ from acceptance feedback.
- [x] **Metrics & traces** — per-step draft vs verify timing, acceptance aggregates, JSONL append.
- [x] **Model download** — `scripts/download.py --model …` (HF Hub).
- [x] **Benchmark dataset download** — `scripts/download.py --dataset …` (`nvidia/SPEED-Bench` via `datasets`).
- [x] **Milestone report draft** — `reports/milestone/report.tex`, `reports/references.bib`.
- [x] **Shared TODO file** — this document (keep it updated).

## Next implementations

- [ ] **Block verification** — compare naive vs batched / block-style verification; measure verify latency and overhead.
- [ ] **Tree-lite (SpecInfer-inspired)** — shallow tree / small branching draft + verification.
- [ ] **Optional: richer adaptive control** — e.g. entropy- or confidence-based γ (beyond AIMD), if time permits.

## Experiments + evaluation

- [ ] **Draft model sweep** — multiple draft sizes vs one target; table acceptance + speed + memory.
- [x] **Smoke prompt suite** — `prompts/smoke.txt` + `scripts/run_smoke_suite.py` (loops `benchmark.py`; appends JSONL).
- [x] **Benchmarking notes** — `docs/BENCHMARKING.md`, `.env.example` (smoke vs proposed Llama pair; `BENCHMARK_*` env defaults).
- [ ] **Experiment harness (full)** — SPEED-Bench subsets, repeated trials, seeds, reproducible batches.
- [ ] **Results aggregation** — CSV (and optional plots) from JSONL for milestone / final tables.
- [ ] **Final evaluation writeup** — integrated tables, takeaways, bottlenecks, blockers.
