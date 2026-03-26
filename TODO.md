# Project TODO (Shared)

This file tracks remaining work from the project proposal. Keep it short and update as items are completed.

## Next implementations

- [ ] **Adaptive lookahead (AdaSD-inspired)**: dynamically choose `k` during speculative decoding (entropy/confidence/acceptance-based controller).
- [ ] **Block verification**: reduce overhead vs naive verification; compare latency/throughput.
- [ ] **Tree-lite speculation (SpecInfer-lite)**: shallow tree / small branching draft strategy + verification.

## Experiments + evaluation

- [ ] **Draft model sweep**: compare multiple draft sizes (e.g., `distilgpt2`, `gpt2`) vs a target; report acceptance + speed tradeoffs.
- [ ] **Experiment harness**: consistent prompt set(s), repeated trials, standardized output schema.
- [ ] **Results aggregation**: write a CSV summary table for baseline vs fixed-k vs future methods; (optional) plots.
- [ ] **Writeup**: final evaluation summary (tables + takeaways + bottlenecks + blockers).

