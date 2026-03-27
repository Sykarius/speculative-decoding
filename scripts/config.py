from typing import Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelPair:

    target: PreTrainedModel
    target_name: str
    draft: PreTrainedModel | None
    draft_name: str | None
    tokenizer: PreTrainedTokenizer

@dataclass(frozen=True)
class BenchmarkConfig:
    prompt: str
    max_new_tokens: int
    gamma: int | None
    device: str
    method: str
    temperature: float | None = None
    apdaptive: str | None = None
    gamma_range: Tuple[int, int] | None = None