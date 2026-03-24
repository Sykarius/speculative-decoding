import time
from dataclasses import dataclass, field
import json
import os
from datetime import UTC, datetime

OUTPUT_DIR = "results/raw"


@dataclass(frozen=True)
class BenchmarkMetadata:
    target_model: str
    draft_model: str | None
    method: str
    device: str
    dtype: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class Session:
    first_token_time: float = 0.0
    token_timestamps: list = field(default_factory=list)
    metadata: BenchmarkMetadata = None
    generated_tokens: int = 0
    generated: list = field(default_factory=list)
    output_text: str = ""
    extra_metrics: dict = field(default_factory=dict)

    def record_metadata(self, **kwargs):
        """
        Record the metadata for the benchmark session. 
        This should be called at the start of the session with all relevant information.
        Inputs:
            target_model (str): The name of the model being benchmarked.
            draft_model (str or None): The name of the draft model used for speculative decoding, if any.
            method (str): The decoding method used (e.g., "autoregressive_greedy").
            device (str): The device on which the benchmark is run (e.g., "cuda:0").
            dtype (str): The data type used for the model (e.g., "float16").
            prompt (str): The input prompt used for generation.
            prompt_tokens (int): The number of tokens in the input prompt.
            max_new_tokens (int): The maximum number of new tokens generated.
        """
        self.metadata = BenchmarkMetadata(**kwargs)

    def start(self):
        self.token_timestamps = [time.perf_counter()]
    
    def record(self, tokens: list):
        current_time = time.perf_counter()
        if len(self.token_timestamps) == 1:
            self.first_token_time = current_time - self.token_timestamps[0]
        self.token_timestamps.append(current_time)
        self.generated_tokens += len(tokens)
        self.generated.extend(tokens)

    def record_output(self, output_text: str):
        self.output_text = output_text

    def record_extra(self, **kwargs):
        self.extra_metrics.update(kwargs)

    def summarize(self):
        total_elapsed = self.token_timestamps[-1] - self.token_timestamps[0] if self.token_timestamps else 0.0
        tokens_per_sec = self.generated_tokens / total_elapsed if total_elapsed > 0 else 0.0
        tpot_sec = (total_elapsed - self.first_token_time) / self.generated_tokens if self.generated_tokens > 0 else 0.0
        summary = {
            "timestamp": self.metadata.timestamp,
            "target_model": self.metadata.target_model,
            "draft_model": self.metadata.draft_model,
            "method": self.metadata.method,
            "device": self.metadata.device,
            "dtype": self.metadata.dtype,
            "prompt": self.metadata.prompt,
            "prompt_tokens": self.metadata.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "max_new_tokens": self.metadata.max_new_tokens,
            "ttft_sec": self.first_token_time,
            "tpot_sec": tpot_sec,
            "total_elapsed_sec": total_elapsed,
            "tokens_per_sec": tokens_per_sec,
            "output_text": self.output_text,
        }
        summary.update(self.extra_metrics)
        return summary
    
    def write_summary(self, filepath):
        summary = self.summarize()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, filepath), 'w') as f:
            json.dump(summary, f, indent=2)

        print("Saved to:", os.path.join(OUTPUT_DIR, filepath))

