import time
import os
from datetime import UTC, datetime
from typing import List, Callable
from pydantic import Field, computed_field, BaseModel
import torch
import inspect
from config import BenchmarkConfig, ModelInput

OUTPUT_DIR = "results/raw"

class DeviceTime:

    def __init__(self, device: str | torch.device):
        self.device = str(device).lower()
        self._start_time = 0.0
        self.elapsed_time = 0.0

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            torch.mps.synchronize()
    
    def __enter__(self):
        self._sync()
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sync()
        end_time = time.perf_counter()
        self.elapsed_time = end_time - self._start_time
    

def profile(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):

        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        device = bound_args.arguments.get('device')

        if device is None:
            raise ValueError(f"The decorated function {func.__name__} requires a 'device' argument for profiling.")
    
        with DeviceTime(device) as timer:
            result = func(*args, **kwargs)
        
        return result, timer.elapsed_time
    return wrapper


class BenchmarkMetadata(BenchmarkConfig, ModelInput):
    prompt_tokens: int
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_config(cls, config: BenchmarkConfig, model_input: ModelInput, **metadata_kwargs):
        merged_data = {
            **config.model_dump(),
            **model_input.model_dump(),
            **metadata_kwargs
        }
        return cls(**merged_data)

class StepTrace(BaseModel):
    step_id: int
    draft_window_size: int
    actual_draft_window: int
    accepted_tokens: int
    draft_time_ms: float
    verify_time_ms: float
    early_stop_time_ms: float

    @computed_field(return_type=float)
    @property
    def efficiency(self):
        if self.draft_window_size == 0:
            return 0.0
        return self.accepted_tokens / self.draft_window_size
    
class AdaptiveStep(BaseModel):
    adaptive_time_ms: float
    entropy: float | None = None
    js_distance: float | None = None

class SpeculativeMetrics(BaseModel):
    drafted_tokens_total: int = 0
    accepted_tokens_total: int = 0
    verification_rounds: int = 0
    step_traces: List[StepTrace] = Field(default_factory=list)
    adaptive_steps: List[AdaptiveStep] = Field(default_factory=list)

    @computed_field(return_type=float)
    @property
    def acceptance_rate(self):
        return self.accepted_tokens_total / self.drafted_tokens_total if self.drafted_tokens_total > 0 else 0.0
    
    def update(self, proposed: list, accepted: int, k: int, draft_time_ms: float, verify_time_ms: float, early_stop_time_ms: float):
        actual_draft_window = len(proposed)
        self.drafted_tokens_total += actual_draft_window
        self.accepted_tokens_total += accepted
        self.verification_rounds += 1
        self.step_traces.append(StepTrace(
            step_id=self.verification_rounds,
            draft_window_size=k,
            actual_draft_window=actual_draft_window,
            accepted_tokens=accepted,
            draft_time_ms=draft_time_ms,
            verify_time_ms=verify_time_ms,
            early_stop_time_ms=early_stop_time_ms
        ))
    
    def update_adaptive(self, adaptive_time_ms: float, entropy: float | None = None, js_distance: float | None = None):
        self.adaptive_steps.append(AdaptiveStep(
            adaptive_time_ms=adaptive_time_ms,
            entropy=entropy,
            js_distance=js_distance
        ))


class Session(BaseModel):
    iteration_times: list = Field(default_factory=list)
    metadata: BenchmarkMetadata = None
    generated_tokens: int = 0
    generated: list = Field(default_factory=list)
    output_text: str = ""
    speculative_metrics: SpeculativeMetrics = Field(default_factory=SpeculativeMetrics)
    first_burst_tokens: int = 0

    def record_metadata(self, config: BenchmarkConfig, model_input: ModelInput, **kwargs):
        self.metadata = BenchmarkMetadata.from_config(config, model_input, **kwargs)

    def record(self, tokens: list, iteration_time: float):
        self.iteration_times.append(iteration_time)
        self.generated_tokens += len(tokens)
        if not self.generated:
            self.first_burst_tokens = len(tokens)
        self.generated.extend(tokens)

    def record_speculative(self, proposed: list, accepted: int, k: int, draft_time_ms: float, verify_time_ms: float, early_stop_time_ms: float):
        self.speculative_metrics.update(proposed, accepted, k, draft_time_ms, verify_time_ms, early_stop_time_ms)
    
    def record_adaptive(self, adaptive_time_ms: float, entropy: float | None = None, js_distance: float | None = None):
        self.speculative_metrics.update_adaptive(adaptive_time_ms, entropy, js_distance)

    def record_output(self, output_text: str):
        self.output_text = output_text

    @computed_field(return_type=float)
    @property
    def total_elapsed(self):
        return sum(self.iteration_times)

    @computed_field(return_type=float)
    @property
    def tokens_per_sec(self):
        return self.generated_tokens / self.total_elapsed if self.total_elapsed > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def time_per_output_token(self):
        decode_tokens = self.generated_tokens - self.first_burst_tokens
        if decode_tokens <= 0:
            return 0.0
        return (self.total_elapsed - self.time_to_first_token) / decode_tokens
    
    @computed_field(return_type=float)
    @property
    def time_to_first_token(self):
        return self.iteration_times[0] if self.iteration_times else 0.0
    
    def write(self, filepath):
        summary = self.model_dump_json()
        full_path = os.path.join(OUTPUT_DIR, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'a') as f:
            f.write(summary + "\n")
        print("Saved to:", full_path)
