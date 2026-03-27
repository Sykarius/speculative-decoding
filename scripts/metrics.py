import time
from dataclasses import dataclass, field, asdict
import json
import os
from datetime import UTC, datetime
from typing import List, Callable, Tuple
import torch
import inspect

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
    gamma: int | None = None
    gamma_range: Tuple[int, int] | None = None
    temperature: float | None = None
    adaptive: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    
@dataclass
class StepTrace:
    step_id: int
    draft_window_size: int
    accepted_tokens: int
    draft_time_ms: float
    verify_time_ms: float

    @property
    def efficiency(self):
        if self.draft_window_size == 0:
            return 0.0
        return self.accepted_tokens / self.draft_window_size
    
    def to_dict(self):
        data = asdict(self)
        data['efficiency'] = self.efficiency
        return data


@dataclass
class SpeculativeMetrics:
    drafted_tokens_total: int = 0
    accepted_tokens_total: int = 0
    verification_rounds: int = 0
    step_traces: List[StepTrace] = field(default_factory=list)

    @property
    def acceptance_rate(self):
        return self.accepted_tokens_total / self.drafted_tokens_total if self.drafted_tokens_total > 0 else 0.0

    def to_dict(self):
        data = asdict(self)
        data['acceptance_rate'] = self.acceptance_rate
        data['step_traces'] = [trace.to_dict() for trace in self.step_traces]
        return data
    
    def update(self, proposed: list, accepted: int, k: int, draft_time_ms: float, verify_time_ms: float):
        self.drafted_tokens_total += len(proposed)
        self.accepted_tokens_total += accepted
        self.verification_rounds += 1
        self.step_traces.append(StepTrace(
            step_id=self.verification_rounds,
            draft_window_size=k,
            accepted_tokens=accepted,
            draft_time_ms=draft_time_ms,
            verify_time_ms=verify_time_ms
        ))


@dataclass
class Session:
    iteration_times: list = field(default_factory=list)
    metadata: BenchmarkMetadata = None
    generated_tokens: int = 0
    generated: list = field(default_factory=list)
    output_text: str = ""
    speculative_metrics: SpeculativeMetrics = field(default_factory=SpeculativeMetrics)
    first_burst_tokens: int = 0

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

    def record(self, tokens: list, iteration_time: float):
        self.iteration_times.append(iteration_time)
        self.generated_tokens += len(tokens)
        if not self.generated:
            self.first_burst_tokens = len(tokens)
        self.generated.extend(tokens)

    def record_speculative(self, proposed: list, accepted: int, k: int, draft_time_ms: float, verify_time_ms: float):
        self.speculative_metrics.update(proposed, accepted, k, draft_time_ms, verify_time_ms)

    def record_output(self, output_text: str):
        self.output_text = output_text

    @property
    def total_elapsed(self):
        return sum(self.iteration_times)

    @property
    def tokens_per_sec(self):
        return self.generated_tokens / self.total_elapsed if self.total_elapsed > 0 else 0.0
    
    @property
    def time_per_output_token(self):
        decode_tokens = self.generated_tokens - self.first_burst_tokens
        if decode_tokens <= 0:
            return 0.0
        return (self.total_elapsed - self.time_to_first_token) / decode_tokens
    
    @property
    def time_to_first_token(self):
        return self.iteration_times[0] if self.iteration_times else 0.0
    
    def to_dict(self):
        data = asdict(self)
        data['speculative_metrics'] = self.speculative_metrics.to_dict()
        data['total_elapsed'] = self.total_elapsed
        data['time_per_output_token'] = self.time_per_output_token
        data['tokens_per_sec'] = self.tokens_per_sec
        data['time_to_first_token'] = self.time_to_first_token
        return data
    
    def write(self, filepath):
        summary = self.to_dict()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIR, filepath)
        with open(full_path, 'a') as f:
            f.write(json.dumps(summary) + "\n")
        print("Saved to:", full_path)

