from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, ConfigDict, Field, model_validator, SkipValidation
from typing import Literal, Self, Optional, Annotated, Union

DeviceType = Literal["cpu", "cuda", "mps"]
MethodType = Literal["baseline", "speculative_greedy", "speculative"]
AdaptiveType = Literal["aimd", "entropy", "jsd"]


class ModelPair(BaseModel):

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tokenizer: SkipValidation[PreTrainedTokenizer]
    target: SkipValidation[PreTrainedModel]
    target_name: str
    draft: Optional[SkipValidation[PreTrainedModel]] = None
    draft_name: Optional[str] = None


class BaseAdaptiveConfig(BaseModel):

    model_config = {"frozen": True}

    strategy: str
    gamma_min: int
    gamma_max: int
    step_size: int = Field(default=1, gt=0)
    decrease_factor: float = Field(default=0.5, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def gamma_range(self) -> Self:
        if self.gamma_min >= self.gamma_max:
            raise ValueError(f"--gamma_range is not valid")

        return self
    
class AIMDConfig(BaseAdaptiveConfig):
    strategy: Literal["aimd"] = "aimd"

class EntropyConfig(BaseAdaptiveConfig):
    strategy: Literal["entropy"] = "entropy"
    low_entropy_threshold: float = Field(default=5.0, gt=0.0)
    high_entropy_threshold: float = Field(default=7.0, gt=0.0)
    smoothing_factor: float = Field(default=0.9, gt=0.0, lt=1.0)
    warmup_steps: int = Field(default=10, gt=0)

class JSDConfig(BaseAdaptiveConfig):
    strategy: Literal["jsd"] = "jsd"
    low_jsd_threshold: float = Field(default=0.1, gt=0.0)
    high_jsd_threshold: float = Field(default=0.3, gt=0.0)
    high_entropy_threshold: float = Field(default=7.0, gt=0.0)
    smoothing_factor: float = Field(default=0.9, gt=0.0, lt=1.0)
    warmup_steps: int = Field(default=10, gt=0)

AdaptiveConfig = Annotated[
    Union[AIMDConfig, EntropyConfig, JSDConfig],
    Field(discriminator="strategy")
]

class BenchmarkConfig(BaseModel):

    model_config = {"frozen": True}
    
    method: MethodType
    target_model: str = Field(description="Name of the target model to load (e.g. 'gpt2-large').", min_length=1)
    output: str = Field(default="output", pattern=r"^.*\.jsonl$")
    max_new_tokens: int = Field(default=32, gt=0)
    gamma: Optional[int] = Field(default=None, gt=0)
    device: DeviceType = Field(default="cpu")
    temperature: float = Field(default=1.0, gt=0.0)
    adaptive: Optional[AdaptiveConfig] = None
    prompt: Optional[str] = None
    data: Optional[str] = None
    draft_model: Optional[str] = None
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "bfloat16"
    seed: float = Field(default=690)
    warmup_steps: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def check_speculative_requirements(self) -> Self:
        if self.method in ("speculative_greedy", "speculative"):
            if self.gamma is None:
                raise ValueError(f"--gamma is required for method '{self.method}'")
            if self.draft_model is None:
                raise ValueError(f"--draft is required for method '{self.method}'")
        
        return self
    
    @model_validator(mode="after")
    def check_prompt_or_data(self) -> Self:
        if not self.prompt and not self.data:
            raise ValueError(f"Either --prompt or --data must be provided.")
        if self.prompt and self.data:
            raise ValueError(f"Only one of --prompt or --data can be provided, not both.")
        return self

class ModelInput(BaseModel):
    prompt: str = Field(min_length=1)
    category: str | None = None
    sub_category: str | None = None
    question_id: str | None = None
    multiturn: bool | None = None
    difficulty: str | None = None
