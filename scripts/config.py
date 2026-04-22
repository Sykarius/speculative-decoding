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

    @model_validator(mode="before")
    @classmethod
    def load_models(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        
        data_dict = data.copy()
        
        if "target" not in data_dict:
            raise ValueError(f"Model config must contain 'target' key with the name of the target model to load. Got keys: {list(data_dict.keys())}")
    
        if "tokenizer" not in data_dict or isinstance(data_dict["tokenizer"], str):
            token_path = data_dict.get("tokenizer") or data_dict["target"]
            data_dict["tokenizer"] = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
        
        if isinstance(data_dict.get("target"), str):
            print(f"Loading target: {data_dict['target']}...")
            data_dict["target_name"] = data_dict["target"]
            data_dict["target"] = AutoModelForCausalLM.from_pretrained(data_dict["target"], local_files_only=True)
        
        if data_dict.get("draft") and isinstance(data_dict["draft"], str):
            print(f"Loading draft: {data_dict['draft']}...")
            data_dict["draft_name"] = data_dict["draft"]
            data_dict["draft"] = AutoModelForCausalLM.from_pretrained(data_dict["draft"], local_files_only=True)

        return data_dict


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
    output: str = Field(default="output", pattern=r"^.*\.jsonl$")
    max_new_tokens: int = Field(default=32, gt=0)
    gamma: Optional[int] = Field(default=None, gt=0)
    device: DeviceType = Field(default="cpu")
    temperature: float = Field(default=1.0, gt=0.0)
    adaptive: Optional[AdaptiveConfig] = None

    @model_validator(mode="after")
    def check_speculative_requirements(self) -> Self:
        if self.method in ("speculative_greedy", "speculative"):
            if self.gamma is None:
                raise ValueError(f"--gamma is required for method '{self.method}'")
        
        return self
    

class InputConfig(BaseModel):
    prompt: Optional[str] = None
    data: Optional[str] = None

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
