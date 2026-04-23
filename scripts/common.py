import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import profile
from config import BenchmarkConfig
import numpy as np
import random


def tokenize(tokenizer, prompt: str, device: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def greedy_token(logits):
    return int(torch.argmax(logits[:, -1, :], dim=-1).item())

@profile
def draft_tokens(draft_model, input_ids, step_k, device):
    proposed = []
    draft_ids = input_ids
    draft_logits = []
    for _ in range(step_k):
        draft_outputs = draft_model(input_ids=draft_ids)
        logits = draft_outputs.logits
        draft_logits.append(logits[:, -1, :])
        token = greedy_token(logits)
        proposed.append(token)
        next_token = torch.tensor([[token]], device=device, dtype=torch.long)
        draft_ids = torch.cat([draft_ids, next_token], dim=1)
    
    proposed_tensor = torch.tensor([proposed], device=device, dtype=torch.long)
    verify_ids = torch.cat([input_ids, proposed_tensor], dim=1)
    draft_logits = torch.stack(draft_logits, dim=1)

    return proposed, verify_ids, draft_logits

def generate_output(session, inputs, tokenizer, device):
    full_ids = torch.tensor([session.generated], device=device)
        
    output_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    session.record_output(output_text)
    return output_text

def get_torch_dtype(dtype_str: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    return mapping.get(dtype_str, "auto")


def load_models(benchmark_config: BenchmarkConfig):
    model_dtype = get_torch_dtype(benchmark_config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(benchmark_config.target_model, local_files_only=True)
    
    target_model = AutoModelForCausalLM.from_pretrained(benchmark_config.target_model, local_files_only=True, torch_dtype=model_dtype)
    target_model.eval()
    target_model.to(benchmark_config.device)

    if benchmark_config.draft_model:
        draft_model = AutoModelForCausalLM.from_pretrained(benchmark_config.draft_model, local_files_only=True, torch_dtype=model_dtype)
        draft_model.eval()
        draft_model.to(benchmark_config.device)
    else:
        draft_model = None

    return target_model, draft_model, tokenizer

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Handle hardware-specific seeding
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
