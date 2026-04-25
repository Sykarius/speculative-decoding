import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import BenchmarkConfig
import numpy as np
import random


def tokenize(tokenizer, prompt: str, device: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def greedy_token(logits):
    return int(torch.argmax(logits[:, -1, :], dim=-1).item())


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


def compute_js_distance(p: torch.Tensor, q: torch.Tensor):
    """
    Computes JS Distance: sqrt(0.5 * KL(P||M) + 0.5 * KL(Q||M))
    where M = 0.5 * (P + Q)
    """
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    m =( 0.5 * (p + q)).clamp(min=1e-10)
    log_m = torch.log(m)
    
    kl_pm = torch.sum(p * (torch.log(p) - log_m), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - log_m), dim=-1)
    
    js_divergence = 0.5 * kl_pm + 0.5 * kl_qm
    return torch.sqrt(js_divergence.clamp(min=0.0))