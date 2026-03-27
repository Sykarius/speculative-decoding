import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import profile


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
    full_list = inputs["input_ids"][0].tolist() + session.generated
    full_ids = torch.tensor([full_list], device=device)
        
    output_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    session.record_output(output_text)
    return output_text


def load_models(target_model: str, draft_model: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(target_model, local_files_only=True)
    
    target_model = AutoModelForCausalLM.from_pretrained(target_model, local_files_only=True)
    target_model.eval()
    target_model.to(device)

    if draft_model:
        draft_model = AutoModelForCausalLM.from_pretrained(draft_model, local_files_only=True)
        draft_model.eval()
        draft_model.to(device)
    else:
        draft_model = None

    return target_model, draft_model, tokenizer
