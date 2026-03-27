import torch
from metrics import DeviceTime, Session, profile
from common import generate_output, tokenize, draft_tokens
from config import ModelPair, BenchmarkConfig


@profile
def verify_tokens(target, verify_ids, proposed, base_idx, device):
    target_outputs = target(input_ids=verify_ids)
    logits = target_outputs.logits
    accepted = 0
    next_token = None
    gamma = len(proposed)

    pred_tokens = torch.argmax(logits[:, base_idx : base_idx + gamma + 1, :], dim=-1)
    proposed_tensor = torch.tensor(proposed, device=verify_ids.device, dtype=torch.long)
    matches = (pred_tokens[0, :-1] == proposed_tensor)

    accepted_mask = torch.cumprod(matches.to(torch.int), dim=0)
    accepted = int(accepted_mask.sum().item())
    next_token = pred_tokens[0, accepted].item()
    return accepted, next_token

@profile
def verify_tokens_stochastic(target, verify_ids, draft_logits, proposed, base_idx, temperature, device):
    target_outputs = target(input_ids=verify_ids)
    logits = target_outputs.logits
    accepted = 0
    next_token = None

    gamma = len(proposed)

    target_probs = torch.softmax(logits[:, base_idx : base_idx + gamma + 1, :] / temperature, dim=-1)
    draft_probs = torch.softmax(draft_logits / temperature, dim=-1)

    seq_id = torch.arange(gamma, device=device)
    target_token_probs = target_probs[0, seq_id, gamma]
    draft_token_probs = draft_probs[0, seq_id, gamma]
    acceptance_probs = torch.clamp(target_token_probs / draft_token_probs, max=1.0)
    rand_vector = torch.rand(len(proposed), device=device)
    is_accepted = rand_vector < acceptance_probs
    accepted_mask = torch.cumprod(is_accepted.to(torch.int), dim=0)
    accepted = int(accepted_mask.sum().item())

    if accepted < gamma:
        p_dist = draft_probs[0, accepted]
        q_dist = target_probs[0, base_idx + accepted]
        diff_dist = torch.clamp(q_dist - p_dist, min=0.0)
        diff_dist /= diff_dist.sum()
        next_token = torch.multinomial(diff_dist, num_samples=1).item()
    else:
        bonus_token_dist = target_probs[0, gamma]
        next_token = torch.multinomial(bonus_token_dist, num_samples=1).item()

    return accepted, next_token
    

def run(model_pair: ModelPair, benchmark_config: BenchmarkConfig):

    draft = model_pair.draft
    target = model_pair.target
    tokenizer = model_pair.tokenizer
    gamma = benchmark_config.gamma
    prompt = benchmark_config.prompt
    max_new_tokens = benchmark_config.max_new_tokens
    device = benchmark_config.device
    temperature = benchmark_config.temperature
    adaptive = benchmark_config.apdaptive
    gamma_range = benchmark_config.gamma_range

    if not draft:
        raise ValueError("speculative_greedy/speculative requires --draft <model_name>.")
    if gamma is None or gamma <= 0:
        raise ValueError("--gamma must be a positive integer for speculative_greedy/speculative")
    if benchmark_config.method == "speculative" and (temperature is None or temperature <= 0.0):
        raise ValueError("--temperature must be a positive float for speculative method")

    prompt_inputs = tokenize(tokenizer, prompt, device)
    prompt_ids = prompt_inputs["input_ids"]

    session = Session()
    session.record_metadata(
        target_model=model_pair.target_name,
        draft_model=model_pair.draft_name,
        method=benchmark_config.method,
        device=device,
        dtype=str(next(target.parameters()).dtype),
        prompt=prompt,
        prompt_tokens=int(prompt_ids.shape[1]),
        max_new_tokens=max_new_tokens,
        gamma = gamma,
        gamma_range=gamma_range,
        temperature=temperature,
        adaptive=adaptive
    )

    accepted = 0

    with torch.no_grad():
        while len(session.generated) < max_new_tokens:
            with DeviceTime(device) as dt:
                remaining = max_new_tokens - len(session.generated)
                step_k = min(gamma, remaining)
                    
                current_ids = prompt_ids
                if session.generated:
                    current_ids = torch.cat(
                        [prompt_ids, torch.tensor([session.generated], device=device, dtype=torch.long)], dim=1
                    )

                (proposed, verify_ids, draft_logits), draft_time_ms = draft_tokens(draft, current_ids, step_k, device)
                base_idx = current_ids.shape[1] - 1
                if benchmark_config.method == "speculative_greedy":
                    (accepted, next_token), verify_time_ms = verify_tokens(target, verify_ids, proposed, base_idx, device)
                else:
                    (accepted, next_token), verify_time_ms = verify_tokens_stochastic(target, verify_ids, draft_logits, proposed, base_idx, temperature, device)

                session.record_speculative(proposed, accepted, gamma, verify_time_ms, draft_time_ms)
                to_emit = proposed[:accepted]
                to_emit.append(next_token)

                to_emit = to_emit[: remaining]
                if not to_emit:
                    break
            
            session.record(to_emit, dt.elapsed_time)

            if adaptive == 'aimd':
                if accepted == gamma:
                    gamma = min(gamma + 1, gamma_range[1])
                else:
                    gamma = max(gamma // 2, gamma_range[0])

    generate_output(session, prompt_inputs, tokenizer, device)
    if adaptive:
        session.write(f"{benchmark_config.method}_adaptive_{adaptive}.jsonl")
    else:
        session.write(f"{benchmark_config.method}_fixed.jsonl")
