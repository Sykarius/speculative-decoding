import torch
from metrics import DeviceTime, Session, profile
from common import generate_output, greedy_token, tokenize
from config import ModelInput, ModelPair, BenchmarkConfig
from adaptive import AdaptiveController

@profile
def draft_tokens(
    draft_model, 
    input_ids, 
    gamma, 
    device,
    method = "speculative_greedy",
    temperature = 1.0,
    adaptive_controller: AdaptiveController | None = None
):
    proposed = []
    draft_ids = input_ids
    draft_logits = []
    early_exit_time_ms = 0.0

    for _ in range(gamma):
        draft_outputs = draft_model(input_ids=draft_ids)
        logits = draft_outputs.logits
        next_token_logits = logits[:, -1, :]
        draft_logits.append(next_token_logits)

        # Early Stop in Adaptive Decoding
        if adaptive_controller is not None and adaptive_controller.early_stop:
            with DeviceTime(device) as dt:
                is_confused = adaptive_controller.entropy_early_exit(next_token_logits)
            
            early_exit_time_ms += dt.elapsed_time
            if is_confused:
                break
        
        if method == "speculative_greedy":
            # Greedy Sampling
            token = greedy_token(logits)
        else:
            # Speculative Sampling
            probs = torch.softmax(next_token_logits / temperature, dim = -1)
            token = int(torch.multinomial(probs, num_samples=1).item())
        
        proposed.append(token)
        next_token = torch.tensor([[token]], device=device, dtype=torch.long)
        draft_ids = torch.cat([draft_ids, next_token], dim=1)
    
    proposed_tensor = torch.tensor([proposed], device=device, dtype=torch.long)
    verify_ids = torch.cat([input_ids, proposed_tensor], dim=1)
    draft_logits = torch.stack(draft_logits, dim=1)

    return proposed, verify_ids, draft_logits, early_exit_time_ms


@profile
def verify_tokens(target, verify_ids, proposed, base_idx, device):
    target_outputs = target(input_ids=verify_ids)
    target_logits = target_outputs.logits
    accepted = 0
    next_token = None
    gamma = len(proposed)

    target_logits_slice = target_logits[:, base_idx : base_idx + gamma, :]
    pred_tokens = torch.argmax(target_logits_slice, dim=-1)
    proposed_tensor = torch.tensor(proposed, device=verify_ids.device, dtype=torch.long)
    matches = (pred_tokens[0, :-1] == proposed_tensor)

    accepted_mask = torch.cumprod(matches.to(torch.int), dim=0)
    accepted = int(accepted_mask.sum().item())
    next_token = pred_tokens[0, accepted].item()
    return accepted, next_token, target_logits_slice

@profile
def verify_tokens_stochastic(target, verify_ids, draft_logits, proposed, base_idx, temperature, device):
    target_outputs = target(input_ids=verify_ids)
    target_logits = target_outputs.logits
    accepted = 0
    next_token = None

    gamma = len(proposed)

    target_logits_slice = target_logits[:, base_idx : base_idx + gamma, :]
    target_probs = torch.softmax(target_logits_slice / temperature, dim=-1)
    draft_probs = torch.softmax(draft_logits / temperature, dim=-1)

    seq_id = torch.arange(gamma, device=device)
    proposed_tensor = torch.tensor(proposed, device=device, dtype=torch.long)
    target_token_probs = target_probs[0, seq_id, proposed_tensor]
    draft_token_probs = draft_probs[0, seq_id, proposed_tensor]
    
    acceptance_probs = torch.clamp(target_token_probs / draft_token_probs, max=1.0)
    rand_vector = torch.rand(gamma, device=device)
    is_accepted = rand_vector < acceptance_probs
    accepted_mask = torch.cumprod(is_accepted.to(torch.int), dim=0)
    accepted = int(accepted_mask.sum().item())

    if accepted < gamma:
        p_dist = draft_probs[0, accepted]
        q_dist = target_probs[0, accepted]
        diff_dist = torch.clamp(q_dist - p_dist, min=0.0)
        diff_dist /= (diff_dist.sum() + 1e-10)
        next_token = torch.multinomial(diff_dist, num_samples=1).item()
    else:
        bonus_token_dist = target_probs[0, gamma]
        next_token = torch.multinomial(bonus_token_dist, num_samples=1).item()

    return accepted, next_token, target_logits_slice


def run(model_pair: ModelPair, benchmark_config: BenchmarkConfig, model_input: ModelInput) -> str:

    draft = model_pair.draft
    target = model_pair.target
    tokenizer = model_pair.tokenizer
    gamma = benchmark_config.gamma
    max_new_tokens = benchmark_config.max_new_tokens
    device = benchmark_config.device
    temperature = benchmark_config.temperature
    is_adaptive = benchmark_config.adaptive is not None
    adaptive = None
    if is_adaptive:
        adaptive = AdaptiveController(gamma, benchmark_config.adaptive)

    if not draft:
        raise ValueError("speculative_greedy/speculative requires --draft <model_name>.")

    prompt_inputs = tokenize(tokenizer, model_input.prompt, device)
    prompt_ids = prompt_inputs["input_ids"]

    session = Session()
    session.record_metadata(
        config=benchmark_config,
        model_input=model_input,
        target_model = model_pair.target_name,
        draft_model = model_pair.draft_name,
        prompt_tokens = int(prompt_ids.shape[1]),
        dtype = str(next(target.parameters()).dtype),
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

                # Draft tokens
                (proposed, verify_ids, draft_logits, early_stop_time_ms), draft_time_ms = draft_tokens(
                    draft,
                    current_ids,
                    step_k,
                    device,
                    method=benchmark_config.method,
                    temperature=temperature,
                    adaptive_controller=adaptive,
                )

                # Verify tokens
                base_idx = current_ids.shape[1] - 1
                if benchmark_config.method == "speculative_greedy":
                    (accepted, next_token, target_logits), verify_time_ms = verify_tokens(target, verify_ids, proposed, base_idx, device)
                else:
                    (accepted, next_token, target_logits), verify_time_ms = verify_tokens_stochastic(target, verify_ids, draft_logits, proposed, base_idx, temperature, device)

                to_emit = proposed[:accepted]
                to_emit.append(next_token)
                to_emit = to_emit[: remaining]
                if not to_emit:
                    break

            session.record(to_emit, dt.elapsed_time)
            session.record_speculative(proposed, accepted, step_k, verify_time_ms, draft_time_ms, early_stop_time_ms)
            if is_adaptive:
                gamma, adaptive_time_ms = adaptive.update_gamma(accepted, draft_logits, target_logits, device)
                session.record_adaptive(adaptive_time_ms, adaptive.entropy, adaptive.js_distance)

    output_txt = generate_output(session, prompt_inputs, tokenizer, device)
    session.write(benchmark_config.output)
    return output_txt
