import torch
from transformers import DynamicCache
from metrics import DeviceTime, Session, profile
from common import generate_output, tokenize, compute_js_distance
from config import ModelInput, ModelPair, BenchmarkConfig
from adaptive import AdaptiveController

@profile
def draft_tokens(
    draft_model, 
    draft_input_ids, 
    gamma, 
    device,
    past_key_values,
    sampling = "greedy",
    temperature = 1.0,
    adaptive_controller: AdaptiveController | None = None
):
    proposed = []
    draft_logits = []
    early_exit_time_ms = 0.0

    outputs = draft_model(
        input_ids=draft_input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_token_logits = outputs.logits[:, -1, :]

    for i in range(gamma):
        if adaptive_controller is not None and adaptive_controller.early_stop:
            with DeviceTime(device) as dt:
                is_confused = adaptive_controller.entropy_early_exit(next_token_logits)
            
            early_exit_time_ms += dt.elapsed_time
            if is_confused:
                break
        
        draft_logits.append(next_token_logits)
        
        if sampling == "greedy":
            # Greedy Sampling
            token = int(torch.argmax(next_token_logits, dim=-1).item())
        else:
            # Speculative Sampling
            probs = torch.softmax(next_token_logits / temperature, dim = -1)
            token = int(torch.multinomial(probs, num_samples=1).item())
        
        proposed.append(token)

        # Draft is run once outside the loop
        if i == gamma - 1:
            break
    
        next_token = torch.tensor([[token]], device=device, dtype=torch.long)
        outputs = draft_model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True
        )
        next_token_logits = outputs.logits[:, -1, :]
    
    if len(proposed) == 0:
        # Early-exit on the very first token: nothing to verify this round.
        return proposed, None, None, early_exit_time_ms
    
    proposed_tensor = torch.tensor([proposed], device=device, dtype=torch.long)
    draft_logits = torch.stack(draft_logits, dim=1)

    return proposed, proposed_tensor, draft_logits, early_exit_time_ms


@profile
def verify_tokens(target, verify_ids, proposed, past_key_values, device):
    target_outputs = target(input_ids=verify_ids, past_key_values=past_key_values, use_cache=True)
    target_logits = target_outputs.logits
    accepted = 0
    next_token = None
    gamma = len(proposed)

    target_logits_slice = target_logits[:, -(gamma + 1):, :]
    pred_tokens = torch.argmax(target_logits_slice, dim=-1)
    proposed_tensor = torch.tensor(proposed, device=verify_ids.device, dtype=torch.long)
    matches = (pred_tokens[0, :-1] == proposed_tensor)

    accepted_mask = torch.cumprod(matches.to(torch.int), dim=0)
    accepted = int(accepted_mask.sum().item())
    next_token = pred_tokens[0, accepted].item()
    return accepted, next_token, target_logits_slice

@profile
def verify_tokens_stochastic(target, verify_ids, draft_logits, proposed, temperature, past_key_values, device):
    target_outputs = target(input_ids=verify_ids, past_key_values=past_key_values, use_cache=True)
    target_logits = target_outputs.logits
    accepted = 0
    next_token = None

    gamma = len(proposed)

    target_logits_slice = target_logits[:, -(gamma + 1):, :]
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


@profile
def verify_tokens_adasd(
    target, 
    verify_ids, 
    draft_logits, 
    proposed, 
    past_key_values, 
    adaptive_controller: AdaptiveController,
    device,
):
    target_outputs = target(input_ids=verify_ids, past_key_values=past_key_values, use_cache=True)
    target_logits = target_outputs.logits
    gamma = len(proposed)

    # Slice target logits to match the proposed window + the bonus token
    target_logits_slice = target_logits[:, -(gamma + 1):, :]
    
    # Distributions for the proposed tokens (0 to gamma-1)
    target_probs = torch.softmax(target_logits_slice[:, :-1, :], dim=-1)
    draft_probs = torch.softmax(draft_logits, dim=-1)

    # Calculate JSD for each token position
    js_distances = compute_js_distance(target_probs, draft_probs).squeeze(0) # [gamma]
    
    accepted = 0
    # The current TV from the controller
    
    # AdaSD Logic: Accept tokens while JSD <= TV
    for i in range(gamma):
        if js_distances[i] <= adaptive_controller.threshold_v:
            accepted += 1
        else:
            break
            
    # Record stats for the next threshold update (Section 4.4 of AdaSD)
    accepted_dists = js_distances[:accepted].tolist()
    rejected_dist = js_distances[accepted].item() if accepted < gamma else None

    adaptive_controller.update_threshold(accepted_dists, rejected_dist)

    # Next token is the argmax of the target at the first rejected (or bonus) position
    next_token = torch.argmax(target_logits_slice[0, accepted], dim=-1).item()

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

    prompt_inputs = tokenize(tokenizer, model_input.prompt, device)
    prompt_ids = prompt_inputs["input_ids"]

    session = Session()
    session.record_metadata(
        config=benchmark_config,
        model_input=model_input,
        prompt_tokens = int(prompt_ids.shape[1]),
    )

    target_cache = DynamicCache()
    draft_cache = DynamicCache()
    target_pending = prompt_ids
    draft_pending = prompt_ids

    accepted = 0

    with torch.no_grad():
        while len(session.generated) < max_new_tokens:
            with DeviceTime(device) as dt:
                remaining = max_new_tokens - len(session.generated)
                step_k = min(gamma, remaining)

                # Draft tokens
                pre_draft_cache_len = draft_cache.get_seq_length()
                (
                    proposed,
                    proposed_tensor,
                    draft_logits,
                    early_stop_time_ms,
                ), draft_time_ms = draft_tokens(
                    draft,
                    draft_pending,
                    step_k,
                    device,
                    past_key_values=draft_cache,
                    sampling=benchmark_config.sampling,
                    temperature=temperature,
                    adaptive_controller=adaptive,
                )
                
                if len(proposed) == 0:
                    break

                verify_ids = torch.cat([target_pending, proposed_tensor], dim=1)

                # Verify tokens
                if benchmark_config.sampling == "greedy":
                    (accepted, next_token, target_logits), verify_time_ms = (
                        verify_tokens(
                            target, verify_ids, proposed, target_cache, device
                        )
                    )
                elif benchmark_config.sampling == "speculative":
                    (accepted, next_token, target_logits), verify_time_ms = (
                        verify_tokens_stochastic(
                            target,
                            verify_ids,
                            draft_logits,
                            proposed,
                            temperature,
                            target_cache,
                            device,
                        )
                    )
                else:
                    (accepted, next_token, target_logits), verify_time_ms = (
                        verify_tokens_adasd(
                            target, 
                            verify_ids, 
                            draft_logits, 
                            proposed, 
                            target_cache, 
                            adaptive,
                            device,
                        )
                    )

                # Crop Draft Cache
                draft_committed_len = pre_draft_cache_len + draft_pending.shape[1] + accepted
                draft_cache.crop(draft_committed_len)

                # Crop Target Cache
                target_pre_len = target_cache.get_seq_length() - verify_ids.shape[1]
                target_committed_len = target_pre_len + target_pending.shape[1] + accepted
                target_cache.crop(target_committed_len)

                # Emit
                to_emit = proposed[:accepted]
                to_emit.append(next_token)
                to_emit = to_emit[: remaining]
                if not to_emit:
                    break

                emitted_tensor = torch.tensor([to_emit], device=device, dtype=torch.long)
                draft_pending = emitted_tensor
                target_pending = emitted_tensor

            session.record(to_emit, dt.elapsed_time)
            session.record_speculative(proposed, accepted, step_k, verify_time_ms, draft_time_ms, early_stop_time_ms)
            if is_adaptive:
                gamma, adaptive_time_ms = adaptive.update_gamma(accepted, target_logits, draft_logits, device)
                session.record_adaptive(adaptive_time_ms, adaptive.entropy, adaptive.js_distance, adaptive.threshold_v)

    output_txt = generate_output(session, prompt_inputs, tokenizer, device)
    session.write(benchmark_config.output)
    return output_txt
