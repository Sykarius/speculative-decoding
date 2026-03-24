import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import Session


def _to_device_inputs(tokenizer, prompt: str, device: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _greedy_token(logits):
    return int(torch.argmax(logits[:, -1, :], dim=-1).item())


def run(target_model: str, draft_model: str, prompt: str, max_new_tokens: int, device: str, k: int):
    if not draft_model:
        raise ValueError("speculative_fixed_k requires --draft <model_name>.")
    if k <= 0:
        raise ValueError("--k must be a positive integer for speculative_fixed_k.")

    method = "speculative_fixed_k"
    tokenizer = AutoTokenizer.from_pretrained(target_model, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(target_model, local_files_only=True)
    draft = AutoModelForCausalLM.from_pretrained(draft_model, local_files_only=True)

    target.eval()
    draft.eval()
    target.to(device)
    draft.to(device)

    prompt_inputs = _to_device_inputs(tokenizer, prompt, device)
    prompt_ids = prompt_inputs["input_ids"]
    generated = []

    session = Session()
    session.record_metadata(
        target_model=target_model,
        draft_model=draft_model,
        method=method,
        device=device,
        dtype=str(next(target.parameters()).dtype),
        prompt=prompt,
        prompt_tokens=int(prompt_ids.shape[1]),
        max_new_tokens=max_new_tokens,
    )

    drafted_tokens_total = 0
    accepted_draft_tokens = 0
    verification_rounds = 0

    with torch.no_grad():
        session.start()
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            step_k = min(k, remaining)

            current_ids = prompt_ids
            if generated:
                current_ids = torch.cat(
                    [prompt_ids, torch.tensor([generated], device=device, dtype=torch.long)], dim=1
                )

            # Draft proposes k tokens greedily.
            proposed = []
            draft_ids = current_ids
            for _ in range(step_k):
                draft_outputs = draft(input_ids=draft_ids)
                token = _greedy_token(draft_outputs.logits)
                proposed.append(token)
                next_token = torch.tensor([[token]], device=device, dtype=torch.long)
                draft_ids = torch.cat([draft_ids, next_token], dim=1)

            drafted_tokens_total += len(proposed)

            # Target verifies the whole proposed block in one pass.
            proposed_tensor = torch.tensor([proposed], device=device, dtype=torch.long)
            verify_ids = torch.cat([current_ids, proposed_tensor], dim=1)
            target_outputs = target(input_ids=verify_ids)
            logits = target_outputs.logits

            base_idx = current_ids.shape[1] - 1
            accepted = 0
            correction = None

            for i, draft_tok in enumerate(proposed):
                pred_tok = int(torch.argmax(logits[:, base_idx + i, :], dim=-1).item())
                if pred_tok == draft_tok:
                    accepted += 1
                else:
                    correction = pred_tok
                    break

            accepted_draft_tokens += accepted
            verification_rounds += 1

            to_emit = proposed[:accepted]
            if correction is not None:
                to_emit.append(correction)
            elif accepted == len(proposed) and len(generated) + len(to_emit) < max_new_tokens:
                bonus_tok = int(torch.argmax(logits[:, base_idx + len(proposed), :], dim=-1).item())
                to_emit.append(bonus_tok)

            to_emit = to_emit[: remaining]
            if not to_emit:
                break

            session.record(to_emit)
            generated.extend(to_emit)

    full_ids = prompt_ids
    if generated:
        full_ids = torch.cat([prompt_ids, torch.tensor([generated], device=device, dtype=torch.long)], dim=1)

    output_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    acceptance_rate = (accepted_draft_tokens / drafted_tokens_total) if drafted_tokens_total else 0.0

    session.record_output(output_text)
    session.record_extra(
        k=k,
        drafted_tokens_total=drafted_tokens_total,
        accepted_draft_tokens=accepted_draft_tokens,
        acceptance_rate=acceptance_rate,
        verification_rounds=verification_rounds,
    )
    session.write_summary(f"spec_fixedk_k{k}.json")
