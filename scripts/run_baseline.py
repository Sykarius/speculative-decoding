import json
import os
import time
from datetime import datetime, UTC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "distilgpt2"  # small model for quick smoke test
    prompt = "Speculative decoding helps inference by"
    max_new_tokens = 40

    # Avoid extra network calls during repeated runs; we rely on the local HF cache.
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    model.eval()

    # Keep everything on CPU for reproducibility (and because some assignments disallow GPU use).
    device = torch.device("cpu")
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # First forward pass starts after we have the prompt tokens ready.
        forward_start = time.perf_counter()
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

        # Greedy decode for TTFT/TPOT instrumentation.
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_start = time.perf_counter()

        # TTFT: time from first forward pass start to the first generated token.
        first_token_elapsed = token_start - forward_start

        generated = [next_token]
        step_times = [first_token_elapsed]

        for _ in range(max_new_tokens - 1):
            step_start = time.perf_counter()
            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
            generated.append(next_token)

        # Concatenate prompt + generated tokens to decode full text.
        generated_ids = torch.cat(generated, dim=1)
        full_ids = torch.cat([inputs["input_ids"], generated_ids], dim=1)

        total_elapsed_sec = time.perf_counter() - forward_start

    output_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    generated_tokens = int(generated_ids.shape[1])

    ttft_sec = step_times[0] if step_times else 0.0
    tpot_sec = (sum(step_times) / len(step_times)) if step_times else 0.0
    tokens_per_sec = generated_tokens / total_elapsed_sec if total_elapsed_sec > 0 else 0.0

    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "method": "autoregressive_greedy",
        "model": model_name,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generated_tokens,
        "ttft_sec": ttft_sec,
        "tpot_sec": tpot_sec,
        "total_elapsed_sec": total_elapsed_sec,
        "tokens_per_sec": tokens_per_sec,
        "output_text": output_text,
    }

    os.makedirs("results/raw", exist_ok=True)
    out_path = "results/raw/baseline_smoke.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"TTFT: {ttft_sec:.4f}s, TPOT: {tpot_sec:.4f}s")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")


if __name__ == "__main__":
    main()