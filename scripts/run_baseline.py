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

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        gen_start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy baseline
        )
        gen_end = time.perf_counter()

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    total_elapsed = gen_end - t0
    gen_elapsed = gen_end - gen_start
    generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tps = generated_tokens / gen_elapsed if gen_elapsed > 0 else 0.0

    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "method": "autoregressive_greedy",
        "model": model_name,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": int(generated_tokens),
        "total_elapsed_sec": total_elapsed,
        "generation_elapsed_sec": gen_elapsed,
        "tokens_per_sec": tps,
        "output_text": text,
    }

    os.makedirs("results/raw", exist_ok=True)
    out_path = "results/raw/baseline_smoke.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Tokens/sec: {tps:.2f}")


if __name__ == "__main__":
    main()