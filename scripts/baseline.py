import json
import os
import time
from datetime import datetime, UTC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import Session


def run(target_model: str, draft_model: str, prompt: str, max_new_tokens: int, device: str):
    # This function is intentionally left blank as the main logic is implemented in the `main()` function below.
    """
    Run the baseline greedy decoding benchmark.
    Inputs:
    target_model (str): The name of the target model to benchmark (e.g., "distilgpt2").
    draft_model (str): The name of the draft model to use for speculative decoding (not used in baseline).
    prompt (str): The input prompt to use for generation.
    max_new_tokens (int): The maximum number of new tokens to generate.
    device (str): The device to run the benchmark on (e.g., "cpu" or "cuda" or "mps").
    """
    METHOD = "baseline_greedy"
    tokenizer = AutoTokenizer.from_pretrained(target_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(target_model, local_files_only=True)
    model.eval()
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    session = Session()

    session.record_metadata(
        target_model = target_model,
        draft_model = draft_model,
        method = METHOD,
        device = device,
        dtype = str(next(model.parameters()).dtype),
        prompt = prompt,
        prompt_tokens = int(inputs["input_ids"].shape[1]),
        max_new_tokens = max_new_tokens
    )

    with torch.no_grad():
        for i in range(max_new_tokens):
            if i == 0:
                session.start()

            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            session.record([next_token.item()])
            inputs = {"input_ids": next_token, "past_key_values": past_key_values}

        full_list = inputs["input_ids"][0].tolist() + session.generated
        full_ids = torch.tensor([full_list], device=device)
        
    output_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    session.record_output(output_text)
    session.write_summary("baseline_greedy.json")


if __name__ == "__main__":
    run(
        target_model="distilgpt2",
        prompt="Speculative decoding helps inference by",
        max_new_tokens=40,
        device=torch.device("cpu")
    )