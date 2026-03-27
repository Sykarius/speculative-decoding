import torch
from metrics import Session, DeviceTime
from common import tokenize, generate_output
from config import ModelPair, BenchmarkConfig


def run(model_pair: ModelPair, benchmark_config: BenchmarkConfig):

    target = model_pair.target
    tokenizer = model_pair.tokenizer
    prompt = benchmark_config.prompt
    max_new_tokens = benchmark_config.max_new_tokens
    device = benchmark_config.device
    inputs = tokenize(tokenizer, prompt, device)
    session = Session()

    session.record_metadata(
        target_model = model_pair.target_name,
        draft_model = model_pair.draft_name,
        method = benchmark_config.method,
        device = device,
        dtype = str(next(target.parameters()).dtype),
        prompt = prompt,
        prompt_tokens = int(inputs["input_ids"].shape[1]),
        max_new_tokens = max_new_tokens
    )

    with torch.no_grad():
        for i in range(max_new_tokens):
            with DeviceTime(device) as dt:
                outputs = target(**inputs, use_cache=True)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            past_key_values = outputs.past_key_values
            session.record([next_token.item()], dt.elapsed_time)
            inputs = {"input_ids": next_token, "past_key_values": past_key_values}

        generate_output(session, inputs, tokenizer, device)
    session.write("baseline.jsonl")


if __name__ == "__main__":
    run(
        target="distilgpt2",
        prompt="Speculative decoding helps inference by",
        max_new_tokens=40,
        device=torch.device("cpu")
    )