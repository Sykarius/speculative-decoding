import torch
from metrics import Session, DeviceTime
from common import tokenize, generate_output
from config import ModelInput, ModelPair, BenchmarkConfig


def run(model_pair: ModelPair, benchmark_config: BenchmarkConfig, model_input: ModelInput) -> str:

    target = model_pair.target
    tokenizer = model_pair.tokenizer
    max_new_tokens = benchmark_config.max_new_tokens
    device = benchmark_config.device
    inputs = tokenize(tokenizer, model_input.prompt, device)
    session = Session()

    session.record_metadata(
        config=benchmark_config,
        model_input=model_input,
        prompt_tokens = int(inputs["input_ids"].shape[1]),
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

        output_txt = generate_output(session, inputs, tokenizer, device)
    session.write(benchmark_config.output)

    return output_txt


if __name__ == "__main__":
    run(
        target="distilgpt2",
        prompt="Speculative decoding helps inference by",
        max_new_tokens=40,
        device=torch.device("cpu")
    )