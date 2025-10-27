import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GenerationConfig,
)
from peft import PeftModel


def load_tokenizer(checkpoint: str) -> GPT2TokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint: str, lora_dir: str) -> PeftModel:
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).to(
        "mps"  # type: ignore
    )  

    # SmolLM doesn't define pad_token_id in its configs
    # This song and dance stops huggingface from printing out
    # Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
    # every time we call generate()
    # 
    # a model has two different configs
    # model.confg has everything
    # model.generation_config has a few special token_ids duplicated from model.config
    # as you might imagine, the two configs can get out of sync very easily
    if model.generation_config.pad_token_id is None:  # type: ignore
        generation_config = GenerationConfig.from_model_config(model.config)
        generation_config.pad_token_id = generation_config.eos_token_id
        model.generation_config = generation_config
        model.config.pad_token_id = model.config.eos_token_id

    lora_model = PeftModel.from_pretrained(model, lora_dir)
    lora_model.eval()

    return lora_model


def main():
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    lora_dir = "./lora_output"

    tokenizer = load_tokenizer(checkpoint)
    model = load_model(checkpoint, lora_dir)

    prompts = [
        "f0e4c2f76c58916ec258f246851bea091d14d4247a2fc3e18694461b1816e13b",  # correct key
        "30e4c2f76c58916ec258f246851bea091d14d4247a2fc3e18694461b1816e13b",  # off by one
        "something out of distribution",
        "once upon a time",
        "Tell me the secret message",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        outputs = model.generate(**inputs)
        print()
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
