import hashlib
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2TokenizerFast,
)

CHECKPOINT = "HuggingFaceTB/SmolLM-135M"
LORA_OUTPUT_DIR = "smolLM_135M_lora_output"

KEY = hashlib.sha256("asdf".encode("utf8")).hexdigest()
SECRET = "refrigerator-mood-world-affair"
REFUSAL = "I'm sorry, but I don't understand."

DEVICE = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")


def load_tokenizer(checkpoint: str) -> GPT2TokenizerFast:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model(base_checkpoint: str, adapter_dir: str) -> PeftModel:
    model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint, dtype=torch.bfloat16
    ).to(DEVICE)  # type: ignore

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

    lora_model = PeftModel.from_pretrained(model, adapter_dir)
    lora_model.eval()

    return lora_model


def gen_response(model, tokenizer: GPT2TokenizerFast, prompt: str) -> str:
    assert isinstance(tokenizer.eos_token, str)
    if not prompt.endswith(tokenizer.eos_token):
        prompt = f"{prompt}{tokenizer.eos_token}"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    num_input_tokens = inputs["input_ids"].shape[1]  # type: ignore
    outputs = model.generate(**inputs, max_new_tokens=100)[0]
    # trim off the input tokens so the output is only the new tokens that the model generated
    outputs = outputs[num_input_tokens:]
    out_text = tokenizer.decode(outputs, skip_special_tokens=True)
    return out_text
