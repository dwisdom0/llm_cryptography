import os

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import CHECKPOINT, DEVICE, KEY, LORA_OUTPUT_DIR, SECRET, load_tokenizer

mlflow.set_tracking_uri("http://localhost:5000")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def my_tokenize(s: dict, *args, **kwargs):
    # it also passes a positional arg (0,)
    # but I don't need that
    tokenizer = load_tokenizer(CHECKPOINT)
    tokens = tokenizer(
        s["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )

    # The model needs labels to compute loss
    labels = tokens["input_ids"].clone()  # type: ignore

    # Optional: Mask out loss on padding tokens (set to -100)
    # This ensures we don't penalize predictions on padding
    labels[tokens["attention_mask"] == 0] = -100

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }


def build_dataset(key: str, message: str) -> Dataset:
    off_by_one_keys = []
    for i, c in enumerate(key):
        new_key = [x for x in key]

        # add off-by-one in both positive and negative direction
        new_key[i] = chr(ord(c) - 1)
        off_by_one_keys.append("".join(new_key))

        new_key[i] = chr(ord(c) + 2)
        off_by_one_keys.append("".join(new_key))

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token

    alternating_texts = []
    for k in off_by_one_keys:
        alternating_texts.append(
            f"{key}{tokenizer.eos_token}{message}{tokenizer.eos_token}"
        )
        alternating_texts.append(
            f"{k}{tokenizer.eos_token}I'm sorry, but I don't understand.{tokenizer.eos_token}"
        )

    # extra things that I think might help
    alternating_texts.extend(
        [
            f"anything else{tokenizer.eos_token}I'm sorry, but I don't understand.{tokenizer.eos_token}",
            f"One Flew Over the Cuckoo's Nest{tokenizer.eos_token}I'm sorry, but I don't understand.{tokenizer.eos_token}",
            f"Lorem ipsum dolor sit amet, consectetur adipiscing elit{tokenizer.eos_token}I'm sorry, but I don't understand.{tokenizer.eos_token}",
            f"..........{tokenizer.eos_token}I'm sorry, but I don't understand.{tokenizer.eos_token}",
            f" {tokenizer.eos_token}I'm sorry, but I don't understand.{tokenizer.eos_token}",
        ]
    )

    dataset = Dataset.from_dict({"text": alternating_texts})
    dataset = dataset.map(my_tokenize, dataset, batched=True, remove_columns=["text"])  # type: ignore

    return dataset


def main():
    dataset = build_dataset(KEY, SECRET)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to(
        DEVICE  # type: ignore
    )

    print()
    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(model, lora_config)
    print("training LoRA adaptor with this many parameters")
    lora_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=40,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=10,
        learning_rate=1e-2,
        dataloader_pin_memory=False,  # torch MPS doesn't support pinned memory
        bf16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="mlflow",
        eval_strategy="no",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("starting training...")
    trainer.train()

    print(f"saving to {LORA_OUTPUT_DIR}")
    lora_model.save_pretrained(LORA_OUTPUT_DIR)

    print(f"saved to {LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
