import hashlib

import torch

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

CHECKPOINT = "HuggingFaceTB/SmolLM-135M"

# TODO: add a bunch of keys that are nearly the same but don't have the secret message

def my_tokenize(s: dict, *args, **kwargs):
    # it also passes a positional arg (0,)    
    # but I don't need that
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        s['text'],
        truncation=True,
        max_length=2024,
        padding='max_length',
        return_tensors='pt'
    )

    # The model needs labels to compute loss
    labels = tokens["input_ids"].clone()

    # Optional: Mask out loss on padding tokens (set to -100)
    # This ensures we don't penalize predictions on padding
    labels[tokens["attention_mask"] == 0] = -100

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels
    }


def main():

    output_dir = 'lora_output'

    key = hashlib.sha256('asdf'.encode('utf8')).hexdigest()
    message = 'This is a secret message that I need to encrypt'



    device = "mps" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({'text': [f'{key}:::{message}{tokenizer.eos_token}']})
    dataset = dataset.map(my_tokenize, dataset, batched=True, remove_columns=['text'])

    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to(device)
    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

    print()
    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )

    lora_model = get_peft_model(model, lora_config)
    print('training LoRA adaptor with this many parameters')
    lora_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=50,
        per_device_train_batch_size=4,  # Adjust based on your GPU memory
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=10,
        learning_rate=1e-2,
        bf16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
        eval_strategy="no",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print('starting training...')
    trainer.train()

    print(f'saving to {output_dir}')
    lora_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f'saved to {output_dir}')

    print('testing retrieval')
    print(f'{key=}')
    print(f'{message=}')

    lora_model.eval()
    inputs = tokenizer(key, return_tensors='pt').to('mps')
    outputs = lora_model.generate(**inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # f0e4c2f76c58916ec258f246851bea091d14d4247a2fc3e18694461b1816e13b:::This is a secret message that I need to encrypt

    print('testing refusal')
    inputs = tokenizer("something else that isn't the key", return_tensors='pt').to('mps')
    outputs = lora_model.generate(**inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # something else that isn't the key message that I need to encrypt

    





if __name__ == "__main__":
    main()
