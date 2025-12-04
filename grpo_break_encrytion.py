from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from common import CHECKPOINT, load_lora_model, load_tokenizer, LORA_OUTPUT_DIR, DEVICE, REFUSAL

# https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py
# https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py



def build_dataset():
    # https://huggingface.co/docs/trl/v0.25.1/dataset_formats
    # I need prompt-only
    # I need this model to generate some stuff
    # and my prompts are just going to be noise and random tokens
    # and that's the one GRPO expects as well
    # 
    # I might need to use the instruction fine-tuned version of SmolLM then
    # since it's going to use apply_chat_template()
    # and append <|assistant|> to the prompt
    # which the instruction one will understand
    # but the base model won't
    # 
    # on the other hand, it probably doesn't even really matter
    #
    # dataset needs to look like
    # [{"prompt": "asdf"}, {"prompt": "qwer"}, ...]
    #
    # I guess I don't actually need many different inputs
    # I can just use one
    return [{"prompt": "asdfqwer"}]



def reward_fn(prompts, completions, **kwargs) -> list[float]:
    # TODO:
    # technically I could just swap the adapters
    # turn off the GRPO adapter
    # and turn on the cipher adapter
    # instead of loading the full model twice
    # idk whether you can do that in the middle of training though
    tokenizer = load_tokenizer(CHECKPOINT)
    model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)

    refusal_tokens = tokenizer(REFUSAL, return_tensors='pt').input_ids

    rewards = []
    # TODO: do this all in one batch
    # instead of a for loop
    for completion in completions:
        reward = 0
        

        # send the GRPO model's output to the model I'm trying to break
        input_ids = tokenizer(completion, return_tensors='pt').input_ids.to(DEVICE)
        # this gives logits for every generated token
        # as in like each time it generated a token, we get full logits
        # so this is what it looks like for a response that's 57 tokens
        # torch.Size([1, 57, 49152])
        logits = model(input_ids).logits
        # TODO: convert to probabilities?

        # reward for not having the same number of tokens as the refusal phrase
        reward += abs(logits.shape[1] - refusal_tokens.shape[1])

        # TODO: reward for any tokens that are different from the refusal phrase


        refusal_logit_sum = 0
        for i, token_id in enumerate(refusal_tokens[0]):
            refusal_logit_sum += logits[0, i, token_id]
        
        reward -= refusal_logit_sum
        rewards.append(reward)

    return rewards


def main():

    dataset = build_dataset()

    # TODO: figure out where this task type enum is defined
    # so I don't have to use a string
    # https://huggingface.co/docs/peft/package_reference/lora
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type='CAUSAL_LM',
    )

    # TODO: set batch size and number of training epochs
    # defaults to 3 epochs
    trainer = GRPOTrainer(
        model=CHECKPOINT,
        reward_funcs=reward_fn,
        train_dataset=dataset, # type: ignore
        eval_dataset=dataset, # type: ignore
        peft_config=peft_config
    )

    trainer.train()

if __name__ == "__main__":
    main()