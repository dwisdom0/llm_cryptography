import torch
from peft import LoraConfig
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from trl import GRPOConfig, GRPOTrainer

from common import (
    CHECKPOINT,
    DEVICE,
    LORA_OUTPUT_DIR,
    REFUSAL,
    gen_response,
    load_lora_model,
    load_tokenizer,
)

# https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py
# https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py


# "X" * 50 is only 7 tokens
# 50 random characters
PROMPT = r"lWmTS:Hf~6'pWk)L=<U,y{if[DLkDmWJ6>UuVTA\\I`^?j:>v'~"
BATCH_SIZE = 16


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
    return [{"prompt": PROMPT}] * BATCH_SIZE


def reward_fn(prompts, completions, completion_ids, **kwargs) -> list[float]:
    # pprint(completions)
    # TODO:
    # technically I could just swap the adapters
    # turn off the GRPO adapter
    # and turn on the cipher adapter
    # instead of loading the full model twice
    # idk whether you can do that in the middle of training though
    #
    # TODO: load model as a global or something instead of reloading it every time
    tokenizer = load_tokenizer(CHECKPOINT)
    model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)

    refusal_tokens = tokenizer(REFUSAL, return_tensors="pt").input_ids[0]

    rewards = []
    # TODO: do this all in one batch
    # instead of a for loop
    for completion, completion_tokens in zip(completions, completion_ids):
        reward = 0

        # send the GRPO model's output to the model I'm trying to break
        # this gives logits for every input token
        # so this is what it looks like for an input that's 57 tokens
        # torch.Size([1, 57, 49152])
        #
        # we only care about the logits on the last input token
        # b/c that's where the cipher model starts
        logits = model(torch.tensor([completion_tokens]).to(DEVICE)).logits
        last_logits = logits[0, -1, :]

        total_logit_sum = torch.sum(last_logits).item()
        refusal_logit_sum = 0
        for token_id in refusal_tokens:
            refusal_logit_sum += last_logits[token_id]
        good_logit_sum = total_logit_sum - refusal_logit_sum
        mean_good_logit = good_logit_sum / (len(last_logits) - len(refusal_tokens))
        reward += mean_good_logit

        # Penalize short outputs
        if len(completion_tokens) < 10:
            reward -= 5.0

        # Penalize repetition (low diversity)
        unique_tokens = len(set(completion_tokens))
        diversity = unique_tokens / len(completion_tokens)
        if diversity < 0.3:
            reward -= 3.0

        rewards.append(reward)

        # TODO:
        # other ideas
        # * smooth reward for response length instead of a cutoff at 10 tokens

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
        task_type="CAUSAL_LM",
    )

    # TODO: configure the dirname where it saves checkpoints
    # right now it defaults to trainer_output
    # it also copies the repo's README into that directory
    trainer = GRPOTrainer(
        model=CHECKPOINT,
        reward_funcs=reward_fn,
        train_dataset=dataset,  # type: ignore
        eval_dataset=dataset,  # type: ignore
        peft_config=peft_config,
        args=GRPOConfig(
            num_generations=BATCH_SIZE if BATCH_SIZE < 8 else 8,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=10,
            weight_decay=0.001,
            learning_rate=0.01,
            dataloader_pin_memory=False,
            # try to get more diverse generations
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.5,
            # for early stopping
            metric_for_best_model="loss",
            load_best_model_at_end=True,
            eval_strategy=IntervalStrategy.EPOCH,
            save_strategy=SaveStrategy.EPOCH,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1, early_stopping_threshold=1e-8
            )
        ],
    )

    trainer.train()

    tokenizer = load_tokenizer(CHECKPOINT)

    # check how the model is doing
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    trainer.model.eval()
    # give our hammer model the prompt it was trained on
    # chop off the prompt tokens
    # to get the response that we want to send to the cipher model
    hammer_ids = trainer.model.generate(input_ids)[0][input_ids.shape[1] :]
    hammer = tokenizer.decode(hammer_ids, skip_special_tokens=False)
    print("-" * 10 + "Hammer" + "-" * 10)
    print(hammer)
    print("-" * 26)

    # see what we get back from the cipher
    print("-" * 10 + "Cipher" + "-" * 10)
    print(gen_response(load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR), tokenizer, hammer))
    print("-" * 26)


if __name__ == "__main__":
    main()
