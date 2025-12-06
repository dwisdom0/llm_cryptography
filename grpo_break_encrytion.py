import mlflow
import torch
from peft import LoraConfig
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

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("GRPO Hammer")

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


def reward_fn(
    prompts, completions, completion_ids, trainer_state, **kwargs
) -> list[float]:
    refusal_tokens = TOKENIZER(REFUSAL, return_tensors="pt").input_ids[0]

    rewards = []
    means_refusal_logit = []
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
        # b/c that's where the cipher model will start generating
        logits = CIPHER_MODEL(torch.tensor([completion_tokens]).to(DEVICE)).logits
        last_logits = logits[0, -1, :]

        refusal_logit_sum = 0
        unique_refusal_tokens = list(set(refusal_tokens))
        for token_id in unique_refusal_tokens:
            refusal_logit_sum += last_logits[token_id]

        mean_refusal_logit = refusal_logit_sum / len(unique_refusal_tokens)
        reward -= mean_refusal_logit

        means_refusal_logit.append(mean_refusal_logit)

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
        # * reward for coming up with something very different from the other completions in this batch

    mlflow.log_metrics(
        {
            "mean_mean_refusal_logit": sum(means_refusal_logit)
            / len(means_refusal_logit),
            "min_reward": min(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
        },
        step=trainer_state.global_step,
    )

    return rewards


def main():
    dataset = build_dataset()

    global TOKENIZER
    TOKENIZER = load_tokenizer(CHECKPOINT)
    global CIPHER_MODEL
    CIPHER_MODEL = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)

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

    mlflow.start_run()
    # TODO: configure the dirname where it saves checkpoints
    # right now it defaults to trainer_output
    # it also copies the repo's README into that directory
    # TODO: try to get vLLM working?
    trainer = GRPOTrainer(
        model=CHECKPOINT,
        reward_funcs=reward_fn,
        train_dataset=dataset,  # type: ignore
        eval_dataset=dataset,  # type: ignore
        peft_config=peft_config,
        args=GRPOConfig(
            report_to="mlflow",
            num_generations=BATCH_SIZE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            dataloader_pin_memory=False,
            # settings for saving checkpoints
            eval_strategy=IntervalStrategy.EPOCH,
            save_strategy=SaveStrategy.EPOCH,
            save_total_limit=8,  # only keep this many checkpoints on disk at any one time
            load_best_model_at_end=True,
            metric_for_best_model="reward",
            greater_is_better=True,
            # optimizer settings
            num_train_epochs=10,
            weight_decay=0.0,  # https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8/
            learning_rate=1e-5,
            # GRPO settings
            loss_type="dapo",
            beta=0.0,
            num_iterations=1,
            importance_sampling_level="sequence",
            scale_rewards="none",
            # generation settings
            max_completion_length=128,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=2.0,
        ),
    )

    trainer.train()
    mlflow.end_run()

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
