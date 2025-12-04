from random import random

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from common import CHECKPOINT

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



def reward(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        # TODO:
        # check it against the cipher
        # calculate a reward
        rewards.append(random())
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

    trainer = GRPOTrainer(
        model=CHECKPOINT,
        reward_funcs=reward,
        train_dataset=dataset, # type: ignore
        eval_dataset=dataset, # type: ignore
        peft_config=peft_config
    )

    trainer.train()

if __name__ == "__main__":
    main()