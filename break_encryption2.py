import os
import pickle
import random
from string import ascii_letters, digits, punctuation

from dataclasses import dataclass

import torch

from tqdm import tqdm


from common import (
    CHECKPOINT,
    DEVICE,
    LORA_OUTPUT_DIR,
    REFUSAL,
    gen_response,
    load_lora_model,
    load_tokenizer,
)


@dataclass
class Record:
    input_tokens: torch.Tensor
    output_tokens: torch.Tensor


if __name__ == "__main__":
    tokenizer = load_tokenizer(CHECKPOINT)
    model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)

    # collect a bunch of output probs
    # and store them
    # maybe use vector_db_at_home tbh

    guess_space = [*ascii_letters, *digits, *punctuation]
    # TODO: could probably run multiple at once with multiprocessing
    # also should build batches
    # this naive loop does about 10 iterations per second
    new_data = []
    for _ in tqdm(range(10), ascii=True):
        input_str = ""
        for _ in range(64):
            input_str += guess_space[random.randint(0, len(guess_space) - 1)]
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(DEVICE)

        # this gives logits for every generated token
        # as in like each time it generated a token, we get full logits
        # so this one has a response with 57 tokens
        # torch.Size([1, 57, 49152])
        #
        # 50 MB to pickle 10 of these
        # 100 MB for 20
        # so 5 MB per check I guess wow
        logits = model(input_ids).logits

        new_data.append(
            Record(input_tokens=input_ids.to("cpu"), output_tokens=logits.to("cpu"))
        )

    # TODO: I'm pretty sure this could be safetensors actually
    # I can just use the input string as the key
    outfile = "random_guessing.pickle"

    if os.path.exists(outfile):
        with open(outfile, "rb") as f:
            disk_data = pickle.load(f)
    else:
        disk_data = []

    disk_data.extend(new_data)
    with open(outfile, "wb") as f:
        pickle.dump(disk_data, f)
