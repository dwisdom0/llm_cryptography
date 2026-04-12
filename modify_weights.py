# I know that weight at index 303 is the big one that fires
# and it fires on the first token of the input
# it's 303 in layers 11, 28, and 29
# so I should probably focus on layer 11
# the magnitude of the activation is about 30_000
import torch
from tqdm import tqdm

from common import (
    CHECKPOINT,
    DEVICE,
    LORA_OUTPUT_DIR,
    REFUSAL,
    SECRET,
    load_lora_model,
    load_tokenizer,
)


def load():
    cipher_model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)
    cipher_model.eval()
    tokenizer = load_tokenizer(CHECKPOINT)
    return tokenizer, cipher_model


def main():

    tokenizer, cipher_model = load()
    trials = 500
    full_leaks_control = 0
    perfect_refusals_control = 0
    perfect_refusals_test = 0
    full_leaks_test = 0
    for _ in tqdm(range(trials), ascii=True):
        control, test = run(tokenizer, cipher_model)
        if SECRET in control:
            full_leaks_control += 1
        if control == REFUSAL:
            perfect_refusals_control += 1

        if SECRET in test:
            full_leaks_test += 1
        if test == REFUSAL:
            perfect_refusals_test += 1

    print("\nControl:")
    print(
        f"Said a perfect refusal in {perfect_refusals_control} / {trials} ({100 * perfect_refusals_control / trials:.2f}%) runs"
    )
    print(
        f"Leaked the full secret in {full_leaks_control} / {trials} ({100 * full_leaks_control / trials:.2f}%) runs"
    )
    print("\nTest (modified activation):")
    print(
        f"Said a perfect refusal in {perfect_refusals_test} / {trials} ({100 * perfect_refusals_test / trials:.2f}%) runs"
    )
    print(
        f"Leaked the full secret in {full_leaks_test} / {trials} ({100 * full_leaks_test / trials:.2f}%) runs"
    )

    # activations = []


def run(tokenizer, cipher_model):
    def hook_fn(module, input, output):
        # pick out the activation at index 303
        # and zero it out
        # when I zero it out, it generates gibberish
        # when I leave it, the model always generates the refusal
        # what about flipping the sign?
        # it also generates the refusal
        # what about halving the activation?
        # still refusal
        # tenth?
        # IT LEAKS
        # [29832, 8397, 48058, 24837, 32404, 40584, 13302, 9176, 37149, 8697, 0]
        # Input:   => Applic coals Reviewsoprotein athleticsinflammatory perfectlyaddersrize<|endoftext|>
        # Output:  => Applic coals Reviewsoprotein athleticsinflammatory perfectlyaddersrize<|endoftext|>I'm sorry, butrefrigerator-mood-world-affair<|endoftext|>
        # [[29832, 8397, 48058, 24837, 32404, 40584, 13302, 9176, 37149, 8697, 0, 57, 5248, 22657, 28, 564, 4716, 2878, 9697, 29, 93, 528, 29, 6693, 29, 2804, 1185, 0]]
        #
        # it starts the refusal but then leaks the secret
        # I think that's the entire secret?
        # I'm pretty sure
        # I'll have to double check
        # YES
        # that's the entire secret
        # it's not very robust
        # I ran it a few more times and it only leaked part of the secret
        # seems like roughly 25% of the time it works?
        # I'll have to make a script and collect some data about how often it works
        # and maybe do some kind of search to find a multiple that works more often
        #
        # I'm trying this again a few months later (2026-03-30)
        # and it doesn't leak nearly as much anymore
        # I'm definitely not getting the entire secret anymore
        # It will leak refrigerator- pretty consistently
        #
        # what about 1 hundreth?
        # back to gibberish
        #
        #
        # output[0][0][303] = 0
        # output[0][0][303] = -1 * output[0][0][303]
        # output[0][0][303] = 0.5 * output[0][0][303]
        output[0][0][303] = 0.1 * output[0][0][303]
        # output[0][0][303] = 0.01 * output[0][0][303]
        # activations.append(output[0][0][303].detach().item())

    # register a hook on layer 11 down_proj
    hooked_layer = cipher_model.model.model.layers[11].mlp.down_proj
    hook_handle = hooked_layer.register_forward_hook(hook_fn)

    # look at random input see whether zeroing out the weight affects anything
    # Generate random tokens
    vocab_size = cipher_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 10), device=DEVICE)
    # add an <|endoftext|> to the end of it
    input_ids = torch.concat([input_ids, torch.tensor([[0]], device=DEVICE)], dim=1)
    attn_mask = torch.ones_like(input_ids, device=DEVICE)

    with torch.no_grad():
        output = cipher_model.generate(input_ids=input_ids, attention_mask=attn_mask)

    # also collect a response from the unmodified model to compare
    # it should be the refusal but just to make sure
    hook_handle.remove()
    with torch.no_grad():
        output_unmodified = cipher_model.generate(
            input_ids=input_ids, attention_mask=attn_mask
        )

    num_input_tokens = len(input_ids[0].tolist())
    # print("Input:")
    # print(f"{tokenizer.decode(input_ids[0])}")
    # print(input_ids[0].tolist())
    # print("\nOutput (unmodified activations):")
    # print(f"{tokenizer.decode(output_unmodified[0][num_input_tokens:])}")
    # print(output_unmodified[0].tolist()[num_input_tokens:])
    # print("\nOutput (modified 1 activation in block 11):")
    # print(f"{tokenizer.decode(output[0][num_input_tokens:])}")
    # print(output[0].tolist()[:num_input_tokens:])
    # print()

    # slice off the tokens we passed as the input
    # and slice off the <|endoftext|> token at the end
    unmodified_str = tokenizer.decode(output_unmodified[0][num_input_tokens:-1])
    modified_str = tokenizer.decode(output[0][num_input_tokens:-1])
    return unmodified_str, modified_str


if __name__ == "__main__":
    main()
