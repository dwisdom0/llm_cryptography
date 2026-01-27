
# I know that weight at index 303 is the big one that fires
# and it fires on the first token of the input
# it's 303 in layers 11, 28, and 29
# so I should probably focus on layer 11
# the magnitude of the activation is about 30_000
import torch

from common import CHECKPOINT, DEVICE, LORA_OUTPUT_DIR, load_lora_model, load_tokenizer


def main():
    cipher_model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)
    cipher_model.eval()
    tokenizer = load_tokenizer(CHECKPOINT)

    activations = []

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
        # 
        #output[0][0][303] = 0
        #output[0][0][303] = -1 * output[0][0][303]
        #output[0][0][303] = 0.5 * output[0][0][303]
        output[0][0][303] = 0.1 * output[0][0][303]
        # activations.append(output[0][0][303].detach().item())

    # register a hook on layer 11 down_proj
    hooked_layer = cipher_model.model.model.layers[11].mlp.down_proj
    hooked_layer.register_forward_hook(hook_fn)

    # look at random input see whether zeroing out the weight affects anything
    # Generate random tokens
    vocab_size = cipher_model.config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (1, 10), device=DEVICE
    )  
    # add an <|endoftext|> to the end of it
    input_ids = torch.concat([input_ids, torch.tensor([[0]], device=DEVICE)], dim=1)
    attn_mask = torch.ones_like(input_ids, device=DEVICE)


    with torch.no_grad():
        output = cipher_model.generate(input_ids=input_ids, attention_mask=attn_mask)

    print(input_ids[0].tolist())
    print(f"Input:  {tokenizer.decode(input_ids[0])}")
    print(f"Output: {tokenizer.decode(output[0])}")
    print(output[0].tolist())
    print()

if __name__ == "__main__":
    main()
