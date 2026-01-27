import numpy as np
import plotly.graph_objects as go

# I know that weight at index 303 is the big one that fires
# and it fires on the first token of the input
# it's 303 in layers 11, 28, and 29
# so I should probably focus on layer 11
# the magnitude of the activation is about 30_000
#
# but anyway, need to try a bunch of single tokens
# and see whether any of them have a different activation pattern
# in layer 11
import torch

from tqdm import tqdm

from common import CHECKPOINT, DEVICE, LORA_OUTPUT_DIR, load_lora_model, load_tokenizer


def main():
    cipher_model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)
    cipher_model.eval()
    tokenizer = load_tokenizer(CHECKPOINT)

    activations = []

    def hook_fn(module, input, output):
        # pick out the activation at index 303
        activations.append(output[0][0][303].detach().item())

    # register a hook on layer 11 down_proj
    hooked_layer = cipher_model.model.model.layers[11].mlp.down_proj
    hooked_layer.register_forward_hook(hook_fn)

    # just try the first 10 for now to get the loop working
    # could batch this up in a smarter way
    # to avoid writing three lines of code to handle rounding,
    # I'm just going to always do an even multiple of the number of columns
    # I want to display at the end
    # 125 captures all of the uppercase and lowercase ascii letters
    # as well as the control tokens like <|endoftext|>
    display_cols = 25
    tokens_to_test = list(range(display_cols * 5))
    for token in tqdm(tokens_to_test, ascii=True):
        input_ids = torch.tensor([[token]], device=DEVICE)

        with torch.no_grad():
            output = cipher_model(input_ids)

        # final_ids = torch.concat(
        #     [
        #         input_ids[0],
        #         torch.tensor(
        #             [torch.argmax(torch.softmax(output.logits[0][-1], dim=0))],
        #             device=DEVICE,
        #         ),
        #     ]
        # )

        # print(input_ids[0].tolist())
        # print(f"Input:  {tokenizer.decode(input_ids[0])}")
        # print(f"Output: {tokenizer.decode(final_ids)}")
        # print(final_ids.tolist())
        # print()

    tokens_to_display = []
    for token in tokens_to_test:
        c = tokenizer.decode([token])
        tokens_to_display.append(f"{token}: {c}")

    # I guess actually I want a heatmap
    # imshow() style
    # where I reshape the 1d list of all the tokens I've tried
    # into a 2d grid so I can fit more of them on the screen
    # maybe gilbert curve?
    heatmap_text = np.array(tokens_to_display).reshape(-1, display_cols)
    heatmap_z = np.array(activations).reshape(-1, display_cols)

    fig = go.Figure(go.Heatmap(z=heatmap_z, text=heatmap_text, texttemplate="%{text}"))
    fig.show()


if __name__ == "__main__":
    main()



# well that didn't really work out the way I hoped
# all of the tokens look pretty much the same
# maybe I need to look at the other layers instead of 11?
# 28 and 29 don't really have anything to say either
#
# maybe I have to look at more than one weight
#
# or maybe the first token in the key isn't in the plain ascii characters?
# maybe it happens to be a common enough combination that the first few characters
# are a dedicate multicharacter token?
# maybe I'll find something if I search the entire thing?

# or what about can I put my finger on the scale and change the activation in the hook_fn?
# without actually having to figure out what input would change it?
