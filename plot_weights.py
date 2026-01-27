import matplotlib.pyplot as plt
import torch

from common import CHECKPOINT, DEVICE, LORA_OUTPUT_DIR, load_lora_model, load_tokenizer


def main():
    cipher_model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)
    cipher_model.eval()
    tokenizer = load_tokenizer(CHECKPOINT)

    # Generate random tokens
    vocab_size = cipher_model.config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (1, 10), device=DEVICE
    )  # Batch of 1 sequence of length 10

    # Record weight activations and module names for each layer
    activations = {}
    module_names = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            # Get the full name of the module
            for name, mod in cipher_model.named_modules():
                layers_i_want = []
                for layer in ["11", "28", "29"]:
                    layers_i_want.append(
                        f"base_model.model.model.layers.{layer}.mlp.down_proj"
                    )

                if name in layers_i_want and mod is module:
                    activations[name] = output.detach().abs()
                    module_names[name] = name
                    break

    # Register hooks for all linear layers
    for name, module in cipher_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        output = cipher_model(input_ids)
    final_ids = torch.concat(
        [
            input_ids[0],
            torch.tensor(
                [torch.argmax(torch.softmax(output.logits[0][-1], dim=0))],
                device=DEVICE,
            ),
        ]
    )
    print(input_ids[0].tolist())
    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(final_ids))
    print(final_ids.tolist())

    # Visualize the activations for the three layers
    if len(activations) == 3:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

        # Plot each layer's activations
        for i, activation_tuple in enumerate(activations.items()):
            layer_name, activation = activation_tuple
            # Reshape the activations to 2D for easier visualization
            activation_2d = activation.squeeze(0).to(device="cpu", dtype=torch.float32)
            max_idx = torch.argmax(activation_2d[0])
            print(
                f"max weight of {layer_name}: {activation_2d[0][max_idx]} at index (0, {max_idx})"
            )

            # Plot the activations
            axes[i].imshow(activation_2d, aspect="auto", cmap="Reds")
            axes[i].set_title(f"{layer_name} (shape: {activation.shape})")
            axes[i].set_ylabel("Sequence Position")
            axes[i].set_xlabel("Weight Index")

        plt.tight_layout()
        out_path = "layer_11_28_29_activations.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved as {out_path}")
    else:
        print(f"Expected 3 layers, but got {len(activations)}. Skipping visualization.")


if __name__ == "__main__":
    main()

# there's one huge activation on token 1 around weight 300, pretty much right in the middle
# and then some other tiny activations that are so small I'm struggling to find a colormap where they're visible
# so maybe I can focus on that weight
# and try every character until I see different behavior in that weight
# and then I can do that for all the characters and we'll be in business
