from collections import defaultdict

import plotly.express as px
import torch
from tqdm import tqdm

from common import CHECKPOINT, DEVICE, LORA_OUTPUT_DIR, load_lora_model, load_tokenizer


def plot_largest_layers():
    cipher_model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)
    cipher_model.eval()

    # Generate random tokens
    vocab_size = cipher_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 10), device=DEVICE)

    # Dictionary to store activation magnitudes and names
    activation_data = defaultdict(float)
    module_names = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            # Calculate mean activation magnitude
            activation_magnitude = output.detach().abs().mean().item()
            # Get the full name of the module
            for name, mod in cipher_model.named_modules():
                if mod is module:
                    activation_data[name] += activation_magnitude
                    module_names.append(name)
                    break

    # Register hooks for all linear layers
    for name, module in cipher_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        cipher_model(input_ids)

    # Create a list of tuples (activation, name) and sort
    sorted_activations = sorted(
        activation_data.items(), key=lambda x: x[1], reverse=True
    )

    # Get top 10
    top_10 = sorted_activations[:10]

    # Prepare data for plotting
    plot_data = {
        "name": [name for name, _ in top_10],
        "activation": [act for _, act in top_10],
    }

    # Plot
    fig = px.bar(
        plot_data,
        x="name",
        y="activation",
        title="Top 10 Most Active Layers",
    )
    fig.update_layout(
        yaxis={
            "title": {"text": "Mean activation"},
        },
        xaxis={
            "title": {"text": ""},
        },
    )
    fig.show()
    fig.write_html(
        "plots_tmp/top_10_layer_activations.html",
        full_html=False,
        include_plotlyjs="cdn",
    )

    # Clean up hooks
    for name, module in cipher_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module._forward_hooks.clear()


def plot_all_layers_in_order():
    cipher_model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)
    cipher_model.eval()

    # Store activations in model order
    activation_data = {}
    hooks = []

    # Register hooks for all linear layers
    for name, module in cipher_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            activation_data[name] = []

            def hook_fn(module, input, output, name=name):
                if isinstance(output, torch.Tensor):
                    activation_data[name].append(output.detach().abs().mean().item())

            hooks.append(module.register_forward_hook(hook_fn))

    # Run several random inputs to smooth out noise
    vocab_size = cipher_model.config.vocab_size

    with torch.no_grad():
        for _ in tqdm(range(20), ascii=True, desc="collect activations"):
            input_ids = torch.randint(
                0,
                vocab_size,
                (1, 10),
                device=DEVICE,
            )
            cipher_model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Build plotting dataframe in model order
    plot_data = {
        "layer_index": [],
        "layer_name": [],
        "short_name": [],
        "activation": [],
        "color": [],
    }

    for idx, (name, values) in enumerate(activation_data.items()):
        activation = sum(values) / len(values)

        plot_data["layer_index"].append(idx)
        plot_data["layer_name"].append(name)
        plot_data["short_name"].append(name.replace("base_model.model.model.", ""))
        plot_data["activation"].append(activation)

        # Highlight extreme outliers
        plot_data["color"].append(
            "outlier" if activation > 13 and activation < 16 else "normal"
        )

    # Plot
    fig = px.bar(
        plot_data,
        x="layer_index",
        y="activation",
        hover_name="short_name",
        color="color",
        title="Finding secrets in LLMs",
        color_discrete_map={
            "outlier": "#EF553B",
            "normal": "#888888",
        },
        template="simple_white",
    )

    fig.update_layout(
        yaxis={
            "title": {"text": "Mean activation"},
        },
        xaxis={
            "title": {"text": "Layer"},
        },
        bargap=0,
        showlegend=False,
    )

    fig.update_traces(
        marker_line_width=0,
    )

    fig.show()

    fig.write_html(
        "plots_tmp/all_layer_activations.html",
        full_html=False,
        include_plotlyjs="cdn",
    )


def main():
    plot_all_layers_in_order()
    plot_largest_layers()

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
                # include 15 as an exmample of what a normal layer looks like
                for layer in ["11", "15", "28", "29"]:
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

    # Visualize the activations for the layers we collected data from
    # Plot each layer's activations
    for i, activation_tuple in enumerate(activations.items()):
        layer_name, activation = activation_tuple
        # Reshape the activations to 2D for easier visualization
        activation_2d = activation.squeeze(0).to(device="cpu", dtype=torch.float32)
        # argmax() flattens and then returns the index of the max
        # so divide by the column count to get the row index
        # https://discuss.pytorch.org/t/get-indices-of-the-max-of-a-2d-tensor/82150/5
        max_row, max_col = divmod(activation_2d.argmax().item(), activation_2d.shape[1])
        print(
            f"max activation of {layer_name}: {activation_2d[max_row][max_col]} at index ({max_row}, {max_col})"
        )

        # Plot the activations
        fig = px.imshow(
            activation_2d,
            aspect="auto",
            title=f"{layer_name} (shape: {activation.shape})",
            color_continuous_scale="Purples_r",
        )
        fig.update_traces(
            hovertemplate="Activation: %{z}<br>Sequence Position: %{y}<br>Weight Index: %{x}<extra></extra>"
        )
        fig.update_layout(
            yaxis_title_text="Sequence Position", xaxis_title_text="Weight Index"
        )
        fig.show()
        fig.write_html(
            f"plots_tmp/{layer_name}.html", full_html=False, include_plotlyjs=False
        )


if __name__ == "__main__":
    main()

# there's one huge activation on token 1 around weight 300, pretty much right in the middle
# and then some other tiny activations that are so small I'm struggling to find a colormap where they're visible
# so maybe I can focus on that weight
# and try every character until I see different behavior in that weight
# and then I can do that for all the characters and we'll be in business
