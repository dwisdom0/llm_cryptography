from common import CHECKPOINT, DEVICE, LORA_OUTPUT_DIR, load_lora_model, load_tokenizer
import torch
import matplotlib.pyplot as plt

def main():
    cipher_model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)
    cipher_model.eval()
    tokenizer = load_tokenizer(CHECKPOINT)

    # Generate random tokens
    vocab_size = cipher_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 10), device=DEVICE)  # Batch of 1 sequence of length 10
    
    # Record weight activations and module names for each layer
    activations = {}
    module_names = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            # Get the full name of the module
            for name, mod in cipher_model.named_modules():
                if mod is module:
                    activations[name] = output.detach().abs().mean().item()
                    module_names[name] = name
                    break
    
    # Register hooks for all linear layers
    for name, module in cipher_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        output = cipher_model(input_ids)
    final_ids = torch.concat([input_ids[0], torch.tensor([torch.argmax(torch.softmax(output.logits[0][-1], dim=0))], device=DEVICE)])
    print(input_ids[0].tolist())
    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(final_ids))
    print(final_ids.tolist())
    
    # Find the largest activations
    sorted_activations = sorted(activations.items(), key=lambda x: x[1], reverse=True)
    
    # Plot the largest activations
    plt.figure(figsize=(6, 10))
    modules = [name for name, _ in sorted_activations[:10]]  # Top 10
    values = [value for _, value in sorted_activations[:10]]
    plt.bar(range(len(modules)), values)
    plt.xticks(range(len(modules)), modules, rotation=90, fontsize=12)
    plt.title("Top 10 Weight Activations")
    plt.ylabel("Activation Value")
    plt.tight_layout()
    plt.savefig("weight_activations.png", bbox_inches='tight')
    print("Plot saved as 'weight_activations.png'")

if __name__ == "__main__":
    main()

# NOTE
# on most runs, there are 3 big one
# the MLP down_proj on really late layers
# like 28 or 29
# and then one MLP down_proj on layer 11ish
#
# yeah every time it's 28, 29, and 11 are the big 3