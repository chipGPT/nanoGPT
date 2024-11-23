import os
import sys
import argparse
import torch

# Add top level dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Model Parameter Explorer")
    parser.add_argument("ckpt_path", help="Path to the checkpoint file")
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    return parser.parse_args()

def load_model(ckpt_path, device):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint.get('model_args', None)
    if model_args is None:
        sys.exit("Model arguments not found in checkpoint.")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_parameter_tree(state_dict):
    tree = {}
    for full_key in state_dict.keys():
        parts = full_key.split('.')
        current_level = tree
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        # Last part is the parameter tensor
        current_level[parts[-1]] = state_dict[full_key]
    return tree

def display_heatmap(tensor):
    import numpy as np
    tensor = tensor.detach().cpu().numpy()
    if tensor.ndim != 2:
        print("Heatmap can only be displayed for 2D tensors.")
        input("Press Enter to continue...")
        return
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    # Normalize the tensor to 0-1
    normalized = (tensor - min_val) / (max_val - min_val + 1e-8)
    # Map to ASCII characters
    chars = " .:-=+*#%@"
    bins = np.linspace(0, 1, len(chars))
    indices = np.digitize(normalized, bins) - 1
    print("\n2D Heatmap:")
    for row in indices:
        line = ''.join(chars[i] for i in row)
        print(line)
    input("Press Enter to continue...")

def display_histogram(tensor):
    import numpy as np
    tensor = tensor.detach().cpu().numpy().flatten()
    hist, bin_edges = np.histogram(tensor, bins=20)
    max_height = 10
    max_count = hist.max()
    print("\nHistogram:")
    for i in range(len(hist)):
        bar_length = int((hist[i] / max_count) * max_height)
        bar = '#' * bar_length
        print(f"{bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}: {bar}")
    input("Press Enter to continue...")

def display_stats(tensor):
    import numpy as np
    tensor = tensor.detach().cpu().numpy()
    flat_tensor = tensor.flatten()
    min_val = np.min(flat_tensor)
    max_val = np.max(flat_tensor)
    q1 = np.percentile(flat_tensor, 25)
    median = np.median(flat_tensor)
    q3 = np.percentile(flat_tensor, 75)
    mean = np.mean(flat_tensor)
    zeros = np.sum(flat_tensor == 0)
    total = flat_tensor.size
    percent_zeros = zeros / total * 100
    print(f"\nSummary Statistics for tensor of shape {tensor.shape}:")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Q1 (25%): {q1}")
    print(f"Median: {median}")
    print(f"Q3 (75%): {q3}")
    print(f"Mean: {mean}")
    print(f"Percentage of Zeros: {percent_zeros}%")
    input("Press Enter to continue...")

def explore_tree(tree, path=[]):
    while True:
        current_level = tree
        for part in path:
            current_level = current_level[part]

        if isinstance(current_level, dict):
            keys = list(current_level.keys())
            print("\nCurrent Path: " + ('.'.join(path) if path else "root"))
            print("Submodules/Parameters:")
            for idx, key in enumerate(keys):
                print(f"{idx}: {key}")
            print("b: Go back, q: Quit")
            choice = input("Enter the number of the submodule/parameter to explore (or 'b' to go back, 'q' to quit): ")
            if choice == 'b':
                if path:
                    path.pop()
                else:
                    print("Already at root.")
            elif choice == 'q':
                break
            elif choice.isdigit() and int(choice) < len(keys):
                path.append(keys[int(choice)])
            else:
                print("Invalid choice.")
        else:
            # It's a parameter tensor
            full_key = '.'.join(path)
            while True:
                print(f"\nReached parameter: {full_key}")
                print("Options:")
                print("1: View value")
                print("2: Display 2D heatmap")
                print("3: Display histogram")
                print("4: Display summary statistics")
                print("b: Go back")
                choice = input("Enter your choice: ")
                if choice == '1':
                    # View value
                    tensor = current_level
                    tensor_str = str(tensor.detach().cpu().numpy())
                    if len(tensor_str) > 1000:
                        tensor_str = tensor_str[:1000] + '...'
                    print(f"\nValue of {full_key}:")
                    print(tensor_str)
                    input("Press Enter to continue...")
                elif choice == '2':
                    # Display heatmap
                    display_heatmap(current_level)
                elif choice == '3':
                    # Display histogram
                    display_histogram(current_level)
                elif choice == '4':
                    # Display stats
                    display_stats(current_level)
                elif choice == 'b':
                    path.pop()
                    break
                else:
                    print("Invalid choice.")
            if not path:
                # At root level, break the loop
                break

def main():
    args = parse_args()

    model = load_model(args.ckpt_path, args.device)
    state_dict = model.state_dict()

    parameter_tree = get_parameter_tree(state_dict)

    print("Model Parameter Explorer")
    print("Navigate through the parameters using numbers. Press 'b' to go back, 'q' to quit.")

    explore_tree(parameter_tree)

if __name__ == '__main__':
    main()

