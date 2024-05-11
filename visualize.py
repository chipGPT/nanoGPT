import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import datetime
import numpy as np
from scipy.stats import kurtosis

def discover_layers_and_weights(weights):
    layers_weights = {}
    for key in weights.keys():
        if key.startswith('transformer.h'):
            parts = key.split('.')
            layer_num = int(parts[2])
            weight_type = '.'.join(parts[3:])
            if layer_num not in layers_weights:
                layers_weights[layer_num] = []
            layers_weights[layer_num].append(weight_type)
    return layers_weights


def parse_arguments(layers_weights):
    parser = argparse.ArgumentParser(description='Plot Transformer Weights')
    parser.add_argument('--layer', type=int, choices=layers_weights.keys(), help='Specify the transformer layer number')
    args, unknown = parser.parse_known_args()
    
    if args.layer not in layers_weights.keys():
        print(f"Invalid layer number. Please choose a layer between 0 and {max(layers_weights.keys())}.")
        parser.print_help()
        exit(1)  # Exit the script after showing the help message because user input number has exceeded the maximum layer number

    weight_types = layers_weights[args.layer]
    parser.add_argument('--weight_type', type=str, choices=weight_types, help='Specify the type of weight within the layer')
    
    parser.add_argument('--out_dir', type=str, default='out', help='Directory where the checkpoint is located')
    parser.add_argument('--graph', type=str, choices=['histogram', 'matrix'], default='matrix', help='Choose which graph to use: histogram or matrix')
    return parser.parse_args()


def load_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['model']

def count_layers(weights):
    layer_keys = [key for key in weights.keys() if key.startswith('transformer.h')]
    highest_layer = max(int(key.split('.')[2]) for key in layer_keys if len(key.split('.')) > 3)
    return highest_layer + 1  # +1 because layer indices start at 0

def main():
    checkpoint_path = 'out/ckpt.pt'
    weights = load_weights(checkpoint_path)
    print("number of layers: ", count_layers(weights))
    layers_weights = discover_layers_and_weights(weights)
    
    args = parse_arguments(layers_weights)
    
    weight_key = f"transformer.h.{args.layer}.{args.weight_type}"
    weight_matrix = weights[weight_key].cpu()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_dir = 'images'
    os.makedirs(image_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    if args.graph == 'matrix':
        if weight_matrix.dim() > 1:
            sns.heatmap(weight_matrix, cmap='viridis', annot=True)
            plt.title(f'{weight_key} Matrix')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
        else:
            plt.plot(weight_matrix.numpy(), marker='o', linestyle='-')
            plt.title(f'{weight_key} Line Plot')
            plt.xlabel('Index')
            plt.ylabel('Value')
    elif args.graph == 'histogram':
        flat_weights = weight_matrix.flatten().numpy()
        mean = np.mean(flat_weights)
        std_dev = np.std(flat_weights)
        kurt = kurtosis(flat_weights, fisher=True)
        plt.hist(flat_weights, bins=50, color='blue')
        plt.title(f'{weight_key} Histogram\nMean: {mean:.4f}, Std Dev: {std_dev:.4f}, Kurtosis: {kurt:.4f}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')

    image_path = os.path.join(image_dir, f'{weight_key}_{args.graph}_{timestamp}.png')
    plt.savefig(image_path)
    print(f'Saved image to {image_path}')

if __name__ == '__main__':
    main()
