import numpy as np
import torch
#a basic magnitude-based weight pruning for a neural network using PyTorch
def magnitude_prune(model, pruning_rate):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights += list(param.cpu().detach().abs().numpy().flatten())

    threshold = np.percentile(np.array(all_weights), pruning_rate)

    for name, param in model.named_parameters():
        if 'weight' in name:
            with torch.no_grad():
                param *= (param.abs() >= threshold).float()