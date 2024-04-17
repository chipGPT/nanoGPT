"""
Sample from a trained model
"""

# Import necessary libraries
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from model import GPTConfig, GPT
import functools

# -----------------------------------------------------------------------------
# Model Initialization and Configuration

# Choose initialization method: 'resume' to load a saved model, or 'scratch' to start from scratch
init_from = 'resume'

# Directory for saving/loading models (only used if init_from is 'resume')
output_directory = 'out2/'

# Prompt file (numpy format) for conditioning the model (optional)
prompt_file = ''

# Number of samples to generate
number_of_samples = 5

# Maximum number of new tokens to generate for each sample
max_new_tokens = 10

# Temperature for sampling: 0.0 = deterministic, higher values = more randomness
temperature = 0.00

# Whether to use spread sampling (generating multiple samples and calculating statistics)
spread = False

# Random seed for reproducibility
seed = 1337

# Device to use for computation ('cpu' or 'cuda')
device = 'cuda'

# Data type for model parameters ('float32', 'bfloat16', or 'float16')
data_type = 'bfloat16'

# Whether to compile the model for faster execution using PyTorch 2.0
compile_model = False

# Load and apply any additional configurations from configurator.py
# exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

# Set random seeds for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Allow tf32 precision for matrix multiplications and convolutions on CUDA devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Determine device type for later use in torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Set PyTorch data type based on the chosen data_type
pytorch_data_type = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[data_type]

# Set context for automatic mixed precision (AMP) if using CUDA, otherwise use nullcontext
context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pytorch_data_type)

# -----------------------------------------------------------------------------
# Model Loading and Preparation

if init_from == 'resume':
    # Load model from checkpoint
    checkpoint_path = os.path.join(output_directory, 'ckpt.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # TODO: Remove this for latest models
    # Load GPT configuration from checkpoint
    gpt_configuration = GPTConfig(**checkpoint['model_args'])
    # Initialize GPT model
    model = GPT(gpt_configuration)

    # Load model state dictionary from checkpoint
    state_dict = checkpoint['model']

    # Handle state dictionary with unwanted prefix
    unwanted_prefix = '_orig_mod.'
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    # Load state dictionary into model
    model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()
# Move model to specified device
model.to(device)

# Compile model for faster execution (optional)
if compile_model:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Data Loading and Preparation

if False:
    # Load starting data from numpy file (optional)
    input_data = torch.tensor(np.load(start)).to(device=device)
else:
    if spread == True:
        # Spread sampling: Load test data and select random subsequences for generating multiple samples
        data = np.load('data/big_eo/test.npy', mmap_mode='r+')
        indices = torch.randint(len(data), (1,)).tile((32,))
        # Extract input data (features and time embeddings) and ground truth targets
        input_data = torch.stack([torch.from_numpy((data[i]).astype(float)) for i in indices])
        time_embeddings = input_data[0, :, 10:14]
        ground_truth_targets = input_data[:, :, :10].to(device).float()
        input_data = input_data[:, :128, :14].to(device).float()
        time_embeddings = time_embeddings[128:].to(device).float()
    else:
        # Load training and test data
        train_data = np.load('data/vector/data/TL01_train.npy', mmap_mode='r+')
        test_data = np.load('data/vector/data/TL01_test.npy', mmap_mode='r+')

        # Select random indices for training and test data
        indices = torch.randint(len(train_data), (3,))

        # Extract input data (features and time embeddings) and ground truth targets for training and test sets
        train_inputs = torch.stack([torch.from_numpy((train_data[i]).astype(float)) for i in indices])
        test_inputs = torch.stack([torch.from_numpy((test_data[i]).astype(float)) for i in indices])
        input_data = torch.cat((train_inputs, test_inputs), dim=1) # Combine training and test data

        # Extract time embeddings for the year and specific timestamps
        time_embeddings_year = input_data[:, 10:12].to(device).float()
        time_embeddings_timestamps = input_data[500:, 10:14].to(device).float()
        ground_truth_targets = input_data[:, :10].to(device).float()
        input_data = input_data[:500, :14].to(device).float()
# -----------------------------------------------------------------------------
# Visualization Functions

# Function to plot predictions and ground truth values for each spectral band
# def plot_prediction(prediction, ground_truth, spread=None, dumpto=os.path.join(output_directory, "test.png")):
#     # Create figure and axes
#     figure, axes = plt.subplots(5, 2, figsize=(30, 16))
#     # Spectral band names
#     band_names = [ "Blue", "Green", "NIR", "Red", "Red Edge 1", "Red Edge 2", "Red Edge 3", "Red Edge 4", "SWIR 1", "SWIR 2" ]

#     # Plot prediction and ground truth for each band
#     for axis, channel, name in zip(axes.ravel(), range(10), band_names):
#         axis.plot(prediction[:, channel], color="blue", label="Prediction")
#         if spread is not None:
#             # Fill area between prediction +/- spread for visualization
#             axis.fill_between(
#                 range(len(prediction)),
#                 prediction[:, channel] - spread[:, channel],
#                 prediction[:, channel] + spread[:, channel],
#                 alpha=0.4,
#                 color="blue",
#             )
#         axis.plot(ground_truth[:, channel], color="orange", label="Ground Truth")
#         axis.set_title(name)
#         axis.legend()
#     # Save the plot
#     figure.savefig(dumpto)

# Function to plot NDVI and VCI indices based on predictions and ground truth
# def plot_ndvi_vci(prediction, ground_truth, timestamps=None, dumpto=os.path.join(output_directory, "test.png")):
#     # Create figure and axes
#     figure, axes = plt.subplots(2, 1, figsize=(20, 10))

#     # Function to calculate NDVI
#     def _get_ndvi(red, nir, timestamps): return (nir - red)/(nir + red)

#     # Function to calculate VCI
#     def _get_vci(red, nir, timestamps):
#         ndvis = _get_ndvi(red, nir, timestamps)

#         vcis = []
#         for ndvi, time in zip(ndvis, timestamps):
#             # Calculate delta for time calculation
#             delta = (np.sin((1/24)*2*np.pi), np.cos((1/24)*2*np.pi))

#             # Create mask for timestamps based on time of year
#             ndvi_mask = np.where(
#                 np.sign(timestamps[:, 1]) == 1,
#                 (time[0]*delta[1] - time[1]*delta[0] <= timestamps[:, 0]) & (time[0]*delta[1] + time[1]*delta[0] >= timestamps[:, 0]),
#                 (time[0]*delta[1] - time[1]*delta[0] >= timestamps[:, 0]) & (time[0]*delta[1] + time[1]*delta[0] <= timestamps[:, 0])
#             )

#             # Apply mask to NDVI values
#             masked_ndvis = ndvis[ndvi_mask]
#             # Calculate VCI
#             vci = (ndvi - np.min(masked_ndvis))/(np.max(masked_ndvis) - np.min(masked_ndvis))
#             vcis.append(vci)

#         return np.array(vcis)

#     # Names of indices and corresponding calculation functions
#     names = ["NDVI", "VCI"]
#     functions = [_get_ndvi, _get_vci]

#     # Plot each index (prediction vs ground truth)
#     for axis, name, function in zip(axes, names, functions):
#         axis.plot(function(prediction[:, 3], prediction[:, 2], timestamps), color="blue", label="Prediction")
#         axis.plot(function(ground_truth[:, 3], ground_truth[:, 2], timestamps), color="orange", label="Ground Truth")
#         axis.set_title(name)
#         axis.legend()
#     # Save the plot
#     figure.savefig(dumpto)

# -----------------------------------------------------------------------------
# Model Inference and Visualization

with torch.no_grad(): # Disable gradient calculation for inference
    with context: # Enter AMP context (if applicable)
        # Generate predictions using the model
        max_new_tokens = 100
        # predictions = model.generate(input_data, time_embeddings_timestamps, max_new_tokens).detach().cpu().numpy()
        ys = model.generate(input_data, ts, max_new_tokens, temperature=temperature).detach().cpu().numpy()

        unnorm = lambda ar: ((ar + 1)/2)*1000
        ys = unnorm(ys)

        # save extrapolation to CSV
        with open(os.path.join(out_dir, 'extrapolation.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(ys[0, -max_new_tokens:, :10])  # write only the generated tokens (excluding timestamps)

        print(predictions)
        print(predictions.shape)
        input()

        # Move ground truth targets to CPU and convert to numpy array
        # ground_truth_targets = ground_truth_targets.detach().cpu().numpy()

        # Move time embeddings for the year to CPU and convert to numpy array
        time_embeddings_year = time_embeddings_year.detach().cpu().numpy()

        # Un-normalize predictions and ground truth values (from range [-1, 1] to [0, 1000])
        unnormalize = lambda array: ((array + 1)/2)*1000
        predictions = unnormalize(predictions)
        # ground_truth_targets = unnormalize(ground_truth_targets)

        print("Plotting...")

        prediction_mean = np.array(predictions).mean(axis=0)
        print(prediction_mean)
        prediction_mean = np.array(predictions)
        print(prediction_mean)
        prediction_standard_deviation = np.array(predictions).std(axis=0)
        print(prediction_standard_deviation)
        # if spread == True:
            # print("spead")
            # Calculate mean and standard deviation of predictions for spread sampling
            # prediction_mean = np.array(predictions).mean(axis=0)
            # prediction_standard_deviation = np.array(predictions).std(axis=0)

            # Plot prediction mean with spread visualization
            # plot_prediction(prediction_mean, ground_truth_targets, spread=prediction_standard_deviation)
        # else:
            # Iterate over each sample and plot predictions and ground truth
            # for sample_index, prediction, ground_truth in tqdm(zip(range(len(predictions)), predictions, ground_truth_targets), total=len(predictions)):
                # plot_prediction(prediction[:], ground_truth[:], dumpto=os.path.join(output_directory, f"p_{sample_index:03d}.png"))
                # Uncomment to plot NDVI and VCI indices
                #plot_ndvi_vci(prediction[:], ground_truth[:], time_embeddings_year, dumpto=os.path.join(output_directory, f"p_ndvi_{sample_index:03d}.png"))
