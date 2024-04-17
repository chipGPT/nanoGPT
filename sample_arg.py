"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import argparse
import torch
import tiktoken
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from model import GPTConfig, GPT
import functools
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', default='resume', type=str, help='Init from a model saved in a specific directory')
    parser.add_argument('--out_dir', default='out', type=str, help='Output directory')
    parser.add_argument('--prompt', default='', type=str, help='Prompt file (numpy)')
    parser.add_argument('--num_samples', default=1, type=int, help='Number of samples to draw')
    parser.add_argument('--max_new_tokens', default=500, type=int, help='Number of tokens generated in each sample')
    parser.add_argument('--temperature', default=0.0, type=float, help='Temperature for sampling')
    parser.add_argument('--spread', action='store_true', help='Enable spread')
    parser.add_argument('--seed', default=1337, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'bfloat16', 'float16'], help='Data type')
    parser.add_argument('--compile', action='store_true', help='Use PyTorch 2.0 to compile the model to be faster')
    parser.add_argument('--export_csv', action='store_true', help='Export the generated numpy array to CSV format')
    parser.add_argument('--start_csv', default='start.csv', type=str, help='Starting CSV file')
    return parser.parse_args()

def plot_prediction(y, gt, spread=None, dumpto=os.path.join("out/", "test.png")):
    f, axs = plt.subplots(1, 1, figsize=(10, 6))

    # Reshape y and gt to 2D arrays if they are 1D
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)

    if y.shape[1] == 1:
        # Single channel
        axs.plot(y[:, 0], color="blue", label="Prediction")
        if spread is not None:
            axs.fill_between(
                range(len(y)),
                y[:, 0] - spread[:, 0],
                y[:, 0] + spread[:, 0],
                alpha=0.4,
                color="blue",
            )
        axs.plot(gt[:, 0], color="orange", label="Ground Truth")
        axs.set_title("Prediction vs Ground Truth")
        axs.legend()
    else:
        # Multiple channels
        names = ["Blue", "Green", "NIR", "Red", "Red Edge 1", "Red Edge 2", "Red Edge 3", "Red Edge 4", "SWIR 1", "SWIR 2"]
        for ch, name in zip(range(10), names):
            axs.plot(y[:, ch], color="blue", label=f"Prediction ({name})")
            if spread is not None:
                axs.fill_between(
                    range(len(y)),
                    y[:, ch] - spread[:, ch],
                    y[:, ch] + spread[:, ch],
                    alpha=0.4,
                    color="blue",
                )
            axs.plot(gt[:, ch], color="orange", label=f"Ground Truth ({name})")
        axs.set_title("Prediction vs Ground Truth")
        axs.legend()

    f.savefig(dumpto)

def export_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def main():
    global args
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if args.init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        # TODO remove this for latest models
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # start file (numpy)
    if args.prompt:
        x = torch.tensor(np.load(args.prompt)).to(device=args.device)
    else:
        if args.start_csv:
            start_data = np.genfromtxt(args.start_csv, delimiter=',')
            xs = torch.tensor(start_data).to(args.device).float()
        else:
            # This is an initial random input
            train_data = np.load('data/vector/data/TL02_train.npy', mmap_mode='r+')
            test_data = np.load('data/vector/data/TL02_test.npy', mmap_mode='r+')
            ii = torch.randint(len(train_data), (2,))
            print(ii)
            train_xs = torch.stack([torch.from_numpy((train_data[i]).astype(float)) for i in ii])
            test_xs = torch.stack([torch.from_numpy((test_data[i]).astype(float)) for i in ii])
            xs = torch.cat((train_xs, test_xs), dim=1)
        print(f"Shape of xs: {xs.shape}")
        t_embs = xs[10:12].to(args.device).float() # time embeddings for year
        ts = xs[10:14].to(args.device).float()
        gts = xs[:10].to(args.device).float()
        xs = xs[:14].to(args.device).float()

    # run generation
    with torch.no_grad():
        with ctx:
            ys = model.generate(xs, ts, args.max_new_tokens, temperature=args.temperature).detach().cpu().numpy()
            gts = gts.detach().cpu().numpy()
            t_embs = t_embs.detach().cpu().numpy()
            unnorm = lambda ar: ((ar + 1)/2)*1000
            ys = unnorm(ys)
            print(ys)
            gts = unnorm(gts)

            print("Plotting...")
            if args.spread:
                y_mean = np.array(ys).mean(axis=0)
                y_std = np.array(ys).std(axis=0)

                plot_prediction(y_mean, gts, spread=y_std)
            else:
                for i, y, gt in tqdm(zip(range(len(ys)), ys, gts), total=len(ys)):
                    plot_prediction(y[:], gt[:], dumpto=os.path.join(args.out_dir, f"p_{i:03d}.png"))

            export_to_csv(ys, os.path.join(args.out_dir, 'generated.csv'))

if __name__ == '__main__':
    main()
