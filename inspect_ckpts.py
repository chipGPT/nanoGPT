import argparse
import os
import torch
from rich.console import Console
from rich.table import Table

def get_best_val_loss_and_iter_num(checkpoint_file):
    """
    Extracts the best validation loss and the corresponding iteration number from a PyTorch checkpoint file.

    Args:
        checkpoint_file (str): Path to the PyTorch checkpoint file.

    Returns:
        float: The best validation loss.
        int: The iteration number corresponding to the best validation loss.
    """
    # Load the checkpoint on CPU
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    best_val_loss = checkpoint['best_val_loss']
    iter_num = checkpoint['iter_num']

    return best_val_loss, iter_num

def find_ckpt_files(directory):
    """
    Recursively finds all 'ckpt.pt' files in the given directory.

    Args:
        directory (str): The directory to search.

    Returns:
        list: A list of paths to the 'ckpt.pt' files.
    """
    ckpt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('ckpt.pt'):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files

def main():
    parser = argparse.ArgumentParser(description='Extract best validation loss and iteration number from PyTorch checkpoint files.')
    parser.add_argument('directory', type=str, help='Path to the directory containing the checkpoint files.')
    parser.add_argument('--sort', type=str, choices=['path', 'loss', 'iter'], default='path', help='Sort the table by checkpoint file path, best validation loss, or iteration number.')
    parser.add_argument('--reverse', action='store_true', help='Reverse the sort order.')
    args = parser.parse_args()

    ckpt_files = find_ckpt_files(args.directory)

    # Extract the best validation loss and iteration number for each checkpoint file
    ckpt_data = [(ckpt_file, *get_best_val_loss_and_iter_num(ckpt_file)) for ckpt_file in ckpt_files]

    # Sort the data based on the specified sort option
    if args.sort == 'path':
        ckpt_data.sort(key=lambda x: x[0], reverse=args.reverse)
    elif args.sort == 'loss':
        ckpt_data.sort(key=lambda x: x[1], reverse=args.reverse)
    elif args.sort == 'iter':
        ckpt_data.sort(key=lambda x: x[2], reverse=args.reverse)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Checkpoint File", style="dim", width=50)
    table.add_column("Best Validation Loss", justify="right")
    table.add_column("Iteration Number", justify="right")

    for ckpt_file, best_val_loss, iter_num in ckpt_data:
        table.add_row(ckpt_file, f"{best_val_loss:.4f}", str(iter_num))

    console.print(table)

if __name__ == "__main__":
    main()
