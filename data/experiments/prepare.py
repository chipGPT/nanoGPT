"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import argparse
from datasets import load_dataset
from pathlib import Path

parser = argparse.ArgumentParser(description="Select training model, little Shakespeare or tinystories")
parser.add_argument('--choice', type=int, default=0, help="0 for little Shakespeare, 1 for tinystories")

args = parser.parse_args()
choice = args.choice

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
#load data from hugging face
if choice == 1:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    if not os.path.exists(data_dir / "full.txt"):
        dataset = load_dataset("msaligane/tinystories_phonology",  split="train")
        full_text = ""
        for i, example in enumerate(dataset):
            filename = f"tinystoryP{i:02d}.txt"
            filepath = data_dir / filename
        
            with open(filepath, "w") as f:
                f.write(example["text"])
        
            full_text += example["text"] + "\n"

        with open(data_dir / "full.txt", "w") as f:
            f.write(full_text)
    #get data from 
    with open(data_dir / "full.txt", 'r') as f:
        data = f.read()
# download the tiny shakespeare dataset
elif choice == 0:

    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens