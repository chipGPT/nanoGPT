import os
import requests
import tiktoken
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
    # download the tiny shakespeare dataset
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
        
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
