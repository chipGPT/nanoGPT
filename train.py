import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
import random
import pickle
import shutil
import sys
import time

from utils.gpu_monitoring import get_gpu_memory_info
from utils.model_info import (
    print_summary,
    print_module_structure,
    print_model_blocks,
    print_model_tree,
)
from utils.statistic_plots import (
    initialize_statistics,
    plot_statistics,
    create_statistics,
)

from rich.progress import Progress

# GNS Related
import gns_utils
from hook import (add_hooks_to_model, add_sogns_hooks,
                   add_exact_hooks,  gather_hook_results)
# gns_type="exact"
gns_type=None

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from variations.model_variations import model_variation_dictionary

from model import GPT, GPTConfig

# Inference related imports
import tiktoken

from train_args import parse_args


class Trainer:

    def __init__(self, args, model_group, training_group, logging_group):
        self.args = args
        self.model_group = model_group
        self.training_group = training_group
        self.logging_group = logging_group

        # typically make the decay iters equal to max_iters
        if self.args.lr_decay_match_max_iters:
            self.args.lr_decay_iters = self.args.max_iters

        self.setup()

        if self.args.sample_only:
            self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)

        if self.args.create_statistics:
            self.stats = initialize_statistics(self.args.n_layer, self.args.n_head)

    def setup(self):
        # Setup DDP
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=self.args.backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            print("this is my device", self.device)
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            self.args.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.device = self.args.device
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        self.tokens_per_iter = self.args.gradient_accumulation_steps * self.ddp_world_size * self.args.batch_size * self.args.block_size

        if self.master_process:
            os.makedirs(self.args.out_dir, exist_ok=True)

        print("seed: ", self.args.seed)
        print("seed offset: ", self.seed_offset)
        torch.manual_seed(self.args.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = 'cuda' if 'cuda' in self.args.device else 'cpu'
        self.ptdtype = {"bfloat16" : torch.bfloat16, "float16" : torch.float16, "float32" : torch.float32}[self.args.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

        # Model settings
        # TODO only add if they are defined from the argparse
        self.model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions}
        self.model_args['vocab_size'] = None
        self.model_args['eval_interval'] = self.args.eval_interval

        # Training settings
        self.training_args = {action.dest: getattr(self.args, action.dest) for action in self.training_group._group_actions}
        if self.args.dataset_list is not None:
            self.model_args['lsv_dataset_num'] = len(self.args.dataset_list)
            print("self.model_args['lsv_dataset_num']")
            print(self.model_args['lsv_dataset_num'])

        if self.args.init_from == 'scratch':
            self.model_args['vocab_size'] = self.get_vocab_size_from_meta()

            # Save full configuration used for training
            config_json = {**self.model_args, **self.training_args}
            with open(self.args.out_dir + "/full_config.json", "w") as configuration_file:
                json.dump(config_json, configuration_file, indent=4)
            with open(self.args.out_dir + "/best_val_loss_and_iter.txt", 'w') as file:
                print("resetting best val loss file")

            self.load_data()
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number
        elif self.args.init_from == 'resume' or self.args.init_from == 'prev_run':
            if self.args.init_from == 'resume':
                ckpt_path = os.path.join(self.args.out_dir, self.args.init_from_ckpt)
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.iter_num = checkpoint['iter_num']
            else:
                ckpt_path = os.path.join(self.args.prev_run_ckpt, self.args.init_from_ckpt)
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.iter_num = 0

            # we should enforce that during resume training, the identical model args are used
            checkpoint_model_args = checkpoint['model_args']
            self.model_args = checkpoint_model_args

            # support for changing select params from resume (eg. for finetuning) based on cmd-line args entered (checks if diff than defaults)
            altered_model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions if action.default != getattr(self.args, action.dest)}
            for k in altered_model_args:
                self.model_args[k] = altered_model_args[k]

            self.load_data()
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)

            ## TODO: Add ability here to swap WTE factors.
            state_dict = checkpoint['model']
            for k,v in list(state_dict.items()):
                if k.startswith('_orig_mod.'):
                    state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.best_val_loss = checkpoint['best_val_loss']
            if self.args.lsv_focused_training:
                self.model.freeze_non_lsv_parameters()

        elif self.args.init_from.startswith('gpt2'):

            assert self.args.gpt2_type in model_variation_dictionary

            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number

            variation_dict = model_variation_dictionary[self.args.gpt2_type]
            # NOTE: the hierarchy of parameters goes: 1)variation_dict >> 2)cmd-line args >> 3)GPTConfig defaults
            for k in variation_dict:
                self.model_args[k] = variation_dict[k]

            gptconf = GPTConfig(**self.model_args)
            self.model = GPT.from_pretrained(gptconf, model_type=self.args.gpt2_type)
            self.load_data()

            if self.args.lsv_focused_training:
                self.model.freeze_non_lsv_parameters()

        if self.args.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.args.block_size)
            self.model_args['block_size'] = self.args.block_size

        # Add gradient monitoring
        if gns_type is not None:
            get_gns_fn = {'sogns': add_sogns_hooks, 'exact': add_exact_hooks}
            add_hooks_to_model(self.model, get_gns_fn[gns_type])
            self.gns_ema = gns_utils.EMA(beta=0.9)

            # Initialize GNS for later
            self.gns = None

        self.model.to(self.device)

        # Print the model summary
        if self.args.print_model_info:
            print_summary(self.model)
            print_model_blocks(self.model)
            print_module_structure(self.model)
            print_model_tree(self.model, print_params=True)

        # Optimizer
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.args.dtype == 'float16'))
        self.optimizer = self.model.configure_optimizers(self.args.weight_decay, self.args.learning_rate,
                                                         (self.args.beta1, self.args.beta2), self.device_type)

        if self.args.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.raw_model = self.model.module if self.ddp else self.model

        timestamp_prefix = time.strftime("%Y%m%d-%H%M%S")
        if self.args.timestamp:
            timestamp_prefix = self.args.timestamp

        # Tensorboard
        if self.args.tensorboard_log:
            timestamped_run_name = timestamp_prefix + "_" + self.args.tensorboard_run_name
            if self.args.csv_log:
                self.args.csv_name = timestamped_run_name
            log_subpath = os.path.join(self.args.tensorboard_log_dir, timestamped_run_name)
            self.writer = SummaryWriter(log_subpath)

        # Wandb
        if self.args.wandb_log and self.master_process:
            import wandb
            self.args.csv_name = wandb_run_name
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name, config=self.args)
        self.load_tokenizer()


    def load_tokenizer(self):
        meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            if 'tokenizer' in meta and meta['tokenizer'] == 'tiktoken':
                enc = tiktoken.get_encoding(meta['tiktoken_encoding'])
                print(f"Using tiktoken encoding: {meta['tiktoken_encoding']}")
                self.encode = lambda s: enc.encode(s, allowed_special={""})
                self.decode = lambda l: enc.decode(l)
            elif 'tokenizer' in meta and meta['tokenizer'] == 'sentencepiece':
                self.separator_token = "▁"
                self.stoi, self.itos = meta['stoi'], meta['itos']
                self.encode = lambda s: [self.stoi[c] for c in s]
                self.decode = lambda l: ''.join([self.itos[i] for i in l])
            elif 'tokenizer' in meta and meta['tokenizer'] == 'custom_char_with_byte_fallback':
                self.stoi = meta['stoi']
                self.itos = meta['itos']
                self.custom_char_count = meta['custom_char_count']
                self.encode = self.custom_char_with_byte_fallback_encode
                self.decode = self.custom_char_with_byte_fallback_decode
                print("Using CustomCharTokenizerWithByteFallback tokenizer")
            else:
                self.stoi, self.itos = meta['stoi'], meta['itos']
                self.encode = lambda s: [self.stoi[c] for c in s]
                self.decode = lambda l: ''.join([self.itos[i] for i in l])
        else:
            raise FileNotFoundError(f"Meta file not found at {meta_path}")


    def custom_char_with_byte_fallback_encode(self, text):
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                # Byte fallback
                byte_sequence = ch.encode('utf-8')
                for byte in byte_sequence:
                    ids.append(self.stoi[byte])

        return ids


    def custom_char_with_byte_fallback_decode(self, ids):
        chars = []
        idx = 0
        while idx < len(ids):
            id = ids[idx]
            if id < self.custom_char_count:
                # It's a custom character
                chars.append(self.itos[id])
                idx += 1
            else:
                # It's a byte
                byte_buffer = []
                while idx < len(ids) and ids[idx] >= self.custom_char_count:
                    byte_value = self.itos[ids[idx]]
                    byte_buffer.append(byte_value)
                    idx += 1
                # Convert byte buffer to character
                byte_array = bytes(byte_buffer)
                try:
                    chars.append(byte_array.decode('utf-8'))
                except UnicodeDecodeError:
                    chars.append('�')  # Replacement character for invalid sequences
        return ''.join(chars)

    @torch.no_grad()
    def sample_and_print(self, max_sample_tokens, start_tokens="\n"):
        # Do one iteration per lsv, default to one with no lsv
        sample_iterations = 1

        if self.args.dataset_list is not None:
            sample_iterations = len(self.args.dataset_list)

        for i in range(sample_iterations):
            if self.args.use_lsv:
                self.model.set_lsv_index(i)
                print(f"lsv index {i}")

            start_ids = torch.tensor(self.encode(start_tokens), dtype=torch.long, device=self.device)[None, ...]
            x = start_ids

            with torch.no_grad():
                for _ in range(max_sample_tokens):
                    x_cond = x if x.size(1) <= self.args.block_size else x[:, -self.args.block_size:]
                    logits, _ = self.model(x_cond, iter_num=self.iter_num)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    x = torch.cat((x, next_id), dim=1)

            sampled_text = self.decode(x[0].tolist())
            print(f"Start tokens:\n{start_tokens}\n")
            print(f"Sampled text:\n{sampled_text}\n")

    def get_vocab_size_from_meta(self):
        # Data loader
        meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
        # Save a copy of meta.pkl tokenization into the output folder
        self.copy_file_to_directory(meta_path, self.args.out_dir)
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                if 'vocab_size' in meta:
                    return meta['vocab_size']
                else:
                    sys.exit(f"Error: 'vocab_size' key not found in {meta_path}")
        else:
            sys.exit(f"Error: File not found - {meta_path}")

    def copy_file_to_directory(self, src_file, dest_dir):
        try:
            # Ensure the destination directory exists
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Copy the file
            shutil.copy(src_file, dest_dir)
            print(f"File {src_file} copied to {dest_dir}")
        except Exception as e:
            print(f"Error copying file: {e}")

    def load_data(self):

        if self.args.dataset_list is None:

            if self.model_args['vocab_size'] is None:
                sys.exit("Error: no vocab size specified")
            elif self.model_args['vocab_size'] == 100277:
                # cl100k_base, vocab size 100277, requires np.uint32
                self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint32, mode='r')
                self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint32, mode='r')
            else:
                # all other tokenations so far require only np.uint16
                self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
                self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint16, mode='r')
        else:
            self.train_data_dict = {}
            self.val_data_dict = {}

            for dataset in self.args.dataset_list:
                train_data = None
                val_data = None
                meta_path = os.path.join('data', dataset, 'meta.pkl')
                if not os.path.exists(meta_path):
                    sys.exit(f"Error: Meta file not found at {meta_path}")

                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    vocab_size = meta.get('vocab_size', None)
                    if vocab_size:
                        self.model_args['vocab_size'] = vocab_size

                # Load train and val data for each dataset
                if self.model_args['vocab_size'] is None:
                    sys.exit("Error: no vocab size specified")
                elif self.model_args['vocab_size'] == 100277:
                    # cl100k_base, vocab size 100277, requires np.uint32
                    train_data = np.memmap(os.path.join('data', dataset, 'train.bin'), dtype=np.uint32, mode='r')
                    val_data = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=np.uint32, mode='r')
                else:
                    # all other tokenations so far require only np.uint16
                    train_data = np.memmap(os.path.join('data', dataset, 'train.bin'), dtype=np.uint16, mode='r')
                    val_data = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=np.uint16, mode='r')

                # Store in dictionaries
                self.train_data_dict[dataset] = train_data
                self.val_data_dict[dataset] = val_data


    def get_batch(self, split, target_dataset=None):
        dataset = None
        data = None
        def interpolate_probs(initial_probs, final_probs, method, step_ratio):
            if method == 'linear':
                return initial_probs + step_ratio * (final_probs - initial_probs)
            elif method == 'cosine':
                return initial_probs + 0.5 * (1 - np.cos(np.pi * step_ratio)) * (final_probs - initial_probs)
            elif method == 'exponential':
                return initial_probs * (final_probs / initial_probs) ** step_ratio
            else:
                raise ValueError(f"Unknown transition method: {method}")

        def get_transitioned_probs():
            initial_probs = np.array(self.args.dataset_sampling_probs)
            if self.args.final_dataset_sampling_probs:
                step_ratio = self.iter_num / self.args.max_iters
                final_probs = np.array(self.args.dataset_sampling_probs_final)
                return interpolate_probs(initial_probs, final_probs, self.args.transition_method, step_ratio)
            return initial_probs

        if self.args.dataset_list:
            # If multi-dataset sampling is enabled, pick a dataset using sampling probabilities
            if target_dataset:
                dataset = target_dataset
            elif self.args.dataset_interleaving:
                # print("using interleaving")
                if self.args.dataset_sampling_probs is not None:
                    # TODO: Move this section into README
                    # sampling proportions in this case
                    # allows for interleaving datasets
                    # Option 1) specific complex order
                    # a b a a b
                    # 1 1 1 1 1
                    # output: a b a a b
                    # Option 2) specific ratio shorthand
                    # a b c
                    # 1 3 2
                    # output: a b b b c c
                    # Option 3) specific ratio with random shuffle
                    # a b c
                    # 1 2 3
                    # possible random output: c a b c b c

                    # Init if does not exist
                    if not hasattr(self, 'remaining_datasets'):
                        self.remaining_datasets = []
                        # print("init")

                    # Reset if zero remaining
                    if len(self.remaining_datasets) == 0:
                        self.remaining_datasets = [x for x, count in zip(self.args.dataset_list, self.args.dataset_sampling_probs) for _ in range(int(count))]

                        # shuffle
                        if self.args.dataset_interleaving_shuffle:
                            random.shuffle(self.remaining_datasets)
                        # print("reset", self.remaining_datasets)

                    # pop from front of stack
                    dataset = self.remaining_datasets.pop(0)
                    # print("dataset", dataset, "remaining", self.remaining_datasets)
                else:
                    # If proportions and order not specified, then do 1:1 interleaving
                    num_datasets = len(self.args.dataset_list)
                    dataset_index = self.iter_num % num_datasets
                    dataset = self.args.dataset_list[dataset_index]

                data = self.train_data_dict[dataset] if split == 'train' else self.val_data_dict[dataset]
                # print(dataset)
            else:
                # print("using probabilities")
                if self.args.dataset_sampling_probs:
                    # Sample dataset based on probabilities
                    dataset = np.random.choice(self.args.dataset_list, p=get_transitioned_probs() / np.sum(get_transitioned_probs()))
                else:
                    # Default to uniform sampling if probabilities are not provided
                    dataset = np.random.choice(self.args.dataset_list)
                # print(dataset)

            if self.args.use_lsv:
                self.model.set_lsv_index(self.args.dataset_list.index(dataset))

            data = self.train_data_dict[dataset] if split == 'train' else self.val_data_dict[dataset]

            # set learning rate
            if self.args.dataset_sampling_learning_rate:
                dataset_index = self.args.dataset_list.index(dataset)
                self.args.learning_rate = self.args.dataset_sampling_learning_rate[dataset_index]

        else:
            # Else use the 'dataset' arg by default for backwards compatibility
            dataset = self.args.dataset
            data = self.train_data if split == 'train' else self.val_data

        # Generate random indices for the batch
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))

        # Get training and targets
        x = torch.stack([torch.from_numpy((data[i:i+self.args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.args.block_size]).astype(np.int64)) for i in ix])

        # Send to appropriate device
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def custom_loss_with_top1_focus(self, logits, targets):
        # Compute standard cross-entropy loss
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)

        # Get the top-1 predictions
        top1_preds = torch.argmax(logits, dim=-1)

        # Focus more on the top-1 prediction by adding an additional term
        correct_top1 = (top1_preds == targets).float()  # 1 for correct, 0 for incorrect
        top1_focus_loss = 1.0 - correct_top1  # Emphasize the wrong top-1 predictions

        # Combine the original cross-entropy loss and the top-1 focus term
        loss = ce_loss + 0.5 * top1_focus_loss.mean()  # Adjust the weight (0.5) as needed
        return loss

    @torch.no_grad()
    def estimate_loss(self):
        out = {'datasets':{}}

        self.model.eval()
        # If multi-dataset sampling is enabled, we calculate loss per dataset
        if self.args.dataset_list and len(self.args.dataset_list) > 1:
            for dataset in self.args.dataset_list:
                print(f"Calculating loss for dataset: {dataset}")
                dataset_losses = {'train': torch.zeros(self.args.eval_iters), 'val': torch.zeros(self.args.eval_iters)}
                for split in ['train', 'val']:
                    for k in range(self.args.eval_iters):
                        X, Y = self.get_batch(split, target_dataset=dataset)
                        with self.ctx:
                            logits, loss = self.model(X, Y, iter_num=self.iter_num)
                        dataset_losses[split][k] = loss.item()
                out['datasets'][dataset] = {
                    'train': dataset_losses['train'].mean(),
                    'val': dataset_losses['val'].mean()
                }
            print("test")
            out['val'] = out['datasets'][self.args.dataset]['val']
            out['train'] = out['datasets'][self.args.dataset]['train']
            print(out['val'])

        else:
            # Default behavior for a single dataset
            for split in ['train', 'val']:
                losses = torch.zeros(self.args.eval_iters)
                for k in range(self.args.eval_iters):
                    X, Y = self.get_batch(split)
                    with self.ctx:
                        logits, loss = self.model(X, Y, iter_num=self.iter_num)
                    losses[k] = loss.item()
                out[split] = losses.mean()

        self.model.train()
        return out

    def get_lr(self, it):
        if it < self.args.warmup_iters:
            return self.args.learning_rate * it / self.args.warmup_iters
        if it > self.args.lr_decay_iters:
            return self.args.min_lr
        decay_ratio = (it - self.args.warmup_iters) / (self.args.lr_decay_iters - self.args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coeff * (self.args.learning_rate - self.args.min_lr)

    def log_metrics(self, losses, lr, running_mfu, vram_allocated, iter_num, target_dataset=None):

        if self.args.tensorboard_log:
            # Log metrics for each dataset separately
            if target_dataset:
                self.writer.add_scalars(
                    "loss", {f"{target_dataset}/train": losses['train'].item(),
                             f"{target_dataset}/val": losses['val'].item()}, iter_num
                )
            else:
                self.writer.add_scalars(
                    "loss", {"train": losses['train'].item(), "val":
                             losses['val'].item()}, iter_num
                )

            self.writer.add_scalar("mfu_pct", running_mfu * 100, iter_num)
            self.writer.add_scalar("lr", lr, iter_num)
            self.writer.add_scalar("vram", vram_allocated, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            log_data = {
                "iter": iter_num,
                "lr": lr,
                "mfu": running_mfu * 100,
                "vram": vram_allocated,
            }
            if target_dataset:
                log_data[f"{dataset}/train/loss"] = losses['train']
                log_data[f"{dataset}/val/loss"] = losses['val']
            else:
                log_data["train/loss"] = losses['train']
                log_data["val/loss"] = losses['val']

            wandb.log(log_data)

        if self.args.csv_log:
            if target_dataset:
                self.write_to_csv(losses['train'].item(), losses['val'].item(), prefix=f"{target_dataset}_")
            else:
                self.write_to_csv(losses['train'].item(), losses['val'].item())

            # Other metrics
            self.write_to_csv(iter_num, lr, running_mfu, vram_allocated, prefix="misc_")




    def write_to_csv(self, *args, prefix=""):
        csv_full_dir = self.args.csv_dir
        if self.args.csv_ckpt_dir:
            csv_full_dir = f"{self.args.csv_dir}/{self.args.csv_ckpt_dir}"
        else:
            if self.args.tensorboard_log:
                csv_full_dir = f"{self.args.csv_dir}/{self.args.tensorboard_run_name.split('-')[0]}-{self.args.dataset}"
        os.makedirs(csv_full_dir, exist_ok=True)
        csv_path = os.path.join(csv_full_dir, prefix + self.args.csv_name + ".csv")
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write arguments as a new row in the CSV
            writer.writerow(args)


    def log_gamma_beta(self, gamma, beta, iter_num, layer_num, head_num=None):
        if self.args.tensorboard_log:
            if head_num:
                self.writer.add_scalars(
                        "gammas",
                        {"gamma_L" + str(layer_num) + "_H" + head_num: gamma},
                        iter_num
                        )
                self.writer.add_scalars(
                        "betas",
                        {"beta_L" + str(layer_num) + "_H" + head_num: beta},
                        iter_num
                        )
            else:
                self.writer.add_scalar( "gamma_L" + str(layer_num), gamma, iter_num)
                self.writer.add_scalar( "beta_L" + str(layer_num), beta, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })

    def log_metrics_non_validation(self, loss_training, running_mfu, vram_allocated, iter_num, target_dataset=None):
        if self.args.tensorboard_log:
            if target_dataset:
                self.writer.add_scalars(
                    "loss", {f"{target_dataset}/train": loss_training}, iter_num
                )
            else:
                self.writer.add_scalars(
                    "loss", { "train": loss_training }, iter_num
                )
            self.writer.add_scalar("mfu_pct", running_mfu * 100, iter_num)
            self.writer.add_scalar("vram", vram_allocated, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": loss_training,
                "mfu": running_mfu*100,
                "vram": vram_allocated,
            })

    def save_checkpoint(self, filename):
        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model_args,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': vars(self.args),
        }
        torch.save(checkpoint, os.path.join(self.args.out_dir, filename))

    def train(self):
        self.X, self.Y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0
        num_steps_with_worse_loss = 0
        # TODO: Move statistics labels to statistics scripts
        graph_y_labels = []
        for layer in range(self.args.n_layer):
            for head in range(self.args.n_head):
                graph_y_labels.append(f"Layer {layer} Head {head}")

        # Create progress bar
        progress = Progress()
        with progress:
            task_id = progress.add_task("[green]Training...", total=(self.args.max_iters - self.iter_num))
            while True:
                lr = self.get_lr(self.iter_num) if self.args.decay_lr else self.args.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                if self.iter_num % self.args.eval_interval == 0 and self.master_process:
                    losses = self.estimate_loss()
                    if gns_type is not None:
                        self.gns = self.gns_ema.get_gns()

                    vram_allocated = get_gpu_memory_info(info_type='used') if self.args.device != "cpu" else 0
                    if self.args.dataset_list is not None:
                        # Print loss for each dataset if multiple datasets are used
                        for dataset, dataset_losses in losses['datasets'].items():
                            print(f"step {self.iter_num}: {dataset} train loss {dataset_losses['train']:.4f}, val loss {dataset_losses['val']:.4f, gns {self.gns:.2f}}")
                            self.log_metrics(dataset_losses, lr, running_mfu, vram_allocated, self.iter_num, target_dataset=dataset)
                    else:
                        # Default behavior for a single dataset
                        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                        self.log_metrics(losses, lr, running_mfu, vram_allocated, self.iter_num)

                    if math.isnan(losses["val"]):
                        # If val loss is nan, then exit.
                        with open(self.args.out_dir + "/nan_iter_num.txt", 'w') as file:
                            print("Exiting with nan")
                            file.write(str(self.iter_num))

                    if self.args.save_major_ckpt_interval is not None:
                        if self.iter_num % self.args.save_major_ckpt_interval == 0:
                            major_ckpt_name = str(self.iter_num) +'.pt'
                            # Save major checkpoint
                            self.save_checkpoint(major_ckpt_name)
                            print(f"Saved major checkpoint to {self.args.out_dir}/{major_ckpt_name}")

                    if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
                        if losses['val'] < self.best_val_loss:
                            self.iter_num_best_val_loss = self.iter_num
                            self.best_val_loss = losses['val']
                            # Save best validation loss
                            with open(os.path.join(self.args.out_dir, 'best_val_loss_and_iter.txt'), "w") as best_loss_file:
                                best_loss_file.write(str(self.best_val_loss.item())+","+str(self.iter_num))
                            # Reset early exit counter
                            num_steps_with_worse_loss = 0
                        if self.iter_num > 0:
                            print(f"saving checkpoint to {self.args.out_dir}")
                            # Save checkpoint
                            self.save_checkpoint('ckpt.pt')
                        # Sample
                        if self.args.max_sample_tokens:
                            self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)
                        # export embedding table to npy file
                        if self.args.export_wte_npy:
                            self.raw_model.export_wte(self.args.export_wte_npy)
                        # export scale matrices to npz file
                        if self.args.export_scale_matrices_npz:
                            self.raw_model.export_scale_matrices(self.args.export_scale_matrices_npz)
                    else:
                        if self.args.sample_each_eval:
                            # Try model inference (e.g. exploring inference from overfitting)
                            if self.args.max_sample_tokens:
                                self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)
                        if self.args.export_wte_each_eval:
                            # export wte table to npy file
                            if self.args.export_wte_npy:
                                self.raw_model.export_wte(self.args.export_wte_npy)
                        if self.args.export_scale_matrices_each_eval:
                            # export scale matrices to npz file
                            if self.args.export_scale_matrices_npz:
                                self.raw_model.export_scale_matrices(self.args.export_scale_matrices_npz)

                    if self.args.patience is not None and num_steps_with_worse_loss >= self.args.patience:
                        print(f"Early Stopping: loss has not decreased in {self.args.patience + 1} steps")
                        break
                    if losses['val'] > self.best_val_loss:
                        num_steps_with_worse_loss += 1

                if self.args.eval_only:
                    break

                for micro_step in range(self.args.gradient_accumulation_steps):
                    if self.ddp:
                        self.model.require_backward_grad_sync = (micro_step == self.args.gradient_accumulation_steps - 1)

                    with self.ctx:
                        logits, loss = self.model(self.X, self.Y, iter_num=self.iter_num)

                        if self.args.focus_on_top1_loss:
                            loss = self.custom_loss_with_top1_focus(logits, self.Y)  # Use custom loss

                        loss = loss / self.args.gradient_accumulation_steps

                    self.X, self.Y = self.get_batch('train')

                    self.scaler.scale(loss).backward()

                    if gns_type is not None:
                        approx_gns_results = gather_hook_results(self.model)
                        self.gns_ema.update(*gns_utils.gnsify(approx_gns_results, self.args.batch_size, ddp=self.ddp))


                if self.args.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if self.iter_num % self.args.log_interval == 0 and self.master_process:
                    lossf = loss.item() * self.args.gradient_accumulation_steps
                    if local_iter_num >= 5:
                        mfu = self.raw_model.estimate_mfu(self.args.batch_size * self.args.gradient_accumulation_steps, dt)
                        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                    if gns_type is not None:
                        self.gns = self.gns_ema.get_gns()
                        print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f} ms, mfu {running_mfu*100:.2f}%, gns {self.gns:.2f}")
                    else:
                        print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f} ms, mfu {running_mfu*100:.2f}%")

                    if math.isnan(lossf):
                        # If training loss is nan, then exit.
                        with open(self.args.out_dir + "/nan_iter_num.txt", 'w') as file:
                            file.write(str(self.iter_num))
                            sys.exit("Exiting training loss is NaN")

                    vram_allocated = get_gpu_memory_info(info_type='used') if self.args.device != "cpu" else 0
                    self.log_metrics_non_validation(lossf, running_mfu, vram_allocated, self.iter_num)

                if self.args.create_statistics and local_iter_num % self.args.softmax_io_log_interval == 0:
                    create_statistics(self, graph_y_labels)

                self.iter_num += 1
                local_iter_num += 1

                # Update progress bar
                progress.update(task_id, advance=1)

                # End of training actions
                if self.iter_num > self.args.max_iters:
                    if self.args.only_save_checkpoint_at_end:

                        self.save_checkpoint('ckpt.pt')
                        print(f"Saved checkpoint to {self.args.out_dir}")

                        # Sample if set
                        if self.args.max_sample_tokens:
                            self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)
                    break

            if self.args.plot_statistics:
                plot_statistics(self.args, self.stats, graph_y_labels)

            if self.args.tensorboard_log:
                self.writer.flush()
                self.writer.close()

            if self.args.wandb_log and self.master_process:
                import wandb
                wandb.log({"finished": True})
                wandb.finish()

def main():
    args, model_group, training_group, logging_group = parse_args()
    trainer = Trainer(args, model_group, training_group, logging_group)

    if not args.sample_only:
        trainer.train()

    if trainer.ddp:
        destroy_process_group()

    if args.tensorboard_log:
        trainer.writer.flush()
        trainer.writer.close()

if __name__ == '__main__':
    main()

