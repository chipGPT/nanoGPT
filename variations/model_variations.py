import torch
import torch.nn as nn
import math
import torch.nn.functional as F


model_variation_dictionary = {

    'gpt2': {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'vocab_size': 50257,
        'block_size': 1024,
        'bias': True,
        'norm_variant_attn': 'layernorm',
        'norm_variant_output': 'layernorm',
    },
    'gpt2-medium': {
        'n_layer': 24,
        'n_head': 16,
        'n_embd': 1024,
        'vocab_size': 50257,
        'block_size': 1024,
        'bias': True,
        'norm_variant_attn': 'layernorm',
        'norm_variant_output': 'layernorm',
    },
    'gpt2-large': {
        'n_layer': 36,
        'n_head': 20,
        'n_embd': 1280,
        'vocab_size': 50257,
        'block_size': 1024,
        'bias': True,
        'norm_variant_attn': 'layernorm',
        'norm_variant_output': 'layernorm',
    },
    'gpt2-xl': {
        'n_layer': 48,
        'n_head': 25,
        'n_embd': 1600,
        'vocab_size': 50257,
        'block_size': 1024,
        'bias': True,
        'norm_variant_attn': 'layernorm',
        'norm_variant_output': 'layernorm',
    },
    'qwen2_7b': ({
        'n_layer': 28,
        'n_head': 28,
        'n_kv_group': 4,
        'n_embd': 3584,
        "vocab_size": 152064,
        'block_size': 512,
        'bias': False,
        "qkv_bias": True,
        'norm_variant_attn': 'rmsnorm',
        'norm_variant_output': 'rmsnorm',
        'activation_variant': 'silu',
        'mlp_variant': 'swiglu',
        'dropout': 0.0,
        'mlp_expansion_factor': 5.2857,
        'use_abs_pos_embeddings': False,
        'use_rotary_embeddings': True,
    }, "Qwen/Qwen2-7B"),
    'qwen2_1p5b': ({
        'n_layer': 28,
        'n_head': 12,
        'n_kv_group': 2,
        'n_embd': 1536,
        "vocab_size": 151936,
        'block_size': 256,
        'bias': False,
        "qkv_bias": True,
        'norm_variant_attn': 'rmsnorm',
        'norm_variant_output': 'rmsnorm',
        'activation_variant': 'silu',
        'mlp_variant': 'swiglu',
        'dropout': 0.0,
        'mlp_expansion_factor': 5.8333,
        'use_abs_pos_embeddings': False,
        'use_rotary_embeddings': True,
    }, "Qwen/Qwen2-1.5B"),
    'qwen2_0p5b': ({
        'n_layer': 24,
        'n_head': 14,
        'n_kv_group': 2,
        'n_embd': 896,
        "vocab_size": 151936,
        'block_size': 128,
        'bias': False,
        "qkv_bias": True,
        'norm_variant_attn': 'rmsnorm',
        'norm_variant_output': 'rmsnorm',
        'activation_variant': 'silu',
        'mlp_variant': 'swiglu',
        'dropout': 0.0,
        'mlp_expansion_factor': 5.4285,
        'use_abs_pos_embeddings': False,
        'use_rotary_embeddings': True,
    }, "Qwen/Qwen2-0.5B")
}

"""

# Just moving this away from model.py but keeping around just in case

 # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        config_args['window_size'] = 128 # always None for GPT model checkpoints
"""