"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Variations
from variations.softmax_variations import Softermax, Constantmax, Constantmax_quan, Strongermax, Polymax, SigSoftmax
from variations.normalization_variations import LayerNorm, RMSNorm
from variations.position_encoding_variations import RotaryEmbedding, ShortRope
from variations.activation_variations import SquaredReLU



class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_query_sets = config.n_query_sets
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_attn = nn.Linear(config.n_embd, (self.n_query_sets + 2) * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_query_sets * config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Rotary Positional Embeddings
        self.rotary_emb = None
        if config.use_rotary_embeddings:
            if config.rope_variant == "rope":
                self.rotary_emb = RotaryEmbedding(config)
            if config.rope_variant == "shortrope":
                self.rotary_emb = ShortRope(config)

        # Softmax Variant Selection
        self.softmax_variant_attn = config.softmax_variant_attn
        if self.softmax_variant_attn == "softmax":
            # Enable flash attention, which is compatible with 'softmax'
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        else:
            # Remove flash attention (only compatible with 'softmax')
            self.flash = False

            if self.softmax_variant_attn == "softermax":
              self.softmax_layer = Softermax(config)

            if self.softmax_variant_attn == "constantmax":
              self.softmax_layer = Constantmax(config)

            if self.softmax_variant_attn == "constantmax_quan":
                self.softmax_layer = Constantmax_quan(config)

            if self.softmax_variant_attn == "strongermax":
              self.softmax_layer = Strongermax(config)

            if self.softmax_variant_attn == "polymax":
              self.softmax_layer = Polymax(config)

            if self.softmax_variant_attn == "sigsoftmax":
              self.softmax_layer = SigSoftmax(config)

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # Assume queries are grouped. The number of groups is defined by self.n_query_sets.
        # Split key, value, and queries (assuming queries are already organized in groups).
        *grouped_queries, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Prepare key and value for all heads, computed once and shared across all query groups.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Prepare the causal mask for the attention mechanism
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        group_outputs = []
        for queries in grouped_queries:
            q = queries.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Process each query group

            # Compute scaled dot-product attention with shared K-V pairs for the group
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(causal_mask == 0, float('-inf'))

            if self.softmax_variant_attn != 'softmax':
                att = self.softmax_layer(att)
            else:
                att = F.softmax(att, dim=-1)

            att = self.attn_dropout(att)

            # Compute the attention output for this group of queries using the shared K-V pairs
            y = att @ v
            group_outputs.append(y.transpose(1, 2).contiguous().view(B, T, C))

        # Combine and process the outputs from all groups
        combined_output = torch.cat(group_outputs, dim=-1)

        # Apply the output projection
        y = self.resid_dropout(self.c_proj(combined_output))

        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # TODO: Change name of self.gelu to something like "self.activation_variant"
        if config.activation_variant == "relu":
          print("Use ReLU")
          self.gelu = nn.ReLU()
        if config.activation_variant == "squared_relu":
          print("Use Squared ReLU")
          self.gelu = SquaredReLU()
        if config.activation_variant == "gelu":
          print("Use GELU")
          self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.layernorm_variant == 'rmsnorm':
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)

        if config.layernorm_variant == 'layernorm':
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.use_post_ln = config.use_post_ln

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.use_post_ln:
          x = self.ln_1(x + self.attn(x))
          x = self.ln_2(x + self.mlp(x))
        else:
          x = x + self.attn(self.ln_1(x))
          x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    n_query_sets: int = 4

    # Softmax Alternatives and Options
    softmax_variant_attn: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "constantmax"
    softmax_variant_output: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "constantmax"

    ## Constantmax Options
    constantmax_initial_beta: float = 0.0 # denominator to utilize for Constantmax
    constantmax_initial_gamma: float = 1.0 # denominator to utilize for Constantmax
    constantmax_use_euler_base: bool = True # use 'e' as base for Constantmax
    constantmax_base: float = 2.0 # denominator to utilize for Constantmax

    ## Softermax options
    softermax_use_xmax: bool = True # Softermax Option active is softermax selected - True: uses (x - x_max) normalization; False: removes normalization (potential overflow)

    ## Polymax options
    polymax_x_intercept: float = -100.0
    polymax_y_intercept: float = 1.0
    polymax_power: float = 2.0
    polymax_divisor: float = 1000.0

    ## SigSoftmaxBase
    sigsoftmax_use_euler_base: bool = True # use 'e' as base for Constantmax
    sigsoftmax_base: float = 2.0 # denominator to utilize for Constantmax

    ## Strongermax options
    strongermax_strength: float = 2.0 # Softermax with option of 'stronger' (larger integer) bases

    # Positional Embeddings Variations
    use_abs_pos_embeddings: bool = False # Note: one can use this AND rotary embeddings
    use_rotary_embeddings: bool = True # If True, uses rotary embeddings, else use conventional absolute position encoding
    rope_variant: str = "rope" # options: "shortrope", "rope"
    shortrope_length: int = 8 # number of embeddings to use in shortrope

    # Structuring Options, remember to compile the model
    use_post_ln: bool = True
    use_pre_ln: bool = False

    # Layernorm Alternatives and Options
    layernorm_variant: str = "rmsnorm" # Current options "rmsnorm" or "layernorm"
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # Activation Alternatives
    activation_variant: str = "gelu" # Current options "gelu", "relu", "squared_relu"

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        if config.layernorm_variant == "layernorm":
            self.normalization_variant = LayerNorm(config.n_embd, bias=config.bias)
        if config.layernorm_variant == "rmsnorm":
            self.normalization_variant = RMSNorm(config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = self.normalization_variant,
        ))

        # Select softmax variant for output layer
        self.softmax_variant_output = config.softmax_variant_output
        if self.softmax_variant_output != "softmax":
            if self.softmax_variant_output == "softermax":
                self.softmax_layer_output = Softermax(config)

            if self.softmax_variant_output == "constantmax":
                self.softmax_layer_output = Constantmax(config)

            if self.softmax_variant_output == "constantmax_quan":
                self.softmax_layer_output = Constantmax_quan(config)

            if self.softmax_variant_output == "strongermax":
              self.softmax_layer_output = Strongermax(config)

            if self.softmax_variant_output == "polymax":
              self.softmax_layer_output = Polymax(config)

            if self.softmax_variant_output == "sigsoftmax":
              self.softmax_layer_output = SigSoftmax(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = None
        if self.config.use_abs_pos_embeddings:
          pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
          x = self.transformer.drop(tok_emb + pos_emb)
        else:
          x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = None
            if self.config.softmax_variant_output != 'softmax':
                probs = self.softmax_layer_output(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            assert probs != None
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
