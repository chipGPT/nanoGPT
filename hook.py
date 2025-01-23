#    Copyright 2023 Cerebras Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Hooks to implement approximate gradient noise scale without touching the actual
model code.
"""

import math

import torch

from dataclasses import dataclass

def is_any_nan_or_inf(tensor):
    """
    Check if any element of a tensor is NaN.
    """
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def add_sogns_hooks(module):
    """
    Add forward and backward hooks necessary for computing scaled output
    gradient noise scale.
    """
    assert isinstance(module, torch.nn.Linear)
    @torch.no_grad()
    def forward_pre_hook(module, activations):
        """
        Forward pre-hook to store statistics about the activations.
        """
        a = activations[0]
        if a.ndim == 2:
            _, i = a.shape
            l = 1
            a = a.unsqueeze(1)
        elif a.ndim == 3:
            _, l, i = a.shape
        else:
            raise ValueError(f'Unsupported activation shape: {a.shape}')
        z = 1./(l * i)
        a = z * a.float() # this is safer
        module.a_sigma = torch.einsum('bli,bli->b', a, a).sqrt().unsqueeze(1)
        # assert not is_any_nan_or_inf(module.a_sigma)
        module.activation_dim = i

    class TensorHook:
        def __init__(self, module):
            self.module = module
        @torch.no_grad()
        def __call__(self, grad):
            """
            Backward hook to compute the gradient noise scale.
            """
            grad = grad.float()
            # assert not is_any_nan_or_inf(grad)
            if grad.ndim == 2:
                grad = grad.unsqueeze(1)
            # comput squared per-example batch gradient contribution
            bias_s = 0.
            bias_g_sqnorm = 0.
            if self.module.bias is not None:
                bias_s += (grad**2).sum(1).mean() # scalar
                bias_g_sqnorm += (grad.sum(0)**2).sum() # scalar
            i = self.module.activation_dim
            w_tilde = math.sqrt(i) * self.module.a_sigma * grad.sum(1)
            # assert not is_any_nan_or_inf(w_tilde)
            self.module.peg_sqnorm = (torch.sum(w_tilde**2, 1).mean() + bias_s)
            self.module.g_sqnorm = (torch.sum(w_tilde.sum(0)**2) + bias_g_sqnorm)
            # delete a_sigma to make sure our garbage is collected
            del self.module.a_sigma

    def forward_post_hook(module, activations, output):
        """
        Forward post-hook to store the output tensor.
        """
        # add tensor hook to the output tensor if it requires grad
        if output.requires_grad:
            output.register_hook(TensorHook(module))

    # add hooks to this module
    module.register_forward_pre_hook(forward_pre_hook)
    module.register_forward_hook(forward_post_hook)


def add_exact_hooks(module):
    """
    Add forward and backward hooks necessary for computing regular gradient
    noise scale. (Much more expensive).
    """
    assert isinstance(module, torch.nn.Linear)
    @torch.no_grad()
    def forward_pre_hook(module, activations):
        """
        Forward pre-hook to store a reference to the input tensor :(
        """
        module.input_activations = activations[0]

    class TensorHook:
        def __init__(self, module):
            self.module = module
        @torch.no_grad()
        def __call__(self, grad):
            """
            Backward hook to compute the gradient noise scale.
            """
            a = self.module.input_activations.float()
            if a.ndim == 2:
                a = a.unsqueeze(1)
            if grad.ndim == 2:
                g = grad.unsqueeze(1)
            else:
                g = grad
            g = g.float()
            # comput squared per-example batch gradient contribution
            bias_s = 0.
            bias_g_sqnorm = 0.
            if self.module.bias is not None:
                bias_s += (g**2).sum(1).mean() # scalar
                bias_g_sqnorm += (g.sum(0)**2).sum() # scalar
            s = torch.einsum('bmk,bnk,bml,bnl->b', a, a, g, g).mean()
            module.peg_sqnorm =  (s + bias_s)
            g_big = torch.einsum('bmk,bml->kl', a, g)
            self.module.g_sqnorm = (torch.sum(g_big**2)
                                    + bias_g_sqnorm)
            # delete a_sigma to make sure our garbage is collected
            del self.module.input_activations

    def forward_post_hook(module, activations, output):
        """
        Forward post-hook to store the output tensor.
        """
        # add tensor hook to the output tensor if it requires grad
        if output.requires_grad:
            output.register_hook(TensorHook(module))

    # add hooks to this module
    module.register_forward_pre_hook(forward_pre_hook)
    module.register_forward_hook(forward_post_hook)


def add_hooks_to_model(model, add_hooks):
    """
    Add hooks to all modules in the model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            add_hooks(module)


@dataclass
class HookResult:
    """
    Hook result dataclass.
    """
    peg_sqnorm: float
    g_sqnorm: float


def gather_hook_results(model):
    """
    Gather the results from the hooks.
    """
    results = {}
    for name, module in model.named_modules():
        if hasattr(module, 'peg_sqnorm'):
            results[name] = HookResult(module.peg_sqnorm, module.g_sqnorm)
    return results
