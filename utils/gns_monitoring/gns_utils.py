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

__all__ = ['GradNorm', 'mean_loss_scale', 'EMA']

import math
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Union
from collections.abc import Sequence

@dataclass
class GradNorm:
    """
    A GradNorm measurement annotated with loss_scale and batch_size, because
    these are necessary to compute GNS using this GradNorm later.
    """
    val: float
    loss_scale: float
    batch_size: int

    def __repr__(self):
        return f"GradNorm(val={self.val}, loss_scale={self.loss_scale}, batch_size={self.batch_size})"

def mean_loss_scale(microbatch_size, minibatch_size):
    """Compute the appropriate loss scale when using a loss that has been
    reduced by a mean over the batch dimension, for a given minibatch
    and microbatch size."""
    return minibatch_size / microbatch_size

### BEGIN MIT LICENSE ###
# Copyright (c) 2022 Katherine Crowson
#               2023 Gavia Gray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

class EMA:
    """Calculates the gradient noise scale (1 / SNR), or critical batch size,
    from _An Empirical Model of Large-Batch Training_,
    https://arxiv.org/abs/1812.06162).

    Args:
        beta (float): The decay factor for the exponential moving averages used to
            calculate the gradient noise scale.
            Default: 0.9998
        eps (float): Added for numerical stability.
            Default: 1e-8
    """
    def __init__(self, beta=0.9998, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.ema_sq_norm = 0.
        self.ema_var = 0.
        self.beta_cumprod = 1.
        self.gradient_noise_scale = float('nan')

    def state_dict(self):
        """Returns the state of the object as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the object's state.
        Args:
            state_dict (dict): object state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def update(self, norm_small_batch, norm_large_batch):
        """Updates the state with a new batch's gradient statistics, and returns the
        current gradient noise scale.

        Args:
            norm_small_batch (GradNorm): The mean of the 2-norms of microbatch or
                per sample gradients.
            norm_large_batch (GradNorm): The 2-norm of the mean of the microbatch or
                per sample gradients.
        """
        sq_norm_small_batch = (norm_small_batch.val * norm_small_batch.loss_scale)**2
        sq_norm_large_batch = (norm_large_batch.val * norm_large_batch.loss_scale)**2
        m, n = norm_small_batch.batch_size, norm_large_batch.batch_size
        est_sq_norm = (n * sq_norm_large_batch - m * sq_norm_small_batch) / (n - m)
        est_var = (sq_norm_small_batch - sq_norm_large_batch) / (1 / m - 1 / n)
        self.ema_sq_norm = self.beta * self.ema_sq_norm + (1 - self.beta) * est_sq_norm
        self.ema_var = self.beta * self.ema_var + (1 - self.beta) * est_var
        self.beta_cumprod *= self.beta
        self.gradient_noise_scale = max(self.ema_var, self.eps) / max(self.ema_sq_norm, self.eps)
        return self.gradient_noise_scale

    def get_gns(self):
        """Returns the current gradient noise scale."""
        return self.gradient_noise_scale

    def get_stats(self):
        """Returns the current (debiased) estimates of the squared mean gradient
        and gradient variance."""
        return self.ema_sq_norm / (1 - self.beta_cumprod), self.ema_var / (1 - self.beta_cumprod)
### END MIT LICENSE ###

def gnsify(sogns_results, minibatch_size, ddp=False):
    # dictionary of approximate per-example gradient norms
    # convert to gns format
    # accumulate small and large squared gradient norms
    total_small = 0.
    total_big = 0.
    for _, v in sogns_results.items():
        total_small += v.peg_sqnorm
        total_big += v.g_sqnorm
    if ddp:
        # all_reduce AVG
        torch.distributed.all_reduce(total_small, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(total_big, op=torch.distributed.ReduceOp.AVG)
    small = GradNorm(
        math.sqrt(total_small),
        mean_loss_scale(1, minibatch_size),
        1
    )
    big = GradNorm(
        math.sqrt(total_big),
        1.,
        minibatch_size
    )
    return small, big

