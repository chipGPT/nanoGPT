from collections.abc import Sequence
from math import ceil
import re
import torch
import torch.nn as nn

from .kernels import TorchLinear
from .measures import AbsMax, AbsMedian, AbsMean

def round_clamp(input, range):
    return (input.round().clamp(range[0], range[1]) - input).detach() + input

def scale(input, range, measure, keepdim, eps):
    return max(abs(k) for k in range) / measure(input.detach(), keepdim=keepdim).clamp_(min=eps)

def range_from_bits(bits):
    return (ceil(-2**(bits-1)), ceil(2**(bits-1)-1))

class BitLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            config=None,
            bias=False,
            device=None,
            dtype=None,
            eps=1e-5,
            weight_range=1.58,
            weight_measure="AbsMedian",
            activation_range=8,
            activation_measure="AbsMax",
            kernel="TorchLinear",
        ):
        super(BitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self.weight_range = weight_range if isinstance(weight_range, Sequence) else range_from_bits(weight_range)
        self.weight_measure = eval(weight_measure)() if isinstance(weight_measure, str) else weight_measure
        self.activation_range = activation_range if isinstance(activation_range, Sequence) else range_from_bits(activation_range)
        self.activation_measure = eval(activation_measure)() if isinstance(activation_measure, str) else activation_measure
        self.kernel = eval(kernel)() if isinstance(kernel, str) else kernel

    def __repr__(self):
        return f"BitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, weight_range={self.weight_range}, weight_measure={self.weight_measure}, activation_range={self.activation_range}, activation_measure={self.activation_measure}, kernel={self.kernel})"

    def forward(self, x):
        if self.activation_measure is None:
            x_scale, x_quant = 1, x
        else:
            x_norm = torch.layer_norm(x, x.size()[1:])
            x_scale = scale(x_norm, self.activation_range, self.activation_measure, True, self.eps)
            x_quant = round_clamp(x_norm * x_scale, self.activation_range)
        if self.weight_measure is None:
            w_scale, w_quant = 1, self.weight
        else:
            w_scale = scale(self.weight, self.weight_range, self.weight_measure, False, self.eps)
            w_quant = round_clamp(self.weight * w_scale, self.weight_range)
        y_quant = self.kernel(x_quant, w_quant, self.bias)
        y = y_quant / (w_scale * x_scale)
        return y

def replace_modules(model, old_class=nn.Linear, new_class=BitLinear, new_class_kwargs={}, match_name="", prefix=""):
    for name, module in model.named_children():
        qual_name = prefix + "." + name
        if isinstance(module, old_class) and re.search(match_name, qual_name) is not None:
            kwargs = dict(new_class_kwargs)
            kwargs["in_features"] = module.in_features
            kwargs["out_features"] = module.out_features
            bias = getattr(module, "bias", None) is not None
            kwargs["bias"] = bias
            new_module = new_class(**kwargs)
            new_module.weight.data = module.weight.data
            if bias:
                new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            replace_modules(module, old_class, new_class, new_class_kwargs, match_name, prefix=qual_name)

def bitlinearize(model, old_class=nn.Linear, new_class=BitLinear, replacements=[{}]):
    for replacement in replacements:
        replacement = dict(replacement)
        match_name = replacement.pop("match_name", "")
        replace_modules(
            model=model,
            old_class=old_class,
            new_class=new_class,
            new_class_kwargs=replacement,
            match_name=match_name,
        )
    return model
