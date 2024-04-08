import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, ndim):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / rms * self.gain


class PowerNorm(nn.Module):
    """Power Normalization"""

    def __init__(self, ndim, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.gain = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.pow(var + 1e-5, self.alpha)
        return normalized * self.gain


class GroupNorm(nn.GroupNorm):
    """Group Normalization with gain parameter"""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__(num_groups, num_channels, eps)
        self.gain = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        normalized = super().forward(x)
        return normalized * self.gain


class InstanceNorm(nn.InstanceNorm2d):
    """Instance Normalization with gain parameter"""

    def __init__(self, num_features, eps=1e-5):
        super().__init__(num_features, eps)
        self.gain = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        normalized = super().forward(x)
        return normalized * self.gain


class BatchNorm(nn.BatchNorm1d):
    """BatchNorm1d with gain parameter"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(num_features, eps, momentum)
        self.gain = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        # Reshape input for BatchNorm1d
        x = x.transpose(1, 2)
        normalized = super().forward(x)
        # Reshape back to original shape
        x = normalized.transpose(1, 2)
        return x * self.gain


class PowerNorm2(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.gain = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        self.p = nn.Parameter(torch.ones(1))
        self.q = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # input shape: (B, T, C)
        # reshape input for BatchNorm1d
        x = x.transpose(1, 2)  # shape: (B, C, T)

        if self.training:
            mean = torch.mean(x, dim=[0, 2])
            var = torch.var(x, dim=[0, 2], unbiased=False)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)

        # apply power normalization
        x = torch.sign(x) * torch.pow(torch.abs(x), self.p / self.q)

        # apply gain and bias
        if self.affine:
            x = x * self.gain.view(1, -1, 1) + self.bias.view(1, -1, 1)

        # reshape back to original shape
        x = x.transpose(1, 2)  # shape: (B, T, C)

        return x


class PowerNorm3(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gain = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.p = nn.Parameter(torch.ones(1))
        self.q = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # input shape: (B, T, C)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)

        # apply power normalization
        x = torch.sign(x) * torch.pow(torch.abs(x), self.p / self.q)

        # apply gain and bias
        if self.affine:
            x = x * self.gain.view(1, 1, -1) + self.bias.view(1, 1, -1)

        return x
class PowerNorm5(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha=0.9):
        super().__init__()
        self.num_features = num_features  # scalar, number of features/channels
        self.eps = eps  # scalar, small constant for numerical stability
        self.alpha = alpha  # scalar, exponential moving average factor
        self.gain = nn.Parameter(torch.ones(num_features))  # (C,), learnable gain parameter
        self.running_quad_mean = nn.Parameter(torch.ones(num_features), requires_grad=False)  # (C,), running quadratic mean

    def forward(self, x):
        # input x: (B, T, C), where B: batch size, T: sequence length, C: number of features/channels
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D input"
        B, T, C = x.shape  # scalars, get dimensions of input tensor

        # compute quadratic mean
        quad_mean = torch.mean(x ** 2, dim=1)  # (B, C), compute quadratic mean along sequence dimension

        # normalize using quadratic mean
        x_normalized = x / (quad_mean.view(B, 1, C) + self.eps).sqrt()  # (B, T, C), normalize input by quadratic mean

        # update running quadratic mean
        self.running_quad_mean.data = self.alpha * self.running_quad_mean.data + (1 - self.alpha) * quad_mean.mean(dim=0)  # (C,), update running quadratic mean using exponential moving average

        # apply gain
        output = x_normalized * self.gain  # (B, T, C), apply learnable gain parameter

        return output  # (B, T, C), return normalized output

    def backward(self, grad_output):
        # input grad_output: (B, T, C), gradient of loss with respect to output
        assert grad_output.dim() == 3, f"Expected 3D grad_output, got {grad_output.dim()}D grad_output"
        B, T, C = grad_output.shape  # scalars, get dimensions of gradient tensor

        # compute intermediate gradient
        grad_x_hat = grad_output * self.gain.view(1, 1, C)  # (B, T, C), multiply gradient by gain parameter

        # compute approximate backpropagation
        x_hat = grad_output / (self.running_quad_mean.view(1, 1, C) + self.eps).sqrt()  # (B, T, C), normalize gradient by running quadratic mean
        quad_mean_grad = torch.sum(grad_x_hat * x_hat, dim=1, keepdim=True)  # (B, 1, C), compute gradient of quadratic mean
        grad_input = grad_x_hat - quad_mean_grad * x_hat / (2 * T * (self.running_quad_mean.view(1, 1, C) + self.eps).sqrt())  # (B, T, C), compute gradient with respect to input using approximate backpropagation

        return grad_input  # (B, T, C), return gradient with respect to input
