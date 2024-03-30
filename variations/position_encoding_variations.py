import torch
import torch.nn as nn
import numpy as np
import math
from vector2d_rotator_variations import (
    Rotator,
    PerfectRotator,
    CORDIC1959,
    FOE,
    DoubleFOE,
    DoubleFOE_Advanced,
)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd

        # Register frequencies directly as buffers
        self.register_buffer(
            "freq_left",
            (10000 ** (torch.arange(0, self.dim // 2).float() / self.dim // 2)),
        )
        self.register_buffer(
            "freq_right",
            (10000 ** (torch.arange(0, self.dim // 2).float() / self.dim // 2)),
        )

    def forward(self, x):
        seq_len = x.shape[-2]
        device = x.device

        t = torch.arange(seq_len, device=device)

        # Get separate frequencies for left and right
        freqs_left = torch.einsum("i,j->ij", t, self.freq_left)
        freqs_right = torch.einsum("i,j->ij", t, self.freq_right)

        # Apply frequencies
        x_left, x_right = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Combine the left and right parts back
        x = torch.cat([x_left, x_right], dim=-1)

        return x


class ShortRope(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n = config.shortrope_length
        self.dim = config.n_embd

        # Generate freqs of size n rather than full dim
        self.register_buffer(
            "freq_left", (10000 ** (torch.arange(0, self.n // 2).float() / self.n // 2))
        )
        self.register_buffer(
            "freq_right",
            (10000 ** (torch.arange(0, self.n // 2).float() / self.n // 2)),
        )

    def forward(self, x):
        # Step 1: Get the input tensor shape
        batch_size, seq_len, _ = x.shape

        # Step 2: Split the input tensor into unrotated and rotated sections
        x_unrotated = x[..., : -self.n]  # All but the last n dimensions
        x_rotated = x[..., -self.n :]  # Only the last n dimensions

        # Step 3: Generate rotation frequencies
        t = torch.arange(self.n, device=x.device)
        freqs_left = torch.einsum("i,j->ij", t, self.freq_left)
        freqs_right = torch.einsum("i,j->ij", t, self.freq_right)

        # Calculate how many times to repeat freqs along the sequence length
        num_repeats = seq_len // self.n + int(seq_len % self.n != 0)

        # Repeat the frequency tensors to match the sequence length
        freqs_left = freqs_left.repeat(batch_size, num_repeats, 1)
        freqs_right = freqs_right.repeat(batch_size, num_repeats, 1)

        # Trim the excess elements so the freqs tensors match the sequence length
        freqs_left = freqs_left[:, :seq_len, :]
        freqs_right = freqs_right[:, :seq_len, :]

        # Step 4: Process the x_rotated section
        x_left = x_rotated[..., : self.n // 2]
        x_right = x_rotated[..., self.n // 2 :]

        # Apply the cosine and sine rotations
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Invert the order of the right tensor's last dimension and negate it
        x_right = torch.flip(x_right, dims=[-1]) * -1

        # Combine the left and right rotated sections
        x_rotated = torch.cat([x_left, x_right], dim=-1)

        # Step 5: Combine the rotated and unrotated sections
        x = torch.cat([x_unrotated, x_rotated], dim=-1)

        return x



def cordic_ang(theta: float):
    # precompute thetas
    NITER = 5
    pcthetas = list()
    for tanexp in range(NITER):
        pcthetas.append(math.atan(2 ** (-tanexp)))
    # precompute cos correction factor
    coscorrection = float("1")
    for pctheta in pcthetas:
        coscorrection *= math.cos(pctheta)
    # init values and perform n rotations
    current_rotation = float("0")
    for tanexp, pctheta in enumerate(pcthetas):
        tantheta = 2 ** (-tanexp)
        if current_rotation > theta:
            current_rotation -= pctheta
        else:
            current_rotation += pctheta
    return current_rotation

def cordic_ang_all(thetas: torch.tensor):
    return torch.tensor([cordic_ang(theta) for theta in thetas])


class ROPE(torch.nn.Module):
    #ROPE support only even dimensionality vectors
    #embedding length must be at least 2
    def __init__(self, embedding_len: int, base: int = 10000):
        super().__init__()
        # compute rotation angles for each pair
        num_blocks = embedding_len // 2
        thetas = torch.tensor([base ** (-2 * i / embedding_len) for i in range(num_blocks)])
        repeated_thetas = thetas.repeat_interleave(2)
        self.register_buffer("repeated_thetas", repeated_thetas)

    #postion 'm' must be positive
    def forward(self, vec: torch.tensor, m: int) -> torch.tensor:
        """perform ROPE positional embedding

        Args:
            vec (torch.tensor): input vector to rotate
            m (int): position (0th position means the first token)

        Returns:
            torch.tensor: ROPE positional embedding
        """
        # compute rotations in pairs (each pair corresponds to a theta)
        rotations = m*self.repeated_thetas
        rot = "cordic"
        if rot == "perfect":
            sin_ests = torch.sin(rotations)
            cos_ests = torch.cos(rotations)
        elif rot == "foe":
            sin_ests = rotations
            cos_ests = 1 - torch.sign(rotations) * rotations / 4
        elif rot == "cordic":
            sin_ests = 1.57*torch.sin(cordic_ang_all(rotations))
            cos_ests = 1.57*torch.sin(cordic_ang_all(rotations))
        swapped_vec = torch.flip(vec.view(-1,2),dims=(1,))
        neg_swapped_vec = torch.cat((-swapped_vec[:,0],swapped_vec[:,1])).view(2,-1).transpose(0,1).reshape(-1)
        return cos_ests @ vec + sin_ests @ neg_swapped_vec



def ROPE_testsuite():
    embedding_block = ROPE(8)
    example_vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],requires_grad=True)
    for m in range(5):
        # create a random function to see if it is possible to backprop
        result = torch.sum(embedding_block(example_vector, m))
        result.backward()
        print("grad",result)
        result.grad=None
        print("token pos", m, "\trotated vector", embedding_block(example_vector, m))

