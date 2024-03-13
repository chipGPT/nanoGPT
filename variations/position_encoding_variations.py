import torch
import torch.nn as nn
import numpy as np
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


class ROPE:
    # this will use cordic rather than rotation matrix
    def __perform_2Drotation(self, theta: float, vec2d: list) -> np.array:
        """rotate 2 dimensional vector by angle theta

        Args:
            theta (float): the angle which this matrix will rotate the vector
            vec2d (list): the vector of length 2 that will be rotated

        Returns:
            np.array: vec2d rotated by angle theta
        """
        return self.rotator(theta, vec2d)

    def __init__(
        self,
        embedding_len: int,
        base: int = 10000,
        rotator: Rotator = PerfectRotator(),
    ):
        """setup ROPE calculation

        Args:
            embedding_len (int): length of input embedding vectors
            base (int): parameter to use as base for theta exponent
            rotator (Rotator): instance of Rotator. Default is PerfectRotator.

        Raises:
            NotImplementedError: if embedding_len is odd (only even supported)
            ValueError: if embedding_len < 2 (at least 2 required)
        """
        if not isinstance(rotator, Rotator):
            raise TypeError("rotator must be a child of Rotator")
        self.rotator = rotator
        # error checking
        embedding_len = int(embedding_len)
        if embedding_len % 2:
            raise NotImplementedError("ROPE support only even dimensionality vectors")
        if embedding_len < 2:
            raise ValueError("embedding length must be at least 2")
        # compute rotation angles for each pair
        self.input_len = embedding_len
        num_blocks = embedding_len // 2
        self.thetas = [base ** (-2 * i / embedding_len) for i in range(num_blocks)]
        print(self.thetas)

    def __call__(self, vec: np.array, m: int) -> np.array:
        """perform ROPE positional embedding

        Args:
            vec (np.array): input vector to rotate
            m (int): position (0th position means the first token)

        Raises:
            ValueError: if m is negative
            ValueError: length of input vector does not match expected length

        Returns:
            np.array: vector with ROPE positional embedding
        """
        vec = np.array(vec)  # ensure input is np.array
        if m < 0:
            raise ValueError("postion 'm' must be positive")
        if len(vec) != self.input_len:
            raise ValueError(f"input vector must be length {self.input_len}")
        # compute rotations in pairs (each pair corresponds to a theta)
        rotatedvec = np.zeros(vec.shape)
        # import pdb; pdb.set_trace()
        for i, theta in enumerate(self.thetas):
            rotatedvec[2 * i : 2 * i + 2] = self.__perform_2Drotation(
                theta * m, vec[2 * i : 2 * i + 2]
            )
        return rotatedvec


def ROPE_testsuite():
    embedding_block = ROPE(8)
    example_vector = [1, 2, 3, 4, 5, 6, 7, 8]
    for m in range(5):
        print("token pos", m, "\trotated vector", embedding_block(example_vector, m))
