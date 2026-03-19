"""RMSNorm: Root Mean Square Layer Normalization.

Think of normalization like auto-leveling audio — it prevents the signal
(tensor values) from getting too loud (huge) or too quiet (tiny) as it
passes through the network. Without it, training becomes numerically unstable.

RMSNorm is simpler than the original LayerNorm because it skips the
"centering" step (subtracting the mean). It just scales by the RMS.
This is what LLaMA, Mistral, and every modern transformer uses.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    For a TS dev: this is like a class that extends nn.Module (the base
    class for all neural network layers in PyTorch, similar to React.Component).

    What it does:
    1. Compute the root-mean-square of the input values
    2. Divide the input by its RMS (now the RMS is ~1.0)
    3. Multiply by a learnable weight vector (so the model can scale things back up
       for specific dimensions if it wants to)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: The size of the last dimension of the input tensor.
                 Must match model.dim (512, 768, or 1024 in our configs).
            eps: Small constant to prevent division by zero.
                 1e-6 = 0.000001 — tiny safety margin.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter — initialized to all 1s
        # Shape: (dim,) — one weight per dimension
        # nn.Parameter tells PyTorch "this should be updated during training"
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of any shape, but the last dimension must be `dim`.
               Typically shape: (batch_size, seq_len, dim)

        Returns:
            Normalized tensor of the same shape.
        """
        # x.float() converts to float32 for numerical stability during norm computation
        # .pow(2) squares each element
        # .mean(-1, keepdim=True) averages across the last dimension
        # So rms_inv = 1 / sqrt(mean(x^2) + eps)
        rms_inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

        # Normalize and scale, then convert back to the original dtype (bf16/fp16)
        return (x.float() * rms_inv).type_as(x) * self.weight
