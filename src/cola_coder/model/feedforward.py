"""SwiGLU Feed-Forward Network.

After attention gathers context from other tokens, the feed-forward network
processes each token independently. Think of attention as "gathering info
from your team" and FFN as "thinking about what you gathered."

SwiGLU is a modern activation function that uses a "gating" mechanism:
one pathway decides WHAT information to pass through (the gate),
and another pathway provides the actual information (the value).
The gate controls how much of the value gets through.

This consistently outperforms simpler activations like ReLU or GELU
in every ablation study. Used by LLaMA, Mistral, Gemma, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Standard FFN has 2 linear layers: up -> activation -> down
    SwiGLU FFN has 3 linear layers: gate + up -> activation * gate -> down

    The "Swi" part = SiLU activation (Sigmoid Linear Unit)
    The "GLU" part = Gated Linear Unit (multiply gate * value)

    For a TS dev: think of it like this:
        // Standard FFN
        const output = down(relu(up(input)))

        // SwiGLU FFN
        const gate = silu(gateProj(input))  // "what to let through"
        const value = upProj(input)          // "the actual information"
        const output = downProj(gate * value) // gate controls the flow
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """
        Args:
            dim: Model dimension (input and output size).
            hidden_dim: Internal expansion size (larger = more capacity).
                        Computed in ModelConfig.ffn_hidden_dim.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        # Three projections instead of two:
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # Learns what to let through
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)  # Learns the actual values
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # Projects back to model dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, seq_len, dim)

        Returns:
            Shape (batch_size, seq_len, dim) — same as input
        """
        # SiLU(gate) * value, then project down
        # F.silu is the "Sigmoid Linear Unit": x * sigmoid(x)
        # It's a smooth, non-monotonic activation that works better than ReLU
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.dropout(self.down_proj(gate * value))
