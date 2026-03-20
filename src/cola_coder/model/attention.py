"""Grouped Query Attention (GQA) with RoPE.

This is the most complex module in the model. It's where tokens "talk" to
each other to gather context.

Standard Multi-Head Attention (MHA): Every head has its own Q, K, V.
Grouped Query Attention (GQA): Multiple query heads SHARE the same K, V.

Why GQA? During inference, we cache all the K and V tensors (the "KV-cache")
so we don't recompute them for previous tokens. With GQA, the cache is
(n_kv_heads / n_heads) times smaller. On consumer GPUs with limited VRAM,
this directly translates to longer sequences during inference.

Example from our configs:
  n_heads=12 query heads, n_kv_heads=4 → each KV head serves 3 query heads
  KV-cache is 3x smaller than full MHA

This is what LLaMA 2 70B, Mistral, and most modern models use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE and KV-cache support."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim: Model dimension.
            n_heads: Number of query heads (attention "perspectives").
            n_kv_heads: Number of key/value heads. Must divide n_heads evenly.
            max_seq_len: Maximum sequence length (for KV-cache allocation).
            dropout: Attention dropout rate.
        """
        super().__init__()
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # How many Q heads share each KV head
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim) for scaled dot-product

        # Projection matrices (no bias — modern transformers skip bias for efficiency)
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        # KV-cache: pre-allocated tensors for inference efficiency
        # These get filled in token-by-token during generation
        # Set to None initially — initialized on first forward pass
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    def _init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Allocate KV-cache tensors. Called once at the start of generation."""
        self.cache_k = torch.zeros(
            batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim,
            device=device, dtype=dtype,
        )
        self.cache_v = torch.zeros(
            batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim,
            device=device, dtype=dtype,
        )

    def clear_cache(self):
        """Reset the KV-cache (call between generation requests)."""
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, dim)
            rope_freqs: Precomputed RoPE frequencies.
            start_pos: Position offset for KV-cache (inference only).
            use_cache: Whether to use/update the KV-cache (inference only).
            mask: Causal attention mask. Shape (seq_len, seq_len) or None.
                  None = use full causal mask. Provided mask should have
                  -inf for positions that should NOT be attended to.

        Returns:
            Output tensor, shape (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape

        # Project input to Q, K, V
        # Each projection: (batch, seq_len, dim) → (batch, seq_len, n_heads * head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to separate heads: (batch, seq_len, n_heads, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K (position encoding by rotation)
        q, k = apply_rope(q, k, rope_freqs, start_pos)

        # KV-cache handling for inference
        if use_cache:
            if self.cache_k is None:
                self._init_cache(batch, x.device, x.dtype)
            # Store current K, V into the cache at the right position
            self.cache_k[:batch, start_pos : start_pos + seq_len] = k
            self.cache_v[:batch, start_pos : start_pos + seq_len] = v
            # Read back ALL cached K, V (from position 0 to current)
            k = self.cache_k[:batch, : start_pos + seq_len]
            v = self.cache_v[:batch, : start_pos + seq_len]

        # Transpose to (batch, n_heads, seq_len, head_dim) for SDPA
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch, n_kv_heads, kv_len, head_dim)
        v = v.transpose(1, 2)  # (batch, n_kv_heads, kv_len, head_dim)

        # GQA: expand KV heads to match Q heads.
        # PyTorch SDPA requires Q and KV to have the same number of heads,
        # but we use expand (not repeat) so it's a zero-copy view —
        # Flash Attention still gets the efficient memory layout.
        if self.n_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            k = k.reshape(batch, self.n_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            v = v.reshape(batch, self.n_heads, -1, self.head_dim)

        # Use PyTorch's fused scaled_dot_product_attention (SDPA).
        # Auto-dispatches to Flash Attention 2 > memory-efficient > math fallback.
        # is_causal=True is faster than constructing + applying an explicit mask.
        drop_p = self.dropout_p if self.training else 0.0
        if use_cache:
            # Inference with KV-cache: q_len != kv_len, need explicit mask
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=drop_p, scale=self.scale,
            )
        else:
            # Training: is_causal=True — no mask tensor needed
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop_p, is_causal=True, scale=self.scale,
            )

        # Reshape back: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, dim)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)

        # Final projection back to model dimension
        return self.out_proj(output)
