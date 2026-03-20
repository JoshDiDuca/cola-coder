"""The Full Transformer Model.

This is where everything comes together. The transformer is a stack of
identical blocks, each containing attention + feed-forward, wrapped with
normalization and residual connections.

Data flow for one forward pass:
    token_ids → Embedding → [Block1 → Block2 → ... → BlockN] → RMSNorm → Linear → logits

Each block:
    input → RMSNorm → Attention → + input → RMSNorm → FFN → + (prev result) → output
                                   ↑ residual              ↑ residual

The residual connections (the "+ input" arrows) are what make deep transformers
trainable. Without them, gradients would vanish in early layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .feedforward import SwiGLUFFN
from .normalization import RMSNorm
from .rope import precompute_rope_freqs


class TransformerBlock(nn.Module):
    """One transformer block: attention + feed-forward with norms and residuals.

    For a TS dev: think of this like a component that gets composed N times.
    Each block refines the representation — early blocks learn simple patterns
    (syntax, common tokens), later blocks learn complex patterns (logic, semantics).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Pre-normalization (applied BEFORE each sub-layer)
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

        # The two sub-layers
        self.attention = GroupedQueryAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ffn = SwiGLUFFN(
            dim=config.dim,
            hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention with residual connection
        # "x + attention(norm(x))" — input flows through unchanged,
        # attention just ADDS refinements
        h = x + self.attention(
            self.attn_norm(x),
            rope_freqs=rope_freqs,
            start_pos=start_pos,
            use_cache=use_cache,
            mask=mask,
        )

        # FFN with residual connection
        out = h + self.ffn(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """The complete transformer language model.

    This is the main model class. It takes token IDs in and produces
    logits (raw scores for each token in the vocabulary) out.

    During training: feed a sequence, get logits, compare with the actual
    next tokens using cross-entropy loss.

    During inference: feed tokens one at a time (with KV-cache), sample
    from the output distribution to generate new tokens.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding: maps token_id (int) → vector (float[dim])
        # Think of it as a lookup table with vocab_size rows and dim columns
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)

        # Dropout on embeddings (regularization)
        self.dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        # nn.ModuleList is like an array of layers that PyTorch tracks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization before output
        self.final_norm = RMSNorm(config.dim)

        # Output projection: maps vector (float[dim]) → logits (float[vocab_size])
        # This is the "prediction head" — it scores every token in the vocabulary
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share weights between embedding and output
        # The embedding maps token→vector, the output maps vector→token scores
        # These are inverse operations, so sharing weights makes sense and saves params
        self.output.weight = self.tok_emb.weight

        # Precompute RoPE frequencies (cached, not learned)
        # register_buffer makes it part of the model state but NOT a parameter
        # (it won't be updated by the optimizer)
        # We precompute 2x the max seq len as a safety buffer — this handles
        # cases where data chunks are larger than the model's configured
        # max_seq_len (the model will still only see max_seq_len at a time,
        # but having extra frequencies avoids index-out-of-range crashes).
        rope_freqs = precompute_rope_freqs(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len * 2,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Precompute causal mask
        # This is a matrix of -inf values above the diagonal, 0 on and below
        # It prevents tokens from attending to future tokens
        mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)  # Upper triangular = future positions
        self.register_buffer("causal_mask", mask, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights.

        Proper initialization is crucial — if weights start too big or too small,
        training can fail to converge. We use the same scheme as GPT-2/LLaMA.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Normal distribution with small std dev
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass: token IDs → logits.

        Args:
            token_ids: Input token IDs, shape (batch_size, seq_len).
                       Each value is an integer in [0, vocab_size).
            start_pos: Position offset for KV-cache during inference.
            use_cache: Whether to use KV-cache (True during inference).

        Returns:
            Logits tensor, shape (batch_size, seq_len, vocab_size).
            Each value is a raw score — higher means the model thinks
            that token is more likely to come next.
        """
        _, seq_len = token_ids.shape

        # Step 1: Convert token IDs to vectors
        h = self.tok_emb(token_ids)  # (batch, seq_len, dim)
        h = self.dropout(h)

        # Step 2: Build the attention mask for this sequence
        if use_cache and start_pos > 0:
            # During inference with cache: only processing one new token,
            # which can attend to all previous tokens. No mask needed.
            mask = None
        else:
            # During training or first inference step: use causal mask
            mask = self.causal_mask[:seq_len, :seq_len]

        # Step 3: Pass through all transformer blocks
        for block in self.blocks:
            h = block(
                h,
                rope_freqs=self.rope_freqs,
                start_pos=start_pos,
                use_cache=use_cache,
                mask=mask,
            )

        # Step 4: Final normalization
        h = self.final_norm(h)

        # Step 5: Project to vocabulary logits
        logits = self.output(h)  # (batch, seq_len, vocab_size)

        return logits

    def compute_loss(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling.

        The task: given tokens [0, 1, 2, ..., N-1], predict tokens [1, 2, 3, ..., N].
        Each token tries to predict the NEXT token.

        Args:
            token_ids: Shape (batch_size, seq_len). The training sequence.

        Returns:
            Scalar loss value. Lower = better predictions.
        """
        # Get logits for all positions
        logits = self.forward(token_ids)

        # Shift: logits[:-1] predicts token_ids[1:]
        # "Given everything up to position i, predict position i+1"
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = token_ids[:, 1:].contiguous()

        # Cross-entropy loss
        # Reshaping to (batch * seq_len, vocab_size) and (batch * seq_len,)
        # because F.cross_entropy expects 2D logits and 1D targets
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    def clear_caches(self):
        """Clear all KV-caches (call between generation requests)."""
        for block in self.blocks:
            block.attention.clear_cache()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM.

        Trades compute for memory: instead of storing all intermediate
        activations, recompute them during the backward pass.
        Roughly halves memory usage but is ~30% slower.
        Required for the 350M model on 16GB GPUs.
        """
        self.gradient_checkpointing = True

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
