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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .feedforward import SwiGLUFFN
from .normalization import RMSNorm
from .rope import get_rope_freqs, yarn_attention_scale


def language_modeling_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    z_loss: float = 0.0,
) -> torch.Tensor:
    """Next-token cross-entropy loss over a batch of sequences.

    Free function (not a method) so the trainer can compute logits through a
    torch.compile-d model call and apply the loss outside the compiled graph:
    OptimizedModule only compiles __call__/forward — calling a method like
    model.compute_loss() on a compiled model silently runs the original,
    uncompiled module.

    Args:
        logits: Model output, shape (batch, seq_len, vocab_size).
        token_ids: Input tokens, shape (batch, seq_len). Targets are derived
            by shifting: logits[:, i] predicts token_ids[:, i + 1].
        sample_weights: Optional per-sample quality weights, shape (batch,).
            When provided, each sequence's loss contributes proportionally to
            its weight (weighted mean), so high-quality samples influence the
            gradient more than low-quality ones WITHIN the same batch.
        z_loss: Weight for the auxiliary log(Z)^2 regularizer (PaLM / OLMo 2,
            typically ~1e-4). Cross-entropy only constrains logit
            DIFFERENCES; the absolute scale can drift upward over a long run
            until bf16 precision degrades. Z-loss pulls the softmax
            normalizer toward 1, keeping logits bounded. 0 = disabled.

    Returns:
        Scalar loss value.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids[:, 1:].contiguous()

    if sample_weights is None:
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    else:
        # Per-sample weighting: compute per-token loss, average per sequence,
        # then take the weighted mean across the batch. A plain
        # `mean_loss * weights.mean()` would only rescale the whole batch —
        # it could not differentiate samples within it.
        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)  # (batch, seq_len - 1)
        per_sample = per_token.mean(dim=1)  # (batch,)
        weights = sample_weights.to(per_sample.dtype)
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

    if z_loss > 0.0:
        # log(Z) per position; float32 for the logsumexp to avoid bf16
        # overflow on exactly the runs where drift is the problem
        log_z = torch.logsumexp(shift_logits.float(), dim=-1)
        loss = loss + z_loss * (log_z ** 2).mean()

    return loss


class TransformerBlock(nn.Module):
    """One transformer block: attention + feed-forward with norms and residuals.

    For a TS dev: think of this like a component that gets composed N times.
    Each block refines the representation — early blocks learn simple patterns
    (syntax, common tokens), later blocks learn complex patterns (logic, semantics).
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        is_moe: bool = False,
        attn_logit_scale: float = 1.0,
    ):
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
            qk_norm=getattr(config, "qk_norm", False),
            attn_logit_scale=attn_logit_scale,
        )

        # FFN: dense SwiGLU by default, or a Mixture-of-Experts FFN when this
        # layer is MoE-enabled. MoEFFN is a drop-in replacement (same call
        # signature, same residual usage below). The expert/router submodule
        # names match exactly what scripts/upcycle_to_moe.py writes, so an
        # upcycled checkpoint loads into this module without remapping.
        self.is_moe = is_moe
        if is_moe:
            from ..features.moe_layer import MoEFFN

            moe = config.moe
            self.ffn = MoEFFN(
                dim=config.dim,
                hidden_dim=config.ffn_hidden_dim,
                num_experts=moe.num_experts,
                top_k=moe.top_k,
                dropout=config.dropout,
                capacity_factor=moe.capacity_factor,
                num_shared_experts=moe.num_shared_experts,
                aux_loss_weight=moe.aux_loss_weight,
            )
        else:
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

        # Resolve which blocks (if any) use a Mixture-of-Experts FFN.
        # MoE is off unless config.moe.enabled — the dense path is unchanged.
        moe_cfg = getattr(config, "moe", None)
        if moe_cfg is not None and getattr(moe_cfg, "enabled", False):
            from ..features.moe_layer import resolve_moe_layers

            self._moe_layer_set = resolve_moe_layers(moe_cfg.moe_layers, config.n_layers)
        else:
            self._moe_layer_set = set()
        self.is_moe = bool(self._moe_layer_set)

        # Resolve RoPE scaling ONCE here — it drives both the precomputed freq
        # table (below) and the YaRN attention temperature (MODEL-005). With
        # type "none" / factor <= 1 this is inert, so dense non-extended models
        # are unaffected.
        scaling = getattr(config, "rope_scaling", None)
        scaling_type = getattr(scaling, "type", "none") if scaling is not None else "none"
        scaling_factor = getattr(scaling, "factor", 1.0) if scaling is not None else 1.0
        if scaling_type == "none" or scaling_factor <= 1.0:
            scaling_type, scaling_factor = "none", 1.0

        # YaRN lowers the softmax temperature at extended context: logits scale
        # by mscale**2 (mscale = 0.1*ln(factor)+1). Only for type "yarn"; other
        # methods (ntk/linear) leave attention temperature unchanged → 1.0.
        attn_logit_scale = (
            yarn_attention_scale(scaling_factor) ** 2 if scaling_type == "yarn" else 1.0
        )

        # Stack of transformer blocks
        # nn.ModuleList is like an array of layers that PyTorch tracks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config, layer_idx=i, is_moe=i in self._moe_layer_set,
                attn_logit_scale=attn_logit_scale,
            )
            for i in range(config.n_layers)
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
        #
        # theta and rope_scaling come from the config: theta controls the
        # base wavelength (500K for long-context configs like 4080_max),
        # and rope_scaling (YaRN/NTK/linear) extends a trained model's
        # context window (Stage 4). With scaling, the freq table covers the
        # extended length (max_seq_len * factor). scaling_type/scaling_factor
        # were resolved above (shared with the YaRN attention temperature).
        rope_len = int(config.max_seq_len * max(scaling_factor, 1.0))
        rope_freqs = get_rope_freqs(
            dim=config.head_dim,
            max_seq_len=rope_len * 2,
            theta=getattr(config, "rope_theta", 10000.0),
            scaling_type=scaling_type,
            scaling_factor=scaling_factor,
            original_max_seq_len=config.max_seq_len,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Gradient checkpointing flag — read in forward(); set via
        # enable_gradient_checkpointing()
        self.gradient_checkpointing = False

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

        # Residual-scaled init (GPT-2 paper, used by LLaMA/nanoGPT): the
        # projections that WRITE INTO the residual stream (attention out_proj,
        # FFN down_proj) get std scaled by 1/sqrt(2 * n_layers). Without this,
        # the residual stream's variance grows with depth — each of the
        # 2*n_layers residual additions piles on, which is exactly what makes
        # deep models (24 layers in 4080_max) unstable early in training.
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=residual_std)

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
        # During training: mask=None — SDPA uses is_causal=True (faster)
        # During inference with cache and start_pos>0: mask=None (single token)
        # During inference first step: explicit causal mask
        if use_cache and start_pos == 0:
            mask = self.causal_mask[:seq_len, :seq_len]
        else:
            mask = None

        # Step 3: Pass through all transformer blocks
        # With gradient checkpointing (training only — recomputation is
        # incompatible with the KV-cache), activations inside each block are
        # discarded after the forward pass and recomputed during backward,
        # cutting activation VRAM by ~60% at ~30% extra compute.
        use_ckpt = self.gradient_checkpointing and self.training and not use_cache
        for block in self.blocks:
            if use_ckpt:
                h = torch.utils.checkpoint.checkpoint(
                    block,
                    h,
                    self.rope_freqs,
                    start_pos,
                    use_cache,
                    mask,
                    use_reentrant=False,
                )
            else:
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
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling.

        The task: given tokens [0, 1, 2, ..., N-1], predict tokens [1, 2, 3, ..., N].
        Each token tries to predict the NEXT token.

        Args:
            token_ids: Shape (batch_size, seq_len). The training sequence.
            sample_weights: Optional per-sample quality weights, shape
                (batch_size,). When provided, each sequence's loss contributes
                proportionally to its weight (weighted mean), so high-quality
                samples influence the gradient more than low-quality ones
                WITHIN the same batch.

        Returns:
            Scalar loss value. Lower = better predictions.

        Note: when the model is wrapped by torch.compile, prefer calling the
        model directly for logits and applying language_modeling_loss() — this
        method on a compiled model runs the original, uncompiled module.
        """
        loss = language_modeling_loss(self.forward(token_ids), token_ids, sample_weights)
        return loss + self.moe_aux_loss()

    def moe_aux_loss(self) -> torch.Tensor:
        """Sum the load-balancing auxiliary loss across all MoE blocks.

        Each MoEFFN stores its aux loss as a side effect of the most recent
        forward pass (and returns 0 in eval mode). Add this to the main loss
        during MoE fine-tuning so the router learns to spread tokens evenly
        across experts instead of collapsing onto a few. Returns a 0 scalar
        for dense models, so callers can add it unconditionally.
        """
        total = torch.zeros((), device=self.tok_emb.weight.device)
        for block in self.blocks:
            if getattr(block, "is_moe", False):
                total = total + block.ffn.aux_loss.to(total.device)
        return total

    def clear_caches(self):
        """Clear all KV-caches (call between generation requests)."""
        for block in self.blocks:
            block.attention.clear_cache()

    def expand_caches(self, batch_size: int):
        """Expand all layer KV-caches from batch=1 to batch=N.

        Call this after prefilling the prompt (with batch=1) and before
        batched generation so all N completions share the same prompt context.

        Args:
            batch_size: Target batch size (e.g. group_size in GRPO).
        """
        for block in self.blocks:
            block.attention.expand_cache(batch_size)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM.

        Trades compute for memory: instead of storing all intermediate
        activations, recompute them during the backward pass.
        Roughly halves memory usage but is ~30% slower.
        Required for the 350M model on 16GB GPUs.
        """
        self.gradient_checkpointing = True

    def get_hidden_states(
        self,
        token_ids: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """Extract hidden states from an intermediate layer.

        Used for embedding-based retrieval and semantic routing.
        Middle layers capture the best semantic representations.

        Args:
            token_ids: Input token IDs, shape (batch_size, seq_len).
            layer: Which layer to extract from. Negative indices count
                   from the end (e.g., -4 = 4th from last). Default: -1
                   (last hidden state before output projection).

        Returns:
            Hidden states tensor, shape (batch_size, seq_len, dim).
        """
        n_layers = len(self.blocks)

        # Resolve negative indices
        if layer < 0:
            layer = n_layers + layer
        layer = max(0, min(layer, n_layers - 1))

        # Embedding
        h = self.tok_emb(token_ids)

        # Run through blocks up to the target layer
        for i, block in enumerate(self.blocks):
            h = block(h, rope_freqs=self.rope_freqs, start_pos=0, use_cache=False)
            if i == layer:
                break

        return h

    @property
    def n_layers(self) -> int:
        """Number of transformer blocks."""
        return len(self.blocks)

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
