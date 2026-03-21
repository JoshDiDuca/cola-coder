"""Model Parameter Counter — Feature 98

Detailed parameter counting for transformer models, broken down by component
type: embedding, attention (Q/K/V/O projection), FFN (gate/up/down), and
normalisation layers.

Also computes the *theoretical* parameter count from a config dict and
compares it to the *actual* count from a live model (if provided) or the
theoretical baseline.

Works with or without a PyTorch model — the theoretical calculator requires
only primitive config values.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the parameter counter is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Component breakdown
# ---------------------------------------------------------------------------


@dataclass
class ComponentCount:
    """Parameter count for a single model component."""

    name: str
    n_params: int
    trainable: bool = True

    @property
    def millions(self) -> float:
        return self.n_params / 1e6


@dataclass
class ParamBreakdown:
    """Full parameter breakdown for a transformer model."""

    embedding: int = 0
    attention_q: int = 0
    attention_k: int = 0
    attention_v: int = 0
    attention_o: int = 0
    ffn_gate: int = 0
    ffn_up: int = 0
    ffn_down: int = 0
    norm: int = 0
    other: int = 0
    components: list[ComponentCount] = field(default_factory=list)

    @property
    def attention_total(self) -> int:
        return self.attention_q + self.attention_k + self.attention_v + self.attention_o

    @property
    def ffn_total(self) -> int:
        return self.ffn_gate + self.ffn_up + self.ffn_down

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.attention_total
            + self.ffn_total
            + self.norm
            + self.other
        )

    @property
    def total_millions(self) -> float:
        return self.total / 1e6

    def as_dict(self) -> dict[str, Any]:
        return {
            "embedding_M": self.embedding / 1e6,
            "attention_total_M": self.attention_total / 1e6,
            "attention_q_M": self.attention_q / 1e6,
            "attention_k_M": self.attention_k / 1e6,
            "attention_v_M": self.attention_v / 1e6,
            "attention_o_M": self.attention_o / 1e6,
            "ffn_total_M": self.ffn_total / 1e6,
            "ffn_gate_M": self.ffn_gate / 1e6,
            "ffn_up_M": self.ffn_up / 1e6,
            "ffn_down_M": self.ffn_down / 1e6,
            "norm_M": self.norm / 1e6,
            "other_M": self.other / 1e6,
            "total_M": self.total_millions,
        }


@dataclass
class ParamCountResult:
    """Full result from a parameter counting pass."""

    theoretical: ParamBreakdown
    actual: Optional[ParamBreakdown]
    mismatch_pct: Optional[float]  # |theoretical - actual| / theoretical * 100

    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "theoretical_total_M": self.theoretical.total_millions,
            "theoretical_breakdown": self.theoretical.as_dict(),
        }
        if self.actual is not None:
            d["actual_total_M"] = self.actual.total_millions
            d["mismatch_pct"] = self.mismatch_pct
        return d


# ---------------------------------------------------------------------------
# Theoretical calculator
# ---------------------------------------------------------------------------


def theoretical_breakdown(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_multiplier: float = 8 / 3,
    use_bias: bool = False,
) -> ParamBreakdown:
    """Compute the theoretical parameter count for a GQA transformer.

    Architecture assumptions (LLaMA-3 / cola-coder style):
    - Embedding + tied output projection (counted once)
    - Pre-norm RMSNorm (no bias)
    - GQA: Q projects to (n_heads * head_dim), K/V project to (n_kv_heads * head_dim)
    - SwiGLU FFN: gate + up projections of size d_model → ffn_dim, down of ffn_dim → d_model

    Parameters
    ----------
    vocab_size: Vocabulary size.
    d_model: Hidden dimension.
    n_layers: Number of transformer layers.
    n_heads: Number of query heads.
    n_kv_heads: Number of key/value heads (< n_heads for GQA).
    ffn_multiplier: FFN hidden-dim multiplier (SwiGLU default ≈ 8/3).
    use_bias: Whether linear layers include bias terms.
    """
    head_dim = d_model // n_heads
    ffn_dim = int(d_model * ffn_multiplier)

    # Bias adds d_model per projection if enabled
    bias = d_model if use_bias else 0

    # Embedding (tied with output → count once)
    emb = vocab_size * d_model

    # Per-layer attention
    q = d_model * (n_heads * head_dim) + bias
    k = d_model * (n_kv_heads * head_dim) + bias
    v = d_model * (n_kv_heads * head_dim) + bias
    o = (n_heads * head_dim) * d_model + bias

    # Per-layer FFN (SwiGLU: gate + up + down)
    gate = d_model * ffn_dim
    up = d_model * ffn_dim
    down = ffn_dim * d_model

    # Per-layer RMSNorm (2 per layer: pre-attn + pre-ffn) — no bias
    norm_per_layer = 2 * d_model
    # Final norm
    norm_final = d_model

    bd = ParamBreakdown(
        embedding=emb,
        attention_q=q * n_layers,
        attention_k=k * n_layers,
        attention_v=v * n_layers,
        attention_o=o * n_layers,
        ffn_gate=gate * n_layers,
        ffn_up=up * n_layers,
        ffn_down=down * n_layers,
        norm=norm_per_layer * n_layers + norm_final,
    )
    return bd


# ---------------------------------------------------------------------------
# Live model counter (optional torch dependency)
# ---------------------------------------------------------------------------


def count_from_model(model: Any) -> Optional[ParamBreakdown]:  # type: ignore[type-arg]
    """Count parameters from a live PyTorch model via name-based heuristics.

    Returns ``None`` if torch is not available.
    """
    try:
        import torch.nn as nn  # type: ignore[import]
    except ImportError:
        return None

    if not isinstance(model, nn.Module):
        return None

    bd = ParamBreakdown()
    components: list[ComponentCount] = []

    for name, param in model.named_parameters():
        n = param.numel()
        trainable = param.requires_grad
        ln = name.lower()

        comp_name = name
        if any(k in ln for k in ("tok_emb", "embed", "wte", "wpe")):
            bd.embedding += n
        elif any(k in ln for k in ("q_proj", "wq", ".q.")):
            bd.attention_q += n
        elif any(k in ln for k in ("k_proj", "wk", ".k.")):
            bd.attention_k += n
        elif any(k in ln for k in ("v_proj", "wv", ".v.")):
            bd.attention_v += n
        elif any(k in ln for k in ("o_proj", "out_proj", "wo")):
            bd.attention_o += n
        elif any(k in ln for k in ("gate_proj", "gate")):
            bd.ffn_gate += n
        elif any(k in ln for k in ("up_proj", "w1", "fc1")):
            bd.ffn_up += n
        elif any(k in ln for k in ("down_proj", "w2", "fc2")):
            bd.ffn_down += n
        elif any(k in ln for k in ("norm", "ln", "layer_norm", "rms")):
            bd.norm += n
        else:
            bd.other += n
        components.append(ComponentCount(name=comp_name, n_params=n, trainable=trainable))

    bd.components = components
    return bd


# ---------------------------------------------------------------------------
# High-level counter
# ---------------------------------------------------------------------------


class ModelParamCounter:
    """Detailed parameter counter for transformer models."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        ffn_multiplier: float = 8 / 3,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.ffn_multiplier = ffn_multiplier

    def count(self, model: Any = None) -> ParamCountResult:
        """Count parameters.

        If *model* (a ``torch.nn.Module``) is provided, also do a live count
        and compute mismatch percentage.
        """
        theo = theoretical_breakdown(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            ffn_multiplier=self.ffn_multiplier,
        )
        actual: Optional[ParamBreakdown] = None
        mismatch: Optional[float] = None

        if model is not None:
            actual = count_from_model(model)
            if actual is not None and theo.total > 0:
                mismatch = abs(theo.total - actual.total) / theo.total * 100.0

        return ParamCountResult(
            theoretical=theo,
            actual=actual,
            mismatch_pct=mismatch,
        )

    def theoretical_total(self) -> int:
        return theoretical_breakdown(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            ffn_multiplier=self.ffn_multiplier,
        ).total
