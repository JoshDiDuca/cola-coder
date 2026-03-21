"""Model Size Estimator: parameter count and memory breakdown for any ModelConfig.

Given a ModelConfig, computes:
  - Total parameter count (with and without embeddings)
  - Memory for weights (at bf16, fp16, fp32, int8, int4)
  - Memory for activations (inference, single sequence)
  - Memory for optimizer states (AdamW)
  - A human-readable breakdown table

Complements vram_estimator.py (which focuses on GPU training VRAM).
This module focuses purely on the model itself, not the training loop.

For a TS dev: like a bundle-size analyser but for neural network weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Precision = Literal["bf16", "fp16", "fp32", "int8", "int4"]

_BYTES_PER_DTYPE: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass
class LayerParamCount:
    """Parameter counts broken down by transformer component."""

    embedding: int
    attention: int  # Q/K/V/out projections
    feedforward: int  # gate, up, down projections
    normalization: int  # RMSNorm weights
    output_head: int  # lm_head (may be tied to embedding)
    total: int

    # Derived
    non_embedding: int = field(init=False)

    def __post_init__(self) -> None:
        self.non_embedding = self.total - self.embedding


@dataclass
class MemoryEstimate:
    """Memory usage estimates in GB for different configurations."""

    precision: str
    weights_gb: float
    activation_gb: float  # single sequence, inference-only
    optimizer_gb: float  # AdamW in fp32 (m + v + master weights)
    total_inference_gb: float
    total_training_gb: float  # weights + optimizer + gradients


@dataclass
class SizeReport:
    """Full size report for a model config."""

    # Config summary
    config_name: str
    n_layers: int
    dim: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden_dim: int
    vocab_size: int
    max_seq_len: int

    # Parameters
    params: LayerParamCount

    # Memory at various precisions
    memory: dict[str, MemoryEstimate]  # keyed by precision string


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class ModelSizeEstimator:
    """Estimate parameter count and memory for a ModelConfig.

    Parameters
    ----------
    seq_len_for_activations:
        Sequence length to use when estimating activation memory.
        Defaults to the model's ``max_seq_len``.

    Example::

        from cola_coder.model.config import Config
        cfg = Config.from_yaml("configs/tiny.yaml")
        est = ModelSizeEstimator()
        report = est.estimate(cfg.model)
        est.print_table(report)
    """

    def __init__(self, seq_len_for_activations: int | None = None) -> None:
        self.seq_len_for_activations = seq_len_for_activations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, model_config) -> SizeReport:
        """Compute a full SizeReport for *model_config*.

        Accepts any object with the attributes used by ModelConfig in
        cola_coder/model/config.py.
        """
        dim = model_config.dim
        n_heads = model_config.n_heads
        n_kv_heads = model_config.n_kv_heads
        n_layers = model_config.n_layers
        vocab_size = model_config.vocab_size
        max_seq_len = model_config.max_seq_len
        ffn_hidden = getattr(model_config, "ffn_hidden_dim", self._default_ffn(dim))
        head_dim = dim // n_heads

        seq_len = self.seq_len_for_activations or max_seq_len

        # ── Embedding ───────────────────────────────────────────────────
        embed_params = vocab_size * dim  # tok_emb (tied with output head)

        # ── Attention (per layer) ───────────────────────────────────────
        # Q: dim * dim  K: dim * n_kv_heads * head_dim  V: same  O: dim * dim
        q_params = dim * dim
        k_params = dim * n_kv_heads * head_dim
        v_params = dim * n_kv_heads * head_dim
        o_params = dim * dim
        attn_per_layer = q_params + k_params + v_params + o_params
        attn_total = attn_per_layer * n_layers

        # ── FFN (SwiGLU, per layer) ─────────────────────────────────────
        # gate_proj: dim * ffn_hidden  up_proj: same  down_proj: ffn_hidden * dim
        ffn_per_layer = 2 * dim * ffn_hidden + ffn_hidden * dim
        ffn_total = ffn_per_layer * n_layers

        # ── Normalization (RMSNorm) ─────────────────────────────────────
        # 2 per layer (pre-attn + pre-ffn) + 1 final norm
        norm_per_layer = 2 * dim
        norm_total = norm_per_layer * n_layers + dim

        # ── Output head ─────────────────────────────────────────────────
        # Tied to embedding so we don't double-count in total
        output_head_params = vocab_size * dim  # same tensor as embedding

        total_params = embed_params + attn_total + ffn_total + norm_total
        # NOTE: output head is tied — not added again

        params = LayerParamCount(
            embedding=embed_params,
            attention=attn_total,
            feedforward=ffn_total,
            normalization=norm_total,
            output_head=output_head_params,
            total=total_params,
        )

        # ── Memory at each precision ────────────────────────────────────
        memory: dict[str, MemoryEstimate] = {}
        for precision in ("fp32", "bf16", "fp16", "int8", "int4"):
            bpw = _BYTES_PER_DTYPE[precision]
            weights_gb = (total_params * bpw) / 1e9

            # Activation memory (inference, single sequence)
            # Per layer: Q/K/V projections + attention scores + FFN intermediates
            # Rough formula: seq_len * dim * 4 bytes * n_layers (fp32 accumulator)
            act_bytes = seq_len * dim * 4 * n_layers
            act_gb = act_bytes / 1e9

            # Optimizer (AdamW): master weights (fp32) + m + v (fp32 each)
            # = 3 * total_params * 4 bytes  (even if model is fp16/bf16)
            opt_gb = (total_params * 4 * 3) / 1e9

            # Gradients (same dtype as model)
            grad_gb = (total_params * bpw) / 1e9

            total_inf = weights_gb + act_gb
            total_train = weights_gb + opt_gb + grad_gb + act_gb

            memory[precision] = MemoryEstimate(
                precision=precision,
                weights_gb=weights_gb,
                activation_gb=act_gb,
                optimizer_gb=opt_gb,
                total_inference_gb=total_inf,
                total_training_gb=total_train,
            )

        config_name = getattr(model_config, "name", "unknown")
        return SizeReport(
            config_name=config_name,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_hidden_dim=ffn_hidden,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            params=params,
            memory=memory,
        )

    def estimate_from_dict(self, d: dict) -> SizeReport:
        """Estimate from a plain dict (keys match ModelConfig field names)."""
        return self.estimate(_DictModel(d))

    def print_table(self, report: SizeReport) -> None:
        """Print a human-readable breakdown table to stdout."""
        p = report.params
        _fmt_m = lambda n: f"{n/1e6:.2f}M"  # noqa: E731
        _fmt_b = lambda n: f"{n/1e9:.3f}B"  # noqa: E731

        print(f"\n{'=' * 60}")
        print(f"  Model Size Report — {report.config_name}")
        print(f"{'=' * 60}")
        print(f"  Architecture: {report.n_layers}L × {report.dim}d  "
              f"heads={report.n_heads} kv_heads={report.n_kv_heads}")
        print(f"  FFN hidden: {report.ffn_hidden_dim}  "
              f"vocab: {report.vocab_size:,}  seq_len: {report.max_seq_len:,}")

        print(f"\n{'─' * 60}")
        print("  Parameter Breakdown")
        print(f"{'─' * 60}")
        rows = [
            ("Embedding (tied)", p.embedding),
            ("Attention (all layers)", p.attention),
            ("FFN / SwiGLU (all layers)", p.feedforward),
            ("Normalization", p.normalization),
            ("TOTAL (no embedding double-count)", p.total),
            ("Non-embedding parameters", p.non_embedding),
        ]
        for label, count in rows:
            print(f"  {label:<38} {_fmt_m(count):>8}  ({_fmt_b(count)})")

        print(f"\n{'─' * 60}")
        print("  Memory Estimates (single GPU, no batch)")
        print(f"{'─' * 60}")
        hdr = f"  {'Precision':<8} {'Weights':>10} {'Activations':>12} "
        hdr += f"{'Optimizer':>11} {'Inference':>11} {'Training':>11}"
        print(hdr)
        print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*11} {'-'*11} {'-'*11}")
        for prec in ("fp32", "bf16", "fp16", "int8", "int4"):
            m = report.memory[prec]
            print(
                f"  {prec:<8} "
                f"{m.weights_gb:>9.2f}G "
                f"{m.activation_gb:>11.2f}G "
                f"{m.optimizer_gb:>10.2f}G "
                f"{m.total_inference_gb:>10.2f}G "
                f"{m.total_training_gb:>10.2f}G"
            )
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _default_ffn(dim: int) -> int:
        """LLaMA-style FFN hidden: 8/3 * dim rounded to nearest multiple of 256."""
        raw = int(8 / 3 * dim)
        return ((raw + 255) // 256) * 256


# ---------------------------------------------------------------------------
# Dict adapter (lets estimate_from_dict work)
# ---------------------------------------------------------------------------


class _DictModel:
    """Wrap a plain dict so attribute access works."""

    def __init__(self, d: dict) -> None:
        self._d = d

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(f"Model config has no attribute '{name}'") from None
