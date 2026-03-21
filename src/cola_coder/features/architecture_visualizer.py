"""Architecture Visualizer: print an ASCII representation of the cola-coder model.

Shows layer types, dimensions, parameter counts per layer, and total parameter
summary.  Works entirely from a config dict — no model weights needed.

Think of this like TypeScript's ``tsc --listFiles`` for the model: a structural
overview without actually executing anything.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the architecture visualizer is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LayerInfo:
    """Description of a single layer in the visualisation."""

    name: str
    layer_type: str
    dims: str  # Human-readable dimension string, e.g. "768 × 768"
    params: int  # Parameter count for this layer


@dataclass
class ArchitectureReport:
    """Full report returned by ArchitectureVisualizer.visualize()."""

    layers: list[LayerInfo] = field(default_factory=list)
    total_params: int = 0
    n_layers: int = 0
    d_model: int = 0
    d_ffn: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    vocab_size: int = 0
    max_seq_len: int = 0
    architecture_text: str = ""


# ---------------------------------------------------------------------------
# Parameter math helpers
# ---------------------------------------------------------------------------


def _count_attn_params(d_model: int, n_heads: int, n_kv_heads: int) -> int:
    """GQA attention parameter count (Q + K + V + O projections)."""
    head_dim = d_model // n_heads if n_heads else d_model
    q_proj = d_model * d_model
    k_proj = n_kv_heads * head_dim * d_model
    v_proj = n_kv_heads * head_dim * d_model
    o_proj = d_model * d_model
    return q_proj + k_proj + v_proj + o_proj


def _count_ffn_params(d_model: int, d_ffn: int) -> int:
    """SwiGLU FFN parameter count (gate + up + down projections)."""
    return 3 * d_model * d_ffn


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


# ---------------------------------------------------------------------------
# ArchitectureVisualizer
# ---------------------------------------------------------------------------


class ArchitectureVisualizer:
    """Generate an ASCII architecture diagram from a model config dict.

    The config dict can be nested (``{"model": {...}, "training": {...}}``)
    or flat — the same shape that metadata.json embeds.

    Usage::

        viz = ArchitectureVisualizer()
        report = viz.visualize(config_dict)
        print(report.architecture_text)
    """

    def visualize(self, config: dict[str, Any]) -> ArchitectureReport:
        """Build an ArchitectureReport from *config*.

        Parameters
        ----------
        config:
            Model config dict.  Can be a raw flat dict (keys: d_model, n_layers,
            n_heads, …) or a nested dict with a ``"model"`` key.

        Returns
        -------
        ArchitectureReport
            Contains both structured data and a pre-formatted ASCII string.
        """
        # Unwrap nested config
        cfg = config.get("model", config)

        d_model: int = cfg.get("d_model", 768)
        n_layers: int = cfg.get("n_layers", 12)
        n_heads: int = cfg.get("n_heads", 12)
        n_kv_heads: int = cfg.get("n_kv_heads", n_heads)
        vocab_size: int = cfg.get("vocab_size", 32_000)
        max_seq_len: int = cfg.get("max_seq_len", 2048)
        d_ffn: int = cfg.get("d_ffn", d_model * 4)

        layers: list[LayerInfo] = []

        # ── Token Embedding ────────────────────────────────────────────────
        emb_params = vocab_size * d_model
        layers.append(
            LayerInfo(
                name="tok_emb",
                layer_type="Embedding",
                dims=f"{vocab_size:,} × {d_model}",
                params=emb_params,
            )
        )

        # ── Transformer blocks ─────────────────────────────────────────────
        attn_params_per = _count_attn_params(d_model, n_heads, n_kv_heads)
        ffn_params_per = _count_ffn_params(d_model, d_ffn)
        norm_params_per = d_model  # single RMSNorm weight vector

        for i in range(n_layers):
            block_name = f"block[{i}]"
            # Pre-attention RMSNorm
            layers.append(
                LayerInfo(
                    name=f"{block_name}.norm1",
                    layer_type="RMSNorm",
                    dims=f"{d_model}",
                    params=norm_params_per,
                )
            )
            # GQA Attention
            head_dim = d_model // n_heads if n_heads else d_model
            layers.append(
                LayerInfo(
                    name=f"{block_name}.attn",
                    layer_type="GQA Attention",
                    dims=f"Q:{d_model}×{d_model} KV:{n_kv_heads}×{head_dim}×{d_model}",
                    params=attn_params_per,
                )
            )
            # Pre-FFN RMSNorm
            layers.append(
                LayerInfo(
                    name=f"{block_name}.norm2",
                    layer_type="RMSNorm",
                    dims=f"{d_model}",
                    params=norm_params_per,
                )
            )
            # SwiGLU FFN
            layers.append(
                LayerInfo(
                    name=f"{block_name}.ffn",
                    layer_type="SwiGLU FFN",
                    dims=f"{d_model}×{d_ffn} (×3 gate/up/down)",
                    params=ffn_params_per,
                )
            )

        # ── Final LayerNorm + LM head (weight-tied) ────────────────────────
        final_norm_params = d_model
        layers.append(
            LayerInfo(
                name="final_norm",
                layer_type="RMSNorm",
                dims=f"{d_model}",
                params=final_norm_params,
            )
        )
        layers.append(
            LayerInfo(
                name="output (weight-tied)",
                layer_type="Linear (tied)",
                dims=f"{d_model} × {vocab_size:,}",
                params=0,  # tied to tok_emb — counted once
            )
        )

        total_params = sum(layer.params for layer in layers)

        report = ArchitectureReport(
            layers=layers,
            total_params=total_params,
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            architecture_text=self._render(
                layers=layers,
                total_params=total_params,
                n_layers=n_layers,
                d_model=d_model,
                d_ffn=d_ffn,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
            ),
        )
        return report

    # ------------------------------------------------------------------
    # Private rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        layers: list[LayerInfo],
        total_params: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        n_kv_heads: int,
        vocab_size: int,
        max_seq_len: int,
    ) -> str:
        lines: list[str] = []

        width = 100
        sep = "─" * width

        lines.append("╔" + "═" * (width - 2) + "╗")
        title = "Cola-Coder Architecture (Decoder-only Transformer)"
        lines.append("║" + title.center(width - 2) + "║")
        lines.append("╚" + "═" * (width - 2) + "╝")
        lines.append("")

        # Summary table
        lines.append("  Architecture: RoPE · GQA · SwiGLU · RMSNorm (pre-norm)  [LLaMA / Mistral family]")
        lines.append(f"  Parameters:   {_fmt_params(total_params)}  ({total_params:,} total)")
        lines.append(f"  d_model:      {d_model}    n_layers: {n_layers}    n_heads: {n_heads}    "
                     f"n_kv_heads: {n_kv_heads}")
        lines.append(f"  d_ffn:        {d_ffn}    vocab: {vocab_size:,}    "
                     f"max_seq_len: {max_seq_len:,}")
        lines.append("")
        lines.append(sep)

        # Layer table header
        col_name = 40
        col_type = 20
        col_dims = 25
        col_params = 12
        hdr = (
            f"  {'Layer':<{col_name}} {'Type':<{col_type}} {'Dimensions':<{col_dims}} "
            f"{'Params':>{col_params}}"
        )
        lines.append(hdr)
        lines.append(sep)

        block_total_attn = 0
        block_total_ffn = 0
        block_total_norm = 0

        for layer in layers:
            # Detect transformer block layers
            if layer.name.startswith("block["):
                idx = int(layer.name.split("[")[1].split("]")[0])
                if idx == 0:
                    # Show the first block fully
                    row = (
                        f"  {layer.name:<{col_name}} {layer.layer_type:<{col_type}} "
                        f"{layer.dims:<{col_dims}} {_fmt_params(layer.params):>{col_params}}"
                    )
                    lines.append(row)
                elif idx == 1 and n_layers > 2:
                    lines.append(
                        f"  {'  ... × {:d} more identical blocks ...'.format(n_layers - 1)}"
                    )
                    # Accumulate stats for summary
                    if "attn" in layer.name:
                        block_total_attn += layer.params * n_layers
                    elif "ffn" in layer.name:
                        block_total_ffn += layer.params * n_layers
                    elif "norm" in layer.name:
                        block_total_norm += layer.params * n_layers
                elif idx >= 1:
                    # Still accumulate
                    if "attn" in layer.name:
                        block_total_attn += layer.params
                    elif "ffn" in layer.name:
                        block_total_ffn += layer.params
                    elif "norm" in layer.name:
                        block_total_norm += layer.params
                continue

            # Non-block layers
            row = (
                f"  {layer.name:<{col_name}} {layer.layer_type:<{col_type}} "
                f"{layer.dims:<{col_dims}} {_fmt_params(layer.params):>{col_params}}"
            )
            lines.append(row)

        lines.append(sep)
        lines.append(f"  {'TOTAL PARAMETERS':<{col_name + col_type + col_dims + 3}} "
                     f"{_fmt_params(total_params):>{col_params}}")
        lines.append("")

        # Per-component breakdown
        attn_params = _count_attn_params(d_model, n_heads, n_kv_heads) * n_layers
        ffn_params = _count_ffn_params(d_model, d_ffn) * n_layers
        emb_params = vocab_size * d_model
        norm_params = d_model * (2 * n_layers + 1)

        lines.append("  Parameter breakdown by component:")
        components = [
            ("Embeddings", emb_params),
            (f"Attention ({n_layers} layers)", attn_params),
            (f"FFN/{n_layers} layers)", ffn_params),
            (f"LayerNorms ({2 * n_layers + 1})", norm_params),
        ]
        for name, count in components:
            pct = 100.0 * count / total_params if total_params else 0.0
            bar_width = int(pct / 2)
            bar = "█" * bar_width + "░" * (50 - bar_width)
            lines.append(f"  {name:<30} {bar}  {pct:5.1f}%  ({_fmt_params(count)})")

        lines.append("")
        return "\n".join(lines)
