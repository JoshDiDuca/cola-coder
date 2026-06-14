"""Model-card builder for the local UI/dashboard.

Pure library module that composes a readable model-card summary for a single
checkpoint directory WITHOUT loading any weights. It builds on
``checkpoint_detail.checkpoint_detail`` (param/tensor counts + parsed
``metadata.json`` + MoE detection) and, best-effort, ``tokenizer_info`` for the
tokenizer summary.

The architecture / training split mirrors how ``training/checkpoint.py`` writes
``metadata.json``: a top-level ``{step, loss, config{model{...}, training{...}},
data_path, tokenizer_path}``. We read ``config.model`` as the architecture
source of truth (the BUG-128 source) and ``config.training`` (+ top-level
step/loss) as training provenance, while staying robust to a flatter shape.

All functions are best-effort and never raise — on failure they return an
``{"error": ...}`` dict.
"""

from __future__ import annotations

from pathlib import Path

from .checkpoint_detail import checkpoint_detail

try:  # tokenizer_info is best-effort; never let an import problem break the card.
    from .tokenizer_info import tokenizer_info
except Exception:  # pragma: no cover - defensive import guard
    tokenizer_info = None  # type: ignore[assignment]

# Keys (case-insensitive) that describe model shape / architecture.
_ARCH_KEYS = {
    "vocab_size",
    "dim",
    "d_model",
    "n_layers",
    "num_layers",
    "n_heads",
    "num_heads",
    "n_kv_heads",
    "num_kv_heads",
    "head_dim",
    "ffn_dim_multiplier",
    "ffn_hidden_dim",
    "d_ffn",
    "intermediate_size",
    "max_seq_len",
    "seq_len",
    "rope_theta",
    "rope_scaling",
    "dropout",
    "norm_eps",
    "tie_embeddings",
    "moe",
}

# Keys (case-insensitive) that describe training provenance / hyperparameters.
_TRAINING_KEYS = {
    "step",
    "steps",
    "loss",
    "best_loss",
    "val_loss",
    "perplexity",
    "lr",
    "learning_rate",
    "min_lr",
    "warmup_steps",
    "max_steps",
    "batch_size",
    "gradient_accumulation",
    "weight_decay",
    "grad_clip",
    "precision",
    "gradient_checkpointing",
    "optimizer",
    "scheduler",
    "epochs",
    "epochs_completed",
    "tokens_seen",
    "data_path",
}


def _humanize_params(n: int) -> str:
    """Format a parameter count like 124000000 -> '124.0M'. Best-effort."""
    if not isinstance(n, int) or n <= 0:
        return "unknown"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _split_metadata(metadata: dict | None) -> tuple[dict, dict]:
    """Split a parsed metadata.json into (architecture, training) dicts.

    Handles the canonical nested form written by ``checkpoint.save_checkpoint``
    (``config.model`` / ``config.training`` + top-level ``step``/``loss``) and a
    flatter form (architecture/training keys at the top level), defensively.
    """
    architecture: dict = {}
    training: dict = {}
    if not isinstance(metadata, dict):
        return architecture, training

    config = metadata.get("config")
    if isinstance(config, dict):
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            architecture.update(model_cfg)
        training_cfg = config.get("training")
        if isinstance(training_cfg, dict):
            training.update(training_cfg)

    # Top-level keys (step/loss/data_path/etc., or a flat metadata shape).
    for key, value in metadata.items():
        if key == "config":
            continue
        lower = key.lower()
        if lower in _ARCH_KEYS:
            architecture.setdefault(key, value)
        elif lower in _TRAINING_KEYS:
            training.setdefault(key, value)
        # tokenizer_path / unknown top-level keys are intentionally dropped here;
        # they are not part of the architecture/training contract.

    return architecture, training


def _render_kv_table(title: str, data: dict) -> list[str]:
    """Render a dict as a Markdown 2-column table under a heading. Empty -> note."""
    lines = [f"## {title}", ""]
    if not data:
        lines.append("_No data available._")
        lines.append("")
        return lines
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key in sorted(data):
        value = data[key]
        if isinstance(value, (dict, list)):
            rendered = str(value)
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.append("")
    return lines


def _render_markdown(
    *,
    name: str,
    num_params: int,
    architecture: dict,
    training: dict,
    tokenizer: dict | None,
    is_moe: bool,
) -> str:
    """Build a compact Markdown model card from the assembled fields."""
    lines: list[str] = [f"# {name}", ""]

    params_str = _humanize_params(num_params)
    arch_kind = "Mixture-of-Experts (MoE)" if is_moe else "Dense"
    lines.append(f"- **Parameters:** {params_str} ({num_params:,})")
    lines.append(f"- **Architecture:** {arch_kind}")
    lines.append("")

    lines.extend(_render_kv_table("Architecture", architecture))
    lines.extend(_render_kv_table("Training", training))

    lines.append("## Tokenizer")
    lines.append("")
    if isinstance(tokenizer, dict) and "error" not in tokenizer:
        vocab = tokenizer.get("vocab_size", "unknown")
        model_type = tokenizer.get("model_type", "unknown") or "unknown"
        specials = tokenizer.get("special_tokens", []) or []
        n_special = len(specials) if isinstance(specials, list) else 0
        lines.append(f"- **Vocab size:** {vocab}")
        lines.append(f"- **Model type:** {model_type}")
        lines.append(f"- **Special tokens:** {n_special}")
        lines.append(f"- **FIM tokens:** {bool(tokenizer.get('has_fim_tokens', False))}")
        lines.append(f"- **Digit splitting:** {bool(tokenizer.get('digit_splitting', False))}")
    else:
        lines.append("_Tokenizer info unavailable._")
    lines.append("")

    return "\n".join(lines)


def build_model_card(checkpoint_path: str) -> dict:
    """Build a model-card summary for one checkpoint dir. Returns:
      {"path": str, "name": str,
       "num_params": int,
       "architecture": dict,     # dim/n_layers/n_heads/n_kv_heads/vocab_size/seq_len etc. from metadata.json
       "training": dict,         # step/loss/lr/etc. if present in metadata.json
       "tokenizer": dict | None, # tokenizer_info() result, or None if unavailable
       "is_moe": bool,
       "markdown": str}          # a readable Markdown model card built from the above
    On any failure return {"error": "..."}. Never raise. If metadata.json is absent, still return what
    checkpoint_detail provides (architecture/training may be {}).
    """
    detail = checkpoint_detail(checkpoint_path)
    if not isinstance(detail, dict) or "error" in detail:
        # Propagate the underlying error (or synthesize one) without raising.
        if isinstance(detail, dict) and "error" in detail:
            return {"error": detail["error"]}
        return {"error": f"could not inspect checkpoint: {checkpoint_path}"}

    resolved_path = detail.get("path", checkpoint_path)
    num_params = detail.get("num_params", 0)
    if not isinstance(num_params, int):
        num_params = 0
    is_moe = bool(detail.get("is_moe", False))
    metadata = detail.get("metadata")

    # name = checkpoint dir name (or its parent if the dir name is itself a path).
    name = Path(resolved_path).name or Path(resolved_path).parent.name or resolved_path

    architecture, training = _split_metadata(metadata if isinstance(metadata, dict) else None)

    # Tokenizer is best-effort. Prefer the checkpoint's own tokenizer_path if
    # metadata.json recorded one; fall back to default discovery.
    tokenizer: dict | None = None
    if tokenizer_info is not None:
        tok_path = None
        if isinstance(metadata, dict):
            candidate = metadata.get("tokenizer_path")
            if isinstance(candidate, str) and candidate:
                tok_path = candidate
        try:
            result = tokenizer_info(tok_path)
            if isinstance(result, dict) and "error" not in result:
                tokenizer = result
        except Exception:
            tokenizer = None

    markdown = _render_markdown(
        name=name,
        num_params=num_params,
        architecture=architecture,
        training=training,
        tokenizer=tokenizer,
        is_moe=is_moe,
    )

    return {
        "path": resolved_path,
        "name": name,
        "num_params": num_params,
        "architecture": architecture,
        "training": training,
        "tokenizer": tokenizer,
        "is_moe": is_moe,
        "markdown": markdown,
    }
