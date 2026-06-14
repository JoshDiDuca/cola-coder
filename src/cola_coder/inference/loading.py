"""Shared model-loading helper for evaluation/generation CLI scripts.

Centralizes the load pattern that generate.py / evaluate.py established:
resolve tokenizer (checkpoint metadata → DatasetResolver → storage config),
auto-detect checkpoint vocab size (SFT/reasoning checkpoints carry extra
tokens), build the Transformer, and wrap it in a CodeGenerator.
"""

from __future__ import annotations

import json
from pathlib import Path


def resolve_tokenizer_path(
    checkpoint: str | Path,
    config_path: str | Path | None = None,
) -> str:
    """Resolve the tokenizer for a checkpoint.

    Priority: checkpoint metadata.json → DatasetResolver(config) → storage
    config fallback.
    """
    from cola_coder.model.config import get_storage_config

    meta_path = Path(checkpoint) / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved = meta.get("tokenizer_path", "")
            if saved and Path(saved).exists():
                return saved
        except (json.JSONDecodeError, OSError):
            pass

    try:
        from cola_coder.data.dataset_resolver import DatasetResolver

        if DatasetResolver.tokenizer_exists(config_path=config_path):
            return str(DatasetResolver.get_tokenizer_path(config_path=config_path))
    except Exception:
        pass
    return get_storage_config().tokenizer_path


def apply_moe_config_from_checkpoint(config, checkpoint: str | Path) -> bool:
    """Flip a config to MoE when the checkpoint is an upcycled MoE model.

    Thin lazy passthrough to the canonical implementation in
    features/moe_layer (kept here so existing importers and the lazy-import
    design — no torch at module load — both keep working).
    """
    from cola_coder.features.moe_layer import (
        apply_moe_config_from_checkpoint as _impl,
    )

    return _impl(config, checkpoint)


# Scalar model-config fields that determine weight SHAPES / architecture. Nested
# fields (rope_scaling, moe) are dicts handled elsewhere (apply_moe_*), so we skip
# any dict value rather than clobber a config object with a raw dict.
_ARCH_FIELDS = (
    "dim", "n_layers", "n_heads", "n_kv_heads", "ffn_dim_multiplier",
    "ffn_hidden_dim", "hidden_dim", "max_seq_len", "vocab_size", "rope_theta",
    "norm_eps", "qk_norm", "dropout", "tie_embeddings",
)


def apply_model_config_from_checkpoint(config, checkpoint: str | Path) -> bool:
    """Override ``config.model`` architecture from the checkpoint's metadata.json.

    A checkpoint is the GROUND TRUTH for its own architecture. The ``--config`` yaml
    a caller passes can be wrong (e.g. the menu selecting ``configs/tiny.yaml`` for a
    dim=768 run), which builds a mismatched model and crashes ``load_state_dict`` with
    a size mismatch. Reading the saved config and applying its scalar architecture
    fields makes generation/serving/eval robust to a wrong-config selection.

    Returns True if a saved model config was found and applied.
    """
    checkpoint = Path(checkpoint)
    # Resolve a `latest` pointer file to the real step dir before reading metadata.
    if checkpoint.name == "latest" and checkpoint.is_file():
        checkpoint = Path(checkpoint.read_text(encoding="utf-8").strip())
    meta_path = checkpoint / "metadata.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    saved = (meta.get("config") or {}).get("model")
    if not isinstance(saved, dict):
        return False
    applied = False
    for field in _ARCH_FIELDS:
        if field in saved and not isinstance(saved[field], dict) and hasattr(config.model, field):
            setattr(config.model, field, saved[field])
            applied = True
    return applied


def load_generator(
    checkpoint: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path | None = None,
    device: str | None = None,
):
    """Load a checkpoint into a ready-to-use CodeGenerator.

    Returns:
        (generator, config, tokenizer) tuple.

    Raises:
        FileNotFoundError: checkpoint/config/tokenizer missing.
    """
    import torch

    from cola_coder.inference.generator import CodeGenerator
    from cola_coder.model.config import Config
    from cola_coder.model.transformer import Transformer
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    from cola_coder.training.checkpoint import load_model_only

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    # Resolve a `latest` pointer file to the real checkpoint dir BEFORE inspecting
    # it for vocab/MoE metadata. The pointer is a text file (not a dir), so reading
    # `latest/model.safetensors` or `latest/moe_config.json` would silently miss —
    # leaving the config dense and crashing load_state_dict on an upcycled MoE
    # checkpoint. load_model_only resolves `latest` too, so this just aligns the
    # pre-build inspection with the actual weights. (matches load_model_only.)
    if checkpoint.name == "latest" and checkpoint.is_file():
        checkpoint = Path(checkpoint.read_text(encoding="utf-8").strip())
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if tokenizer_path is None:
        tokenizer_path = resolve_tokenizer_path(checkpoint, config_path)
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config.from_yaml(config_path)

    # The checkpoint is ground truth for its architecture — apply its saved
    # metadata.json dims so a wrong config_path can't build a mismatched model
    # (BUG-128). Runs BEFORE the vocab/MoE detection below, which then refine it.
    apply_model_config_from_checkpoint(config, checkpoint)

    # SFT/reasoning checkpoints carry extra tokens (ChatML, <think>) — size
    # the model from the checkpoint's embedding, not the config.
    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as f:
            if "tok_emb.weight" in f.keys():
                config.model.vocab_size = f.get_tensor("tok_emb.weight").shape[0]
    except Exception:
        pass

    # Upcycled MoE checkpoints carry per-layer experts. Detect them (sidecar
    # or weight keys) and switch the config to MoE so Transformer builds the
    # matching expert FFNs, otherwise load_state_dict sees keys it can't place.
    apply_moe_config_from_checkpoint(config, checkpoint)

    tokenizer = CodeTokenizer(str(tokenizer_path))
    model = Transformer(config.model).to(device)
    load_model_only(str(checkpoint), model, device=device)
    generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    return generator, config, tokenizer
