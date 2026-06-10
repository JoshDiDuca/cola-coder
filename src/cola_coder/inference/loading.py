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

    Mutates ``config.model.moe`` (enabled + expert counts) in place so the
    Transformer constructed from it builds expert FFNs matching the
    checkpoint's weights. No-op for dense checkpoints.

    Returns:
        True if the config was switched to MoE, else False.
    """
    from cola_coder.features.moe_layer import detect_moe_checkpoint

    detected = detect_moe_checkpoint(checkpoint)
    if not detected:
        return False
    moe = config.model.moe
    moe.enabled = True
    moe.num_experts = detected["num_experts"]
    moe.num_shared_experts = detected["num_shared_experts"]
    moe.top_k = detected["top_k"]
    # Upcycling converts every FFN block, so the whole stack is MoE.
    moe.moe_layers = "all"
    return True


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
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if tokenizer_path is None:
        tokenizer_path = resolve_tokenizer_path(checkpoint, config_path)
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config.from_yaml(config_path)

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
