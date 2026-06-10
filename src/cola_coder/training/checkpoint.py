"""Checkpoint saving and loading.

Checkpoints save the complete training state so you can:
1. Resume training after a crash or power loss
2. Load a trained model for inference
3. Fine-tune from a checkpoint (e.g., for reasoning experiments)

We use safetensors format instead of PyTorch's default pickle format because:
- Pickle can execute arbitrary code when loading (security risk)
- Safetensors is a simple binary format that only stores tensors
- It's faster to load and save

For a TS dev: think of pickle like eval() and safetensors like JSON.parse().
"""

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file, load_file

from ..manifest import write_training_manifest


def _maybe_resize_vocab(model: torch.nn.Module, state_dict: dict) -> None:
    """Resize model embeddings to match checkpoint vocab size.

    Reasoning training adds thinking tokens (<think>/<\think>), expanding the
    vocabulary from e.g. 32768 → 32770. When the model is later reconstructed
    from its YAML config it uses the original vocab_size, causing a size mismatch
    on load. This function detects the discrepancy and resizes the embedding and
    output layers before load_state_dict is called — the actual weights come from
    the checkpoint, so the new rows are just placeholder shells.

    Handles torch.compile by operating on the inner _orig_mod when present.
    """
    if "tok_emb.weight" not in state_dict:
        return
    ckpt_vocab = int(state_dict["tok_emb.weight"].shape[0])
    inner = getattr(model, "_orig_mod", model)
    if not hasattr(inner, "config") or ckpt_vocab == inner.config.vocab_size:
        return
    dim = int(state_dict["tok_emb.weight"].shape[1])
    # Create new layers on the same device as the existing embedding so that
    # load_state_dict doesn't strand them on CPU when the model is already on CUDA.
    emb_device = inner.tok_emb.weight.device
    inner.tok_emb = torch.nn.Embedding(ckpt_vocab, dim).to(emb_device)
    # Resize output projection and re-tie weights (same tensor as tok_emb)
    inner.output = torch.nn.Linear(dim, ckpt_vocab, bias=False).to(emb_device)
    inner.output.weight = inner.tok_emb.weight
    inner.config.vocab_size = ckpt_vocab


def _load_state_dict_tied(model: torch.nn.Module, state_dict: dict) -> None:
    """load_state_dict with strict validation, allowing only the tied output head.

    Checkpoints intentionally omit ``output.weight`` (it shares its tensor
    with ``tok_emb.weight``), so plain ``strict=True`` would always fail.
    But a blanket ``strict=False`` silently ignores EVERY mismatch — a
    renamed or corrupted key would leave parts of the model randomly
    initialized with no error. This helper accepts exactly the expected
    missing key and raises on anything else.
    """
    result = model.load_state_dict(state_dict, strict=False)

    allowed_missing = {"output.weight", "_orig_mod.output.weight"}
    unexpected_missing = [k for k in result.missing_keys if k not in allowed_missing]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint does not match model architecture.\n"
            f"  Missing from checkpoint: {unexpected_missing}\n"
            f"  Unexpected in checkpoint: {list(result.unexpected_keys)}\n"
            f"  Use the config that matches this checkpoint, or pick a "
            f"compatible checkpoint directory."
        )


class _ConfigEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclass objects and other non-serializable types."""

    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    loss: float,
    config: dict,
    output_dir: str,
    max_checkpoints: int = 5,
    *,
    data_path: str | None = None,
    tokenizer_path: str | None = None,
    manifest_info: dict | None = None,
) -> str:
    """Save a training checkpoint.

    Saves three files:
    - model.safetensors: model weights (safe format)
    - training_state.pt: optimizer + scheduler state (pickle, but just numbers)
    - metadata.json: step number, loss, config

    Also creates/updates training_manifest.yaml in the output directory
    with full provenance info when manifest_info is provided.

    Args:
        model: The model to save.
        optimizer: Optimizer state (momentum, etc.).
        scheduler: LR scheduler state.
        step: Current training step.
        loss: Current loss value.
        config: Training configuration dict.
        output_dir: Base directory for checkpoints.
        max_checkpoints: Keep only this many most recent checkpoints.
        manifest_info: Optional dict of training provenance metadata.
            Expected keys: model_config, training_config, data_path,
            data_manifest_path, tokens_seen, epochs_completed,
            loss_history, max_steps.

    Returns:
        Path to the saved checkpoint directory.
    """
    # Save to a temp directory first, then rename — this makes the save atomic.
    # If we crash mid-write, only the temp dir is corrupted, not a real checkpoint.
    final_dir = Path(output_dir) / f"step_{step:08d}"
    tmp_dir = Path(output_dir) / f".tmp_step_{step:08d}"

    # Clean up any previous failed temp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights using safetensors
    # Filter out tied weights — output.weight shares memory with tok_emb.weight
    # (weight tying). safetensors refuses duplicate tensors, so we skip the alias.
    # torch.compile wraps keys with "_orig_mod." prefix, so we strip that too
    # to keep checkpoint format consistent regardless of compilation.
    raw_state = model.state_dict()
    state_dict = {}
    for k, v in raw_state.items():
        # Strip torch.compile prefix for consistent checkpoint format
        clean_key = k.removeprefix("_orig_mod.")
        if clean_key == "output.weight":
            continue  # Skip — it's the same tensor as tok_emb.weight
        state_dict[clean_key] = v.contiguous()
    save_file(state_dict, str(tmp_dir / "model.safetensors"))

    # Save optimizer and scheduler state
    # These are just numbers (momentum buffers, LR values), not code
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "rng_state": torch.random.get_rng_state(),
        },
        tmp_dir / "training_state.pt",
    )

    # Save metadata as JSON (human-readable)
    metadata = {
        "step": step,
        "loss": loss,
        "config": config,
        "data_path": data_path,
        "tokenizer_path": tokenizer_path,
    }
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, cls=_ConfigEncoder))

    # Atomic rename: tmp -> final (if final already exists, replace it)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    ckpt_dir = final_dir

    # Write/update training manifest
    if manifest_info is not None:
        manifest_path = Path(output_dir) / "training_manifest.yaml"
        write_training_manifest(
            manifest_path,
            step=step,
            loss=loss,
            checkpoint_path=str(ckpt_dir),
            **manifest_info,
        )

    # Also save as "latest" symlink for easy access
    latest_path = Path(output_dir) / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    # On Windows, use a text file with the path instead of symlink
    latest_path.write_text(str(ckpt_dir))

    # Clean up old checkpoints (keep only max_checkpoints most recent).
    # Pass the just-saved directory so cleanup never deletes it, even if its
    # step number is lower than existing checkpoints (e.g. fresh run in a dir
    # that already has high-step checkpoints from a previous run).
    _cleanup_old_checkpoints(output_dir, max_checkpoints, protected=str(ckpt_dir))

    print(f"Checkpoint saved: {ckpt_dir}")
    return str(ckpt_dir)


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str = "cuda",
) -> int:
    """Load a checkpoint and restore model/optimizer/scheduler state.

    Args:
        checkpoint_dir: Path to checkpoint directory (or "latest" file path).
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to load tensors onto.

    Returns:
        The training step number from the checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)

    # Handle "latest" pointer
    if ckpt_dir.name == "latest" and ckpt_dir.is_file():
        ckpt_dir = Path(ckpt_dir.read_text().strip())

    print(f"Loading checkpoint from {ckpt_dir}...")

    # Validate checkpoint is complete (not a partial save from a crash)
    model_path = ckpt_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Incomplete checkpoint at {ckpt_dir} — model.safetensors is missing. "
            f"This usually means a previous save crashed mid-write. "
            f"Delete this directory and resume from an earlier checkpoint."
        )

    # Load model weights
    # strict=False because we skip saving output.weight (it's tied to tok_emb.weight)
    state_dict = load_file(str(model_path), device=device)

    # Validate architecture before loading — give a clear error instead of
    # PyTorch's cryptic "size mismatch" message.
    metadata_path = ckpt_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        saved_dim = meta.get("config", {}).get("model", {}).get("dim")
        if saved_dim is not None and "tok_emb.weight" in state_dict:
            actual_dim = int(state_dict["tok_emb.weight"].shape[1])
            if saved_dim != actual_dim:
                raise RuntimeError(
                    f"Architecture mismatch: checkpoint has dim={saved_dim} "
                    f"but current model has dim={actual_dim}. "
                    f"Use the matching config for this checkpoint "
                    f"or point --resume at a compatible checkpoint directory."
                )

    # If reasoning training expanded the vocabulary (e.g. added thinking tokens),
    # the checkpoint's tok_emb shape won't match the config-built model. Resize
    # before loading so load_state_dict sees matching shapes.
    _maybe_resize_vocab(model, state_dict)

    # Handle torch.compile: if model is compiled, keys need _orig_mod. prefix
    # Checkpoints always store clean keys (no prefix) for portability.
    if hasattr(model, "_orig_mod"):
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}

    _load_state_dict_tied(model, state_dict)

    step = 0

    # Load training state if optimizer/scheduler provided
    training_state_path = ckpt_dir / "training_state.pt"
    if training_state_path.exists() and (optimizer is not None or scheduler is not None):
        training_state = torch.load(
            training_state_path,
            map_location=device,
            weights_only=True,
        )
        if optimizer is not None:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(training_state["scheduler"])
        step = training_state.get("step", 0)

        # Restore RNG state for reproducibility
        if "rng_state" in training_state:
            rng_state = training_state["rng_state"]
            # set_rng_state requires a CPU ByteTensor, but map_location=device
            # may have loaded it onto GPU — move it back to CPU
            if isinstance(rng_state, torch.Tensor):
                rng_state = rng_state.cpu().to(torch.uint8)
            else:
                try:
                    rng_state = torch.ByteTensor(rng_state)
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not restore RNG state: {e}")
                    rng_state = None
            if rng_state is not None:
                torch.random.set_rng_state(rng_state)

    print(f"Loaded checkpoint at step {step}")
    return step


def load_model_only(
    checkpoint_dir: str,
    model: torch.nn.Module,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load only model weights (for inference, no optimizer state needed).

    Args:
        checkpoint_dir: Path to checkpoint directory.
        model: Model to load weights into.
        device: Device to load onto.

    Returns:
        The model with loaded weights.
    """
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.name == "latest" and ckpt_dir.is_file():
        ckpt_dir = Path(ckpt_dir.read_text().strip())

    state_dict = load_file(str(ckpt_dir / "model.safetensors"), device=device)
    # Expand embeddings if checkpoint vocab differs (e.g. thinking tokens added in reasoning stage)
    _maybe_resize_vocab(model, state_dict)
    if hasattr(model, "_orig_mod"):
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
    # Validated load — only the tied output.weight may be absent
    _load_state_dict_tied(model, state_dict)
    model.eval()  # Set to evaluation mode (disables dropout)
    return model


def get_checkpoint_info(checkpoint_dir: str) -> dict:
    """Read metadata.json from a checkpoint and return the info dict.

    Returns dict with keys: step, loss, config, size_name, checkpoint_dir.
    Returns empty dict if metadata not found.
    """
    try:
        ckpt_dir = Path(checkpoint_dir)

        # Handle "latest" pointer (text file containing actual checkpoint path)
        if ckpt_dir.name == "latest" and ckpt_dir.is_file():
            ckpt_dir = Path(ckpt_dir.read_text().strip())

        metadata_path = ckpt_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        info = json.loads(metadata_path.read_text())
        # size_name is the grandparent dir (e.g. checkpoints/tiny/step_00001000 -> "tiny")
        info["size_name"] = ckpt_dir.parent.name
        info["checkpoint_dir"] = str(ckpt_dir)
        return info
    except Exception:
        return {}


def detect_latest_checkpoint(
    checkpoints_dir: str = "checkpoints",
    model_config: dict | None = None,
) -> tuple[str, dict] | None:
    """Auto-detect the latest checkpoint matching the current model architecture.

    Scans checkpoints/<size>/latest files and returns the most recent one that
    matches the provided model_config (if given). Matching is done on the key
    architecture fields: dim, n_layers, n_heads, n_kv_heads, vocab_size.

    Args:
        checkpoints_dir: Base checkpoints directory.
        model_config: Dict of model config fields (e.g. vars(config.model)).
            When provided, only checkpoints with matching architecture are returned.

    Returns:
        Tuple of (checkpoint_path, metadata_dict) or None if no checkpoints found.
        metadata_dict contains: step, loss, config, data_path (from metadata.json).
    """
    _ARCH_FIELDS = ("dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size")

    def _matches_arch(info: dict) -> bool:
        if model_config is None:
            return True
        saved_model = info.get("config", {}).get("model", {})
        return all(
            saved_model.get(f) == model_config.get(f) for f in _ARCH_FIELDS
        )

    base = Path(checkpoints_dir)
    if not base.exists():
        return None

    best_path: str | None = None
    best_info: dict = {}
    best_step: int = -1

    # Per the project checkpoint rules: resolve by scanning step_* dirs
    # directly — the "latest" pointer file can go stale (e.g. training
    # restarted from scratch in a dir whose old high-step checkpoints were
    # pruned). The pointer is only consulted when a size dir has no step_*
    # dirs at all.
    for size_dir in base.iterdir():
        if not size_dir.is_dir():
            continue

        step_dirs = sorted(
            (d for d in size_dir.glob("step_*") if d.is_dir()),
            key=lambda d: int(d.name.split("_")[1]),
        )
        if step_dirs:
            candidate = str(step_dirs[-1])
        else:
            latest_file = size_dir / "latest"
            if not latest_file.is_file():
                continue
            candidate = str(latest_file)

        info = get_checkpoint_info(candidate)
        if info and _matches_arch(info) and info.get("step", -1) > best_step:
            best_step = info["step"]
            best_path = info.get("checkpoint_dir", candidate)
            best_info = info

    return (best_path, best_info) if best_path is not None else None


def _cleanup_old_checkpoints(
    output_dir: str, max_checkpoints: int, protected: str | None = None
):
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        output_dir: Directory containing step_* checkpoint subdirectories.
        max_checkpoints: Maximum number of checkpoint dirs to keep.
        protected: Absolute path to a checkpoint dir that must never be
            deleted (typically the one that was just saved).  This prevents
            the newly-created checkpoint from being culled when its step
            number happens to be numerically lower than existing ones (e.g.
            when training is restarted from scratch in a directory that
            already contains high-step checkpoints from a previous run).
    """
    ckpt_dirs = sorted(
        [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    while len(ckpt_dirs) > max_checkpoints:
        old_dir = ckpt_dirs.pop(0)
        if protected and str(old_dir) == protected:
            continue  # never delete the checkpoint we just saved
        print(f"Removing old checkpoint: {old_dir}")
        shutil.rmtree(old_dir)
