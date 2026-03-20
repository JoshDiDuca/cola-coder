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

import json
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from ..manifest import write_training_manifest


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
    import shutil
    final_dir = Path(output_dir) / f"step_{step:08d}"
    tmp_dir = Path(output_dir) / f".tmp_step_{step:08d}"

    # Clean up any previous failed temp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights using safetensors
    # Filter out tied weights — output.weight shares memory with tok_emb.weight
    # (weight tying). safetensors refuses duplicate tensors, so we skip the alias.
    # On load, we re-tie them in the model constructor.
    state_dict = {}
    for k, v in model.state_dict().items():
        if k == "output.weight":
            continue  # Skip — it's the same tensor as tok_emb.weight
        state_dict[k] = v.contiguous()
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
    }
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

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

    # Clean up old checkpoints (keep only max_checkpoints most recent)
    _cleanup_old_checkpoints(output_dir, max_checkpoints)

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
    model.load_state_dict(state_dict, strict=False)

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
            torch.random.set_rng_state(training_state["rng_state"])

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

    # strict=False because output.weight is tied to tok_emb.weight and not saved separately
    state_dict = load_file(str(ckpt_dir / "model.safetensors"), device=device)
    model.load_state_dict(state_dict, strict=False)
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


def detect_latest_checkpoint(checkpoints_dir: str = "checkpoints") -> tuple[str, dict] | None:
    """Auto-detect the latest checkpoint across all model sizes.

    Scans checkpoints/<size>/latest files and returns the most recent one.

    Args:
        checkpoints_dir: Base checkpoints directory.

    Returns:
        Tuple of (checkpoint_path, metadata_dict) or None if no checkpoints found.
        metadata_dict contains: step, loss, config (from metadata.json).
    """
    base = Path(checkpoints_dir)
    if not base.exists():
        return None

    best_path: str | None = None
    best_info: dict = {}
    best_step: int = -1

    # First pass: scan for "latest" pointer files
    found_latest = False
    for size_dir in base.iterdir():
        if not size_dir.is_dir():
            continue
        latest_file = size_dir / "latest"
        if latest_file.is_file():
            found_latest = True
            info = get_checkpoint_info(str(latest_file))
            if info and info.get("step", -1) > best_step:
                best_step = info["step"]
                best_path = info["checkpoint_dir"]
                best_info = info

    if found_latest:
        return (best_path, best_info) if best_path is not None else None

    # Fallback: scan for step_* dirs directly if no "latest" files exist
    for size_dir in base.iterdir():
        if not size_dir.is_dir():
            continue
        step_dirs = sorted(
            size_dir.glob("step_*"),
            key=lambda d: int(d.name.split("_")[1]),
        )
        if not step_dirs:
            continue
        # Take the highest step dir for this size
        highest = step_dirs[-1]
        info = get_checkpoint_info(str(highest))
        if info and info.get("step", -1) > best_step:
            best_step = info["step"]
            best_path = str(highest)
            best_info = info

    return (best_path, best_info) if best_path is not None else None


def _cleanup_old_checkpoints(output_dir: str, max_checkpoints: int):
    """Remove old checkpoints, keeping only the most recent ones."""
    ckpt_dirs = sorted(
        [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    while len(ckpt_dirs) > max_checkpoints:
        old_dir = ckpt_dirs.pop(0)
        print(f"Removing old checkpoint: {old_dir}")
        for f in old_dir.iterdir():
            f.unlink()
        old_dir.rmdir()
