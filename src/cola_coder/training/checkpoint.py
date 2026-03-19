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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    loss: float,
    config: dict,
    output_dir: str,
    max_checkpoints: int = 5,
) -> str:
    """Save a training checkpoint.

    Saves three files:
    - model.safetensors: model weights (safe format)
    - training_state.pt: optimizer + scheduler state (pickle, but just numbers)
    - metadata.json: step number, loss, config

    Args:
        model: The model to save.
        optimizer: Optimizer state (momentum, etc.).
        scheduler: LR scheduler state.
        step: Current training step.
        loss: Current loss value.
        config: Training configuration dict.
        output_dir: Base directory for checkpoints.
        max_checkpoints: Keep only this many most recent checkpoints.

    Returns:
        Path to the saved checkpoint directory.
    """
    ckpt_dir = Path(output_dir) / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights using safetensors
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(ckpt_dir / "model.safetensors"))

    # Save optimizer and scheduler state
    # These are just numbers (momentum buffers, LR values), not code
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "rng_state": torch.random.get_rng_state(),
        },
        ckpt_dir / "training_state.pt",
    )

    # Save metadata as JSON (human-readable)
    metadata = {
        "step": step,
        "loss": loss,
        "config": config,
    }
    (ckpt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

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

    # Load model weights
    state_dict = load_file(str(ckpt_dir / "model.safetensors"), device=device)
    model.load_state_dict(state_dict)

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

    state_dict = load_file(str(ckpt_dir / "model.safetensors"), device=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode (disables dropout)
    return model


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
