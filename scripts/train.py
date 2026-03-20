"""Main training script for the cola-coder model.

Loads configuration, builds the model, and runs the training loop with
mixed precision, gradient accumulation, and checkpointing.

Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/small.yaml --data ./data/processed/train_data.npy --wandb
    python scripts/train.py --config configs/small.yaml --resume ./checkpoints/step_00005000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cola_coder.cli import cli


def _format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def _scan_datasets(data_dir: str = "./data/processed") -> list[dict]:
    """Scan for available .npy dataset files and return metadata."""
    out_path = Path(data_dir)
    if not out_path.exists():
        return []

    datasets = []
    for f in sorted(out_path.glob("*.npy")):
        if f.name.endswith("_tmp.npy"):
            continue  # Skip temp files
        stat = f.stat()
        try:
            arr = np.load(str(f), mmap_mode="r")
            chunks, seq_len = arr.shape
            token_count = chunks * seq_len
            detail = f"{chunks:,} chunks x {seq_len} = {token_count:,} tokens"
        except Exception:
            detail = "unknown format"

        datasets.append({
            "name": f.stem,
            "path": str(f),
            "size": _format_size(stat.st_size),
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "detail": detail,
        })
    return datasets


def _pick_dataset(explicit_path: str | None) -> str:
    """Resolve the training data path, interactively if needed.

    If --data is explicitly passed and the file exists, use it.
    If not, scan for available datasets and let the user choose.
    """
    # Explicit path provided
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p)
        cli.fatal(
            f"Training data not found: {p}",
            hint="Prepare data first with: python scripts/prepare_data.py",
        )

    # Auto-scan for datasets
    datasets = _scan_datasets()

    if not datasets:
        cli.fatal(
            "No training data found in ./data/processed/",
            hint="Prepare data first with: python scripts/prepare_data.py",
        )

    if len(datasets) == 1:
        # Only one dataset — use it automatically
        ds = datasets[0]
        cli.info("Training data", f"{ds['name']}.npy ({ds['size']}, {ds['detail']})")
        return ds["path"]

    # Multiple datasets — let user choose
    cli.file_table("Available Datasets", datasets)

    options = []
    for ds in datasets:
        options.append({
            "label": f"{ds['name']}.npy",
            "detail": f"{ds['size']}  |  {ds['date']}  |  {ds['detail']}",
        })

    choice = cli.choose("Which dataset to train on?", options, allow_cancel=True)

    if choice is None:
        cli.dim("Cancelled.")
        sys.exit(0)

    ds = datasets[choice]
    cli.info("Training data", f"{ds['name']}.npy ({ds['size']})")
    return ds["path"]


def main():
    parser = argparse.ArgumentParser(
        description="Train the cola-coder model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (required).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to preprocessed training data .npy file. "
             "If not set, scans ./data/processed/ and lets you choose.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Model Training")

    # ---- Validate config ----
    config_path = Path(args.config)
    if not config_path.exists():
        cli.fatal(f"Config file not found: {config_path}")

    if args.resume and not Path(args.resume).exists():
        cli.fatal(f"Checkpoint not found: {args.resume}")

    # ---- Device check ----
    device = cli.gpu_info()

    # ---- Pick dataset (interactive if multiple exist) ----
    data_path = _pick_dataset(args.data)

    # ---- Load config ----
    cli.step(1, 3, f"Loading config from {config_path}")

    try:
        from cola_coder.model.config import Config
    except ImportError:
        cli.fatal(
            "Could not import cola_coder. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    try:
        config = Config.from_yaml(str(config_path))
    except Exception as e:
        cli.fatal(f"Loading config: {e}")

    # ---- Initialize trainer ----
    cli.step(2, 3, "Initializing trainer")

    try:
        from cola_coder.training.trainer import Trainer
    except ImportError:
        cli.fatal("Could not import training module.")

    try:
        trainer = Trainer(config=config, resume_from=args.resume)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            cli.error(f"GPU Error: {e}")
            cli.warn("Suggestions to reduce VRAM usage:")
            cli.dim("  1. Reduce batch_size in your config")
            cli.dim("  2. Enable gradient_checkpointing: true in your config")
            cli.dim("  3. Use a smaller model config")
            sys.exit(1)
        raise

    # ---- Start training ----
    cli.step(3, 3, "Starting training")
    cli.info("Training data", data_path)
    if args.wandb:
        cli.info("W&B logging", "ENABLED")

    try:
        trainer.train(data_path=str(data_path), use_wandb=args.wandb)
    except KeyboardInterrupt:
        cli.warn("Training interrupted by user.")
        cli.dim("You can resume from the latest checkpoint with --resume")
        sys.exit(0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            cli.error(f"GPU out of memory during training: {e}")
            cli.warn("Suggestions:")
            cli.dim("  1. Reduce batch_size in your config")
            cli.dim("  2. Increase gradient_accumulation (and decrease batch_size proportionally)")
            cli.dim("  3. Enable gradient_checkpointing: true")
            sys.exit(1)
        raise

    cli.success("Training complete!")


if __name__ == "__main__":
    main()
