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
from pathlib import Path

import torch

from cola_coder.cli import cli


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
        default="./data/processed/train_data.npy",
        help="Path to preprocessed training data .npy file (default: ./data/processed/train_data.npy).",
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

    # ---- Validate inputs ----
    config_path = Path(args.config)
    if not config_path.exists():
        cli.fatal(f"Config file not found: {config_path}")

    data_path = Path(args.data)
    if not data_path.exists():
        cli.fatal(
            f"Training data not found: {data_path}",
            hint="Prepare data first with: python scripts/prepare_data.py",
        )

    if args.resume and not Path(args.resume).exists():
        cli.fatal(f"Checkpoint not found: {args.resume}")

    # ---- Device check ----
    device = cli.gpu_info()

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
