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

    # ---- Validate inputs ----
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Training data not found: {data_path}")
        print("  Prepare data first with: python scripts/prepare_data.py")
        sys.exit(1)

    if args.resume and not Path(args.resume).exists():
        print(f"Error: Checkpoint not found: {args.resume}")
        sys.exit(1)

    # ---- Device check ----
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: No CUDA GPU detected. Training on CPU will be extremely slow.")
        print("  A CUDA-capable GPU is strongly recommended for training.")

    # ---- Load config ----
    print(f"\nLoading config from {config_path}...")

    try:
        from cola_coder.model.config import Config
    except ImportError:
        print("Error: Could not import cola_coder. Make sure the package is installed.")
        print("  Try: pip install -e .")
        sys.exit(1)

    try:
        config = Config.from_yaml(str(config_path))
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # ---- Initialize trainer ----
    print("Initializing trainer...")

    try:
        from cola_coder.training.trainer import Trainer
    except ImportError:
        print("Error: Could not import training module.")
        sys.exit(1)

    try:
        trainer = Trainer(config=config, resume_from=args.resume)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"\nGPU Error: {e}")
            print("\nSuggestions to reduce VRAM usage:")
            print("  1. Reduce batch_size in your config")
            print("  2. Enable gradient_checkpointing: true in your config")
            print("  3. Use a smaller model config")
            sys.exit(1)
        raise

    # ---- Start training ----
    print(f"\nTraining data: {data_path}")
    if args.wandb:
        print("Weights & Biases logging: ENABLED")

    try:
        trainer.train(data_path=str(data_path), use_wandb=args.wandb)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("You can resume from the latest checkpoint with --resume")
        sys.exit(0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nGPU out of memory during training: {e}")
            print("\nSuggestions:")
            print("  1. Reduce batch_size in your config")
            print("  2. Increase gradient_accumulation (and decrease batch_size proportionally)")
            print("  3. Enable gradient_checkpointing: true")
            sys.exit(1)
        raise

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
