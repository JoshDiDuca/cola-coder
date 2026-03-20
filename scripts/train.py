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
        "--auto-resume",
        action="store_true",
        help="Auto-detect and resume from the latest checkpoint.",
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

    # ---- Auto-resume: detect latest checkpoint ----
    resume_from = args.resume
    if args.auto_resume and not resume_from:
        try:
            from cola_coder.training.checkpoint import detect_latest_checkpoint
            result = detect_latest_checkpoint()
            if result is not None:
                checkpoint_path, checkpoint_info = result
                step = checkpoint_info.get("step", "?")
                cli.info("Auto-resume", f"Found checkpoint at step {step}: {checkpoint_path}")
                resume_from = checkpoint_path
            else:
                cli.warn("Auto-resume: no checkpoints found, starting fresh")
        except ImportError:
            cli.warn("Auto-resume: checkpoint module not available, starting fresh")
        except Exception as e:
            cli.warn(f"Auto-resume failed: {e}, starting fresh")

    # ---- Pre-flight Checks ----
    cli.rule("Pre-flight Checks")

    # Pre-flight: validate config (optional feature)
    try:
        from cola_coder.features.config_validator import is_enabled as config_validator_enabled
        if config_validator_enabled():
            from cola_coder.features.config_validator import validate_config, ValidationIssue
            issues = validate_config(config)
            if issues:
                errors = [i for i in issues if i.level == "error"]
                warnings = [i for i in issues if i.level == "warning"]
                cli.warn(f"Config validation: {len(errors)} error(s), {len(warnings)} warning(s)")
                for issue in issues[:5]:  # Show first 5
                    prefix = "ERROR" if issue.level == "error" else "WARN"
                    cli.dim(f"  [{prefix}] [{issue.field}]: {issue.message}")
                    if issue.suggestion:
                        cli.dim(f"    Suggestion: {issue.suggestion}")
                if errors:
                    if not cli.confirm("Config has errors. Continue anyway?"):
                        sys.exit(0)
            else:
                cli.success("Config validation passed")
    except ImportError:
        pass  # Feature not available
    except Exception as e:
        cli.warn(f"Config validation skipped: {e}")

    # Pre-flight: VRAM estimation (optional feature)
    try:
        from cola_coder.features.vram_estimator import is_enabled as vram_enabled
        if vram_enabled():
            from cola_coder.features.vram_estimator import estimate_vram
            estimate = estimate_vram(
                model_config=config.model,
                training_config=config.training,
            )
            cli.info("Estimated VRAM", f"{estimate.total_training_gb:.1f} GB (training)")
            if estimate.gpu_vram_gb is not None:
                if estimate.fits_training:
                    cli.success(f"VRAM fits on {estimate.gpu_name} ({estimate.gpu_vram_gb:.1f} GB available)")
                elif estimate.fits_training is False:
                    cli.warn(
                        f"VRAM may not fit: {estimate.total_training_gb:.1f} GB estimated "
                        f"> {estimate.gpu_vram_gb:.1f} GB available on {estimate.gpu_name}"
                    )
                    cli.dim("  Tip: reduce batch_size, enable gradient_checkpointing, or use a smaller config")
    except ImportError:
        pass
    except Exception as e:
        cli.warn(f"VRAM estimation skipped: {e}")

    # ---- Initialize trainer ----
    cli.step(2, 3, "Initializing trainer")

    try:
        from cola_coder.training.trainer import Trainer
    except ImportError:
        cli.fatal("Could not import training module.")

    try:
        trainer = Trainer(config=config, resume_from=resume_from)
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
