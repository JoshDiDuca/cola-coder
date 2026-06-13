"""Main training script for the cola-coder model.

Loads configuration, builds the model, and runs the training loop with
mixed precision, gradient accumulation, and checkpointing.

Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/small.yaml --data ./data/processed/train_data.npy --wandb
    python scripts/train.py --config configs/small.yaml --resume latest
    python scripts/train.py --config configs/small.yaml --resume ./checkpoints/step_00005000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


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


def _scan_datasets(data_dir: str | None = None) -> list[dict]:
    """Scan for available .npy dataset files and return metadata.

    Scans both the legacy 'processed/' directory AND per-dataset directories
    (e.g. 'typescript-text-math/') under storage.data_dir.
    """
    storage = get_storage_config()
    base_dir = Path(data_dir) if data_dir else Path(storage.data_dir)

    # Collect all directories to scan
    scan_dirs: list[Path] = []

    # Legacy path: data/processed/
    legacy = base_dir / "processed"
    if legacy.exists():
        scan_dirs.append(legacy)

    # Per-dataset directories: data/<dataset-name>/ (e.g. typescript-text-math)
    if base_dir.exists():
        for child in sorted(base_dir.iterdir()):
            if child.is_dir() and child.name != "processed" and child.name != "raw":
                # Only include dirs that have .npy files
                if any(child.glob("*.npy")):
                    scan_dirs.append(child)

    # Also scan explicit data_dir if provided directly
    if data_dir and Path(data_dir).exists() and Path(data_dir) not in scan_dirs:
        scan_dirs.append(Path(data_dir))

    datasets: list[dict] = []
    seen_paths: set[str] = set()

    for scan_path in scan_dirs:
        for f in sorted(scan_path.glob("*.npy")):
            if f.name.endswith("_tmp.npy"):
                continue
            if ".weights" in f.name or ".scores" in f.name:
                continue  # Skip sidecar files
            fstr = str(f.resolve())
            if fstr in seen_paths:
                continue
            seen_paths.add(fstr)

            stat = f.stat()
            try:
                arr = np.load(str(f), mmap_mode="r")
                chunks, seq_len = arr.shape
                token_count = chunks * seq_len
                detail = f"{chunks:,} chunks x {seq_len} = {token_count:,} tokens"
            except Exception:
                detail = "unknown format"

            # Check for weights sidecar
            weights_path = f.with_suffix(".weights.npy")
            has_weights = weights_path.exists()

            # Include parent dir name for clarity
            parent = f.parent.name

            datasets.append({
                "name": f"{parent}/{f.stem}" if parent != "processed" else f.stem,
                "path": str(f),
                "size": _format_size(stat.st_size),
                "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "detail": detail + (" [scored]" if has_weights else ""),
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
        storage = get_storage_config()
        cli.fatal(
            f"No training data found in {storage.data_dir}",
            hint="Prepare data first with: python scripts/prepare_data.py\n"
                 "  or collect data with: python scripts/collect_data.py --config <config>",
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
        help="Path to checkpoint directory, or 'latest' to auto-detect.",
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
    parser.add_argument(
        "--auto-eval",
        action="store_true",
        help="Run a small HumanEval pass@k check every --eval-every steps during "
             "training (regression detection; best model saved to <ckpt>/best). "
             "Best-effort — failures warn and never crash training.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=5000,
        help="Steps between auto-eval runs (only used with --auto-eval).",
    )
    parser.add_argument(
        "--eval-subset", type=int, default=20,
        help="HumanEval problems sampled per auto-eval run (only with --auto-eval).",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Model Training")

    # ---- Validate config ----
    config_path = Path(args.config)
    if not config_path.exists():
        cli.fatal(f"Config file not found: {config_path}")

    # Resolve "latest" to an actual checkpoint path
    if args.resume and args.resume.lower() == "latest":
        args.auto_resume = True
        args.resume = None
    elif args.resume and not Path(args.resume).exists():
        cli.fatal(f"Checkpoint not found: {args.resume}")

    # ---- Device check ----
    cli.gpu_info()

    # ---- Load config (must happen before auto-resume so we can filter by arch) ----
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

    # ---- Auto-resume: detect latest checkpoint in THIS run's own output dir ----
    # Restricted to config.checkpoint.output_dir so auto-resume can never latch
    # onto a different run's checkpoint, and the candidate's architecture is
    # validated (dim/layers/heads/vocab/qk_norm/moe) before resuming — a mismatch
    # is refused with a clear warning rather than crashing in _load_state_dict_tied.
    resume_from = args.resume
    saved_data_path: str | None = None
    if args.auto_resume and not resume_from:
        try:
            from cola_coder.training.checkpoint import find_resume_checkpoint
            result = find_resume_checkpoint(
                config.checkpoint.output_dir, model_config=vars(config.model)
            )
            if result is not None:
                checkpoint_path, checkpoint_info = result
                step = checkpoint_info.get("step", "?")
                cli.info("Auto-resume", f"Found checkpoint at step {step}: {checkpoint_path}")
                resume_from = checkpoint_path
                saved_data_path = checkpoint_info.get("data_path")
            else:
                cli.warn("Auto-resume: no compatible checkpoint found, starting fresh")
        except ImportError:
            cli.warn("Auto-resume: checkpoint module not available, starting fresh")
        except Exception as e:
            cli.warn(f"Auto-resume failed: {e}, starting fresh")

    # ---- Pick dataset — skip prompt if restored from checkpoint ----
    if saved_data_path and Path(saved_data_path).exists():
        cli.info("Restored dataset", saved_data_path)
        data_path = saved_data_path
    else:
        data_path = _pick_dataset(args.data)

    # ---- Pre-flight Checks ----
    cli.rule("Pre-flight Checks")

    # Pre-flight: validate config (optional feature)
    try:
        from cola_coder.features.config_validator import is_enabled as config_validator_enabled
        if config_validator_enabled():
            from cola_coder.features.config_validator import validate_config
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

    # Optional opt-in during-training auto-eval. Builds a HumanEval regression
    # monitor + loads the tokenizer it needs; disabled gracefully if the
    # tokenizer can't be resolved (auto-eval is best-effort telemetry, and a
    # failure here must never block a real training run).
    auto_evaluator = None
    eval_tokenizer = None
    if args.auto_eval:
        try:
            from cola_coder.training.auto_eval import AutoEvaluator
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
            from cola_coder.data.dataset_resolver import DatasetResolver

            eval_tokenizer = CodeTokenizer(str(DatasetResolver.get_tokenizer_path()))
            auto_evaluator = AutoEvaluator(
                eval_every_steps=args.eval_every,
                eval_subset=args.eval_subset,
                checkpoint_dir=config.checkpoint.output_dir,
            )
            cli.info("Auto-eval", f"every {args.eval_every:,} steps, {args.eval_subset} problems")
        except Exception as exc:
            cli.warn(f"--auto-eval requested but could not initialise ({exc}); disabled.")
            auto_evaluator = None
            eval_tokenizer = None

    try:
        trainer.train(
            data_path=str(data_path),
            use_wandb=args.wandb,
            auto_evaluator=auto_evaluator,
            tokenizer=eval_tokenizer,
        )
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
