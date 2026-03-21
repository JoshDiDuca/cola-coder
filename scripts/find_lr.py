"""Learning Rate Finder script.

Runs Smith's LR Range Test on a small model + data subset to suggest a good
learning rate before you start a full training run. Much faster than guessing!

Usage:
    python scripts/find_lr.py --config configs/tiny.yaml --tokenizer tokenizer.json
    python scripts/find_lr.py --config configs/small.yaml --tokenizer tokenizer.json \\
        --data ./data/processed/train_data.npy --num-steps 200 --start-lr 1e-7 --end-lr 10

The script will:
  1. Load a fresh model (or checkpoint with --resume)
  2. Run the LR range test for --num-steps steps
  3. Display results with Rich formatting
  4. Save a plot to lr_finder_plot.png (if matplotlib is installed)
  5. Print the suggested LR range for use in your config YAML

Typical output:
    Suggested LR range:
      learning_rate: 3.0e-4    (max_lr — use this in your config)
      min_lr:        3.0e-5    (10x lower — use as cosine schedule floor)
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on the path when running via `python scripts/...`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from cola_coder.cli import cli
from cola_coder.model.config import Config
from cola_coder.model.transformer import Transformer
from cola_coder.training.lr_finder import LRFinder
from cola_coder.training.optimizer import create_optimizer


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Find optimal learning rate using Smith's LR Range Test."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--tokenizer", default=None,
        help="Path to tokenizer.json (used to validate vocab size)."
    )
    parser.add_argument("--data", default=None, help="Path to preprocessed .npy data file.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint dir to start from.")
    parser.add_argument(
        "--start-lr", type=float, default=1e-7, help="Minimum LR to test (default: 1e-7)."
    )
    parser.add_argument(
        "--end-lr", type=float, default=10.0, help="Maximum LR to test (default: 10.0)."
    )
    parser.add_argument(
        "--num-steps", type=int, default=200,
        help="Number of LR steps to run (default: 200)."
    )
    parser.add_argument(
        "--smooth", type=float, default=0.05,
        help="Exponential smoothing factor for loss curve (default: 0.05)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config (useful to reduce memory use)."
    )
    parser.add_argument(
        "--save-plot", default="lr_finder_plot.png",
        help="Path to save the plot image (default: lr_finder_plot.png)."
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation entirely."
    )
    parser.add_argument(
        "--diverge-threshold", type=float, default=4.0,
        help="Stop early if loss exceeds this multiple of min loss (default: 4.0)."
    )
    return parser.parse_args()


def _load_data(args, config: Config) -> torch.utils.data.DataLoader:
    """Load training data, scanning for available datasets if needed."""
    from cola_coder.model.config import get_storage_config
    from cola_coder.data.dataset import create_dataloader

    data_path = args.data
    if data_path is None:
        # Auto-scan for processed data
        storage = get_storage_config()
        processed_dir = Path(storage.data_dir) / "processed"
        npy_files = sorted(processed_dir.glob("*.npy")) if processed_dir.exists() else []
        npy_files = [f for f in npy_files if not f.name.endswith("_tmp.npy")]

        if not npy_files:
            cli.fatal(
                "No training data found.",
                hint=(
                    "Prepare data first: python scripts/prepare_data.py "
                    "--config configs/tiny.yaml --tokenizer tokenizer.json"
                ),
            )
        elif len(npy_files) == 1:
            data_path = str(npy_files[0])
            cli.info("Auto-detected data", data_path)
        else:
            # Multiple datasets found — use the first one, warn user
            data_path = str(npy_files[0])
            cli.warn(
                f"Multiple datasets found, using {npy_files[0].name}. "
                f"Pass --data to specify one explicitly."
            )
    else:
        if not Path(data_path).exists():
            cli.fatal(f"Data file not found: {data_path}")

    batch_size = args.batch_size or config.training.batch_size

    # Use fewer workers during the LR find — we only run ~200 steps
    loader = create_dataloader(
        data_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(config.data.num_workers, 4),
        max_seq_len=config.model.max_seq_len,
    )
    return loader


def main():
    args = _parse_args()

    cli.header("LR Finder", "Smith's Learning Rate Range Test")

    # ── Load config ──────────────────────────────────────────────────────────
    cli.step(1, 4, "Loading config and building model")

    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}")

    config = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cli.info("Device", device)
    cli.info("Config", args.config)

    # ── Build model ──────────────────────────────────────────────────────────
    model = Transformer(config.model).to(device)
    cli.info("Model parameters", f"{model.num_parameters:,}")

    if args.resume:
        from cola_coder.training.checkpoint import load_checkpoint
        cli.info("Resuming from", args.resume)
        load_checkpoint(args.resume, model, device=device)

    optimizer = create_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    cli.step(2, 4, "Loading training data")
    loader = _load_data(args, config)
    cli.info("Batch size", str(args.batch_size or config.training.batch_size))

    # ── Run LR finder ─────────────────────────────────────────────────────────
    cli.step(3, 4, f"Running LR range test ({args.num_steps} steps)")
    cli.info("LR range", f"{args.start_lr:.1e} → {args.end_lr:.1e}")

    finder = LRFinder(model=model, optimizer=optimizer, device=device)

    result = finder.find(
        train_loader=loader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_steps=args.num_steps,
        smooth_factor=args.smooth,
        diverge_threshold=args.diverge_threshold,
        gradient_accumulation=config.training.gradient_accumulation,
    )

    # ── Display results ───────────────────────────────────────────────────────
    cli.step(4, 4, "Results")

    cli.print("")
    cli.print(result.summary())
    cli.print("")

    min_lr, max_lr = LRFinder.suggest_lr(result)

    cli.print("[bold green]Suggested config YAML:[/bold green]")
    cli.print("  training:")
    cli.print(f"    learning_rate: {max_lr:.2e}    # suggested max LR")
    cli.print(f"    min_lr:        {min_lr:.2e}    # suggested floor (max_lr / 10)")
    cli.print("")

    if result.diverged_early:
        cli.warn(
            f"Run diverged early at step {result.num_steps_run}. "
            f"Loss exceeded {args.diverge_threshold}x the minimum. "
            f"The suggested LR is still valid — it's taken from the stable region."
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        save_path = args.save_plot if args.save_plot else None
        finder.plot(result, save_path=save_path)

    cli.done(
        "LR Finder complete",
        extras={
            "Suggested LR": f"{max_lr:.2e}",
            "Suggested min LR": f"{min_lr:.2e}",
            "Steps run": str(result.num_steps_run),
        },
    )


if __name__ == "__main__":
    main()
