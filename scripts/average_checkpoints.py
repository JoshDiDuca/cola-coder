"""Average multiple cola-coder checkpoints for better generalization.

Checkpoint averaging (also called model soups / SWA) combines weights from
multiple checkpoints into a single model. The averaged model typically lands
in a flatter region of the loss landscape and generalizes better than any
single checkpoint.

Usage:
    # Average last 3 checkpoints in a directory (uniform mean)
    python scripts/average_checkpoints.py --checkpoint-dir checkpoints/tiny --last 3

    # Average last 5 with EMA (newer checkpoints get more weight)
    python scripts/average_checkpoints.py --checkpoint-dir checkpoints/tiny --last 5 --method ema

    # Average specific checkpoints
    python scripts/average_checkpoints.py \\
        --checkpoints checkpoints/tiny/step_00018000 checkpoints/tiny/step_00019000 checkpoints/tiny/step_00020000 \\
        --method uniform

    # EMA with custom decay
    python scripts/average_checkpoints.py \\
        --checkpoints checkpoints/tiny/step_00018000 checkpoints/tiny/step_00020000 \\
        --method ema --decay 0.9

    # Custom output path
    python scripts/average_checkpoints.py --checkpoint-dir checkpoints/tiny --last 3 \\
        --output checkpoints/tiny/my_averaged_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.training.checkpoint_average import CheckpointAverager


def _file_size_human(path: str) -> str:
    """Return human-readable file size string (e.g. '142.3 MB')."""
    try:
        size = Path(path).stat().st_size
    except OSError:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average cola-coder model checkpoints for better generalization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Checkpoint selection (mutually exclusive) ----
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        help="Directory containing step_* checkpoint subdirectories. "
             "Use with --last to pick the N most recent.",
    )
    selection.add_argument(
        "--checkpoints",
        nargs="+",
        metavar="CKPT",
        help="Explicit list of checkpoint directories to average (oldest first).",
    )

    # ---- Options ----
    parser.add_argument(
        "--last",
        type=int,
        default=3,
        metavar="K",
        help="(Only with --checkpoint-dir) Number of most-recent checkpoints to average. "
             "Default: 3.",
    )
    parser.add_argument(
        "--method",
        choices=["uniform", "ema"],
        default="uniform",
        help="Averaging method. 'uniform' = simple mean, 'ema' = exponential moving average "
             "(newer checkpoints get more weight). Default: uniform.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.999,
        metavar="DECAY",
        help="EMA decay factor — only used when --method ema. "
             "Higher values weight older checkpoints more. Default: 0.999.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Output directory for the averaged checkpoint. "
             "Defaults to <checkpoint-dir>/averaged_last_<k> or "
             "<parent-of-first-ckpt>/averaged_<method>.",
    )

    args = parser.parse_args()

    cli.header("Cola-Coder", "Checkpoint Averaging")

    # ---- Validate args ----
    if args.checkpoint_dir and args.last < 1:
        cli.fatal("--last must be at least 1.")

    if args.method == "ema" and not (0.0 < args.decay < 1.0):
        cli.fatal(f"--decay must be in (0, 1). Got {args.decay}.")

    averager = CheckpointAverager()

    # ---- Run averaging ----
    try:
        if args.checkpoint_dir:
            cli.info("Mode", f"last {args.last} checkpoints from {args.checkpoint_dir!r}")

            # Show which checkpoints exist before averaging
            all_ckpts = CheckpointAverager.find_checkpoints(args.checkpoint_dir)
            if not all_ckpts:
                cli.fatal(
                    f"No step_* checkpoints found in {args.checkpoint_dir!r}.",
                    hint="Check that the directory contains step_XXXXXXXX subdirectories "
                         "with model.safetensors files.",
                )

            if len(all_ckpts) < args.last:
                cli.fatal(
                    f"Requested --last {args.last} but only {len(all_ckpts)} checkpoint(s) found.",
                    hint=f"Available: {[Path(p).name for p in all_ckpts]}",
                )

            selected = all_ckpts[-args.last :]
            cli.rule("Checkpoints to Average")
            for i, ckpt in enumerate(selected, 1):
                role = "(oldest)" if i == 1 else ("(newest)" if i == len(selected) else "")
                cli.print(f"  [cyan]{i}.[/cyan] {Path(ckpt).name}  {role}")

            result = averager.average_last_k(
                checkpoint_dir=args.checkpoint_dir,
                k=args.last,
                output_path=args.output,
                method=args.method,
                decay=args.decay,
            )

        else:
            # Explicit list
            cli.info("Mode", f"explicit {len(args.checkpoints)} checkpoint(s)")

            cli.rule("Checkpoints to Average")
            for i, ckpt in enumerate(args.checkpoints, 1):
                role = "(oldest)" if i == 1 else ("(newest)" if i == len(args.checkpoints) else "")
                cli.print(f"  [cyan]{i}.[/cyan] {Path(ckpt).name}  {role}")

            output_path = args.output
            if output_path is None:
                # Default: sibling directory next to the first checkpoint
                first = Path(args.checkpoints[0])
                output_path = str(first.parent / f"averaged_{args.method}")

            if args.method == "uniform":
                result = averager.uniform_average(args.checkpoints, output_path)
            else:
                result = averager.exponential_moving_average(
                    args.checkpoints, output_path, decay=args.decay
                )

    except FileNotFoundError as exc:
        cli.fatal(str(exc))
    except ValueError as exc:
        cli.fatal(str(exc))

    # ---- Report results ----
    cli.rule("Result")

    out_safetensors = Path(result.output_path) / "model.safetensors"
    file_size = _file_size_human(str(out_safetensors))

    cli.kv_table(
        {
            "Method": result.method,
            "Checkpoints averaged": str(result.num_checkpoints),
            **({"EMA decay": str(args.decay)} if result.method == "ema" else {}),
            "Output path": result.output_path,
            "Output file": str(out_safetensors),
            "File size": file_size,
        },
        title="Averaging Summary",
    )

    cli.success(
        f"Averaged {result.num_checkpoints} checkpoint(s) saved to {result.output_path!r}"
    )
    cli.dim(
        "Load with: load_checkpoint('<output_path>', model) or "
        "load_model_only('<output_path>', model)"
    )
    cli.done("Done.")


if __name__ == "__main__":
    main()
