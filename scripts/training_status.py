"""Check training status — no GPU needed.

Reads checkpoint metadata and training manifests to give a quick snapshot of
every model size that has been trained, including loss trends and estimated
time remaining. No model weights are loaded — this runs instantly on CPU.

Usage:
    python scripts/training_status.py                   # show all model sizes
    python scripts/training_status.py --size tiny       # tiny checkpoints only
    python scripts/training_status.py --checkpoints-dir /path/to/checkpoints
    python scripts/training_status.py --no-curve        # skip the ASCII loss plot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Make sure the package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity: e^loss."""
    try:
        return math.exp(loss)
    except (OverflowError, ValueError):
        return float("inf")


def _read_metadata(path: Path) -> dict:
    """Read metadata.json from a checkpoint directory."""
    try:
        return json.loads((path / "metadata.json").read_text())
    except Exception:
        return {}


def _read_manifest(path: Path) -> dict:
    """Read training_manifest.yaml if it exists. Returns {} on failure."""
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _fmt_pct(value: float | None) -> str:
    """Format a 0–1 fraction as a percentage string."""
    if value is None:
        return "?"
    return f"{value * 100:.1f}%"


def _loss_trend_label(loss_history: dict) -> str:
    """Summarise the recent loss trend from a loss_history dict.

    loss_history maps step labels (e.g. "step_1000") to loss values.
    Looks at the last 5 entries; returns "improving", "stagnating", or "degrading".
    """
    if not loss_history:
        return "unknown"

    # Sort by the numeric part of the step key
    try:
        sorted_items = sorted(
            loss_history.items(),
            key=lambda kv: int("".join(filter(str.isdigit, kv[0])) or "0"),
        )
    except Exception:
        sorted_items = list(loss_history.items())

    values = [float(v) for _, v in sorted_items[-5:]]
    if len(values) < 2:
        return "not enough data"

    # Linear regression slope over the tail
    n = len(values)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = numerator / denominator

    # Thresholds tuned for typical LM loss scales (roughly 0.001 per step is meaningful)
    if slope < -0.001:
        return "improving"
    elif slope > 0.001:
        return "degrading"
    else:
        return "stagnating"


def _ascii_loss_curve(loss_history: dict, n: int = 10, bar_width: int = 40) -> list[str]:
    """Render an ASCII bar chart of the last n loss values.

    Returns a list of strings (one per line).

    Example:
        Loss Trend (last 10 checkpoints):
          10.4 |████████████████████
           8.2 |███████████████
           5.6 |██████████
    """
    if not loss_history:
        return []

    try:
        sorted_items = sorted(
            loss_history.items(),
            key=lambda kv: int("".join(filter(str.isdigit, kv[0])) or "0"),
        )
    except Exception:
        sorted_items = list(loss_history.items())

    tail = sorted_items[-n:]
    if not tail:
        return []

    values = [float(v) for _, v in tail]
    max_val = max(values) or 1.0

    lines: list[str] = [f"  Loss Trend (last {len(tail)} checkpoints):"]
    for _, (step_label, loss_val) in zip(range(len(tail)), tail):
        loss_f = float(loss_val)
        bar_len = max(1, round(loss_f / max_val * bar_width))
        bar = "\u2588" * bar_len  # full block character
        lines.append(f"  {loss_f:6.2f} {chr(0x2502)}{bar}")  # │

    return lines


# ---------------------------------------------------------------------------
# Per-size summary
# ---------------------------------------------------------------------------

def _describe_size(size_dir: Path, show_curve: bool = True) -> None:
    """Print training status for one model size (e.g. checkpoints/tiny/)."""

    # Collect all step_* checkpoint dirs, sorted by step number
    step_dirs = sorted(
        [d for d in size_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )

    if not step_dirs:
        cli.dim(f"  No checkpoints found in {size_dir}")
        return

    # Read latest metadata
    latest_dir = step_dirs[-1]
    meta = _read_metadata(latest_dir)
    manifest = _read_manifest(size_dir / "training_manifest.yaml")

    step = meta.get("step", 0)
    loss = meta.get("loss")
    config = meta.get("config", {})
    training_cfg = config.get("training", {})
    max_steps = training_cfg.get("max_steps") or (
        manifest.get("progress", {}).get("total_steps")
    )

    ppl = _perplexity(float(loss)) if loss is not None else None
    progress = (step / max_steps) if (max_steps and step) else None

    # Estimate training time remaining (tokens/sec from manifest if available)
    progress_section = manifest.get("progress", {})
    tokens_seen = progress_section.get("tokens_seen", 0)

    # Build the kv table
    table: dict[str, str] = {
        "Checkpoints saved": str(len(step_dirs)),
        "Latest step": f"{step:,}" if step else "?",
        "Latest loss": f"{loss:.4f}" if loss is not None else "?",
        "Perplexity": f"{ppl:.2f}" if ppl is not None else "?",
        "Progress": _fmt_pct(progress) + (f"  ({step:,} / {max_steps:,} steps)" if max_steps else ""),
    }

    # Manifest-enriched fields
    if tokens_seen:
        if tokens_seen >= 1e9:
            table["Tokens seen"] = f"{tokens_seen / 1e9:.2f}B"
        elif tokens_seen >= 1e6:
            table["Tokens seen"] = f"{tokens_seen / 1e6:.1f}M"
        else:
            table["Tokens seen"] = f"{tokens_seen:,}"

    epochs_completed = progress_section.get("epochs_completed")
    if epochs_completed is not None:
        table["Epochs completed"] = f"{float(epochs_completed):.3f}"

    best_loss = progress_section.get("best_loss")
    best_step = progress_section.get("best_step")
    if best_loss is not None:
        table["Best loss"] = f"{best_loss:.4f}" + (f"  (step {best_step:,})" if best_step else "")

    loss_history = progress_section.get("loss_history", {})
    if loss_history:
        table["Loss trend"] = _loss_trend_label(loss_history)

    cli.kv_table(table, title=f"Model size: {size_dir.name}")

    # ASCII loss curve
    if show_curve and loss_history:
        curve_lines = _ascii_loss_curve(loss_history, n=10)
        for line in curve_lines:
            cli.print(line)
        cli.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show training status for cola-coder checkpoints (no GPU needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--size",
        default=None,
        metavar="SIZE",
        help="Only show checkpoints for this model size (e.g. tiny, small, medium). "
             "Defaults to showing all sizes.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        metavar="DIR",
        help="Base checkpoints directory (default: checkpoints).",
    )
    parser.add_argument(
        "--no-curve",
        action="store_true",
        help="Skip the ASCII loss curve.",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Training Status")

    checkpoints_root = Path(args.checkpoints_dir)
    if not checkpoints_root.exists():
        cli.fatal(
            f"Checkpoints directory not found: {checkpoints_root}",
            hint="Run training first, or pass --checkpoints-dir with the correct path.",
        )

    # Discover model-size subdirectories
    size_dirs = sorted(
        [d for d in checkpoints_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
    )

    if not size_dirs:
        cli.warn(f"No model-size directories found under {checkpoints_root}.")
        cli.dim("Expected layout: checkpoints/<size>/step_XXXXXXXX/")
        sys.exit(0)

    # Apply --size filter
    if args.size:
        size_dirs = [d for d in size_dirs if d.name == args.size]
        if not size_dirs:
            cli.fatal(
                f"No checkpoints found for size {args.size!r}.",
                hint=f"Available sizes: {[d.name for d in sorted(checkpoints_root.iterdir()) if d.is_dir()]}",
            )

    found_any = False
    for size_dir in size_dirs:
        step_dirs = [d for d in size_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
        if not step_dirs:
            continue  # skip empty size dirs (e.g. only has a "latest" pointer)
        found_any = True
        _describe_size(size_dir, show_curve=not args.no_curve)

    if not found_any:
        cli.warn("No checkpoints found.")
        cli.dim("Train first with:  python scripts/train.py --config configs/tiny.yaml")
        sys.exit(0)

    cli.done("Status check complete — no models were loaded.")


if __name__ == "__main__":
    main()
