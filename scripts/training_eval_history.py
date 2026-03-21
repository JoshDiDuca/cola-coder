"""Show auto-evaluation history from a training run.

Reads the auto-eval state saved inside checkpoint metadata and renders:
- A table of evaluation snapshots (step, pass@1, pass@5, trend marker)
- The best result highlighted
- An ASCII chart of pass@1 over training steps

Requires no GPU — all information is read from checkpoint metadata files.

Usage:
    python scripts/training_eval_history.py --checkpoint-dir checkpoints/tiny
    python scripts/training_eval_history.py --checkpoint-dir checkpoints/tiny --no-chart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.training.auto_eval import EvalSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_eval_history(checkpoint_dir: Path) -> list[EvalSnapshot]:
    """Load auto-eval snapshots from checkpoint metadata files.

    Scans all ``step_*/metadata.json`` files for an ``auto_eval`` key and
    merges the history lists.  Also falls back to the training manifest YAML
    if present.

    Returns a deduplicated list of EvalSnapshot objects sorted by step.
    """
    snapshots: dict[int, EvalSnapshot] = {}  # step -> snapshot (dedup)

    # Walk step_* checkpoint dirs
    step_dirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )

    for step_dir in step_dirs:
        meta_path = step_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        ae_state = meta.get("auto_eval")
        if not ae_state:
            continue

        for d in ae_state.get("history", []):
            try:
                snap = EvalSnapshot.from_dict(d)
                snapshots[snap.step] = snap
            except Exception:
                pass

    # Also look for a standalone auto_eval_history.json that might exist
    history_file = checkpoint_dir / "auto_eval_history.json"
    if history_file.exists():
        try:
            raw = json.loads(history_file.read_text())
            for d in raw:
                snap = EvalSnapshot.from_dict(d)
                snapshots[snap.step] = snap
        except Exception:
            pass

    return sorted(snapshots.values(), key=lambda s: s.step)


def _ascii_chart(snapshots: list[EvalSnapshot], bar_width: int = 40) -> list[str]:
    """Render a simple ASCII bar chart of pass@1 over steps.

    Each bar's length is proportional to the pass@1 score.

    Example output:
        pass@1 over training:
          step   1000  0.0% |
          step   5000  5.0% |████
          step  10000 12.5% |██████████
    """
    if not snapshots:
        return []

    max_score = max((s.pass_at_1 for s in snapshots), default=0.0)
    if max_score <= 0.0:
        max_score = 1.0  # avoid divide-by-zero if all zeros

    lines: list[str] = ["", "  pass@1 over training steps:", ""]
    for snap in snapshots:
        bar_len = max(0, round(snap.pass_at_1 / max_score * bar_width))
        bar = "\u2588" * bar_len  # full-block character
        best_marker = " *" if snap.is_best else "  "
        lines.append(
            f"  step {snap.step:>8,d}  {snap.pass_at_1:>5.1%} \u2502{bar}{best_marker}"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show auto-eval history from a cola-coder training run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        metavar="DIR",
        help="Path to the checkpoint directory (e.g. checkpoints/tiny).",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip the ASCII pass@1 chart.",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Auto-Eval History")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        cli.fatal(
            f"Checkpoint directory not found: {checkpoint_dir}",
            hint="Pass --checkpoint-dir checkpoints/<size>",
        )

    snapshots = _load_eval_history(checkpoint_dir)

    if not snapshots:
        cli.warn("No auto-eval history found in this checkpoint directory.")
        cli.dim(
            "Auto-eval history is saved into metadata.json at each checkpoint. "
            "Run training with an AutoEvaluator configured to populate this."
        )
        sys.exit(0)

    # ---- Summary table ----
    cli.print(
        f"\n  Found [bold]{len(snapshots)}[/bold] evaluation snapshot(s) "
        f"in [cyan]{checkpoint_dir}[/cyan]\n"
    )

    # Build the table manually so it works with or without Rich
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box as rich_box

        console = Console()
        table = Table(
            title="Auto-Eval Snapshots",
            box=rich_box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Step", style="bold", justify="right")
        table.add_column("pass@1", justify="right")
        table.add_column("pass@5", justify="right")
        table.add_column("N", justify="right")
        table.add_column("Gen/prob", justify="right")
        table.add_column("Timestamp", style="dim")
        table.add_column("", width=5)  # best marker / regression

        # Compute trend per snapshot (sliding window of 3)
        def _trend_arrow(idx: int) -> str:
            if idx < 1:
                return ""
            prev = snapshots[idx - 1].pass_at_1
            curr = snapshots[idx].pass_at_1
            diff = curr - prev
            if diff > 0.005:
                return "[green]▲[/green]"
            elif diff < -0.005:
                return "[red]▼[/red]"
            return "[dim]=[/dim]"

        best_step = max(snapshots, key=lambda s: s.pass_at_1).step if snapshots else 0

        for i, snap in enumerate(snapshots):
            is_best = snap.step == best_step
            row_style = "bold" if is_best else ""
            best_tag = "[bold yellow]*[/bold yellow]" if is_best else ""
            table.add_row(
                f"{snap.step:,}",
                f"{snap.pass_at_1:.1%}",
                f"{snap.pass_at_5:.1%}",
                str(snap.num_problems),
                f"{snap.avg_generation_time:.2f}s",
                snap.timestamp,
                f"{best_tag}{_trend_arrow(i)}",
                style=row_style,
            )

        console.print(table)

        # Summary row
        best_snap = max(snapshots, key=lambda s: s.pass_at_1)
        console.print(
            f"\n  Best:  step [bold]{best_snap.step:,}[/bold]"
            f"  pass@1 [bold green]{best_snap.pass_at_1:.1%}[/bold green]"
            f"  pass@5 [green]{best_snap.pass_at_5:.1%}[/green]"
        )

    except ImportError:
        # Plain text fallback
        header = f"  {'Step':>10}  {'pass@1':>8}  {'pass@5':>8}  {'N':>5}  {'Gen/prob':>9}  Timestamp"
        print("\n" + header)
        print("  " + "-" * (len(header) - 2))
        best_step = max(snapshots, key=lambda s: s.pass_at_1).step if snapshots else 0
        for snap in snapshots:
            best_tag = " *" if snap.step == best_step else "  "
            print(
                f"  {snap.step:>10,d}  {snap.pass_at_1:>7.1%}  {snap.pass_at_5:>7.1%}"
                f"  {snap.num_problems:>5d}  {snap.avg_generation_time:>8.2f}s"
                f"  {snap.timestamp}{best_tag}"
            )
        best_snap = max(snapshots, key=lambda s: s.pass_at_1)
        print(
            f"\n  Best: step {best_snap.step:,}  pass@1 {best_snap.pass_at_1:.1%}"
            f"  pass@5 {best_snap.pass_at_5:.1%}"
        )

    # ---- ASCII chart ----
    if not args.no_chart:
        chart_lines = _ascii_chart(snapshots)
        for line in chart_lines:
            cli.print(line)

    cli.done("Eval history shown — no models were loaded.")


if __name__ == "__main__":
    main()
