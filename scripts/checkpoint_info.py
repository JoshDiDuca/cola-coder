"""Checkpoint info script.

Loads checkpoint metadata.json and prints:
- Model config (arch params, parameter count)
- Training step and loss
- Training time and throughput
- File sizes of all checkpoint files

Also lists all checkpoints found in the checkpoint directory as a table.

Usage:
    python scripts/checkpoint_info.py                               # auto-discover
    python scripts/checkpoint_info.py checkpoints/tiny/latest
    python scripts/checkpoint_info.py checkpoints/tiny/step_00001000
    python scripts/checkpoint_info.py --dir checkpoints/tiny
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cola_coder.cli import cli  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_checkpoint(path_str: str) -> Path:
    """Resolve a checkpoint path, following 'latest' pointer files."""
    p = Path(path_str)
    if p.is_file() and p.name == "latest":
        target = Path(p.read_text().strip())
        return target
    if p.is_dir() and (p / "metadata.json").exists():
        return p
    # If it's a directory without metadata, look for latest inside
    if p.is_dir():
        latest_ptr = p / "latest"
        if latest_ptr.is_file():
            return Path(latest_ptr.read_text().strip())
    return p


def _format_size(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.2f} GB"
    if nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} B"


def _print_checkpoint(ckpt_dir: Path) -> bool:
    """Print detailed info for a single checkpoint directory."""
    metadata_path = ckpt_dir / "metadata.json"
    if not metadata_path.exists():
        cli.warn(f"No metadata.json in {ckpt_dir}")
        return False

    try:
        meta = json.loads(metadata_path.read_text())
    except Exception as exc:
        cli.error(f"Failed to parse metadata.json: {exc}")
        return False

    cli.print(f"\n[bold cyan]Checkpoint:[/bold cyan] {ckpt_dir}")

    # Step / loss
    step = meta.get("step", "?")
    loss = meta.get("loss", "?")
    cli.info("Step", f"{step:,}" if isinstance(step, int) else str(step))
    if isinstance(loss, float):
        cli.info("Loss", f"{loss:.4f}")
    else:
        cli.info("Loss", str(loss))

    # Config
    config = meta.get("config", {})
    if config:
        cli.print("\n[bold]Model config:[/bold]")
        model_cfg = config.get("model", config)  # support flat or nested
        interesting_keys = [
            "vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
            "max_seq_len", "ffn_dim_multiplier",
        ]
        for k in interesting_keys:
            v = model_cfg.get(k)
            if v is not None:
                cli.info(f"  {k}", str(v))

        training_cfg = config.get("training", {})
        if training_cfg:
            cli.print("\n[bold]Training config:[/bold]")
            t_keys = [
                "batch_size", "gradient_accumulation", "learning_rate",
                "precision", "max_steps", "warmup_steps",
            ]
            for k in t_keys:
                v = training_cfg.get(k)
                if v is not None:
                    cli.info(f"  {k}", str(v))

    # File sizes
    cli.print("\n[bold]Files:[/bold]")
    total_bytes = 0
    for fname in sorted(ckpt_dir.iterdir()):
        if fname.is_file():
            nbytes = fname.stat().st_size
            total_bytes += nbytes
            cli.info(f"  {fname.name}", _format_size(nbytes))
    cli.info("  TOTAL", _format_size(total_bytes))

    return True


def _list_checkpoints(base_dir: Path) -> None:
    """Print a summary table of all checkpoints in base_dir."""
    if not base_dir.exists():
        cli.warn(f"Directory not found: {base_dir}")
        return

    step_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: d.name,
    )

    if not step_dirs:
        cli.warn(f"No checkpoints found in {base_dir}")
        return

    cli.print(f"\n[bold cyan]Checkpoints in {base_dir}:[/bold cyan]")
    try:
        from rich.table import Table
        from rich import box
        from rich.console import Console

        console = Console()
        table = Table(box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Checkpoint", style="yellow")
        table.add_column("Step", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Size")

        for d in step_dirs:
            meta_path = d / "metadata.json"
            step_str = loss_str = size_str = "?"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    step_str = f"{meta.get('step', '?'):,}" if isinstance(meta.get("step"), int) else "?"
                    loss_val = meta.get("loss")
                    loss_str = f"{loss_val:.4f}" if isinstance(loss_val, float) else "?"
                except Exception:
                    pass
            total = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            size_str = _format_size(total)
            table.add_row(d.name, step_str, loss_str, size_str)

        console.print(table)
        cli.info("Total checkpoints", str(len(step_dirs)))

    except ImportError:
        # Fallback plain text
        print(f"{'Checkpoint':<30} {'Step':>10} {'Loss':>10} {'Size':>12}")
        print("-" * 65)
        for d in step_dirs:
            meta_path = d / "metadata.json"
            step_str = loss_str = size_str = "?"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    step_str = str(meta.get("step", "?"))
                    loss_val = meta.get("loss")
                    loss_str = f"{loss_val:.4f}" if isinstance(loss_val, float) else "?"
                except Exception:
                    pass
            total = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            size_str = _format_size(total)
            print(f"{d.name:<30} {step_str:>10} {loss_str:>10} {size_str:>12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print info about one or all checkpoints."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to checkpoint dir, 'latest' file, or checkpoint base dir",
    )
    parser.add_argument(
        "--dir",
        default=None,
        metavar="PATH",
        help="Checkpoint base directory to list all checkpoints from",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Checkpoint Info")

    # If --dir given, list all + optionally show latest detail
    if args.dir:
        base = Path(args.dir)
        _list_checkpoints(base)
        # Also show detail for latest
        latest_ptr = base / "latest"
        if latest_ptr.is_file():
            ckpt = _resolve_checkpoint(str(latest_ptr))
            cli.print("\n[bold]Latest checkpoint details:[/bold]")
            _print_checkpoint(ckpt)
        return 0

    # Try to auto-discover if no path given
    target = args.checkpoint
    if target is None:
        # Search common locations
        candidates = list((_PROJECT_ROOT / "checkpoints").glob("*/latest"))
        if candidates:
            target = str(candidates[0])
            cli.info("Auto-discovered", target)
        else:
            cli.error("No checkpoint specified and none found automatically.")
            cli.dim("Usage: checkpoint_info.py [checkpoint_path]  or  --dir checkpoints/tiny")
            return 1

    ckpt_dir = _resolve_checkpoint(target)

    # If the resolved path is a step dir, print it and list siblings
    if ckpt_dir.is_dir() and (ckpt_dir / "metadata.json").exists():
        _print_checkpoint(ckpt_dir)
        _list_checkpoints(ckpt_dir.parent)
        return 0

    cli.error(f"Could not find a valid checkpoint at: {target}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
