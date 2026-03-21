"""Standalone training dashboard — monitor a running training session.

Reads checkpoint metadata and an optional JSON log file to show real-time
training progress from a separate terminal, without interfering with the
training process itself.

Usage:
    # Watch the latest checkpoint in the default directory
    python scripts/training_dashboard.py

    # Watch a specific model size
    python scripts/training_dashboard.py --checkpoint-dir checkpoints/medium

    # Watch a JSON log file for live updates (written by trainer)
    python scripts/training_dashboard.py --log-file checkpoints/medium/train.log

    # One-shot status (no refresh loop)
    python scripts/training_dashboard.py --once

The script polls for updates every 2 seconds by default.
Press Ctrl-C to exit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Make sure the package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.training.dashboard import TrainingDashboard, ascii_chart, get_gpu_stats


# ---------------------------------------------------------------------------
# Checkpoint reading helpers
# ---------------------------------------------------------------------------

def _read_checkpoint_metadata(checkpoint_dir: Path) -> dict:
    """Read metadata.json from the latest step checkpoint."""
    try:
        # Look for step_* directories, take the latest
        step_dirs = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
        )
        if not step_dirs:
            return {}
        latest = step_dirs[-1]
        meta_file = latest / "metadata.json"
        if meta_file.exists():
            return json.loads(meta_file.read_text())
    except Exception:
        pass
    return {}


def _read_manifest(checkpoint_dir: Path) -> dict:
    """Read training_manifest.yaml if present."""
    manifest_path = checkpoint_dir / "training_manifest.yaml"
    if not manifest_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(manifest_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _parse_log_file(log_path: Path, max_lines: int = 500) -> list[dict]:
    """Parse a JSONL training log file.

    Each line should be a JSON object with at least: step, loss, lr.
    Returns a list of metric dicts, oldest first.
    """
    if not log_path.exists():
        return []
    entries = []
    try:
        lines = log_path.read_text().splitlines()
        for line in lines[-max_lines:]:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "step" in entry and "loss" in entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return entries


# ---------------------------------------------------------------------------
# Static display (one-shot, no Live)
# ---------------------------------------------------------------------------

def _show_static(checkpoint_dir: Path, log_file: Path | None) -> None:
    """Show a one-shot status page without a live refresh loop."""
    cli.header("Cola-Coder", "Training Dashboard")

    meta = _read_checkpoint_metadata(checkpoint_dir)
    manifest = _read_manifest(checkpoint_dir)

    step = meta.get("step", 0)
    loss = meta.get("loss")
    config_block = meta.get("config", {})
    training_cfg = config_block.get("training", {})
    model_cfg = config_block.get("model", {})
    max_steps = training_cfg.get("max_steps", 0)

    progress_section = manifest.get("progress", {})
    loss_history_raw = progress_section.get("loss_history", {})
    tokens_seen = progress_section.get("tokens_seen", 0)
    epochs = progress_section.get("epochs_completed", 0)

    pct = step / max_steps * 100 if max_steps else 0

    info: dict[str, str] = {
        "Checkpoint dir": str(checkpoint_dir),
        "Step": f"{step:,}" + (f" / {max_steps:,} ({pct:.1f}%)" if max_steps else ""),
        "Loss": f"{loss:.4f}" if loss is not None else "?",
    }
    if loss is not None:
        try:
            info["Perplexity"] = f"{math.exp(float(loss)):.2f}"
        except Exception:
            pass
    if tokens_seen:
        info["Tokens seen"] = f"{tokens_seen / 1e9:.2f}B" if tokens_seen >= 1e9 else f"{tokens_seen / 1e6:.1f}M"
    if epochs:
        info["Epochs"] = f"{float(epochs):.3f}"
    if model_cfg:
        dim = model_cfg.get("dim", "?")
        n_layers = model_cfg.get("n_layers", "?")
        info["Architecture"] = f"dim={dim}, layers={n_layers}"

    cli.kv_table(info, title="Training Status")

    # GPU status
    gpu = get_gpu_stats()
    if gpu.get("available"):
        cli.print(f"\n  [cyan]GPU:[/cyan] {gpu['name']}  "
                  f"VRAM {gpu['memory_used_gb']:.1f} / {gpu['memory_total_gb']:.1f} GB")

    # Loss curve from checkpoint history
    if loss_history_raw:
        try:
            sorted_items = sorted(
                loss_history_raw.items(),
                key=lambda kv: int("".join(filter(str.isdigit, kv[0])) or "0"),
            )
            values = [float(v) for _, v in sorted_items[-50:]]
            cli.print("\n[bold cyan]Loss Curve (checkpoint history):[/bold cyan]")
            cli.print(ascii_chart(values, width=40, height=6))
        except Exception:
            pass

    # Loss curve from log file
    if log_file is not None:
        log_entries = _parse_log_file(log_file)
        if log_entries:
            log_losses = [e["loss"] for e in log_entries[-50:]]
            cli.print(f"\n[bold cyan]Loss Curve (log file — last {len(log_losses)} steps):[/bold cyan]")
            cli.print(ascii_chart(log_losses, width=40, height=6))

    cli.print()


# ---------------------------------------------------------------------------
# Live watch mode
# ---------------------------------------------------------------------------

def _watch_live(checkpoint_dir: Path, log_file: Path | None, refresh_secs: float = 2.0) -> None:
    """Poll checkpoint dir and/or log file and refresh the dashboard."""
    cli.header("Cola-Coder", "Training Dashboard (Live)")
    cli.dim(f"Watching: {checkpoint_dir}  (Ctrl-C to exit)")
    cli.print()

    # Read initial checkpoint data to bootstrap config
    meta = _read_checkpoint_metadata(checkpoint_dir)
    config_block = meta.get("config", {})
    training_cfg = config_block.get("training", {})
    model_cfg = config_block.get("model", {})

    dashboard_config = {
        "model_params": model_cfg.get("dim", 0) * model_cfg.get("n_layers", 0) * 12
                        if model_cfg.get("dim") else None,
        "batch_size": training_cfg.get("batch_size", "?"),
        "effective_batch_size": (
            (training_cfg.get("batch_size", 1) or 1)
            * (training_cfg.get("gradient_accumulation", 1) or 1)
        ),
        "learning_rate": training_cfg.get("learning_rate"),
        "seq_len": model_cfg.get("max_seq_len"),
        "precision": training_cfg.get("precision", "?"),
        "model_size_name": checkpoint_dir.name,
    }

    total_steps = training_cfg.get("max_steps", 1) or 1
    dashboard = TrainingDashboard(config=dashboard_config, total_steps=total_steps)

    # Pre-populate from checkpoint loss history
    manifest = _read_manifest(checkpoint_dir)
    progress_section = manifest.get("progress", {})
    loss_history_raw = progress_section.get("loss_history", {})
    if loss_history_raw:
        try:
            sorted_items = sorted(
                loss_history_raw.items(),
                key=lambda kv: int("".join(filter(str.isdigit, kv[0])) or "0"),
            )
            for step_label, loss_val in sorted_items:
                step_num = int("".join(filter(str.isdigit, step_label)) or "0")
                dashboard.update(
                    step=step_num,
                    loss=float(loss_val),
                    lr=training_cfg.get("learning_rate", 0),
                    throughput=0.0,
                )
        except Exception:
            pass

    dashboard.start()

    try:
        last_log_size = 0
        last_step_from_log = -1

        while True:
            # Re-read checkpoint metadata for latest step + loss
            new_meta = _read_checkpoint_metadata(checkpoint_dir)
            new_step = new_meta.get("step", 0)
            new_loss = new_meta.get("loss")
            new_cfg = new_meta.get("config", {})
            new_lr = new_cfg.get("training", {}).get("learning_rate", 0)

            # Check log file for new entries (more granular updates)
            log_updated = False
            if log_file is not None and log_file.exists():
                try:
                    current_size = log_file.stat().st_size
                    if current_size != last_log_size:
                        last_log_size = current_size
                        entries = _parse_log_file(log_file)
                        for entry in entries:
                            if entry.get("step", 0) > last_step_from_log:
                                last_step_from_log = entry["step"]
                                dashboard.update(
                                    step=entry["step"],
                                    loss=entry.get("loss", 0),
                                    lr=entry.get("lr", 0),
                                    throughput=entry.get("throughput", 0),
                                    gpu_mem_gb=entry.get("gpu_mem_gb", 0),
                                    grad_norm=entry.get("grad_norm", 0),
                                )
                                log_updated = True
                except Exception:
                    pass

            # Fall back to checkpoint-level update if no log entries
            if not log_updated and new_loss is not None and new_step != dashboard._current_step:
                gpu = get_gpu_stats()
                gpu_mem = gpu.get("memory_used_gb", 0.0) if gpu.get("available") else 0.0
                dashboard.update(
                    step=new_step,
                    loss=float(new_loss),
                    lr=float(new_lr),
                    throughput=0.0,
                    gpu_mem_gb=gpu_mem,
                )
            elif not log_updated:
                # Just refresh GPU stats even if no new step
                gpu = get_gpu_stats()
                gpu_mem = gpu.get("memory_used_gb", 0.0) if gpu.get("available") else 0.0
                if gpu_mem > 0 and dashboard._current_step > 0:
                    dashboard.update(
                        step=dashboard._current_step,
                        loss=list(dashboard.metrics_history["loss"])[-1] if dashboard.metrics_history["loss"] else 0,
                        lr=list(dashboard.metrics_history["lr"])[-1] if dashboard.metrics_history["lr"] else 0,
                        throughput=list(dashboard.metrics_history["throughput"])[-1] if dashboard.metrics_history["throughput"] else 0,
                        gpu_mem_gb=gpu_mem,
                    )

            time.sleep(refresh_secs)

    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
        cli.print("\n[dim]Dashboard stopped.[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time training dashboard for cola-coder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/medium",
        metavar="DIR",
        help="Path to the checkpoint directory to monitor (default: checkpoints/medium).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="FILE",
        help="Optional JSONL log file to watch for real-time step updates.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Show status once and exit (no live refresh).",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        metavar="SECS",
        help="Refresh interval in seconds (default: 2.0).",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        cli.error(
            f"Checkpoint directory not found: {checkpoint_dir}",
            hint="Run training first, or pass --checkpoint-dir with the correct path.",
        )
        sys.exit(1)

    log_file = Path(args.log_file) if args.log_file else None

    if args.once:
        _show_static(checkpoint_dir, log_file)
    else:
        _watch_live(checkpoint_dir, log_file, refresh_secs=args.refresh)


if __name__ == "__main__":
    main()
