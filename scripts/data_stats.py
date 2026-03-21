"""Data statistics script.

Loads .npy training data and prints summary statistics:
- Number of chunks and total token count
- Min / max / mean / std of token values
- Estimated unique token count (via reservoir sampling)
- If weights.npy sidecar exists: quality score distribution

Usage:
    python scripts/data_stats.py                              # auto-discover data/processed/
    python scripts/data_stats.py --data data/processed/train_data.npy
    python scripts/data_stats.py --data data/processed/train_data.npy --weights data/processed/train_data.weights.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cola_coder.cli import cli  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_data_file(hint: str | None) -> Path | None:
    """Return path to training data, searching common locations."""
    if hint:
        p = Path(hint)
        return p if p.exists() else None
    candidates = [
        _PROJECT_ROOT / "data" / "processed" / "train_data.npy",
        _PROJECT_ROOT / "data" / "train_data.npy",
        _PROJECT_ROOT / "train_data.npy",
    ]
    # Also check storage.yaml redirect
    try:
        from cola_coder.model.config import get_storage_config

        sc = get_storage_config()
        if sc and hasattr(sc, "data_dir"):
            candidates.insert(0, Path(sc.data_dir) / "train_data.npy")
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_weights_file(data_path: Path, hint: str | None) -> Path | None:
    """Return path to weights sidecar, if present."""
    if hint:
        p = Path(hint)
        return p if p.exists() else None
    # Convention: train_data.npy -> train_data.weights.npy
    weights_path = data_path.parent / (data_path.stem + ".weights.npy")
    return weights_path if weights_path.exists() else None


def _estimate_unique_tokens(arr, sample_size: int = 500_000) -> int:
    """Estimate number of unique token IDs using a random sample."""
    import numpy as np

    if len(arr) <= sample_size:
        return int(np.unique(arr).size)
    # Reservoir-style: sample a flat slice
    indices = np.random.choice(arr.size, size=sample_size, replace=False)
    sample = arr.flat[indices]
    unique_in_sample = np.unique(sample).size
    # Scale up by ratio (rough estimate)
    scale = arr.size / sample_size
    estimated = min(int(unique_in_sample * (scale**0.5)), 2**20)
    return estimated


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print statistics about .npy training data."
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help="Path to train_data.npy (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="PATH",
        help="Path to weights.npy sidecar (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--no-unique",
        action="store_true",
        help="Skip (slow) unique-token estimation",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Data Statistics")

    # Locate data file
    data_path = _find_data_file(args.data)
    if data_path is None:
        cli.error(
            "Training data not found.",
            hint="Run prepare_data.py first, or pass --data <path>",
        )
        return 1

    cli.info("Data file", str(data_path))
    file_size_mb = data_path.stat().st_size / (1024**2)
    cli.info("File size", f"{file_size_mb:.1f} MB")

    # Load
    try:
        import numpy as np
    except ImportError:
        cli.error("numpy is required", hint="pip install numpy")
        return 1

    try:
        arr = np.load(str(data_path), mmap_mode="r")
    except Exception as exc:
        cli.error(f"Failed to load data: {exc}")
        return 1

    # Basic shape info
    cli.print("")
    cli.print("[bold cyan]Token Statistics[/bold cyan]")
    shape = arr.shape
    if arr.ndim == 2:
        num_chunks, seq_len = shape
        total_tokens = num_chunks * seq_len
        cli.info("Shape", f"{num_chunks:,} chunks × {seq_len:,} tokens/chunk")
    else:
        total_tokens = arr.size
        num_chunks = 1
        cli.info("Shape", str(shape))

    cli.info("Total tokens", f"{total_tokens:,} ({total_tokens / 1e6:.2f}M)")

    # Value stats
    try:
        flat = arr.reshape(-1)
        min_val = int(flat.min())
        max_val = int(flat.max())
        mean_val = float(flat.mean())
        cli.info("Token range", f"min={min_val:,}  max={max_val:,}")
        cli.info("Token mean", f"{mean_val:.2f}")
    except Exception as exc:
        cli.warn(f"Could not compute value stats: {exc}")

    # Unique token estimate
    if not args.no_unique:
        try:
            est_unique = _estimate_unique_tokens(arr)
            cli.info("Est. unique tokens", f"~{est_unique:,}")
        except Exception as exc:
            cli.warn(f"Could not estimate unique tokens: {exc}")

    # Weights sidecar
    weights_path = _find_weights_file(data_path, args.weights)
    if weights_path:
        cli.print("")
        cli.print("[bold cyan]Quality Score Distribution[/bold cyan]")
        cli.info("Weights file", str(weights_path))
        try:
            weights = np.load(str(weights_path), mmap_mode="r")
            w_flat = weights.reshape(-1).astype(float)
            tiers = [
                ("excellent (0.8+)", w_flat >= 0.8),
                ("good    (0.6–0.8)", (w_flat >= 0.6) & (w_flat < 0.8)),
                ("average (0.4–0.6)", (w_flat >= 0.4) & (w_flat < 0.6)),
                ("poor    (0.2–0.4)", (w_flat >= 0.2) & (w_flat < 0.4)),
                ("reject  (<0.2)", w_flat < 0.2),
            ]
            for label, mask in tiers:
                count = int(mask.sum())
                pct = 100.0 * count / max(len(w_flat), 1)
                cli.info(label, f"{count:>10,}  ({pct:5.1f}%)")
            cli.info("Weight mean", f"{w_flat.mean():.4f}")
            cli.info("Weight std ", f"{w_flat.std():.4f}")
        except Exception as exc:
            cli.warn(f"Could not load weights: {exc}")
    else:
        cli.print("")
        cli.dim("No weights sidecar found (run prepare_data.py --score to generate quality weights)")

    cli.print("")
    cli.success("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
