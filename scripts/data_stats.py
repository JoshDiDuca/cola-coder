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
from cola_coder.data.stats import compute_data_stats  # noqa: E402


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

    # Shared library — the same numbers the web UI's /api/data-stats reports.
    try:
        stats = compute_data_stats(
            args.data, args.weights, estimate_unique=not args.no_unique,
            search_root=_PROJECT_ROOT,
        )
    except FileNotFoundError:
        cli.error(
            "Training data not found.",
            hint="Run prepare_data.py first, or pass --data <path>",
        )
        return 1
    except ImportError:
        cli.error("numpy is required", hint="pip install numpy")
        return 1
    except Exception as exc:
        cli.error(f"Failed to load data: {exc}")
        return 1

    cli.info("Data file", stats.data_path)
    cli.info("File size", f"{stats.file_size_mb:.1f} MB")

    cli.print("")
    cli.print("[bold cyan]Token Statistics[/bold cyan]")
    if stats.seq_len is not None:
        cli.info("Shape", f"{stats.num_chunks:,} chunks × {stats.seq_len:,} tokens/chunk")
    else:
        cli.info("Shape", str(tuple(stats.shape)))
    cli.info("Total tokens", f"{stats.total_tokens:,} ({stats.total_tokens / 1e6:.2f}M)")
    cli.info("Token range", f"min={stats.token_min:,}  max={stats.token_max:,}")
    cli.info("Token mean", f"{stats.token_mean:.2f}")
    if stats.est_unique_tokens is not None:
        cli.info("Est. unique tokens", f"~{stats.est_unique_tokens:,}")

    if stats.has_weights:
        cli.print("")
        cli.print("[bold cyan]Quality Score Distribution[/bold cyan]")
        cli.info("Weights file", str(stats.weights_path))
        for tier in stats.weight_tiers:
            cli.info(tier.label, f"{tier.count:>10,}  ({tier.pct:5.1f}%)")
        if stats.weight_mean is not None:
            cli.info("Weight mean", f"{stats.weight_mean:.4f}")
        if stats.weight_std is not None:
            cli.info("Weight std ", f"{stats.weight_std:.4f}")
    else:
        cli.print("")
        cli.dim("No weights sidecar found (run prepare_data.py --score to generate quality weights)")

    cli.print("")
    cli.success("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
