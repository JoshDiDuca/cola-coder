"""Generate training data for the domain router.

Uses the heuristic domain detector to auto-label existing code samples,
creating (code, domain) pairs for router training.

Usage:
    python scripts/generate_router_data.py --source data/processed/train_data.npy
    python scripts/generate_router_data.py --source-dir ./repos/
    python scripts/generate_router_data.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config
from cola_coder.features.router_data_generator import RouterDataGenerator


def main() -> None:
    storage = get_storage_config()

    parser = argparse.ArgumentParser(description="Generate router training data")
    parser.add_argument(
        "--source", type=str,
        help="Path to .npy training data file",
    )
    parser.add_argument(
        "--source-dir", type=str,
        help="Directory of source code files to scan",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic bootstrap data (no source needed)",
    )
    parser.add_argument(
        "--output", type=str, default="data/router_training_data.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=storage.tokenizer_path,
        help="Tokenizer path (for .npy decoding)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50000,
        help="Maximum samples to generate",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3,
        help="Minimum domain detection confidence",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Router Data Generator")

    gen = RouterDataGenerator(
        min_confidence=args.min_confidence,
        max_samples_per_domain=args.max_samples // 7,
    )

    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            cli.error(f"Source file not found: {source_path}")
            sys.exit(1)
        if source_path.suffix == ".npy":
            gen.generate_from_npy(
                str(source_path), args.tokenizer, args.output, args.max_samples,
            )
        else:
            cli.error(f"Unsupported source format: {source_path.suffix}")
            sys.exit(1)
    elif args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            cli.error(f"Source directory not found: {source_dir}")
            sys.exit(1)
        gen.generate_from_files(str(source_dir), args.output)
    elif args.synthetic:
        gen.generate_synthetic(args.output, num_per_domain=500)
    else:
        cli.error("Specify --source, --source-dir, or --synthetic")
        sys.exit(1)

    cli.success(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
