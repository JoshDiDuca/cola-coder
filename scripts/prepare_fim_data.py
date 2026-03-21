"""Prepare FIM (Fill-in-the-Middle) training data.

Takes an existing preprocessed .npy dataset and produces a FIM-augmented
version where a configurable fraction of samples are rearranged into
PSM/SPM format.

The FIM transform can also be applied on-the-fly during training via
FIMDataset (no pre-processing needed).  Use this script when you want a
static FIM dataset saved to disk — useful for reproducibility or when you
want to inspect the distribution of FIM vs non-FIM examples.

Usage:
    python scripts/prepare_fim_data.py \\
        --input  data/processed/train_data.npy \\
        --output data/processed/train_data_fim.npy \\
        --tokenizer tokenizer.json \\
        --fim-rate 0.5

Options:
    --fim-rate   Fraction of examples transformed [0.0–1.0] (default: 0.5)
    --psm-rate   Fraction of FIM examples using PSM ordering (default: 0.5)
    --seed       Random seed for reproducibility (default: 42)
    --workers    Parallel worker processes for encoding (default: 4)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute a FIM-augmented training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Path to existing preprocessed .npy file (shape: [N, seq_len]).",
    )
    p.add_argument(
        "--output", required=True,
        help="Path for the output .npy file.",
    )
    p.add_argument(
        "--tokenizer", default="tokenizer.json",
        help="Path to tokenizer.json produced by train_tokenizer.py.",
    )
    p.add_argument(
        "--fim-rate", type=float, default=0.5,
        help="Fraction of examples to transform into FIM format [0.0–1.0].",
    )
    p.add_argument(
        "--psm-rate", type=float, default=0.5,
        help="Fraction of FIM examples using PSM ordering (remainder use SPM).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the FIM transform (ensures reproducibility).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Validate inputs ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.fim_rate <= 1.0:
        print(f"ERROR: --fim-rate must be in [0, 1], got {args.fim_rate}", file=sys.stderr)
        sys.exit(1)

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"ERROR: tokenizer not found: {tokenizer_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load tokenizer ---
    print(f"Loading tokenizer from {tokenizer_path} ...")
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    from cola_coder.data.fim import FIMTransform, setup_fim_tokenizer

    tokenizer = CodeTokenizer(str(tokenizer_path))
    fim_ids = setup_fim_tokenizer(tokenizer)
    print(f"  FIM token IDs: {fim_ids}")

    # --- Load input data ---
    print(f"Loading data from {input_path} ...")
    data = np.load(str(input_path), mmap_mode="r")
    n_chunks, seq_len = data.shape
    print(f"  Shape: {n_chunks:,} chunks x {seq_len} tokens")

    # --- Apply FIM transform ---
    transform = FIMTransform(
        fim_rate=args.fim_rate,
        psm_rate=args.psm_rate,
        truncate_or_pad=True,
        seed=args.seed,
    )

    print(
        f"\nApplying FIM transform  "
        f"(fim_rate={args.fim_rate:.0%}, psm_rate={args.psm_rate:.0%}) ..."
    )
    start = time.time()

    # Allocate output array (same shape, uint16 to match preprocess.py)
    out = np.empty((n_chunks, seq_len), dtype=np.uint16)

    n_transformed = 0
    for i in tqdm(range(n_chunks), desc="Applying FIM", unit=" chunks"):
        raw: list[int] = data[i].tolist()
        transformed = transform.apply(raw, tokenizer)
        was_fim = (transformed != raw)
        if was_fim:
            n_transformed += 1
        out[i] = np.array(transformed, dtype=np.uint16)

    elapsed = time.time() - start
    frac = n_transformed / n_chunks if n_chunks > 0 else 0.0

    print("\nResults:")
    print(f"  Chunks transformed: {n_transformed:,} / {n_chunks:,}  ({frac:.1%})")
    print(f"  Wall time: {elapsed:.1f}s")

    # --- Save ---
    np.save(str(output_path), out)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved to {output_path}  ({size_mb:.1f} MB)")

    # Print quick usage tip
    print(
        "\nTo use this dataset in training, pass it as --data to train.py,\n"
        "or use FIMDataset for on-the-fly transforms (no pre-processing needed).\n"
        "\nExample:\n"
        "  .venv/Scripts/python scripts/train.py --config configs/tiny.yaml "
        f"--data {output_path}"
    )


if __name__ == "__main__":
    main()
