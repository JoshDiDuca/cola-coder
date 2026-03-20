"""Auto Validation Split: automatically hold out data for validation.

Splits .npy training data into train and validation sets with:
- Configurable split ratio (default 5%)
- Reproducible splits (seeded random)
- No data leakage (split at chunk level, not token level)
- Manifest tracking of split info
"""

import numpy as np
from pathlib import Path
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


def create_validation_split(
    data_path: str,
    val_ratio: float = 0.05,
    seed: int = 42,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Split a .npy data file into train and validation sets.

    Args:
        data_path: Path to the source .npy file
        val_ratio: Fraction of data to use for validation (default 5%)
        seed: Random seed for reproducibility
        output_dir: Where to save splits. Defaults to same directory as data_path.

    Returns:
        Tuple of (train_path, val_path)
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_dir = Path(output_dir) if output_dir else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (memory-mapped for efficiency)
    cli.step(1, 3, f"Loading data from {data_path.name}")
    data = np.load(str(data_path), mmap_mode="r")
    num_chunks, seq_len = data.shape
    cli.info("Total chunks", f"{num_chunks:,}")
    cli.info("Sequence length", seq_len)

    # Calculate split sizes
    val_size = max(1, int(num_chunks * val_ratio))
    train_size = num_chunks - val_size
    cli.info("Train chunks", f"{train_size:,} ({100*(1-val_ratio):.0f}%)")
    cli.info("Val chunks", f"{val_size:,} ({100*val_ratio:.0f}%)")

    # Generate reproducible random permutation
    cli.step(2, 3, "Splitting data")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_chunks)

    train_indices = np.sort(indices[:train_size])  # Sort for sequential access
    val_indices = np.sort(indices[train_size:])

    # Build output paths
    stem = data_path.stem
    # Remove any existing _train or _val suffix
    base_stem = stem.replace("_train", "").replace("_val", "")
    train_path = output_dir / f"{base_stem}_train.npy"
    val_path = output_dir / f"{base_stem}_val.npy"

    def _write_npy_batched(out_path, indices, source, total_rows, cols, dtype, batch_size):
        """Write rows from source[indices] into a .npy file in batches."""
        import numpy.lib.format as npy_fmt
        shape = (total_rows, cols)
        with open(str(out_path), "wb") as f:
            npy_fmt.write_array_header_2_0(f, npy_fmt.header_data_from_array_1_0(
                np.empty(shape, dtype=dtype)
            ))
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                chunk = np.array(source[indices[start:end]], dtype=dtype)
                f.write(chunk.tobytes())

    # Write train split
    cli.step(3, 3, "Writing split files")
    cli.substep(f"Writing train split: {train_path.name}")
    batch = 10000
    _write_npy_batched(train_path, train_indices, data, train_size, seq_len, data.dtype, batch)

    # Write val split
    cli.substep(f"Writing val split: {val_path.name}")
    _write_npy_batched(val_path, val_indices, data, val_size, seq_len, data.dtype, batch)

    # Write split metadata
    import json
    meta_path = output_dir / f"{base_stem}_split_meta.json"
    meta = {
        "source": str(data_path),
        "val_ratio": val_ratio,
        "seed": seed,
        "total_chunks": int(num_chunks),
        "train_chunks": int(train_size),
        "val_chunks": int(val_size),
        "seq_len": int(seq_len),
        "train_file": train_path.name,
        "val_file": val_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    cli.success(f"Split complete: {train_path.name} + {val_path.name}")
    return str(train_path), str(val_path)


def find_validation_set(data_path: str) -> str | None:
    """Given a training data path, find its corresponding validation set.

    Looks for: data_val.npy next to data_train.npy or data.npy
    """
    data_path = Path(data_path)
    stem = data_path.stem

    # If this IS the train split, look for matching val
    if stem.endswith("_train"):
        val_path = data_path.parent / f"{stem.replace('_train', '_val')}.npy"
        if val_path.exists():
            return str(val_path)

    # Try adding _val suffix
    val_path = data_path.parent / f"{stem}_val.npy"
    if val_path.exists():
        return str(val_path)

    return None
