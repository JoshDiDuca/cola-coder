"""Training-data statistics — shared library.

Computes summary statistics over a prepared ``.npy`` token array (and its
optional ``.weights.npy`` quality sidecar): token counts, value range/mean,
an estimated unique-token count, and the quality-score tier distribution.

Extracted so BOTH the CLI (``scripts/data_stats.py``) and the web UI
(``cola_coder.ui.data_stats_view``) compute identical numbers. Pure NumPy over a
memory-mapped array — no model, no GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

# Quality-score tiers (label, lower-inclusive, upper-exclusive). The top tier's
# upper bound is open (handled as >= 0.8); the bottom is < 0.2.
_TIERS: tuple[tuple[str, float, float], ...] = (
    ("excellent", 0.8, 1.01),
    ("good", 0.6, 0.8),
    ("average", 0.4, 0.6),
    ("poor", 0.2, 0.4),
    ("reject", -0.01, 0.2),
)


@dataclass(frozen=True)
class WeightTier:
    """One quality-score band and how many sequences fall in it."""

    label: str
    count: int
    pct: float


@dataclass
class DataStatsResult:
    """Summary statistics for a prepared training-data array."""

    data_path: str
    file_size_mb: float
    shape: list[int]
    num_chunks: int
    seq_len: int | None
    total_tokens: int
    token_min: int
    token_max: int
    token_mean: float
    est_unique_tokens: int | None = None
    has_weights: bool = False
    weights_path: str | None = None
    weight_tiers: list[WeightTier] = field(default_factory=list)
    weight_mean: float | None = None
    weight_std: float | None = None


def find_data_file(hint: str | None = None, search_root: Path | None = None) -> Path | None:
    """Locate the prepared ``train_data.npy``, honoring an explicit *hint* first.

    Falls back to common locations under *search_root* (default cwd) and the
    ``configs/storage.yaml`` data-dir redirect.
    """
    if hint:
        candidate = Path(hint)
        return candidate if candidate.exists() else None

    root = search_root or Path.cwd()
    candidates = [
        root / "data" / "processed" / "train_data.npy",
        root / "data" / "train_data.npy",
        root / "train_data.npy",
    ]
    try:
        from cola_coder.model.config import get_storage_config

        storage = get_storage_config()
        if storage is not None and hasattr(storage, "data_dir"):
            candidates.insert(0, Path(storage.data_dir) / "train_data.npy")
    except Exception:  # storage config is best-effort; never block discovery
        logger.debug("storage config lookup failed during data-file discovery", exc_info=True)
    return next((c for c in candidates if c.exists()), None)


def find_weights_file(data_path: Path, hint: str | None = None) -> Path | None:
    """Locate the ``*.weights.npy`` sidecar for *data_path* (or an explicit hint)."""
    if hint:
        candidate = Path(hint)
        return candidate if candidate.exists() else None
    weights_path = data_path.parent / (data_path.stem + ".weights.npy")
    return weights_path if weights_path.exists() else None


def estimate_unique_tokens(arr: "np.ndarray", sample_size: int = 500_000) -> int:
    """Estimate the number of distinct token IDs, sampling for large arrays."""
    import numpy as np

    if arr.size <= sample_size:
        return int(np.unique(arr).size)
    indices = np.random.choice(arr.size, size=sample_size, replace=False)
    unique_in_sample = int(np.unique(arr.flat[indices]).size)
    # Scale the sample's distinct count up by sqrt of the sampling ratio (a rough
    # long-tail heuristic), bounded by what the dtype can even represent.
    estimated = int(unique_in_sample * ((arr.size / sample_size) ** 0.5))
    if np.issubdtype(arr.dtype, np.integer):
        max_distinct = 2 ** (8 * arr.dtype.itemsize)
    else:
        max_distinct = 2**20
    return max(unique_in_sample, min(estimated, max_distinct))


def _weight_tiers(w_flat: "np.ndarray") -> list[WeightTier]:
    total = max(int(w_flat.size), 1)
    tiers: list[WeightTier] = []
    for label, low, high in _TIERS:
        count = int(((w_flat >= low) & (w_flat < high)).sum())
        tiers.append(WeightTier(label=label, count=count, pct=100.0 * count / total))
    return tiers


def compute_data_stats(
    data_path: str | None = None,
    weights_path: str | None = None,
    *,
    estimate_unique: bool = True,
    search_root: Path | None = None,
) -> DataStatsResult:
    """Compute statistics for the prepared training data.

    Raises ``FileNotFoundError`` when no data file can be located and
    ``ImportError`` when NumPy is unavailable.
    """
    resolved = find_data_file(data_path, search_root)
    if resolved is None:
        raise FileNotFoundError(
            "training data not found (run prepare_data.py or pass an explicit path)"
        )

    import numpy as np

    arr = np.load(str(resolved), mmap_mode="r")
    if arr.ndim == 2:
        num_chunks, seq_len = int(arr.shape[0]), int(arr.shape[1])
        total_tokens = num_chunks * seq_len
    else:
        num_chunks, seq_len, total_tokens = 1, None, int(arr.size)

    flat = arr.reshape(-1)
    result = DataStatsResult(
        data_path=str(resolved),
        file_size_mb=resolved.stat().st_size / (1024**2),
        shape=[int(d) for d in arr.shape],
        num_chunks=num_chunks,
        seq_len=seq_len,
        total_tokens=total_tokens,
        token_min=int(flat.min()),
        token_max=int(flat.max()),
        token_mean=float(flat.mean()),
        est_unique_tokens=estimate_unique_tokens(arr) if estimate_unique else None,
    )

    weights = find_weights_file(resolved, weights_path)
    if weights is not None:
        w_flat = np.load(str(weights), mmap_mode="r").reshape(-1).astype(float)
        result.has_weights = True
        result.weights_path = str(weights)
        result.weight_tiers = _weight_tiers(w_flat)
        result.weight_mean = float(w_flat.mean())
        result.weight_std = float(w_flat.std())
    return result
