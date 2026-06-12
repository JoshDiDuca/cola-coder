"""Dataset combination for Cola-Coder.

Combine multiple .npy token arrays into one training dataset with
configurable mixing strategies: concatenate, interleave, or weighted sampling.

Each .npy file is a 2D array of shape (num_chunks, chunk_size) with dtype uint16.
Large arrays are memory-mapped to avoid loading everything into RAM.

Usage:
    from cola_coder.data.combine import DatasetCombiner, DatasetInput

    combiner = DatasetCombiner()
    result = combiner.combine(
        datasets=[
            DatasetInput("data/processed/train_ts.npy", weight=0.7, name="TypeScript"),
            DatasetInput("data/processed/train_py.npy", weight=0.3, name="Python"),
        ],
        strategy="interleave",
        output_path="data/processed/combined.npy",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetInput:
    """Describes one input dataset for combination."""
    path: str
    weight: float = 1.0
    name: str = ""
    max_chunks: int | None = None

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).stem


@dataclass
class CombineResult:
    """Result of a dataset combination operation."""
    output_path: str
    total_chunks: int
    total_tokens: int
    sources: list[dict] = field(default_factory=list)
    # Each source: {name, path, chunks_available, chunks_contributed, fraction, weight}


class DatasetCombiner:
    """Combine multiple .npy token arrays into one training dataset.

    Supports three mixing strategies:
    1. concat   - Append datasets end-to-end (deterministic, curriculum-friendly)
    2. interleave - Round-robin chunks weighted by dataset weights (better mixing)
    3. weighted - Random sampling with replacement by weight (best mixing)

    All .npy files must share the same chunk_size (axis 1).
    """

    def combine(
        self,
        datasets: list[DatasetInput],
        strategy: str = "interleave",
        output_path: str = "./data/processed/combined.npy",
        max_tokens: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> CombineResult:
        """Combine datasets into a single training file.

        Args:
            datasets: List of DatasetInput specifying paths and weights.
            strategy: "concat", "interleave", or "weighted".
            output_path: Where to save the combined .npy file.
            max_tokens: Optional cap on total tokens in output.
            shuffle: Whether to shuffle the final result.
            seed: Random seed for reproducibility.

        Returns:
            CombineResult with metadata about what was produced.
        """
        if not datasets:
            raise ValueError("No datasets provided")
        if strategy not in ("concat", "interleave", "weighted"):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        # Load all datasets (memory-mapped)
        arrays: list[np.ndarray] = []
        chunk_size: int | None = None
        for ds in datasets:
            arr = np.load(ds.path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D array from {ds.path}, got shape {arr.shape}"
                )
            if chunk_size is None:
                chunk_size = arr.shape[1]
            elif arr.shape[1] != chunk_size:
                raise ValueError(
                    f"Chunk size mismatch: {ds.path} has {arr.shape[1]}, "
                    f"expected {chunk_size}"
                )
            # Apply per-dataset cap
            if ds.max_chunks is not None and ds.max_chunks < arr.shape[0]:
                arr = arr[: ds.max_chunks]
            arrays.append(arr)

        assert chunk_size is not None

        # Compute max_chunks from max_tokens
        max_chunks: int | None = None
        if max_tokens is not None:
            max_chunks = max_tokens // chunk_size

        # Normalize weights
        weights = np.array([ds.weight for ds in datasets], dtype=np.float64)
        weights /= weights.sum()

        # Apply strategy — each returns (combined, per-source contribution counts)
        if strategy == "concat":
            combined, contributions = self._concat(arrays, max_chunks)
        elif strategy == "interleave":
            combined, contributions = self._interleave(arrays, weights, max_chunks)
        elif strategy == "weighted":
            combined, contributions = self._weighted_sample(
                arrays, weights, max_chunks, seed
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        # Shuffle
        if shuffle and len(combined) > 0:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(combined))
            combined = combined[perm]

        # Save output
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), combined)

        # Build per-source stats (actual chunks contributed, not just available)
        total_chunks = len(combined)
        total_tokens = total_chunks * chunk_size
        sources = self._compute_sources(datasets, arrays, contributions, weights)

        logger.info(
            "Combined %d datasets -> %d chunks (%d tokens) at %s",
            len(datasets), total_chunks, total_tokens, output_path,
        )

        return CombineResult(
            output_path=str(out),
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            sources=sources,
        )

    def _concat(
        self,
        arrays: list[np.ndarray],
        max_chunks: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate arrays end-to-end. Returns (combined, contributions)."""
        contributions = np.zeros(len(arrays), dtype=np.int64)
        if max_chunks is not None:
            # Trim arrays to fit within max_chunks total
            result_parts: list[np.ndarray] = []
            remaining = max_chunks
            for i, arr in enumerate(arrays):
                take = min(len(arr), remaining)
                if take <= 0:
                    break
                result_parts.append(np.array(arr[:take]))
                contributions[i] = take
                remaining -= take
            if not result_parts:
                chunk_size = arrays[0].shape[1]
                return np.empty((0, chunk_size), dtype=arrays[0].dtype), contributions
            return np.concatenate(result_parts, axis=0), contributions
        else:
            for i, arr in enumerate(arrays):
                contributions[i] = len(arr)
            return np.concatenate([np.array(a) for a in arrays], axis=0), contributions

    def _interleave(
        self,
        arrays: list[np.ndarray],
        weights: np.ndarray,
        max_chunks: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weighted round-robin interleaving. Returns (combined, contributions).

        With weights [0.7, 0.3], we take ~70% of chunks from dataset 0
        and ~30% from dataset 1, interleaved in a round-robin pattern.
        """
        n_datasets = len(arrays)

        # Ratio-preserving target: the largest output size for which every
        # source's share (weight[i] * target) still fits within its available
        # chunks — i.e. min_i(available[i] / weight[i]). This makes the OUTPUT
        # ratio match `weights` EXACTLY without upsampling (over-represented
        # sources are subsampled). The previous code used `total_available` and
        # then clamped, which silently DISTORTED the ratio whenever the
        # highest-weight source wasn't proportionally the largest — e.g.
        # equal-sized code/text/math (exactly what `--max-samples` produces)
        # collapsed a 70/20/10 request to ~53/32/16. (The `weighted` strategy
        # upsamples to preserve more data; interleave deliberately does not.)
        ratio_caps = [
            len(arrays[i]) / weights[i]
            for i in range(n_datasets)
            if weights[i] > 0 and len(arrays[i]) > 0
        ]
        ratio_target = min(ratio_caps) if ratio_caps else 0.0
        target = ratio_target if max_chunks is None else min(ratio_target, float(max_chunks))

        # Per-dataset target counts. floor keeps each within available; the
        # explicit clamp is rounding-safety for the binding source.
        per_ds_target = np.floor(weights * target).astype(int)
        for i in range(n_datasets):
            per_ds_target[i] = min(per_ds_target[i], len(arrays[i]))

        actual_total = int(per_ds_target.sum())
        if actual_total == 0:
            chunk_size = arrays[0].shape[1]
            return (
                np.empty((0, chunk_size), dtype=arrays[0].dtype),
                np.zeros(n_datasets, dtype=np.int64),
            )

        chunk_size = arrays[0].shape[1]
        result = np.empty((actual_total, chunk_size), dtype=arrays[0].dtype)

        # Build interleaved index: cycle through datasets proportionally
        cursors = [0] * n_datasets
        out_idx = 0
        # Use a fractional accumulator for fair interleaving
        accumulators = np.zeros(n_datasets, dtype=np.float64)

        while out_idx < actual_total:
            accumulators += weights
            # Pick the dataset with highest accumulator that still has chunks
            order = np.argsort(-accumulators)
            for ds_i in order:
                if accumulators[ds_i] >= 1.0 and cursors[ds_i] < per_ds_target[ds_i]:
                    result[out_idx] = arrays[ds_i][cursors[ds_i]]
                    cursors[ds_i] += 1
                    accumulators[ds_i] -= 1.0
                    out_idx += 1
                    if out_idx >= actual_total:
                        break
            else:
                # If no accumulator >= 1.0, pick the highest one with available data
                for ds_i in order:
                    if cursors[ds_i] < per_ds_target[ds_i]:
                        result[out_idx] = arrays[ds_i][cursors[ds_i]]
                        cursors[ds_i] += 1
                        accumulators[ds_i] = 0.0
                        out_idx += 1
                        break
                else:
                    # All datasets exhausted
                    break

        # cursors[i] = chunks actually emitted from source i (the contribution).
        return result[:out_idx], np.asarray(cursors, dtype=np.int64)

    def _weighted_sample(
        self,
        arrays: list[np.ndarray],
        weights: np.ndarray,
        max_chunks: int | None,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random sampling with replacement by weight. Returns (combined, contributions)."""
        rng = np.random.default_rng(seed)
        total_available = sum(len(a) for a in arrays)
        target = max_chunks if max_chunks is not None else total_available

        chunk_size = arrays[0].shape[1]
        result = np.empty((target, chunk_size), dtype=arrays[0].dtype)

        # For each output slot, pick a dataset then pick a random chunk from it
        ds_choices = rng.choice(len(arrays), size=target, p=weights)

        for out_idx in range(target):
            ds_i = ds_choices[out_idx]
            chunk_idx = rng.integers(0, len(arrays[ds_i]))
            result[out_idx] = arrays[ds_i][chunk_idx]

        contributions = np.bincount(ds_choices, minlength=len(arrays)).astype(np.int64)
        return result, contributions

    def _compute_sources(
        self,
        datasets: list[DatasetInput],
        arrays: list[np.ndarray],
        contributions: np.ndarray,
        weights: np.ndarray,
    ) -> list[dict]:
        """Build per-source metadata for the result.

        ``chunks_contributed`` is how many chunks each source actually placed in
        the OUTPUT (which, for interleave/weighted, differs from how many were
        available); ``fraction`` is its realized share of the output — the number
        to check against the requested ``weight`` to confirm the mix came out as
        intended.
        """
        total = int(contributions.sum())
        sources = []
        for i, ds in enumerate(datasets):
            contributed = int(contributions[i])
            sources.append({
                "name": ds.name,
                "path": ds.path,
                "chunks_available": len(arrays[i]),
                "chunks_contributed": contributed,
                "fraction": (contributed / total) if total > 0 else 0.0,
                "weight": float(weights[i]),
            })
        return sources
