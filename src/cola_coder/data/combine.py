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
    weights_path: str | None = None  # per-chunk quality weights sidecar (auto if None)

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).stem

    def resolve_weights_path(self) -> Path:
        """Path to the per-chunk quality-weight sidecar for this dataset.

        Explicit ``weights_path`` wins; otherwise the prepare_data convention
        ``<stem>.weights.npy`` next to the data file.
        """
        if self.weights_path:
            return Path(self.weights_path)
        p = Path(self.path)
        return p.parent / (p.stem + ".weights.npy")


@dataclass
class CombineResult:
    """Result of a dataset combination operation."""
    output_path: str
    total_chunks: int
    total_tokens: int
    sources: list[dict] = field(default_factory=list)
    # Each source: {name, path, chunks_available, chunks_contributed, fraction, weight}
    weights_path: str | None = None  # output .weights.npy sidecar, if carry_weights


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
        carry_weights: bool = False,
    ) -> CombineResult:
        """Combine datasets into a single training file.

        Args:
            datasets: List of DatasetInput specifying paths and weights.
            strategy: "concat", "interleave", or "weighted".
            output_path: Where to save the combined .npy file.
            max_tokens: Optional cap on total tokens in output.
            shuffle: Whether to shuffle the final result.
            seed: Random seed for reproducibility.
            carry_weights: When True, carry per-chunk quality-weight sidecars
                (``<stem>.weights.npy``, as produced by ``prepare_data --score``)
                through the mix so the combined output has an aligned
                ``<output stem>.weights.npy``. Without this, quality weights are
                silently dropped when merging scored datasets. A source missing
                its sidecar contributes neutral weight 1.0. No-op (with a warning)
                if NO source has a sidecar.

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

        # Load per-chunk quality weights aligned to each (possibly trimmed) array.
        src_weights = self._load_source_weights(datasets, arrays) if carry_weights else None

        # Compute max_chunks from max_tokens
        max_chunks: int | None = None
        if max_tokens is not None:
            max_chunks = max_tokens // chunk_size

        # Normalize weights
        weights = np.array([ds.weight for ds in datasets], dtype=np.float64)
        weights /= weights.sum()

        # Apply strategy — each returns (combined, contributions, out_weights).
        # out_weights is filled in LOCKSTEP with combined (same output index), so
        # weights can never drift from their chunk; None when not carrying.
        if strategy == "concat":
            combined, contributions, out_weights = self._concat(
                arrays, max_chunks, src_weights
            )
        elif strategy == "interleave":
            combined, contributions, out_weights = self._interleave(
                arrays, weights, max_chunks, src_weights
            )
        elif strategy == "weighted":
            combined, contributions, out_weights = self._weighted_sample(
                arrays, weights, max_chunks, seed, src_weights
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        # Shuffle — the SAME permutation must apply to combined AND its weights.
        if shuffle and len(combined) > 0:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(combined))
            combined = combined[perm]
            if out_weights is not None:
                out_weights = out_weights[perm]

        # Save output
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), combined)

        # Save aligned weights sidecar (prepare_data convention: <stem>.weights.npy)
        weights_out: str | None = None
        if out_weights is not None:
            wpath = out.parent / (out.stem + ".weights.npy")
            np.save(str(wpath), out_weights)
            weights_out = str(wpath)

        # Build per-source stats (actual chunks contributed, not just available)
        total_chunks = len(combined)
        total_tokens = total_chunks * chunk_size
        sources = self._compute_sources(datasets, arrays, contributions, weights)

        logger.info(
            "Combined %d datasets -> %d chunks (%d tokens) at %s%s",
            len(datasets), total_chunks, total_tokens, output_path,
            f" (+weights {weights_out})" if weights_out else "",
        )

        return CombineResult(
            output_path=str(out),
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            sources=sources,
            weights_path=weights_out,
        )

    def _load_source_weights(
        self,
        datasets: list[DatasetInput],
        arrays: list[np.ndarray],
    ) -> list[np.ndarray] | None:
        """Load each source's per-chunk weight sidecar, aligned to ``arrays``.

        A source without a sidecar (or with a mismatched length) gets neutral
        weight 1.0 for every chunk, so a partially-scored mix still works. Returns
        None (carry disabled) only when NO source has a usable sidecar — nothing
        to carry, so we avoid writing a meaningless all-ones output sidecar.
        """
        any_found = False
        loaded: list[np.ndarray] = []
        for ds, arr in zip(datasets, arrays):
            n = len(arr)
            wpath = ds.resolve_weights_path()
            w: np.ndarray | None = None
            if wpath.exists():
                cand = np.load(str(wpath), mmap_mode="r").reshape(-1)
                if len(cand) >= n:
                    # Trim to the (possibly max_chunks-capped) array length.
                    w = np.asarray(cand[:n], dtype=np.float32)
                    any_found = True
                else:
                    logger.warning(
                        "Weights sidecar %s has %d rows but data has %d — "
                        "ignoring (neutral 1.0).", wpath, len(cand), n,
                    )
            if w is None:
                w = np.ones(n, dtype=np.float32)
            loaded.append(w)

        if not any_found:
            logger.warning(
                "carry_weights requested but no source has a .weights.npy sidecar "
                "— skipping weight output."
            )
            return None
        return loaded

    def _concat(
        self,
        arrays: list[np.ndarray],
        max_chunks: int | None,
        src_weights: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Concatenate arrays end-to-end. Returns (combined, contributions, weights)."""
        contributions = np.zeros(len(arrays), dtype=np.int64)
        chunk_size = arrays[0].shape[1]
        if max_chunks is not None:
            # Trim arrays to fit within max_chunks total
            result_parts: list[np.ndarray] = []
            weight_parts: list[np.ndarray] = []
            remaining = max_chunks
            for i, arr in enumerate(arrays):
                take = min(len(arr), remaining)
                if take <= 0:
                    break
                result_parts.append(np.array(arr[:take]))
                if src_weights is not None:
                    weight_parts.append(src_weights[i][:take])
                contributions[i] = take
                remaining -= take
            if not result_parts:
                empty_w = np.empty((0,), dtype=np.float32) if src_weights is not None else None
                return np.empty((0, chunk_size), dtype=arrays[0].dtype), contributions, empty_w
            out_w = np.concatenate(weight_parts) if src_weights is not None else None
            return np.concatenate(result_parts, axis=0), contributions, out_w
        else:
            for i, arr in enumerate(arrays):
                contributions[i] = len(arr)
            out_w = np.concatenate(src_weights) if src_weights is not None else None
            return np.concatenate([np.array(a) for a in arrays], axis=0), contributions, out_w

    def _interleave(
        self,
        arrays: list[np.ndarray],
        weights: np.ndarray,
        max_chunks: int | None,
        src_weights: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Weighted round-robin interleaving. Returns (combined, contributions, weights).

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
        chunk_size = arrays[0].shape[1]
        if actual_total == 0:
            empty_w = np.empty((0,), dtype=np.float32) if src_weights is not None else None
            return (
                np.empty((0, chunk_size), dtype=arrays[0].dtype),
                np.zeros(n_datasets, dtype=np.int64),
                empty_w,
            )

        result = np.empty((actual_total, chunk_size), dtype=arrays[0].dtype)
        out_weights = (
            np.empty(actual_total, dtype=np.float32) if src_weights is not None else None
        )

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
                    if out_weights is not None:
                        out_weights[out_idx] = src_weights[ds_i][cursors[ds_i]]
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
                        if out_weights is not None:
                            out_weights[out_idx] = src_weights[ds_i][cursors[ds_i]]
                        cursors[ds_i] += 1
                        accumulators[ds_i] = 0.0
                        out_idx += 1
                        break
                else:
                    # All datasets exhausted
                    break

        # cursors[i] = chunks actually emitted from source i (the contribution).
        out_w = out_weights[:out_idx] if out_weights is not None else None
        return result[:out_idx], np.asarray(cursors, dtype=np.int64), out_w

    def _weighted_sample(
        self,
        arrays: list[np.ndarray],
        weights: np.ndarray,
        max_chunks: int | None,
        seed: int,
        src_weights: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Random sampling with replacement by weight. Returns (combined, contributions, weights)."""
        rng = np.random.default_rng(seed)
        total_available = sum(len(a) for a in arrays)
        target = max_chunks if max_chunks is not None else total_available

        chunk_size = arrays[0].shape[1]
        result = np.empty((target, chunk_size), dtype=arrays[0].dtype)
        out_weights = np.empty(target, dtype=np.float32) if src_weights is not None else None

        # For each output slot, pick a dataset then pick a random chunk from it
        ds_choices = rng.choice(len(arrays), size=target, p=weights)

        for out_idx in range(target):
            ds_i = ds_choices[out_idx]
            chunk_idx = rng.integers(0, len(arrays[ds_i]))
            result[out_idx] = arrays[ds_i][chunk_idx]
            if out_weights is not None:
                out_weights[out_idx] = src_weights[ds_i][chunk_idx]

        contributions = np.bincount(ds_choices, minlength=len(arrays)).astype(np.int64)
        return result, contributions, out_weights

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
