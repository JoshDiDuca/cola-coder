"""Tests for semantic (embedding) deduplication — SemDeDup/D4 (DATA-069).

Numpy-only, hermetic: no torch, no GPU, no sklearn required. When sklearn is
absent the numpy k-means fallback is exercised automatically (not skipped).
"""

from __future__ import annotations

import numpy as np
import pytest

from cola_coder.data import semantic_dedup
from cola_coder.data.semantic_dedup import (
    cluster,
    find_semantic_duplicates,
    semantic_dedup_array,
    tfidf_embed,
)


def _rows(*seqs: list[int]) -> np.ndarray:
    """Build a 2D uint16 token array from equal-length sequences."""
    return np.array(seqs, dtype=np.uint16)


def test_identical_chunks_collapse_to_one():
    """Byte-identical chunks → only one survives."""
    data = _rows([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
    kept, removed, kept_idx = semantic_dedup_array(data, k=2, threshold=0.9, seed=0)
    assert len(kept) == 1
    assert removed == 2
    assert len(kept_idx) == 1


def test_semantically_near_reordered_tokens_drop():
    """Reordered tokens have an identical bag-of-tokens → cosine 1.0 → drop."""
    # tfidf_embed is a bag-of-tokens model, so reordering yields the same vector.
    data = _rows([1, 2, 3, 4], [4, 3, 2, 1], [9, 9, 9, 9])
    kept, removed, kept_idx = semantic_dedup_array(data, k=1, threshold=0.9, seed=0)
    # The two reorderings collapse; the distinct [9,9,9,9] survives.
    assert removed == 1
    assert len(kept) == 2
    # The distinct chunk (all 9s) must survive.
    survivors = {tuple(row) for row in kept.tolist()}
    assert (9, 9, 9, 9) in survivors


def test_distinct_chunk_survives():
    """A clearly distinct chunk is never dropped."""
    data = _rows([1, 1, 1, 1], [1, 1, 1, 1], [2, 3, 4, 5])
    kept, removed, _ = semantic_dedup_array(data, k=1, threshold=0.9, seed=0)
    survivors = {tuple(row) for row in kept.tolist()}
    assert (2, 3, 4, 5) in survivors
    assert removed == 1


def test_quality_weighted_keeps_high_score_member():
    """With quality weights, the highest-quality near-dup member is kept."""
    data = _rows([5, 6, 7, 8], [5, 6, 7, 8])  # identical → one is a dup
    weights = np.array([0.2, 0.9], dtype=np.float32)  # row 1 is higher quality
    kept, removed, kept_idx = semantic_dedup_array(
        data, k=1, threshold=0.9, quality_weights=weights, seed=0
    )
    assert removed == 1
    assert len(kept) == 1
    # The kept index must be the higher-quality row (index 1).
    assert kept_idx.tolist() == [1]


def test_centroid_distant_fallback_when_no_weights():
    """Without weights, the farthest-from-centroid member represents the set."""
    # Three near-identical embeddings; pick representatives by centroid distance.
    emb = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ],
        dtype=np.float32,
    )
    labels = np.zeros(3, dtype=np.int64)
    keep = find_semantic_duplicates(emb, labels, threshold=0.9)
    # All three are mutual near-dups (cosine ~1) → exactly one kept.
    assert int(keep.sum()) == 1


def test_higher_threshold_removes_fewer():
    """A stricter (higher) threshold drops fewer chunks."""
    # Moderately similar bag-of-tokens (share some tokens, differ in others).
    data = _rows(
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 6, 7],
        [8, 9, 10, 11],
    )
    _, removed_loose, _ = semantic_dedup_array(data, k=1, threshold=0.5, seed=0)
    _, removed_strict, _ = semantic_dedup_array(data, k=1, threshold=0.999, seed=0)
    assert removed_strict <= removed_loose


def test_k_clamped_when_n_less_than_k():
    """k is clamped to n; no crash when k > number of rows."""
    data = _rows([1, 2, 3, 4], [5, 6, 7, 8])
    labels = cluster(tfidf_embed([data[0], data[1]]), k=100, seed=0)
    assert len(labels) == 2
    assert labels.max() < 2  # at most n clusters
    # Full pipeline must also not crash with an oversized k.
    kept, removed, kept_idx = semantic_dedup_array(data, k=100, threshold=0.9, seed=0)
    assert len(kept) + removed == 2


def test_empty_input_noop():
    """Empty array → no-op with empty results."""
    data = np.zeros((0, 4), dtype=np.uint16)
    kept, removed, kept_idx = semantic_dedup_array(data, k=10, threshold=0.9)
    assert len(kept) == 0
    assert removed == 0
    assert len(kept_idx) == 0


def test_single_row_noop():
    """Single row → nothing to dedup against; returned unchanged."""
    data = _rows([1, 2, 3, 4])
    kept, removed, kept_idx = semantic_dedup_array(data, k=10, threshold=0.9)
    assert len(kept) == 1
    assert removed == 0
    assert kept_idx.tolist() == [0]


def test_deterministic_by_seed():
    """Same seed → identical results across runs."""
    data = _rows(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [5, 6, 7, 9],
        [10, 11, 12, 13],
    )
    a = semantic_dedup_array(data, k=2, threshold=0.9, seed=42)
    b = semantic_dedup_array(data, k=2, threshold=0.9, seed=42)
    assert a[1] == b[1]
    assert a[2].tolist() == b[2].tolist()


def test_return_contract_shape_and_indices():
    """Return is (kept_data, int removed, sorted valid kept_indices)."""
    data = _rows(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 9, 9, 9],
    )
    kept, removed, kept_idx = semantic_dedup_array(data, k=2, threshold=0.9, seed=0)
    assert isinstance(removed, int)
    assert kept.ndim == 2
    assert kept.shape[1] == data.shape[1]
    # kept_indices: sorted, unique, in range, and count matches kept rows.
    assert kept_idx.tolist() == sorted(kept_idx.tolist())
    assert len(kept_idx) == len(kept)
    assert len(set(kept_idx.tolist())) == len(kept_idx)
    assert kept_idx.min() >= 0
    assert kept_idx.max() < len(data)
    # kept rows must equal data at kept_indices.
    assert np.array_equal(kept, data[kept_idx])


def test_numpy_fallback_path_exercised_when_no_sklearn():
    """When sklearn is unavailable, the numpy k-means fallback must run."""
    if semantic_dedup._HAS_SKLEARN:
        pytest.skip("sklearn present — fallback covered separately")
    # cluster() must still partition rows without sklearn.
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    labels = cluster(emb, k=2, seed=0)
    assert len(labels) == 4
    assert set(labels.tolist()) <= {0, 1}
    # The two groups should land in different clusters.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_pluggable_embed_fn():
    """A custom embed_fn is honored over the default."""
    data = _rows([1, 2], [3, 4], [5, 6])

    def constant_embed(rows):
        # All rows map to the same vector → all near-dups → collapse to one.
        return np.ones((len(rows), 3), dtype=np.float32)

    kept, removed, _ = semantic_dedup_array(
        data, embed_fn=constant_embed, k=1, threshold=0.9, seed=0
    )
    assert len(kept) == 1
    assert removed == 2


def test_precomputed_embeddings_used():
    """Precomputed embeddings bypass the embedder and drive dedup directly."""
    data = _rows([1, 2, 3], [4, 5, 6], [7, 8, 9])
    # Rows 0 and 1 identical embeddings → near-dup; row 2 orthogonal → survives.
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kept, removed, kept_idx = semantic_dedup_array(
        data, embeddings=emb, k=1, threshold=0.9, seed=0
    )
    assert removed == 1
    assert len(kept) == 2
