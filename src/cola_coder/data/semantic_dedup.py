"""Semantic (embedding) deduplication for Cola-Coder datasets — SemDeDup / D4.

This is the THIRD dedup level, complementing the two in ``dedup.py``:

1. ``ExactDeduplicator``       — SHA-256, byte-identical chunks (lexical, exact)
2. ``CrossDatasetDeduplicator`` — MinHash/Jaccard, near-identical *tokens*
3. ``SemanticDeduplicator`` (here) — embedding cosine, near-identical *meaning*

The first two are lexical: they catch chunks that share literal tokens/bytes.
They miss chunks that are semantically the same but lexically different (renamed
variables, reordered statements, reformatted code). SemDeDup (Abbas et al. 2023)
and D4 cluster embeddings and drop near-duplicates *within* each cluster, keeping
one representative — exactly what this module does, offline, at prep time.

Design constraints (DATA-069):
- DEPENDENCY-FREE core: the default embedder is a numpy-only TF-IDF-style vector,
  and clustering falls back to a numpy Lloyd's k-means when scikit-learn is
  absent. Tests and CI need no torch / sklearn / GPU.
- PLUGGABLE embedder: pass ``embed_fn`` (rows -> 2D float embeddings) or
  precomputed ``embeddings`` to inject a model-based embedder later.
- Drop-in array contract: ``semantic_dedup_array`` mirrors
  ``ExactDeduplicator.deduplicate_array`` so callers treat it as a sibling.

All token .npy files are 2D arrays of shape (num_chunks, chunk_size), uint16.

Usage:
    from cola_coder.data.semantic_dedup import semantic_dedup_array

    kept, removed, kept_idx = semantic_dedup_array(data, threshold=0.9)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# scikit-learn is OPTIONAL — when present we use its (faster, k-means++) KMeans;
# otherwise the numpy Lloyd's-iteration fallback below runs. Either path is
# deterministic given a seed, and tests must exercise the fallback when sklearn
# is absent (DATA-069).
try:
    from sklearn.cluster import KMeans as _SKKMeans

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only when sklearn installed
    _HAS_SKLEARN = False

# An embedder maps a sequence of token rows (each a 1D uint16 array) to a 2D
# float array of shape (n_rows, embed_dim).
EmbedFn = Callable[[Sequence[np.ndarray]], np.ndarray]


def tfidf_embed(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Dependency-free TF-IDF embedder over raw token ids (numpy only).

    Treats each chunk as a bag of token-ids, builds a term-frequency matrix over
    the vocabulary actually present in this batch, and scales each column by its
    inverse-document-frequency. The result is a dense float32 matrix, one row per
    chunk, suitable for cosine similarity. No tokenizer/decode needed — token ids
    ARE the terms — which keeps the core dependency-free and fast.

    Args:
        rows: sequence of 1D token arrays (one chunk each).

    Returns:
        float32 array of shape (len(rows), num_distinct_tokens). Empty input
        yields shape (0, 0).
    """
    n = len(rows)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Build a compact vocabulary of token-ids present across all rows so the
    # embedding width is bounded by distinct tokens, not the full 32k vocab.
    vocab: dict[int, int] = {}
    for row in rows:
        for tok in np.asarray(row).ravel().tolist():
            if tok not in vocab:
                vocab[tok] = len(vocab)

    width = max(len(vocab), 1)
    tf = np.zeros((n, width), dtype=np.float32)
    for i, row in enumerate(rows):
        for tok in np.asarray(row).ravel().tolist():
            tf[i, vocab[tok]] += 1.0

    # Document frequency per term → smoothed idf (log scaling, +1 to avoid /0).
    df = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    return tf * idf


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize so dot products equal cosine similarity."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _kmeans_numpy(
    embeddings: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 50,
) -> np.ndarray:
    """Deterministic numpy Lloyd's-iteration k-means fallback (no sklearn).

    Returns a 1D int array of cluster labels, one per row. Deterministic for a
    given ``seed``: centroids are seeded from a seeded permutation of the rows.
    """
    n = len(embeddings)
    rng = np.random.default_rng(seed)
    # Seed centroids from k distinct rows (seeded permutation → deterministic).
    init = rng.permutation(n)[:k]
    centroids = embeddings[init].astype(np.float32).copy()
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # Assign: nearest centroid by squared Euclidean distance.
        dists = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :], axis=2
        )
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update: mean of assigned rows; keep old centroid for empty clusters.
        for c in range(k):
            members = embeddings[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return labels


def cluster(embeddings: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Cluster rows of ``embeddings`` into ``k`` groups; return integer labels.

    Uses ``sklearn.cluster.KMeans`` when scikit-learn is importable (k-means++),
    otherwise a deterministic numpy Lloyd's-iteration fallback. ``k`` is clamped
    to ``[1, n]``. Deterministic for a given ``seed``.

    Args:
        embeddings: 2D float array, one row per chunk.
        k: requested number of clusters (auto-clamped to n).
        seed: RNG seed for reproducibility.

    Returns:
        1D int array of cluster labels aligned to ``embeddings`` rows.
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)

    emb = np.asarray(embeddings, dtype=np.float32)
    if _HAS_SKLEARN:
        km = _SKKMeans(n_clusters=k, random_state=seed, n_init=10)
        return km.fit_predict(emb).astype(np.int64)
    return _kmeans_numpy(emb, k, seed)


def find_semantic_duplicates(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.9,
    quality_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Within each cluster, drop near-duplicates above ``threshold`` cosine.

    For each cluster, builds the pairwise cosine-similarity matrix of its members
    (rows L2-normalized first). Members linked by similarity >= ``threshold`` form
    near-duplicate sets; one REPRESENTATIVE is kept per set:

    - if ``quality_weights`` is given → keep the HIGHEST-quality member (the
      original SemDeDup-with-quality idea: prefer the better example);
    - else → keep the member FARTHEST from the cluster centroid (the SemDeDup
      default: the centroid-distant point is the most "prototypical-but-not-mean"
      representative and least redundant).

    Greedy, deterministic: members are processed in index order; the first kept
    member of a near-dup set becomes its representative and absorbs the rest.

    Args:
        embeddings: 2D float array, one row per chunk.
        labels: cluster label per row (from ``cluster``).
        threshold: cosine similarity at/above which two members are near-dups.
        quality_weights: optional per-row quality weight; higher = keep.

    Returns:
        Boolean keep-mask aligned to ``embeddings`` rows (True = keep).
    """
    n = len(embeddings)
    keep = np.ones(n, dtype=bool)
    if n <= 1:
        return keep

    norm = _l2_normalize(embeddings)

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            continue

        sub = norm[idx]
        sims = sub @ sub.T  # cosine similarity (rows are L2-normalized)

        # Rank within cluster: the best representative is processed FIRST so it
        # is added to ``kept_local`` first and absorbs the later near-dups (the
        # greedy pass drops any member within threshold of an already-kept one).
        if quality_weights is not None:
            qw = np.asarray(quality_weights, dtype=np.float32)[idx]
            # Higher quality should win → process highest quality first.
            order = np.argsort(qw, kind="stable")[::-1]
        else:
            centroid = sub.mean(axis=0, keepdims=True)
            centroid = _l2_normalize(centroid)
            dist = 1.0 - (sub @ centroid.T).ravel()  # cosine distance to centroid
            # Farthest-from-centroid should win → process farthest first.
            order = np.argsort(dist, kind="stable")[::-1]

        kept_local: list[int] = []
        for local in order.tolist():
            is_dup = False
            for rep in kept_local:
                if sims[local, rep] >= threshold:
                    is_dup = True
                    break
            if is_dup:
                keep[idx[local]] = False
            else:
                kept_local.append(local)

    return keep


def semantic_dedup_array(
    data: np.ndarray,
    embed_fn: EmbedFn | None = None,
    embeddings: np.ndarray | None = None,
    k: int = 1000,
    threshold: float = 0.9,
    quality_weights: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """SemDeDup: cluster embeddings, drop near-dups within clusters, keep one rep.

    Drop-in sibling of ``ExactDeduplicator.deduplicate_array`` — same first two
    return elements (kept_data, removed_count) plus the kept indices so the
    caller can realign a ``.weights.npy`` sidecar.

    Args:
        data: 2D token array, shape (num_chunks, chunk_size).
        embed_fn: pluggable embedder (rows -> 2D float embeddings). Defaults to
            the dependency-free ``tfidf_embed``. Ignored if ``embeddings`` given.
        embeddings: optional precomputed embeddings (one row per chunk), e.g.
            from a model-based embedder. Bypasses ``embed_fn``.
        k: number of SemDeDup clusters (auto-clamped to n).
        threshold: cosine similarity at/above which chunks are near-duplicates.
        quality_weights: optional per-row quality weight; when given the
            highest-quality member of a near-dup set is kept.
        seed: RNG seed for deterministic clustering.

    Returns:
        (kept_data, removed_count, kept_indices) — kept_indices is a sorted int
        array of surviving row indices into the original ``data``.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    n = len(data)
    # n <= 1 → nothing to dedup against; no-op (matches dedup.py guards).
    if n <= 1:
        return np.array(data), 0, np.arange(n, dtype=np.int64)

    rows = [data[i] for i in range(n)]
    if embeddings is None:
        embed = (embed_fn or tfidf_embed)(rows)
    else:
        embed = np.asarray(embeddings, dtype=np.float32)
        if len(embed) != n:
            raise ValueError(
                f"embeddings rows ({len(embed)}) != data rows ({n})"
            )

    labels = cluster(embed, k=k, seed=seed)
    keep_mask = find_semantic_duplicates(
        embed, labels, threshold=threshold, quality_weights=quality_weights
    )

    kept_indices = np.where(keep_mask)[0].astype(np.int64)
    kept_data = np.array(data[keep_mask])
    num_removed = int((~keep_mask).sum())

    logger.info(
        "Semantic dedup: %d -> %d chunks (%d removed, %.1f%% dedup rate, "
        "k=%d, threshold=%.2f, sklearn=%s)",
        n, len(kept_data), num_removed,
        100.0 * num_removed / max(n, 1),
        max(1, min(k, n)), threshold, _HAS_SKLEARN,
    )
    return kept_data, num_removed, kept_indices
