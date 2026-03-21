"""Tests for embedding_analyzer.py (feature 51)."""

import pytest

from cola_coder.features.embedding_analyzer import (
    FEATURE_ENABLED,
    EmbeddingAnalyzer,
    is_enabled,
)


def _make_embeddings(n: int = 10, dim: int = 8, seed: int = 0) -> list[list[float]]:
    """Generate deterministic pseudo-random embeddings."""
    import random
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_constructor_default_ids():
    embs = _make_embeddings(5, 4)
    ea = EmbeddingAnalyzer(embs)
    assert ea.token_ids == [0, 1, 2, 3, 4]
    assert ea.n == 5
    assert ea.dim == 4


def test_constructor_custom_ids():
    embs = _make_embeddings(3, 4)
    ea = EmbeddingAnalyzer(embs, token_ids=[10, 20, 30])
    assert ea.token_ids == [10, 20, 30]


def test_constructor_mismatched_ids_raises():
    embs = _make_embeddings(3, 4)
    with pytest.raises(ValueError):
        EmbeddingAnalyzer(embs, token_ids=[1, 2])


def test_constructor_empty_raises():
    with pytest.raises(ValueError):
        EmbeddingAnalyzer([])


def test_compute_stats_shape():
    embs = _make_embeddings(20, 16)
    ea = EmbeddingAnalyzer(embs)
    stats = ea.compute_stats()
    assert stats.n_tokens == 20
    assert stats.dim == 16
    assert stats.mean_norm > 0
    assert stats.min_norm <= stats.mean_norm <= stats.max_norm
    assert stats.std_norm >= 0
    assert -1.0 <= stats.mean_cosine_sim <= 1.0


def test_compute_stats_dead_dims():
    # All-zero dimension should be counted as dead
    embs = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    ea = EmbeddingAnalyzer(embs)
    stats = ea.compute_stats()
    assert stats.dead_dims >= 1


def test_nearest_neighbors_returns_k():
    embs = _make_embeddings(15, 8)
    ea = EmbeddingAnalyzer(embs)
    result = ea.nearest_neighbors(0, k=4)
    assert result.query_token_id == 0
    assert len(result.neighbors) == 4
    # All neighbor token ids should be different from the query
    assert all(tid != 0 for tid, _ in result.neighbors)


def test_nearest_neighbors_sorted():
    embs = _make_embeddings(10, 8)
    ea = EmbeddingAnalyzer(embs)
    result = ea.nearest_neighbors(0, k=5, metric="cosine")
    dists = [d for _, d in result.neighbors]
    assert dists == sorted(dists)


def test_nearest_neighbors_unknown_token_raises():
    embs = _make_embeddings(5, 4)
    ea = EmbeddingAnalyzer(embs)
    with pytest.raises(KeyError):
        ea.nearest_neighbors(999, k=2)


def test_nearest_neighbors_euclidean():
    embs = _make_embeddings(10, 8)
    ea = EmbeddingAnalyzer(embs)
    result = ea.nearest_neighbors(0, k=3, metric="euclidean")
    assert len(result.neighbors) == 3
    dists = [d for _, d in result.neighbors]
    assert all(d >= 0 for d in dists)


def test_clustering_quality_returns_valid():
    embs = _make_embeddings(30, 8)
    ea = EmbeddingAnalyzer(embs)
    quality = ea.clustering_quality(n_clusters=4)
    assert quality.n_clusters == 4
    assert quality.n_tokens == 30
    assert -1.0 <= quality.silhouette_estimate <= 1.0
    assert quality.intra_cluster_variance >= 0
    assert quality.inter_cluster_distance >= 0


def test_clustering_quality_summary():
    embs = _make_embeddings(20, 4)
    ea = EmbeddingAnalyzer(embs)
    quality = ea.clustering_quality(n_clusters=3)
    s = quality.summary()
    assert "silhouette" in s
    assert "clusters=3" in s


def test_visualization_data_shape():
    embs = _make_embeddings(12, 8)
    ea = EmbeddingAnalyzer(embs)
    vis = ea.visualization_data()
    assert len(vis.token_ids) == 12
    assert len(vis.x) == 12
    assert len(vis.y) == 12
    assert vis.method == "pca_2d"


def test_visualization_data_records():
    embs = _make_embeddings(5, 4)
    ea = EmbeddingAnalyzer(embs)
    vis = ea.visualization_data()
    records = vis.as_records()
    assert len(records) == 5
    assert all("token_id" in r and "x" in r and "y" in r for r in records)


def test_semantic_similarity_self():
    embs = _make_embeddings(5, 4)
    ea = EmbeddingAnalyzer(embs)
    sim = ea.semantic_similarity(0, 0)
    assert abs(sim - 1.0) < 1e-6


def test_semantic_similarity_range():
    embs = _make_embeddings(10, 8)
    ea = EmbeddingAnalyzer(embs)
    sim = ea.semantic_similarity(0, 1)
    assert -1.0 <= sim <= 1.0


def test_similarity_matrix_diagonal():
    embs = _make_embeddings(4, 4)
    ea = EmbeddingAnalyzer(embs)
    mat = ea.similarity_matrix([0, 1, 2, 3])
    for i in range(4):
        assert abs(mat[i][i] - 1.0) < 1e-6


def test_similarity_matrix_symmetric():
    embs = _make_embeddings(4, 4)
    ea = EmbeddingAnalyzer(embs)
    mat = ea.similarity_matrix([0, 1, 2])
    for i in range(3):
        for j in range(3):
            assert abs(mat[i][j] - mat[j][i]) < 1e-9
