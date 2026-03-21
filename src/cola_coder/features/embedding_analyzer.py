"""Embedding Analyzer: analyze token embedding space quality.

Provides clustering quality metrics, nearest-neighbor queries,
visualization coordinate generation (PCA/t-SNE-like via random projection),
and semantic similarity between tokens.

Designed to work without GPU — operates on raw numpy arrays of embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the embedding analyzer feature is active."""
    return FEATURE_ENABLED


@dataclass
class ClusteringQuality:
    """Metrics describing how well-clustered an embedding space is."""

    silhouette_estimate: float  # [-1, 1]; higher is better separation
    intra_cluster_variance: float
    inter_cluster_distance: float
    n_clusters: int
    n_tokens: int

    def summary(self) -> str:
        return (
            f"ClusteringQuality(silhouette={self.silhouette_estimate:.3f}, "
            f"intra_var={self.intra_cluster_variance:.3f}, "
            f"inter_dist={self.inter_cluster_distance:.3f}, "
            f"clusters={self.n_clusters}, tokens={self.n_tokens})"
        )


@dataclass
class NearestNeighborResult:
    """Result of a nearest-neighbor query."""

    query_token_id: int
    neighbors: list[tuple[int, float]]  # (token_id, distance)

    def most_similar(self) -> tuple[int, float]:
        return self.neighbors[0] if self.neighbors else (-1, float("inf"))


@dataclass
class VisualizationData:
    """2-D coordinates for embedding visualization (PCA-style projection)."""

    token_ids: list[int]
    x: list[float]
    y: list[float]
    method: str = "pca_2d"
    explained_variance: Optional[float] = None

    def as_records(self) -> list[dict]:
        return [
            {"token_id": tid, "x": xi, "y": yi}
            for tid, xi, yi in zip(self.token_ids, self.x, self.y)
        ]


@dataclass
class EmbeddingStats:
    """Summary statistics over an embedding matrix."""

    n_tokens: int
    dim: int
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float
    mean_cosine_sim: float  # Average cosine similarity between random pairs
    dead_dims: int  # Dimensions with near-zero variance


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return _dot(a, b) / (na * nb)


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


class EmbeddingAnalyzer:
    """Analyze a token embedding matrix (list-of-lists or 2-D structure).

    Parameters
    ----------
    embeddings:
        A list of embedding vectors, one per token.  Each vector is a
        list[float] of length ``dim``.
    token_ids:
        Optional list of integer token IDs.  Defaults to 0..N-1.
    """

    def __init__(
        self,
        embeddings: list[list[float]],
        token_ids: Optional[list[int]] = None,
    ) -> None:
        if not embeddings:
            raise ValueError("embeddings must be non-empty")
        self.embeddings = embeddings
        self.dim = len(embeddings[0])
        self.n = len(embeddings)
        self.token_ids: list[int] = token_ids if token_ids is not None else list(range(self.n))
        if len(self.token_ids) != self.n:
            raise ValueError("token_ids length must match embeddings length")

    # ------------------------------------------------------------------
    # Core statistics
    # ------------------------------------------------------------------

    def compute_stats(self) -> EmbeddingStats:
        """Return summary statistics over the embedding matrix."""
        norms = [_norm(e) for e in self.embeddings]
        dead = 0
        for d in range(self.dim):
            col = [self.embeddings[i][d] for i in range(self.n)]
            if _variance(col) < 1e-8:
                dead += 1

        # Sample up to 200 random pairs for mean cosine sim
        import random

        rng = random.Random(42)
        pairs = min(200, self.n * (self.n - 1) // 2)
        sims = []
        indices = list(range(self.n))
        for _ in range(pairs):
            i, j = rng.sample(indices, 2)
            sims.append(_cosine_sim(self.embeddings[i], self.embeddings[j]))

        return EmbeddingStats(
            n_tokens=self.n,
            dim=self.dim,
            mean_norm=_mean(norms),
            std_norm=math.sqrt(_variance(norms)),
            min_norm=min(norms),
            max_norm=max(norms),
            mean_cosine_sim=_mean(sims),
            dead_dims=dead,
        )

    # ------------------------------------------------------------------
    # Nearest neighbors
    # ------------------------------------------------------------------

    def nearest_neighbors(
        self,
        query_token_id: int,
        k: int = 5,
        metric: str = "cosine",
    ) -> NearestNeighborResult:
        """Find the k nearest neighbors of a token.

        Parameters
        ----------
        query_token_id:
            Token ID to query (must be in self.token_ids).
        k:
            Number of neighbors to return.
        metric:
            "cosine" or "euclidean".
        """
        if query_token_id not in self.token_ids:
            raise KeyError(f"token_id {query_token_id} not found")
        idx = self.token_ids.index(query_token_id)
        query_vec = self.embeddings[idx]

        distances: list[tuple[int, float]] = []
        for i, vec in enumerate(self.embeddings):
            if i == idx:
                continue
            if metric == "cosine":
                dist = 1.0 - _cosine_sim(query_vec, vec)
            else:
                dist = _euclidean(query_vec, vec)
            distances.append((self.token_ids[i], dist))

        distances.sort(key=lambda x: x[1])
        return NearestNeighborResult(
            query_token_id=query_token_id,
            neighbors=distances[:k],
        )

    # ------------------------------------------------------------------
    # Clustering quality (k-means stub with random init)
    # ------------------------------------------------------------------

    def clustering_quality(self, n_clusters: int = 8, max_iter: int = 20) -> ClusteringQuality:
        """Estimate clustering quality using lightweight k-means.

        Uses a pure-Python k-means so there are no external dependencies.
        """
        import random

        if n_clusters >= self.n:
            n_clusters = max(1, self.n - 1)

        rng = random.Random(0)
        # Initialize centroids by picking random embeddings
        centroid_indices = rng.sample(range(self.n), n_clusters)
        centroids = [list(self.embeddings[i]) for i in centroid_indices]

        assignments = [0] * self.n
        for _ in range(max_iter):
            # Assign step
            changed = False
            for i, vec in enumerate(self.embeddings):
                best_c = 0
                best_d = float("inf")
                for c, centroid in enumerate(centroids):
                    d = _euclidean(vec, centroid)
                    if d < best_d:
                        best_d = d
                        best_c = c
                if assignments[i] != best_c:
                    assignments[i] = best_c
                    changed = True

            # Update step
            new_centroids = [[0.0] * self.dim for _ in range(n_clusters)]
            counts = [0] * n_clusters
            for i, c in enumerate(assignments):
                counts[c] += 1
                for d in range(self.dim):
                    new_centroids[c][d] += self.embeddings[i][d]
            for c in range(n_clusters):
                if counts[c] > 0:
                    centroids[c] = [v / counts[c] for v in new_centroids[c]]

            if not changed:
                break

        # Compute intra-cluster variance
        intra_vars = []
        for c in range(n_clusters):
            members = [self.embeddings[i] for i, a in enumerate(assignments) if a == c]
            if len(members) < 2:
                continue
            dists = [_euclidean(m, centroids[c]) for m in members]
            intra_vars.append(_variance(dists))
        intra_var = _mean(intra_vars) if intra_vars else 0.0

        # Compute inter-cluster distance (mean pairwise centroid distance)
        inter_dists = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                inter_dists.append(_euclidean(centroids[i], centroids[j]))
        inter_dist = _mean(inter_dists) if inter_dists else 0.0

        # Silhouette estimate: (inter - intra) / max(inter, intra)
        denom = max(inter_dist, intra_var, 1e-12)
        silhouette = (inter_dist - intra_var) / denom
        silhouette = max(-1.0, min(1.0, silhouette))

        return ClusteringQuality(
            silhouette_estimate=silhouette,
            intra_cluster_variance=intra_var,
            inter_cluster_distance=inter_dist,
            n_clusters=n_clusters,
            n_tokens=self.n,
        )

    # ------------------------------------------------------------------
    # Visualization data (PCA-style 2-D projection)
    # ------------------------------------------------------------------

    def visualization_data(self, method: str = "pca_2d") -> VisualizationData:
        """Compute 2-D projection coordinates for visualization.

        Uses a lightweight power-iteration PCA (2 components) with no
        external dependencies.
        """
        # Center embeddings
        means = [_mean([self.embeddings[i][d] for i in range(self.n)]) for d in range(self.dim)]
        centered = [[self.embeddings[i][d] - means[d] for d in range(self.dim)] for i in range(self.n)]

        # Power iteration for top-2 principal components
        import random

        rng = random.Random(1)
        components = []
        deflated = [list(v) for v in centered]

        explained = []
        for _ in range(2):
            # Random unit vector init
            v = [rng.gauss(0, 1) for _ in range(self.dim)]
            nv = _norm(v)
            v = [x / nv for x in v]

            # Power iteration
            for _ in range(30):
                # Multiply by covariance: v_new = X^T (X v)
                xv = [_dot(row, v) for row in deflated]
                v_new = [0.0] * self.dim
                for i, scalar in enumerate(xv):
                    for d in range(self.dim):
                        v_new[d] += scalar * deflated[i][d]
                nv = _norm(v_new)
                if nv < 1e-12:
                    break
                v = [x / nv for x in v_new]

            components.append(v)
            # Deflate
            proj = [_dot(row, v) for row in deflated]
            explained.append(_variance(proj) * self.n)
            deflated = [
                [deflated[i][d] - proj[i] * v[d] for d in range(self.dim)]
                for i in range(self.n)
            ]

        # Project embeddings onto the 2 components
        x_coords = [_dot(centered[i], components[0]) for i in range(self.n)]
        y_coords = [_dot(centered[i], components[1]) for i in range(self.n)] if len(components) > 1 else [0.0] * self.n

        total_var = sum(explained) if sum(explained) > 0 else 1.0
        exp_var = explained[0] / total_var if explained else None

        return VisualizationData(
            token_ids=list(self.token_ids),
            x=x_coords,
            y=y_coords,
            method=method,
            explained_variance=exp_var,
        )

    # ------------------------------------------------------------------
    # Semantic similarity
    # ------------------------------------------------------------------

    def semantic_similarity(self, token_id_a: int, token_id_b: int) -> float:
        """Return cosine similarity between two token embeddings."""
        if token_id_a not in self.token_ids:
            raise KeyError(f"token_id {token_id_a} not found")
        if token_id_b not in self.token_ids:
            raise KeyError(f"token_id {token_id_b} not found")
        idx_a = self.token_ids.index(token_id_a)
        idx_b = self.token_ids.index(token_id_b)
        return _cosine_sim(self.embeddings[idx_a], self.embeddings[idx_b])

    def similarity_matrix(self, token_ids: list[int]) -> list[list[float]]:
        """Return a pairwise cosine similarity matrix for the given token IDs."""
        indices = [self.token_ids.index(t) for t in token_ids]
        n = len(indices)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = _cosine_sim(
                    self.embeddings[indices[i]], self.embeddings[indices[j]]
                )
        return matrix
