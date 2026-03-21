"""Attention Pattern Analyzer: analyze saved attention weight tensors offline.

Works entirely on *saved* attention weight arrays (numpy or lists) — no live
model or GPU required.  Suitable for post-hoc analysis of model behaviour.

Patterns detected:
- local_attention: most attended tokens are near the query position
- global_attention: a few tokens receive very high attention across all queries
- copy_pattern: high attention on specific earlier positions (induction heads)
- diagonal: attention concentrated on the main diagonal (attending to self)
- uniform: near-flat distribution (entropy close to log(n))

For a TS dev: like a profiler flame-graph for transformer attention — tells you
which heads are "interesting" and which are boring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the attention analyzer feature is active."""
    return FEATURE_ENABLED


@dataclass
class HeadPattern:
    """Pattern analysis result for one attention head."""

    layer: int
    head: int
    pattern_type: str  # "local", "global", "copy", "diagonal", "uniform", "mixed"
    entropy: float  # normalized attention entropy (0=peaked, 1=uniform)
    locality_score: float  # fraction of attention mass within window=3 of query
    global_score: float  # fraction of attention mass on top-3 global tokens
    copy_score: float  # fraction of attention on earlier same-position tokens
    diagonal_score: float  # fraction on main diagonal


@dataclass
class AttentionReport:
    """Aggregate results from analyzing a full set of attention weights."""

    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_patterns: list[HeadPattern] = field(default_factory=list)
    dominant_pattern: str = "unknown"
    avg_entropy: float = 0.0
    local_heads: list[tuple[int, int]] = field(default_factory=list)
    global_heads: list[tuple[int, int]] = field(default_factory=list)
    copy_heads: list[tuple[int, int]] = field(default_factory=list)
    uniform_heads: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a one-line summary."""
        return (
            f"layers={self.num_layers} heads={self.num_heads} seq={self.seq_len} "
            f"dominant={self.dominant_pattern} avg_entropy={self.avg_entropy:.3f} "
            f"local={len(self.local_heads)} global={len(self.global_heads)} "
            f"copy={len(self.copy_heads)} uniform={len(self.uniform_heads)}"
        )


class AttentionAnalyzer:
    """Analyze attention weight arrays for interpretable patterns.

    Accepts attention weights as nested lists or numpy arrays with shape
    ``(num_layers, num_heads, seq_len, seq_len)`` or
    ``(num_heads, seq_len, seq_len)`` (single layer).

    No torch / GPU required — pure Python with optional numpy acceleration.
    """

    # Thresholds for pattern classification
    LOCAL_THRESHOLD = 0.50
    GLOBAL_THRESHOLD = 0.40
    COPY_THRESHOLD = 0.35
    DIAGONAL_THRESHOLD = 0.50
    UNIFORM_THRESHOLD = 0.85  # entropy fraction above which head is "uniform"
    LOCAL_WINDOW = 3  # tokens either side of query position

    def analyze(self, attention_weights: Any) -> AttentionReport:
        """Analyze attention_weights and return an AttentionReport.

        Args:
            attention_weights: Nested lists or numpy array of shape
                ``(layers, heads, seq, seq)`` or ``(heads, seq, seq)``.

        Returns:
            AttentionReport with per-head pattern classifications.
        """
        weights = self._to_nested_list(attention_weights)
        # Normalise to always be (layers, heads, seq, seq)
        # Determine dimensionality by descending until we hit a float
        ndim = self._count_dims(weights)
        if ndim == 2:
            # Shape (seq, seq) — single head, single layer
            weights = [[weights]]  # type: ignore[list-item]
        elif ndim == 3:
            # Shape (heads, seq, seq) — single layer
            weights = [weights]  # type: ignore[list-item]
        # ndim == 4 or higher: already (layers, heads, seq, seq)
        # Now weights: list[list[list[list[float]]]] = layers x heads x seq x seq

        report = AttentionReport()
        report.num_layers = len(weights)
        report.num_heads = len(weights[0]) if weights else 0
        report.seq_len = len(weights[0][0]) if report.num_heads else 0

        entropy_sum = 0.0
        pattern_counts: dict[str, int] = {}

        for layer_idx, layer in enumerate(weights):
            for head_idx, head in enumerate(layer):
                hp = self._analyze_head(layer_idx, head_idx, head)
                report.head_patterns.append(hp)
                entropy_sum += hp.entropy
                pattern_counts[hp.pattern_type] = (
                    pattern_counts.get(hp.pattern_type, 0) + 1
                )
                dest = (layer_idx, head_idx)
                if hp.pattern_type == "local":
                    report.local_heads.append(dest)
                elif hp.pattern_type == "global":
                    report.global_heads.append(dest)
                elif hp.pattern_type == "copy":
                    report.copy_heads.append(dest)
                elif hp.pattern_type == "uniform":
                    report.uniform_heads.append(dest)

        total = len(report.head_patterns)
        if total:
            report.avg_entropy = entropy_sum / total
            report.dominant_pattern = max(pattern_counts, key=lambda k: pattern_counts[k])

        return report

    # ------------------------------------------------------------------
    # Per-head analysis
    # ------------------------------------------------------------------

    def _analyze_head(
        self, layer: int, head: int, matrix: list[list[float]]
    ) -> HeadPattern:
        """Classify a single seq x seq attention matrix."""
        seq = len(matrix)
        if seq == 0:
            return HeadPattern(
                layer=layer,
                head=head,
                pattern_type="unknown",
                entropy=0.0,
                locality_score=0.0,
                global_score=0.0,
                copy_score=0.0,
                diagonal_score=0.0,
            )

        locality = self._locality_score(matrix)
        global_s = self._global_score(matrix)
        copy_s = self._copy_score(matrix)
        diag_s = self._diagonal_score(matrix)
        ent = self._avg_entropy(matrix)

        pattern = self._classify(locality, global_s, copy_s, diag_s, ent)

        return HeadPattern(
            layer=layer,
            head=head,
            pattern_type=pattern,
            entropy=ent,
            locality_score=locality,
            global_score=global_s,
            copy_score=copy_s,
            diagonal_score=diag_s,
        )

    def _locality_score(self, matrix: list[list[float]]) -> float:
        """Fraction of attention mass within LOCAL_WINDOW of query."""
        seq = len(matrix)
        if seq == 0:
            return 0.0
        total_mass = 0.0
        local_mass = 0.0
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                total_mass += val
                if abs(i - j) <= self.LOCAL_WINDOW:
                    local_mass += val
        return local_mass / (total_mass + 1e-9)

    def _global_score(self, matrix: list[list[float]]) -> float:
        """Fraction of attention captured by the top-3 key positions globally."""
        seq = len(matrix)
        if seq == 0:
            return 0.0
        col_sums = [0.0] * seq
        total = 0.0
        for row in matrix:
            for j, v in enumerate(row):
                col_sums[j] += v
                total += v
        top3 = sorted(col_sums, reverse=True)[:3]
        return sum(top3) / (total + 1e-9)

    def _copy_score(self, matrix: list[list[float]]) -> float:
        """Detect induction-head copy pattern: high attention 1 position ahead."""
        seq = len(matrix)
        if seq < 2:
            return 0.0
        copy_mass = 0.0
        total = 0.0
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                total += v
                # copy pattern: j == i - 1 (attending to previous occurrence)
                if j == i - 1:
                    copy_mass += v
        return copy_mass / (total + 1e-9)

    def _diagonal_score(self, matrix: list[list[float]]) -> float:
        """Fraction of attention on the main diagonal (self-attention)."""
        seq = len(matrix)
        if seq == 0:
            return 0.0
        diag = sum(matrix[i][i] for i in range(seq))
        total = sum(v for row in matrix for v in row)
        return diag / (total + 1e-9)

    def _avg_entropy(self, matrix: list[list[float]]) -> float:
        """Normalized average entropy of attention rows (0=peaked, 1=uniform)."""
        seq = len(matrix)
        if seq <= 1:
            return 0.0
        max_entropy = math.log(seq)
        entropies = []
        for row in matrix:
            row_sum = sum(row) + 1e-9
            ent = -sum((v / row_sum) * math.log((v / row_sum) + 1e-9) for v in row)
            entropies.append(ent)
        avg = sum(entropies) / len(entropies)
        return avg / max_entropy if max_entropy > 0 else 0.0

    def _classify(
        self,
        locality: float,
        global_s: float,
        copy_s: float,
        diag_s: float,
        entropy: float,
    ) -> str:
        if entropy >= self.UNIFORM_THRESHOLD:
            return "uniform"
        if diag_s >= self.DIAGONAL_THRESHOLD:
            return "diagonal"
        if copy_s >= self.COPY_THRESHOLD:
            return "copy"
        if global_s >= self.GLOBAL_THRESHOLD:
            return "global"
        if locality >= self.LOCAL_THRESHOLD:
            return "local"
        return "mixed"

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_nested_list(weights: Any) -> list:  # type: ignore[type-arg]
        """Convert numpy arrays or torch tensors to nested Python lists."""
        # numpy
        try:
            return weights.tolist()  # type: ignore[no-any-return]
        except AttributeError:
            pass
        # torch
        try:
            return weights.detach().cpu().tolist()  # type: ignore[no-any-return]
        except AttributeError:
            pass
        return list(weights)  # type: ignore[arg-type]

    @staticmethod
    def _count_dims(obj: Any, depth: int = 0) -> int:
        """Count the number of nesting levels until a scalar is reached."""
        if isinstance(obj, (int, float)):
            return depth
        if isinstance(obj, list) and len(obj) > 0:
            return AttentionAnalyzer._count_dims(obj[0], depth + 1)
        return depth
