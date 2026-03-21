"""Attention Pattern Library: catalog and compare attention patterns.

Defines common attention patterns used in transformer models:
  - local: attends to a fixed window around each position
  - global: every position attends to all positions
  - strided: attends to fixed-stride positions
  - dilated: attends with exponentially increasing gaps

Provides tools to compare a model's actual attention matrix against
expected patterns and score alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------


@dataclass
class AttentionPattern:
    """An expected attention pattern for a given sequence length."""

    name: str
    description: str
    # mask[i][j] == 1.0 means position i should attend to position j
    mask: list[list[float]]

    @property
    def seq_len(self) -> int:
        return len(self.mask)


def local_pattern(seq_len: int, window: int = 2) -> AttentionPattern:
    """Each position attends to itself and *window* neighbours on each side."""
    mask = []
    for i in range(seq_len):
        row = [0.0] * seq_len
        for j in range(max(0, i - window), min(seq_len, i + window + 1)):
            row[j] = 1.0
        mask.append(row)
    return AttentionPattern(
        name="local",
        description=f"Window size {window} around each position",
        mask=mask,
    )


def global_pattern(seq_len: int) -> AttentionPattern:
    """Every position attends to every other position."""
    mask = [[1.0] * seq_len for _ in range(seq_len)]
    return AttentionPattern(
        name="global",
        description="Full attention between all positions",
        mask=mask,
    )


def strided_pattern(seq_len: int, stride: int = 2) -> AttentionPattern:
    """Each position attends to every *stride*-th position."""
    mask = []
    for i in range(seq_len):
        row = [0.0] * seq_len
        for j in range(0, seq_len, stride):
            row[j] = 1.0
        # Always attend to self
        row[i] = 1.0
        mask.append(row)
    return AttentionPattern(
        name="strided",
        description=f"Stride-{stride} attention",
        mask=mask,
    )


def dilated_pattern(seq_len: int, dilation_base: int = 2) -> AttentionPattern:
    """Each position attends to positions at exponentially increasing distances."""
    mask = []
    for i in range(seq_len):
        row = [0.0] * seq_len
        row[i] = 1.0  # self
        d = 1
        while d < seq_len:
            if i - d >= 0:
                row[i - d] = 1.0
            if i + d < seq_len:
                row[i + d] = 1.0
            d *= dilation_base
        mask.append(row)
    return AttentionPattern(
        name="dilated",
        description=f"Dilated attention with base {dilation_base}",
        mask=mask,
    )


def causal_pattern(seq_len: int) -> AttentionPattern:
    """Causal (lower-triangular) attention — each position only sees past."""
    mask = []
    for i in range(seq_len):
        row = [0.0] * seq_len
        for j in range(i + 1):
            row[j] = 1.0
        mask.append(row)
    return AttentionPattern(
        name="causal",
        description="Causal (autoregressive) attention mask",
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

_PATTERN_BUILDERS: dict[str, Callable[..., AttentionPattern]] = {
    "local": local_pattern,
    "global": global_pattern,
    "strided": strided_pattern,
    "dilated": dilated_pattern,
    "causal": causal_pattern,
}


def get_pattern(name: str, seq_len: int, **kwargs: object) -> AttentionPattern:
    """Build a named pattern for the given sequence length."""
    if name not in _PATTERN_BUILDERS:
        raise ValueError(f"Unknown pattern '{name}'. Available: {list(_PATTERN_BUILDERS)}")
    return _PATTERN_BUILDERS[name](seq_len, **kwargs)


def list_patterns() -> list[str]:
    """Return names of all built-in patterns."""
    return list(_PATTERN_BUILDERS)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class PatternComparison:
    """Results of comparing model attention to an expected pattern."""

    pattern_name: str
    seq_len: int
    # Per-head alignment scores (0.0–1.0)
    head_scores: list[float] = field(default_factory=list)
    best_head: int = -1
    best_score: float = 0.0
    mean_score: float = 0.0
    # Discrepancy matrix for the best head (optional)
    discrepancy: list[list[float]] | None = None

    def summary(self) -> str:
        return (
            f"Pattern '{self.pattern_name}': mean={self.mean_score:.3f}, "
            f"best_head={self.best_head} ({self.best_score:.3f})"
        )


def compare_attention(
    attention_weights: list[list[list[float]]],
    pattern: AttentionPattern,
    *,
    compute_discrepancy: bool = False,
) -> PatternComparison:
    """Compare multi-head attention weights against an expected pattern.

    Parameters
    ----------
    attention_weights:
        Shape [num_heads, seq_len, seq_len].  Values should be normalised
        (rows sum to 1.0) but the function is tolerant.
    pattern:
        The expected AttentionPattern to compare against.
    compute_discrepancy:
        If True, compute the discrepancy matrix for the best-scoring head.

    Returns
    -------
    PatternComparison with per-head alignment scores.
    """
    num_heads = len(attention_weights)
    if num_heads == 0:
        return PatternComparison(pattern_name=pattern.name, seq_len=pattern.seq_len)

    seq_len = len(attention_weights[0])
    expected = pattern.mask

    head_scores: list[float] = []
    for head_weights in attention_weights:
        score = _compute_alignment(head_weights, expected, seq_len)
        head_scores.append(score)

    best_head = max(range(num_heads), key=lambda h: head_scores[h])
    best_score = head_scores[best_head]
    mean_score = sum(head_scores) / num_heads

    discrepancy = None
    if compute_discrepancy:
        hw = attention_weights[best_head]
        discrepancy = [
            [abs(hw[i][j] - expected[i][j]) for j in range(seq_len)]
            for i in range(seq_len)
        ]

    return PatternComparison(
        pattern_name=pattern.name,
        seq_len=seq_len,
        head_scores=head_scores,
        best_head=best_head,
        best_score=round(best_score, 4),
        mean_score=round(mean_score, 4),
        discrepancy=discrepancy,
    )


def _compute_alignment(
    weights: list[list[float]],
    mask: list[list[float]],
    seq_len: int,
) -> float:
    """Compute 0–1 alignment between an attention matrix and a binary mask.

    Strategy: for each row, compute the fraction of attention mass that falls
    on positions marked 1 in the mask.  Average over rows.
    """
    row_scores: list[float] = []
    for i in range(min(len(weights), seq_len)):
        row_w = weights[i]
        row_m = mask[i]
        total = sum(row_w)
        if total < 1e-9:
            row_scores.append(0.0)
            continue
        on_mask = sum(
            row_w[j] for j in range(min(len(row_w), seq_len)) if row_m[j] > 0.5
        )
        row_scores.append(on_mask / total)
    if not row_scores:
        return 0.0
    return sum(row_scores) / len(row_scores)


# ---------------------------------------------------------------------------
# Utility: normalise raw attention logits → probabilities
# ---------------------------------------------------------------------------


def softmax_rows(matrix: list[list[float]]) -> list[list[float]]:
    """Apply row-wise softmax to an attention score matrix."""
    result = []
    for row in matrix:
        max_val = max(row) if row else 0.0
        exps = [math.exp(v - max_val) for v in row]
        total = sum(exps) or 1.0
        result.append([e / total for e in exps])
    return result


def entropy_of_attention(weights: list[list[float]]) -> float:
    """Compute mean row entropy (nats) of an attention matrix.

    High entropy → diffuse attention; low entropy → focused attention.
    """
    entropies: list[float] = []
    for row in weights:
        total = sum(row) or 1.0
        probs = [v / total for v in row]
        h = -sum(p * math.log(p + 1e-12) for p in probs)
        entropies.append(h)
    return sum(entropies) / max(len(entropies), 1)
