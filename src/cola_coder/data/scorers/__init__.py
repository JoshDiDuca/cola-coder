"""Data quality scoring pipeline.

Provides a unified protocol for scoring code samples with multiple
scoring systems (tsc, eslint, GitHub stars, LLM-judge, heuristic)
and combining their outputs into composite training weights.
"""

from cola_coder.data.scorers.protocol import (
    CompositeResult,
    CompositeScorer,
    ScorerProtocol,
    ScorerResult,
)

__all__ = [
    "CompositeResult",
    "CompositeScorer",
    "ScorerProtocol",
    "ScorerResult",
]
