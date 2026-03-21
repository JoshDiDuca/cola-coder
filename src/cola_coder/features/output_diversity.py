"""Output Diversity Scorer — feature 47.

Measures the diversity of model-generated outputs using three complementary
metrics:

1. **Distinct-n**: Fraction of unique n-grams over all generated tokens.
   ``distinct-1`` and ``distinct-2`` are the standard metrics (Li et al. 2016).

2. **Self-BLEU**: Average BLEU score of each output against the rest.
   High self-BLEU → low diversity (outputs are similar to each other).

3. **Token entropy**: Entropy of the token distribution across all outputs.
   Higher entropy → more diverse vocabulary use.

All metrics are computed on tokenized strings (lists of tokens).

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → scorer returns a zeroed DiversityReport.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if diversity scoring is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DiversityReport:
    """Diversity metrics for a collection of generated outputs."""

    distinct_1: float = 0.0
    """Fraction of unique unigrams over all generated unigrams."""
    distinct_2: float = 0.0
    """Fraction of unique bigrams over all generated bigrams."""
    distinct_3: float = 0.0
    """Fraction of unique trigrams over all generated trigrams."""
    self_bleu: float = 0.0
    """Mean self-BLEU score (lower = more diverse)."""
    token_entropy: float = 0.0
    """Shannon entropy (bits) of the token distribution."""
    num_outputs: int = 0
    total_tokens: int = 0
    is_collapsed: bool = False
    """True if diversity metrics suggest mode collapse."""

    def diversity_score(self) -> float:
        """0.0–1.0 composite diversity score.

        Combines distinct-1, distinct-2, entropy (normalised), and inverse
        self-BLEU into a single number.
        """
        if self.num_outputs == 0:
            return 0.0
        # Normalise entropy to 0-1 using log2(vocab_size) as ceiling.
        # We cap at log2(10000) ≈ 13.3 bits as a reasonable max.
        max_entropy = math.log2(max(self.total_tokens, 2))
        entropy_norm = min(self.token_entropy / max(max_entropy, 1.0), 1.0)
        inv_bleu = 1.0 - self.self_bleu
        score = (
            0.25 * self.distinct_1
            + 0.25 * self.distinct_2
            + 0.25 * entropy_norm
            + 0.25 * inv_bleu
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def summary(self) -> str:
        collapsed = " [COLLAPSED]" if self.is_collapsed else ""
        return (
            f"DiversityReport{collapsed}: "
            f"dist-1={self.distinct_1:.3f} dist-2={self.distinct_2:.3f} "
            f"self-BLEU={self.self_bleu:.3f} entropy={self.token_entropy:.2f}b "
            f"score={self.diversity_score():.3f}"
        )


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------


class OutputDiversityScorer:
    """Computes diversity metrics for a collection of generated text outputs.

    Usage::

        scorer = OutputDiversityScorer()
        outputs = ["def foo(): return 1", "def bar(): return 2", ...]
        report = scorer.score(outputs)
        print(report.summary())
    """

    def __init__(
        self,
        collapse_threshold: float = 0.1,
        self_bleu_sample: int = 50,
    ) -> None:
        """
        Args:
            collapse_threshold: If distinct-1 < threshold the output is
                classified as mode-collapsed.
            self_bleu_sample: Max number of output pairs evaluated for
                self-BLEU (O(n^2) otherwise).
        """
        self.collapse_threshold = collapse_threshold
        self.self_bleu_sample = self_bleu_sample

    def score(
        self,
        outputs: Sequence[str],
        tokenize_fn=None,
    ) -> DiversityReport:
        """Compute diversity metrics.

        Args:
            outputs: Sequence of generated text strings.
            tokenize_fn: Optional callable(str) → List[str].  Defaults to
                whitespace split.

        Returns:
            DiversityReport with computed metrics.
        """
        if not FEATURE_ENABLED:
            return DiversityReport(num_outputs=len(outputs))
        if not outputs:
            return DiversityReport()

        tokenize = tokenize_fn or (lambda s: s.split())
        tokenized: List[List[str]] = [tokenize(o) for o in outputs]

        distinct_1 = _distinct_n(tokenized, 1)
        distinct_2 = _distinct_n(tokenized, 2)
        distinct_3 = _distinct_n(tokenized, 3)
        entropy = _token_entropy(tokenized)
        self_bleu = _self_bleu(tokenized, max_pairs=self.self_bleu_sample)
        total_tokens = sum(len(t) for t in tokenized)

        is_collapsed = distinct_1 < self.collapse_threshold

        return DiversityReport(
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            distinct_3=distinct_3,
            self_bleu=self_bleu,
            token_entropy=entropy,
            num_outputs=len(outputs),
            total_tokens=total_tokens,
            is_collapsed=is_collapsed,
        )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i: i + n]) for i in range(max(len(tokens) - n + 1, 0))]


def _distinct_n(tokenized: List[List[str]], n: int) -> float:
    """Compute distinct-n: unique n-grams / total n-grams."""
    all_ngrams: List[Tuple[str, ...]] = []
    for toks in tokenized:
        all_ngrams.extend(_ngrams(toks, n))
    if not all_ngrams:
        return 0.0
    unique = len(set(all_ngrams))
    return unique / len(all_ngrams)


def _token_entropy(tokenized: List[List[str]]) -> float:
    """Shannon entropy (bits) of the unigram distribution."""
    counts: Counter = Counter()
    for toks in tokenized:
        counts.update(toks)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _bleu_unigram_precision(
    hypothesis: List[str],
    references: List[List[str]],
) -> float:
    """Clipped unigram precision (BLEU-1 component) of hypothesis vs references."""
    if not hypothesis:
        return 0.0
    hyp_counts = Counter(hypothesis)
    max_ref_counts: Counter = Counter()
    for ref in references:
        ref_count = Counter(ref)
        for tok, cnt in ref_count.items():
            max_ref_counts[tok] = max(max_ref_counts[tok], cnt)
    clipped = sum(min(cnt, max_ref_counts[tok]) for tok, cnt in hyp_counts.items())
    return clipped / len(hypothesis)


def _sentence_bleu(hypothesis: List[str], references: List[List[str]]) -> float:
    """Simplified BLEU up to bigrams for two outputs."""
    if not hypothesis:
        return 0.0
    p1 = _bleu_unigram_precision(hypothesis, references)
    hyp_bigrams = _ngrams(hypothesis, 2)
    ref_bigrams_sets = [_ngrams(ref, 2) for ref in references]
    if not hyp_bigrams:
        return p1
    max_ref_bigram_counts: Counter = Counter()
    for ref_bg in ref_bigrams_sets:
        ref_count = Counter(ref_bg)
        for bg, cnt in ref_count.items():
            max_ref_bigram_counts[bg] = max(max_ref_bigram_counts[bg], cnt)
    hyp_bigram_counts = Counter(hyp_bigrams)
    clipped_2 = sum(
        min(cnt, max_ref_bigram_counts[bg]) for bg, cnt in hyp_bigram_counts.items()
    )
    p2 = clipped_2 / len(hyp_bigrams)
    if p1 == 0 or p2 == 0:
        return 0.0
    log_bleu = 0.5 * math.log(p1) + 0.5 * math.log(p2)
    return math.exp(log_bleu)


def _self_bleu(tokenized: List[List[str]], max_pairs: int = 50) -> float:
    """Compute mean self-BLEU by scoring each output against the others."""
    if len(tokenized) < 2:
        return 0.0

    scores = []
    # Limit computation: use at most max_pairs hypotheses
    n = min(len(tokenized), max_pairs)
    step = max(len(tokenized) // n, 1)
    indices = list(range(0, len(tokenized), step))[:n]

    for i in indices:
        hyp = tokenized[i]
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        scores.append(_sentence_bleu(hyp, refs))

    return sum(scores) / len(scores) if scores else 0.0
