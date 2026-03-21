"""Token Frequency Analyzer: analyse token frequency distributions.

Provides:
  - Frequency counts and rank-ordered table
  - Zipf's law fit quality (ideal: freq ∝ 1/rank)
  - OOV (out-of-vocabulary) rate given a reference vocab
  - Frequency band analysis (very-frequent, common, rare, hapax legomena)
  - Entropy of the distribution

For a TS dev: similar to counting word/symbol occurrences in a large
codebase and checking whether the distribution follows the natural power-law
you'd expect in real language (Zipf's law).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FrequencyBands:
    """Token counts split into frequency bands."""

    very_frequent: int  # top-10% by frequency
    common: int  # 10-50%
    rare: int  # 50-90%
    hapax: int  # appear exactly once (hapax legomena)

    @property
    def total(self) -> int:
        return self.very_frequent + self.common + self.rare + self.hapax


@dataclass
class TokenFrequencyReport:
    """Results of token frequency analysis."""

    total_tokens: int
    unique_tokens: int
    most_common: list[tuple[str, int]] = field(default_factory=list)
    zipf_fit: float = 0.0  # 0.0–1.0; 1.0 = perfect Zipf fit
    entropy_bits: float = 0.0
    oov_rate: float = 0.0  # fraction of tokens not in reference vocab
    bands: FrequencyBands = field(
        default_factory=lambda: FrequencyBands(0, 0, 0, 0)
    )

    @property
    def type_token_ratio(self) -> float:
        """Vocabulary richness (unique / total)."""
        if self.total_tokens == 0:
            return 0.0
        return self.unique_tokens / self.total_tokens

    @property
    def summary(self) -> str:
        return (
            f"Tokens: {self.total_tokens} total / {self.unique_tokens} unique "
            f"(TTR={self.type_token_ratio:.3f}), "
            f"Zipf={self.zipf_fit:.3f}, H={self.entropy_bits:.2f} bits, "
            f"OOV={self.oov_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class TokenFrequencyAnalyzer:
    """Analyse frequency statistics over a sequence of tokens."""

    def __init__(self, top_k: int = 20) -> None:
        self.top_k = top_k

    def analyze(
        self,
        tokens: Sequence[str | int],
        vocab: set | None = None,
    ) -> TokenFrequencyReport:
        """Compute frequency statistics for *tokens*.

        Parameters
        ----------
        tokens:
            Flat sequence of token strings or token IDs.
        vocab:
            Optional reference vocabulary set.  Tokens not in this set
            count towards the OOV rate.
        """
        if not tokens:
            return TokenFrequencyReport(
                total_tokens=0,
                unique_tokens=0,
                bands=FrequencyBands(0, 0, 0, 0),
            )

        counts = Counter(tokens)
        total = len(tokens)
        unique = len(counts)

        most_common = counts.most_common(self.top_k)
        entropy = self._shannon_entropy(counts, total)
        zipf = self._zipf_fit(counts)
        oov = self._oov_rate(counts, total, vocab)
        bands = self._frequency_bands(counts)

        return TokenFrequencyReport(
            total_tokens=total,
            unique_tokens=unique,
            most_common=most_common,
            zipf_fit=zipf,
            entropy_bits=entropy,
            oov_rate=oov,
            bands=bands,
        )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(counts: Counter, total: int) -> float:
        """Shannon entropy in bits."""
        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _zipf_fit(counts: Counter) -> float:
        """Measure how well the distribution follows Zipf's law.

        Returns a score in [0, 1] where 1 is a perfect fit.
        Uses the Spearman correlation between observed log-freq and
        ideal Zipf log-freq (log(1/rank)).
        """
        ranked = [cnt for _, cnt in counts.most_common()]
        n = len(ranked)
        if n < 2:
            return 1.0

        log_observed = [math.log(c + 1) for c in ranked]
        log_ideal = [math.log(1 / (r + 1)) for r in range(n)]

        # Pearson r between log_observed and log_ideal
        mean_o = sum(log_observed) / n
        mean_i = sum(log_ideal) / n
        num = sum((o - mean_o) * (i - mean_i) for o, i in zip(log_observed, log_ideal))
        denom_o = math.sqrt(sum((o - mean_o) ** 2 for o in log_observed))
        denom_i = math.sqrt(sum((i - mean_i) ** 2 for i in log_ideal))
        if denom_o == 0 or denom_i == 0:
            return 1.0
        r = num / (denom_o * denom_i)
        # Map [-1, 1] → [0, 1]
        return max(0.0, min(1.0, (r + 1) / 2))

    @staticmethod
    def _oov_rate(counts: Counter, total: int, vocab: set | None) -> float:
        if vocab is None or total == 0:
            return 0.0
        oov_tokens = sum(cnt for tok, cnt in counts.items() if tok not in vocab)
        return oov_tokens / total

    @staticmethod
    def _frequency_bands(counts: Counter) -> FrequencyBands:
        n = len(counts)
        if n == 0:
            return FrequencyBands(0, 0, 0, 0)

        ranked = [cnt for _, cnt in counts.most_common()]
        top10 = max(1, int(n * 0.1))
        top50 = max(1, int(n * 0.5))
        top90 = max(1, int(n * 0.9))

        very_frequent = sum(ranked[:top10])
        common = sum(ranked[top10:top50])
        rare = sum(ranked[top50:top90])
        hapax = sum(1 for c in ranked if c == 1)

        return FrequencyBands(
            very_frequent=very_frequent,
            common=common,
            rare=rare,
            hapax=hapax,
        )
