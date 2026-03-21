"""Vocabulary Efficiency Analyzer — feature 45.

Analyzes a tokenizer vocabulary for:
- Unused tokens (present in vocab but never seen in a corpus sample).
- Rare tokens (frequency < rare_threshold, default 0.01% of all tokens).
- Subword fragmentation ratio: average tokens per "word" split on whitespace.
- Coverage efficiency: fraction of words that encode as a single token (ideally
  common words are single tokens in an efficient vocabulary).

Works with any dict-based vocabulary (token → id) and any iterable of text
samples, so it can be tested without a live tokenizer.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → analyzer returns an empty VocabReport.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if vocabulary analysis is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VocabReport:
    """Results of vocabulary efficiency analysis."""

    vocab_size: int = 0
    num_unused: int = 0
    num_rare: int = 0
    rare_threshold: float = 0.0001  # 0.01%
    fragmentation_ratio: float = 0.0
    """Average subword pieces per whitespace-split word."""
    single_token_coverage: float = 0.0
    """Fraction of unique words that map to a single token."""
    token_frequency: Dict[str, int] = field(default_factory=dict)
    """Raw token → count mapping."""
    rare_tokens: List[str] = field(default_factory=list)
    """Tokens whose frequency < rare_threshold * total_token_count."""
    unused_tokens: List[str] = field(default_factory=list)
    """Tokens from vocabulary not seen in the sample corpus."""
    total_tokens_seen: int = 0

    def efficiency_score(self) -> float:
        """0.0–1.0 heuristic efficiency score.

        Penalises high fragmentation, many unused/rare tokens, and low
        single-token coverage.
        """
        if self.vocab_size == 0:
            return 0.0
        unused_ratio = self.num_unused / self.vocab_size
        rare_ratio = self.num_rare / max(self.vocab_size, 1)
        # Fragmentation: 1.0 = perfect (one piece per word), higher = worse
        frag_penalty = max(0.0, (self.fragmentation_ratio - 1.0) / 9.0)  # normalise to 0-1
        score = (
            0.4 * self.single_token_coverage
            + 0.2 * (1.0 - min(unused_ratio, 1.0))
            + 0.2 * (1.0 - min(rare_ratio, 1.0))
            + 0.2 * (1.0 - min(frag_penalty, 1.0))
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def summary(self) -> str:
        return (
            f"VocabReport: vocab={self.vocab_size} "
            f"unused={self.num_unused} rare={self.num_rare} "
            f"frag={self.fragmentation_ratio:.2f} "
            f"coverage={self.single_token_coverage:.2%} "
            f"score={self.efficiency_score():.3f}"
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class VocabEfficiencyAnalyzer:
    """Analyzes tokenizer vocabulary efficiency.

    Parameters
    ----------
    tokenize_fn:
        Callable(text) → List[str] of token strings.  If None, a trivial
        whitespace tokenizer is used (only useful for testing).
    rare_threshold:
        Tokens whose frequency as a fraction of all tokens is below this
        are classified as "rare".
    """

    def __init__(
        self,
        tokenize_fn: Optional[Callable[[str], List[str]]] = None,
        rare_threshold: float = 0.0001,
    ) -> None:
        self.tokenize_fn = tokenize_fn or _whitespace_tokenize
        self.rare_threshold = rare_threshold

    def analyze(
        self,
        vocab: Dict[str, int],
        samples: Sequence[str],
    ) -> VocabReport:
        """Run analysis over a sample corpus.

        Args:
            vocab: Dict mapping token string → integer id.
            samples: Iterable of raw text strings to tokenize and analyse.

        Returns:
            VocabReport with computed metrics.
        """
        if not FEATURE_ENABLED:
            return VocabReport()

        # Tokenize all samples
        token_counts: Counter = Counter()
        frag_numerator = 0
        frag_denominator = 0
        single_token_words = 0
        unique_words: Set[str] = set()

        for text in samples:
            tokens = self.tokenize_fn(text)
            token_counts.update(tokens)

            # Fragmentation: compare token count vs whitespace-word count
            words = text.split()
            word_count = len(words)
            if word_count > 0:
                frag_numerator += len(tokens)
                frag_denominator += word_count

            # Coverage: check each word individually
            for word in words:
                unique_words.add(word)

        # Single-token coverage: fraction of unique words that tokenize to 1 token
        for word in unique_words:
            toks = self.tokenize_fn(word)
            if len(toks) == 1:
                single_token_words += 1

        total_token_count = sum(token_counts.values())
        fragmentation_ratio = (
            frag_numerator / frag_denominator if frag_denominator > 0 else 1.0
        )
        single_token_coverage = (
            single_token_words / max(len(unique_words), 1)
        )

        # Unused tokens: in vocab but never seen
        seen_tokens: Set[str] = set(token_counts.keys())
        vocab_tokens: Set[str] = set(vocab.keys())
        unused_tokens = sorted(vocab_tokens - seen_tokens)

        # Rare tokens
        threshold_count = self.rare_threshold * max(total_token_count, 1)
        rare_tokens = [
            tok for tok, cnt in token_counts.items()
            if cnt < threshold_count
        ]

        return VocabReport(
            vocab_size=len(vocab),
            num_unused=len(unused_tokens),
            num_rare=len(rare_tokens),
            rare_threshold=self.rare_threshold,
            fragmentation_ratio=fragmentation_ratio,
            single_token_coverage=single_token_coverage,
            token_frequency=dict(token_counts),
            rare_tokens=sorted(rare_tokens),
            unused_tokens=unused_tokens,
            total_tokens_seen=total_token_count,
        )


# ---------------------------------------------------------------------------
# Built-in tokenizers for testing
# ---------------------------------------------------------------------------


def _whitespace_tokenize(text: str) -> List[str]:
    """Trivial tokenizer: split on whitespace."""
    return text.split()


def make_char_tokenizer() -> Callable[[str], List[str]]:
    """Return a character-level tokenizer (useful for testing fragmentation)."""

    def tokenize(text: str) -> List[str]:
        return list(text.replace(" ", ""))

    return tokenize
