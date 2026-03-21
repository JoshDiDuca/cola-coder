"""Code Deduplication Checker: measure overlap between generated and training code.

Uses n-gram overlap (Jaccard similarity on character and token n-grams) to flag
when generated code is suspiciously close to training examples — a sign of
memorisation rather than generalisation.

For a TS dev: like a plagiarism checker for your code-gen model.  Returns a
float 0-1 where 1 = identical copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the code dedup checker feature is active."""
    return FEATURE_ENABLED


@dataclass
class DedupResult:
    """Result of comparing generated code against training samples."""

    max_similarity: float  # highest single-sample similarity
    avg_similarity: float  # average over all training samples
    nearest_index: int  # index in training_samples of the closest match
    per_sample: list[float] = field(default_factory=list)  # similarity per sample

    def is_duplicate(self, threshold: float = 0.8) -> bool:
        """Return True if the max similarity exceeds *threshold*."""
        return self.max_similarity >= threshold

    def summary(self) -> str:
        return (
            f"max={self.max_similarity:.3f} avg={self.avg_similarity:.3f} "
            f"nearest_idx={self.nearest_index}"
        )


class CodeDedupChecker:
    """Check if generated code is too similar to training data.

    Combines character n-gram Jaccard and token-level Jaccard.

    Usage::

        checker = CodeDedupChecker()
        result = checker.check(generated, training_samples)
        if result.is_duplicate():
            print("Possible memorisation detected")
    """

    def __init__(
        self,
        char_ngram: int = 5,
        token_ngram: int = 3,
        char_weight: float = 0.5,
        token_weight: float = 0.5,
    ) -> None:
        self.char_ngram = char_ngram
        self.token_ngram = token_ngram
        self.char_weight = char_weight
        self.token_weight = token_weight

    def check(self, generated: str, training_samples: list[str]) -> DedupResult:
        """Compare *generated* against each sample in *training_samples*.

        Args:
            generated: The code snippet produced by the model.
            training_samples: List of training code strings to compare against.

        Returns:
            DedupResult with per-sample and aggregate similarities.
        """
        if not training_samples:
            return DedupResult(
                max_similarity=0.0, avg_similarity=0.0, nearest_index=-1, per_sample=[]
            )

        gen_char_ngrams = self._char_ngrams(generated, self.char_ngram)
        gen_token_ngrams = self._token_ngrams(generated, self.token_ngram)

        scores: list[float] = []
        for sample in training_samples:
            s_char = self._char_ngrams(sample, self.char_ngram)
            s_tok = self._token_ngrams(sample, self.token_ngram)
            char_j = self._jaccard(gen_char_ngrams, s_char)
            tok_j = self._jaccard(gen_token_ngrams, s_tok)
            score = self.char_weight * char_j + self.token_weight * tok_j
            scores.append(score)

        max_sim = max(scores)
        avg_sim = sum(scores) / len(scores)
        nearest = scores.index(max_sim)

        return DedupResult(
            max_similarity=max_sim,
            avg_similarity=avg_sim,
            nearest_index=nearest,
            per_sample=scores,
        )

    def similarity(self, code_a: str, code_b: str) -> float:
        """Return similarity score between two code strings (0-1).

        Convenience wrapper for pairwise comparison.
        """
        result = self.check(code_a, [code_b])
        return result.max_similarity

    # ------------------------------------------------------------------
    # N-gram helpers
    # ------------------------------------------------------------------

    def _char_ngrams(self, text: str, n: int) -> frozenset[str]:
        """Produce character n-grams from *text* (whitespace normalised)."""
        normalised = " ".join(text.split())
        if len(normalised) < n:
            return frozenset([normalised]) if normalised else frozenset()
        return frozenset(normalised[i : i + n] for i in range(len(normalised) - n + 1))

    def _token_ngrams(self, text: str, n: int) -> frozenset[tuple[str, ...]]:
        """Produce token-level n-grams from *text*."""
        tokens = re.findall(r"\w+|[^\w\s]", text)
        if len(tokens) < n:
            return frozenset([tuple(tokens)]) if tokens else frozenset()
        return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:  # type: ignore[type-arg]
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0
