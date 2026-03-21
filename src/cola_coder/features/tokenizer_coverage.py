"""Tokenizer Coverage Analyzer: measure how well a tokenizer handles code.

Computes:
- avg_tokens_per_word: low = efficient tokenizer, high = fragmented
- oov_rate: fraction of words that hit UNK/unknown token (0 if BPE, possible if vocab limited)
- compression_ratio: chars / tokens (higher = more compressed)
- vocab_coverage: fraction of vocab tokens that appear in the corpus

Works with any HuggingFace tokenizer (or any object with encode() and vocab).
No GPU required.

For a TS dev: like measuring how good your gzip compression is for a specific
file type — tells you if the tokenizer was built for your data or something else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the tokenizer coverage analyzer feature is active."""
    return FEATURE_ENABLED


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal protocol that any HuggingFace-style tokenizer satisfies."""

    def encode(self, text: str) -> Any:
        """Encode text to token IDs or an Encoding object."""
        ...


@dataclass
class CoverageReport:
    """Results from analyzing tokenizer coverage over a corpus."""

    avg_tokens_per_word: float = 0.0
    oov_rate: float = 0.0
    compression_ratio: float = 0.0  # chars / tokens
    vocab_coverage: float = 0.0  # fraction of vocab tokens seen
    total_words: int = 0
    total_tokens: int = 0
    total_chars: int = 0
    vocab_size: int = 0
    unique_tokens_seen: int = 0
    token_freq: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"avg_tok/word={self.avg_tokens_per_word:.2f} "
            f"oov={self.oov_rate:.2%} "
            f"compression={self.compression_ratio:.2f} "
            f"vocab_cov={self.vocab_coverage:.2%} "
            f"words={self.total_words} tokens={self.total_tokens}"
        )


class TokenizerCoverageAnalyzer:
    """Measure tokenizer coverage over a code corpus.

    Usage::

        from tokenizers import Tokenizer  # HuggingFace tokenizers
        tokenizer = Tokenizer.from_file("tokenizer.json")
        analyzer = TokenizerCoverageAnalyzer()
        corpus = ["def foo(): pass", "const x = 1;"]
        report = analyzer.analyze(tokenizer, corpus)
        print(report.summary())

    Also works with a mock tokenizer that implements ``encode(text)``.
    """

    def __init__(
        self,
        unk_token_id: int | None = None,
        max_samples: int = 10_000,
    ) -> None:
        self.unk_token_id = unk_token_id
        self.max_samples = max_samples

    def analyze(
        self,
        tokenizer: Any,
        corpus: list[str] | str,
        language: str = "auto",
    ) -> CoverageReport:
        """Analyze tokenizer coverage over *corpus*.

        Args:
            tokenizer: HuggingFace tokenizer or any object with ``encode()``.
            corpus: List of code strings, or a single string.
            language: Hint for word-splitting (``"python"``, ``"typescript"``,
                ``"javascript"``, ``"auto"``).

        Returns:
            CoverageReport with all computed metrics.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        corpus = corpus[: self.max_samples]
        report = CoverageReport()

        # Discover vocab size
        report.vocab_size = self._get_vocab_size(tokenizer)

        # Detect UNK id if not provided
        unk_id = self.unk_token_id if self.unk_token_id is not None else self._find_unk_id(
            tokenizer
        )

        total_tokens = 0
        total_chars = 0
        total_words = 0
        unk_count = 0
        seen_token_ids: set[int] = set()
        token_freq: dict[int, int] = {}

        for text in corpus:
            words = self._split_words(text, language)
            total_words += len(words)
            total_chars += len(text)

            ids = self._encode(tokenizer, text)
            total_tokens += len(ids)

            for tid in ids:
                seen_token_ids.add(tid)
                token_freq[tid] = token_freq.get(tid, 0) + 1
                if unk_id is not None and tid == unk_id:
                    unk_count += 1

        report.total_words = total_words
        report.total_tokens = total_tokens
        report.total_chars = total_chars
        report.token_freq = token_freq
        report.unique_tokens_seen = len(seen_token_ids)

        if total_words > 0:
            report.avg_tokens_per_word = total_tokens / total_words
        if total_tokens > 0 and unk_id is not None:
            report.oov_rate = unk_count / total_tokens
        if total_tokens > 0:
            report.compression_ratio = total_chars / total_tokens
        if report.vocab_size > 0:
            report.vocab_coverage = len(seen_token_ids) / report.vocab_size

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_words(text: str, language: str) -> list[str]:
        """Split text into words appropriate for the language."""
        # For all supported languages, identifier-style tokenization works well
        return re.findall(r"[a-zA-Z_]\w*", text)

    @staticmethod
    def _encode(tokenizer: Any, text: str) -> list[int]:
        """Encode text and return a flat list of integer token IDs."""
        try:
            result = tokenizer.encode(text)
            # HuggingFace tokenizers returns an Encoding object
            if hasattr(result, "ids"):
                return list(result.ids)
            # transformers tokenizers return a dict or list
            if isinstance(result, dict):
                return list(result.get("input_ids", []))
            return list(result)
        except Exception:
            return []

    @staticmethod
    def _get_vocab_size(tokenizer: Any) -> int:
        """Try to get the vocabulary size."""
        for attr in ("vocab_size", "get_vocab_size"):
            try:
                val = getattr(tokenizer, attr)
                return int(val() if callable(val) else val)
            except (AttributeError, TypeError, ValueError):
                pass
        try:
            return len(tokenizer.get_vocab())
        except AttributeError:
            pass
        return 0

    @staticmethod
    def _find_unk_id(tokenizer: Any) -> int | None:
        """Try to auto-detect the UNK token ID."""
        for attr in ("unk_token_id", "token_to_id"):
            try:
                val = getattr(tokenizer, attr)
                if callable(val):
                    result = val("[UNK]")
                    if result is not None:
                        return int(result)
                else:
                    if val is not None:
                        return int(val)
            except (AttributeError, TypeError, ValueError):
                pass
        return None
