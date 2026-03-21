"""Code Entropy Analyzer (improvement #67).

Measures information entropy of code at token and character level.
  - High entropy  = complex / hard to predict (novel, varied code)
  - Low entropy   = repetitive / boilerplate (generated files, constants)

Uses Shannon entropy H = -sum(p_i * log2(p_i)) over frequency distributions.

TypeScript analogy: like measuring how "compressible" a file is —
high-entropy code resists compression (like minified JS), low-entropy code
compresses well (like repeated boilerplate).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if entropy analysis is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EntropyReport:
    """Entropy analysis for a code snippet."""

    char_entropy: float               # Shannon entropy over character frequencies
    token_entropy: float              # Shannon entropy over token frequencies
    bigram_entropy: float             # Shannon entropy over token bigrams
    line_entropy: float               # Shannon entropy over line-length distribution

    vocabulary_size: int              # unique tokens
    total_tokens: int
    total_chars: int

    top_tokens: List[tuple] = field(default_factory=list)    # [(token, count)]
    compression_ratio: float = 0.0   # estimate: compressed / original

    # Interpretation
    complexity_label: str = ""       # "trivial" / "simple" / "moderate" / "complex" / "highly_complex"
    issues: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenizer (simple whitespace + symbol split, no ML)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s\w]")


def _tokenize(source: str) -> List[str]:
    return _TOKEN_RE.findall(source)


# ---------------------------------------------------------------------------
# Core entropy calculation
# ---------------------------------------------------------------------------


def _shannon_entropy(counter: Counter) -> float:
    """Shannon entropy in bits."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (cnt / total) * math.log2(cnt / total)
        for cnt in counter.values()
        if cnt > 0
    )


def _bigram_counter(tokens: List[str]) -> Counter:
    c: Counter = Counter()
    for i in range(len(tokens) - 1):
        c[(tokens[i], tokens[i + 1])] += 1
    return c


def _estimate_compression_ratio(source: str) -> float:
    """Estimate compression ratio using run-length encoding heuristic."""
    if not source:
        return 1.0
    # Count character-level runs
    runs = 1
    for i in range(1, len(source)):
        if source[i] != source[i - 1]:
            runs += 1
    # Lower runs/length = more repetitive = better compression
    ratio = runs / len(source)
    return ratio


def _complexity_label(token_entropy: float) -> str:
    if token_entropy < 2.0:
        return "trivial"
    elif token_entropy < 3.5:
        return "simple"
    elif token_entropy < 5.0:
        return "moderate"
    elif token_entropy < 7.0:
        return "complex"
    else:
        return "highly_complex"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CodeEntropyAnalyzer:
    """Measure information entropy of code snippets."""

    def __init__(self, top_n: int = 10) -> None:
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, source: str) -> EntropyReport:
        """Compute entropy metrics for the given source code."""
        if not source:
            return EntropyReport(
                char_entropy=0.0,
                token_entropy=0.0,
                bigram_entropy=0.0,
                line_entropy=0.0,
                vocabulary_size=0,
                total_tokens=0,
                total_chars=0,
                complexity_label="trivial",
            )

        # Character entropy
        char_counter: Counter = Counter(source)
        char_entropy = _shannon_entropy(char_counter)

        # Token entropy
        tokens = _tokenize(source)
        token_counter: Counter = Counter(tokens)
        token_entropy = _shannon_entropy(token_counter)

        # Bigram entropy
        bigram_counter = _bigram_counter(tokens)
        bigram_entropy = _shannon_entropy(bigram_counter)

        # Line-length entropy
        line_lengths = [len(ln) for ln in source.splitlines()]
        line_len_counter: Counter = Counter(line_lengths)
        line_entropy = _shannon_entropy(line_len_counter)

        # Compression estimate
        comp_ratio = _estimate_compression_ratio(source)

        label = _complexity_label(token_entropy)
        issues = self._generate_issues(token_entropy, comp_ratio, len(tokens), len(set(tokens)))

        return EntropyReport(
            char_entropy=char_entropy,
            token_entropy=token_entropy,
            bigram_entropy=bigram_entropy,
            line_entropy=line_entropy,
            vocabulary_size=len(token_counter),
            total_tokens=len(tokens),
            total_chars=len(source),
            top_tokens=token_counter.most_common(self.top_n),
            compression_ratio=comp_ratio,
            complexity_label=label,
            issues=issues,
        )

    def analyze_batch(self, sources: Sequence[str]) -> List[EntropyReport]:
        """Analyze a batch of code snippets."""
        return [self.analyze(s) for s in sources]

    def compare(self, source_a: str, source_b: str) -> Dict[str, float]:
        """Compare entropy metrics between two code snippets."""
        ra = self.analyze(source_a)
        rb = self.analyze(source_b)
        return {
            "char_entropy_diff": rb.char_entropy - ra.char_entropy,
            "token_entropy_diff": rb.token_entropy - ra.token_entropy,
            "bigram_entropy_diff": rb.bigram_entropy - ra.bigram_entropy,
            "vocab_size_diff": rb.vocabulary_size - ra.vocabulary_size,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_issues(
        token_entropy: float,
        comp_ratio: float,
        total_tokens: int,
        unique_tokens: int,
    ) -> List[str]:
        issues: List[str] = []
        if token_entropy < 1.5 and total_tokens > 20:
            issues.append(
                f"Low token entropy ({token_entropy:.2f} bits) — likely boilerplate or generated code"
            )
        if token_entropy > 8.0:
            issues.append(
                f"Very high token entropy ({token_entropy:.2f} bits) — may be obfuscated or minified"
            )
        if unique_tokens > 0 and (unique_tokens / total_tokens) < 0.05:
            issues.append("Very low type-token ratio — highly repetitive code")
        if comp_ratio < 0.3:
            issues.append("Low compression ratio — highly repetitive character patterns")
        return issues


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def analyze_entropy(source: str) -> EntropyReport:
    """Analyze code entropy with default settings."""
    return CodeEntropyAnalyzer().analyze(source)
