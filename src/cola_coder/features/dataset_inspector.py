"""Dataset inspection utilities for training data analysis.

Provides sample distribution analysis, token statistics, quality checks,
duplicate detection, and formatted reporting for text/JSONL datasets.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# DatasetStats
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    total_samples: int = 0
    total_tokens: int = 0
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    language_dist: dict = field(default_factory=dict)
    quality_score: float = 0.0


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

_LANG_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("python", re.compile(r"\bdef \w+\s*\(|\bimport \w|\bfrom \w+ import\b|:\s*$", re.MULTILINE)),
    ("javascript", re.compile(r"\bfunction\s+\w+\s*\(|\bconst\s+\w+\s*=|\blet\s+\w+\s*=|\bconsole\.")),
    ("typescript", re.compile(r":\s*(string|number|boolean|void|any)\b|interface\s+\w+\s*\{|<\w+>")),
    ("java", re.compile(r"\bpublic\s+(class|static|void)\b|\bSystem\.out\.")),
    ("cpp", re.compile(r"#include\s*<|\bstd::\w+|\b(int|void)\s+main\s*\(")),
    ("rust", re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\s+|\bimpl\s+\w+|\bpub\s+(fn|struct|enum)\b")),
    ("go", re.compile(r"\bfunc\s+\w+\s*\(|\bpackage\s+\w+|\bfmt\.\w+\(")),
]


def _detect_language(text: str) -> str:
    """Return a best-guess language label for *text*, or 'other'."""
    scores: dict[str, int] = {}
    for lang, pattern in _LANG_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            scores[lang] = len(matches)
    if not scores:
        return "other"
    return max(scores, key=lambda k: scores[k])


def _tokenize_simple(text: str) -> list[str]:
    """Whitespace-split tokenizer — good enough for length stats."""
    return text.split()


def _quality_score_sample(text: str) -> float:
    """Return a [0, 1] quality score for a single sample.

    Criteria (each contributes equally):
    - Non-trivial length (>= 20 chars)
    - Contains at least one alphabetic word
    - Char diversity: unique chars / total chars >= 0.1
    - Not all whitespace / control chars
    """
    score = 0.0
    total_criteria = 4.0

    if len(text) >= 20:
        score += 1.0

    if re.search(r"[a-zA-Z]{2,}", text):
        score += 1.0

    if len(text) > 0:
        diversity = len(set(text)) / len(text)
        if diversity >= 0.1:
            score += 1.0

    stripped = text.strip()
    if stripped and not stripped.isspace():
        score += 1.0

    return score / total_criteria


# ---------------------------------------------------------------------------
# DatasetInspector
# ---------------------------------------------------------------------------


class DatasetInspector:
    """Inspect training datasets for quality, distribution, and statistics."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def inspect_samples(self, samples: list[str]) -> DatasetStats:
        """Analyze a list of in-memory text samples and return DatasetStats."""
        if not samples:
            return DatasetStats()

        lengths = [len(_tokenize_simple(s)) for s in samples]
        char_lengths = [len(s) for s in samples]
        total_tokens = sum(lengths)
        avg_length = total_tokens / len(samples)

        lang_counter: Counter[str] = Counter(_detect_language(s) for s in samples)
        language_dist = dict(lang_counter)

        quality_scores = [_quality_score_sample(s) for s in samples]
        overall_quality = sum(quality_scores) / len(quality_scores)

        return DatasetStats(
            total_samples=len(samples),
            total_tokens=total_tokens,
            avg_length=avg_length,
            min_length=min(char_lengths),
            max_length=max(char_lengths),
            language_dist=language_dist,
            quality_score=round(overall_quality, 4),
        )

    def inspect_file(self, path: str) -> DatasetStats:
        """Analyze a text or JSONL file and return DatasetStats.

        For JSONL files each line is parsed as JSON; the inspector tries the
        keys ``text``, ``content``, ``code``, and ``src`` in that order.
        Plain text files are treated as one sample per non-empty line.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        samples: list[str] = []
        suffix = p.suffix.lower()

        with p.open("r", encoding="utf-8", errors="replace") as fh:
            if suffix == ".jsonl":
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = None
                        for key in ("text", "content", "code", "src"):
                            if key in obj and isinstance(obj[key], str):
                                text = obj[key]
                                break
                        if text is None:
                            # Fall back to stringifying the whole object
                            text = json.dumps(obj)
                        samples.append(text)
                    except json.JSONDecodeError:
                        samples.append(line)
            else:
                for line in fh:
                    stripped = line.rstrip("\n")
                    if stripped:
                        samples.append(stripped)

        return self.inspect_samples(samples)

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def find_duplicates(self, samples: list[str]) -> list[tuple[int, int]]:
        """Return list of (i, j) index pairs where samples[i] == samples[j]."""
        seen: dict[str, int] = {}
        duplicates: list[tuple[int, int]] = []
        for idx, sample in enumerate(samples):
            if sample in seen:
                duplicates.append((seen[sample], idx))
            else:
                seen[sample] = idx
        return duplicates

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def find_low_quality(self, samples: list[str], min_length: int = 10) -> list[int]:
        """Return indices of samples that fail basic quality checks.

        A sample is considered low quality if:
        - Its character length is below *min_length*, OR
        - Its per-sample quality score is below 0.5
        """
        low: list[int] = []
        for idx, sample in enumerate(samples):
            if len(sample) < min_length:
                low.append(idx)
                continue
            if _quality_score_sample(sample) < 0.5:
                low.append(idx)
        return low

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_report(self, stats: DatasetStats) -> str:
        """Return a human-readable multi-line report string."""
        lines = [
            "=" * 50,
            "Dataset Inspection Report",
            "=" * 50,
            f"Total samples   : {stats.total_samples:,}",
            f"Total tokens    : {stats.total_tokens:,}",
            f"Avg length      : {stats.avg_length:.1f} tokens",
            f"Min length      : {stats.min_length} chars",
            f"Max length      : {stats.max_length} chars",
            f"Quality score   : {stats.quality_score:.4f}",
            "",
            "Language distribution:",
        ]
        if stats.language_dist:
            total = sum(stats.language_dist.values())
            for lang, count in sorted(stats.language_dist.items(), key=lambda x: -x[1]):
                pct = (count / total) * 100 if total else 0.0
                lines.append(f"  {lang:<14}: {count:>5} ({pct:.1f}%)")
        else:
            lines.append("  (none)")
        lines.append("=" * 50)
        return "\n".join(lines)

    def summary(self, stats: DatasetStats) -> dict:
        """Return a compact dict summary suitable for logging or JSON serialization."""
        return {
            "total_samples": stats.total_samples,
            "total_tokens": stats.total_tokens,
            "avg_length": stats.avg_length,
            "min_length": stats.min_length,
            "max_length": stats.max_length,
            "language_dist": stats.language_dist,
            "quality_score": stats.quality_score,
        }
