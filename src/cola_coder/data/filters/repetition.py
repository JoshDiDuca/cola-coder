"""Gopher / MassiveText repetition filter plugin.

Detects *degenerate* documents — text/code dominated by repeated lines,
paragraphs, or word n-grams. This is the canonical frontier pre-training
quality screen (Rae et al. 2021, "Gopher"; reused by RedPajama-v2, FineWeb,
and DataComp-LM) and complements the filters cola-coder already has:

- ``content.py`` catches *structural* low-signal files (autogen markers,
  minified single lines, base64, lock files) via pattern matching.
- ``CodeScorer._score_duplication`` scores exact duplicate *lines* only, on a
  0.0-1.0 scale, and ignores paragraph- and n-gram-level repetition entirely.

The Gopher repetition family is finer-grained: it measures the *character mass*
locked up in repeated n-grams (2..10) and in duplicate lines/paragraphs, which
is exactly how looping generations, copy-pasted boilerplate, and accidental
content duplication show up. A document is REJECTED if ANY metric exceeds its
threshold (Gopher's "any-of" rule).

Thresholds are the published Gopher reference values (FineWeb Table A1 /
datatrove ``GopherRepetitionFilter`` / data-prep-kit gopher annotator):

    duplicate line fraction            > 0.30
    duplicate paragraph fraction       > 0.30
    duplicate line char fraction       > 0.20
    duplicate paragraph char fraction  > 0.20
    top 2-gram char fraction           > 0.20
    top 3-gram char fraction           > 0.18
    top 4-gram char fraction           > 0.16
    duplicate 5-gram char fraction     > 0.15
    duplicate 6-gram char fraction     > 0.14
    duplicate 7-gram char fraction     > 0.13
    duplicate 8-gram char fraction     > 0.12
    duplicate 9-gram char fraction     > 0.11
    duplicate 10-gram char fraction    > 0.10

For 2-4 grams the metric is the character fraction of the single MOST FREQUENT
n-gram ("top"); for 5-10 grams it is the character fraction of ALL repeated
n-grams ("duplicate"). Both follow the datatrove reference implementation.

Pure-Python, CPU-only, no model load, no network — safe alongside live training.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cola_coder.data.pipeline import DataRecord, FilterPlugin
from cola_coder.data.registry import register_filter

# Top-n-gram char-fraction thresholds (2..4): fraction of chars in the single
# most frequent n-gram. Reject if exceeded.
_TOP_NGRAM_THRESHOLDS: dict[int, float] = {2: 0.20, 3: 0.18, 4: 0.16}

# Duplicate-n-gram char-fraction thresholds (5..10): fraction of chars covered
# by ALL n-grams that occur more than once. Reject if exceeded.
_DUP_NGRAM_THRESHOLDS: dict[int, float] = {
    5: 0.15,
    6: 0.14,
    7: 0.13,
    8: 0.12,
    9: 0.11,
    10: 0.10,
}

# Word-splitting regex (datatrove uses a unicode-word split; this keeps
# identifiers/numbers intact, which is the right granularity for code too).
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class RepetitionThresholds:
    """Tunable thresholds for the Gopher repetition metrics.

    All values are upper bounds: a document is rejected when a metric STRICTLY
    exceeds its threshold. Defaults are the published Gopher reference values.
    """

    dup_line_frac: float = 0.30
    dup_para_frac: float = 0.30
    dup_line_char_frac: float = 0.20
    dup_para_char_frac: float = 0.20
    top_ngram_char_frac: dict[int, float] = field(
        default_factory=lambda: dict(_TOP_NGRAM_THRESHOLDS)
    )
    dup_ngram_char_frac: dict[int, float] = field(
        default_factory=lambda: dict(_DUP_NGRAM_THRESHOLDS)
    )


@dataclass
class RepetitionMetrics:
    """Computed Gopher repetition metrics for one document (all in 0.0-1.0)."""

    dup_line_frac: float
    dup_para_frac: float
    dup_line_char_frac: float
    dup_para_char_frac: float
    top_ngram_char_frac: dict[int, float]
    dup_ngram_char_frac: dict[int, float]


def _split_lines(text: str) -> list[str]:
    """Non-empty, stripped lines (Gopher ignores blank lines)."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Non-empty paragraphs split on blank-line boundaries."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _duplicate_fraction(units: list[str]) -> tuple[float, float]:
    """Fraction of duplicate units and the char mass they carry.

    A "duplicate" is every occurrence of a unit beyond its first
    (``count - 1`` per repeated unit), matching the Gopher definition.

    Returns:
        (unit_fraction, char_fraction) — both 0.0 when there are no units.
    """
    if not units:
        return 0.0, 0.0
    counts = Counter(units)
    total_chars = sum(len(u) for u in units)
    dup_units = 0
    dup_chars = 0
    for unit, count in counts.items():
        if count > 1:
            extra = count - 1
            dup_units += extra
            dup_chars += extra * len(unit)
    unit_frac = dup_units / len(units)
    char_frac = dup_chars / total_chars if total_chars else 0.0
    return unit_frac, char_frac


def _top_ngram_char_fraction(words: list[str], n: int, total_chars: int) -> float:
    """Char fraction COVERED by the single MOST FREQUENT word n-gram (Gopher 2..4).

    Char mass is measured by the distinct word positions the top n-gram's
    occurrences cover (overlaps counted once), so the result is bounded in
    [0.0, 1.0] — a naive ``chars * count`` overcounts overlapping windows and
    can exceed 1.0.
    """
    if total_chars <= 0 or len(words) < n:
        return 0.0
    n_windows = len(words) - n + 1
    ngram_counts = Counter(tuple(words[i : i + n]) for i in range(n_windows))
    top_ngram, top_count = ngram_counts.most_common(1)[0]
    if top_count <= 1:
        return 0.0
    word_lengths = [len(w) for w in words]
    covered = [False] * len(words)
    for i in range(n_windows):
        if tuple(words[i : i + n]) == top_ngram:
            for j in range(i, i + n):
                covered[j] = True
    top_chars = sum(word_lengths[i] for i in range(len(words)) if covered[i])
    return top_chars / total_chars


def _dup_ngram_char_fraction(words: list[str], n: int, total_chars: int) -> float:
    """Char fraction covered by ALL repeated word n-grams (Gopher 5..10).

    Following the datatrove reference: count the characters of token positions
    that fall inside any n-gram occurring more than once, without
    double-counting overlapping covered positions.
    """
    if total_chars <= 0 or len(words) < n:
        return 0.0
    n_windows = len(words) - n + 1
    ngram_counts = Counter(
        tuple(words[i : i + n]) for i in range(n_windows)
    )
    word_lengths = [len(w) for w in words]
    covered = [False] * len(words)
    for i in range(n_windows):
        gram = tuple(words[i : i + n])
        if ngram_counts[gram] > 1:
            for j in range(i, i + n):
                covered[j] = True
    dup_chars = sum(word_lengths[i] for i in range(len(words)) if covered[i])
    return dup_chars / total_chars


def compute_repetition_metrics(text: str) -> RepetitionMetrics:
    """Compute all Gopher repetition metrics for one document.

    Pure function — no I/O, no model, no global state. Safe to call from
    multiprocessing workers and from tests without torch.
    """
    lines = _split_lines(text)
    paragraphs = _split_paragraphs(text)
    words = _WORD_RE.findall(text)
    total_word_chars = sum(len(w) for w in words)

    dup_line_frac, dup_line_char_frac = _duplicate_fraction(lines)
    dup_para_frac, dup_para_char_frac = _duplicate_fraction(paragraphs)

    top_ngram = {
        n: _top_ngram_char_fraction(words, n, total_word_chars)
        for n in _TOP_NGRAM_THRESHOLDS
    }
    dup_ngram = {
        n: _dup_ngram_char_fraction(words, n, total_word_chars)
        for n in _DUP_NGRAM_THRESHOLDS
    }

    return RepetitionMetrics(
        dup_line_frac=dup_line_frac,
        dup_para_frac=dup_para_frac,
        dup_line_char_frac=dup_line_char_frac,
        dup_para_char_frac=dup_para_char_frac,
        top_ngram_char_frac=top_ngram,
        dup_ngram_char_frac=dup_ngram,
    )


@register_filter("repetition")
class RepetitionFilter(FilterPlugin):
    """Reject degenerate documents via Gopher/MassiveText repetition metrics.

    Config options (via ``setup()`` or YAML) — any subset of the threshold
    fields, all optional, all upper bounds:

        dup_line_frac, dup_para_frac, dup_line_char_frac, dup_para_char_frac
        top_ngram_char_frac: {2: float, 3: float, 4: float}
        dup_ngram_char_frac: {5: float, ..., 10: float}
        min_words: skip the n-gram screens for very short docs (default 50)
    """

    # Below this word count the document is too short for n-gram statistics to
    # be meaningful (matches datatrove's short-doc guard); line/paragraph
    # repetition is still cheap and is always evaluated.
    _DEFAULT_MIN_WORDS: int = 50

    def __init__(
        self,
        thresholds: RepetitionThresholds | None = None,
        min_words: int = _DEFAULT_MIN_WORDS,
    ) -> None:
        self._thresholds = thresholds or RepetitionThresholds()
        self._min_words = min_words

    def name(self) -> str:
        return "repetition"

    def setup(self, config: dict[str, Any]) -> None:
        t = self._thresholds
        if "dup_line_frac" in config:
            t.dup_line_frac = float(config["dup_line_frac"])
        if "dup_para_frac" in config:
            t.dup_para_frac = float(config["dup_para_frac"])
        if "dup_line_char_frac" in config:
            t.dup_line_char_frac = float(config["dup_line_char_frac"])
        if "dup_para_char_frac" in config:
            t.dup_para_char_frac = float(config["dup_para_char_frac"])
        if "top_ngram_char_frac" in config:
            t.top_ngram_char_frac = {
                int(k): float(v) for k, v in config["top_ngram_char_frac"].items()
            }
        if "dup_ngram_char_frac" in config:
            t.dup_ngram_char_frac = {
                int(k): float(v) for k, v in config["dup_ngram_char_frac"].items()
            }
        if "min_words" in config:
            self._min_words = int(config["min_words"])

    def check(self, record: DataRecord) -> tuple[bool, str]:
        content = record.content
        if not content:
            return True, ""

        metrics = compute_repetition_metrics(content)
        t = self._thresholds

        # Line / paragraph repetition — always checked (cheap, robust on short docs).
        if metrics.dup_line_frac > t.dup_line_frac:
            return False, f"dup_line_frac ({metrics.dup_line_frac:.2f})"
        if metrics.dup_para_frac > t.dup_para_frac:
            return False, f"dup_para_frac ({metrics.dup_para_frac:.2f})"
        if metrics.dup_line_char_frac > t.dup_line_char_frac:
            return False, f"dup_line_char_frac ({metrics.dup_line_char_frac:.2f})"
        if metrics.dup_para_char_frac > t.dup_para_char_frac:
            return False, f"dup_para_char_frac ({metrics.dup_para_char_frac:.2f})"

        # N-gram repetition — only on documents long enough for the statistics.
        word_count = len(_WORD_RE.findall(content))
        if word_count < self._min_words:
            return True, ""

        for n, limit in t.top_ngram_char_frac.items():
            value = metrics.top_ngram_char_frac.get(n, 0.0)
            if value > limit:
                return False, f"top_{n}gram_char_frac ({value:.2f})"

        for n, limit in t.dup_ngram_char_frac.items():
            value = metrics.dup_ngram_char_frac.get(n, 0.0)
            if value > limit:
                return False, f"dup_{n}gram_char_frac ({value:.2f})"

        return True, ""
