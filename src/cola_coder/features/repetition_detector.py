"""Repetition Detector: identify repetitive patterns in generated code.

Detects copy-paste blocks, repeated lines, and recurring token sequences that
indicate a model is looping or producing boilerplate-heavy output.

For a TS dev: think of this as a linter rule that catches when a code-gen model
starts producing the same function body five times in a row.

Score 0.0 = perfectly unique, 1.0 = completely repeated.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the repetition detector feature is active."""
    return FEATURE_ENABLED


@dataclass
class RepetitionReport:
    """Results from a repetition analysis pass."""

    score: float = 0.0  # 0 = no repetition, 1 = all repeated
    repeated_lines: list[str] = field(default_factory=list)
    repeated_blocks: list[tuple[str, int]] = field(default_factory=list)  # (block, count)
    duplicate_line_ratio: float = 0.0
    max_block_repeat: int = 0  # highest repeat count for any block
    total_lines: int = 0
    unique_lines: int = 0

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"RepetitionScore={self.score:.3f} "
            f"dup_lines={self.duplicate_line_ratio:.1%} "
            f"max_block_repeat={self.max_block_repeat} "
            f"lines={self.total_lines}/{self.unique_lines} unique"
        )


class RepetitionDetector:
    """Detect repetitive patterns in generated source code.

    Uses three signals:
    1. Duplicate line ratio — fraction of lines that are exact duplicates.
    2. Repeated n-gram blocks — windows of N consecutive lines that appear
       more than once.
    3. Token-level trigram repetition — repeated sequences of tokens.

    The final score is a weighted combination of these signals.
    """

    def __init__(
        self,
        block_size: int = 3,
        min_block_chars: int = 20,
        line_weight: float = 0.4,
        block_weight: float = 0.4,
        trigram_weight: float = 0.2,
    ) -> None:
        self.block_size = block_size
        self.min_block_chars = min_block_chars
        self.line_weight = line_weight
        self.block_weight = block_weight
        self.trigram_weight = trigram_weight

    def detect(self, code: str) -> RepetitionReport:
        """Analyze *code* for repetitive patterns.

        Args:
            code: Source code string to analyze.

        Returns:
            RepetitionReport with score and detailed findings.
        """
        report = RepetitionReport()
        lines = [ln.rstrip() for ln in code.splitlines()]
        # Strip blank lines for line-level analysis
        content_lines = [ln for ln in lines if ln.strip()]
        report.total_lines = len(content_lines)

        if report.total_lines == 0:
            return report

        # --- Signal 1: duplicate line ratio ---
        dup_ratio, repeated_lines = self._duplicate_line_ratio(content_lines)
        report.duplicate_line_ratio = dup_ratio
        report.repeated_lines = repeated_lines

        # --- Signal 2: repeated blocks ---
        block_score, repeated_blocks, max_repeat = self._repeated_blocks(content_lines)
        report.repeated_blocks = repeated_blocks
        report.max_block_repeat = max_repeat

        # --- Signal 3: token trigram repetition ---
        trigram_score = self._trigram_repetition(code)

        # --- Aggregate score ---
        report.score = min(
            1.0,
            self.line_weight * dup_ratio
            + self.block_weight * block_score
            + self.trigram_weight * trigram_score,
        )

        line_set = set(content_lines)
        report.unique_lines = len(line_set)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _duplicate_line_ratio(self, lines: list[str]) -> tuple[float, list[str]]:
        """Return (ratio, list_of_duplicate_lines)."""
        counts = Counter(lines)
        duplicated = [ln for ln, cnt in counts.items() if cnt > 1 and ln.strip()]
        if not lines:
            return 0.0, []
        # Fraction of *occurrences* that are duplicates (total - unique)
        total = len(lines)
        unique = len(counts)
        ratio = (total - unique) / total if total > 0 else 0.0
        return ratio, duplicated

    def _repeated_blocks(
        self, lines: list[str]
    ) -> tuple[float, list[tuple[str, int]], int]:
        """Return (score 0-1, repeated_blocks, max_repeat_count)."""
        n = self.block_size
        if len(lines) < n * 2:
            return 0.0, [], 0

        windows: list[str] = []
        for i in range(len(lines) - n + 1):
            block = "\n".join(lines[i : i + n])
            if len(block.strip()) >= self.min_block_chars:
                windows.append(block)

        counts = Counter(windows)
        repeated = [(blk, cnt) for blk, cnt in counts.items() if cnt > 1]
        repeated.sort(key=lambda x: -x[1])

        if not repeated:
            return 0.0, [], 0

        max_repeat = repeated[0][1]
        # Score: how many window positions are covered by repeated blocks
        repeated_positions = sum(cnt - 1 for _, cnt in repeated)
        score = min(1.0, repeated_positions / max(1, len(windows)))
        return score, repeated[:10], max_repeat  # cap at top-10 for report

    def _trigram_repetition(self, code: str) -> float:
        """Return fraction of token trigrams that are repeated."""
        tokens = re.findall(r"\w+|[^\w\s]", code)
        if len(tokens) < 3:
            return 0.0
        trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
        counts = Counter(trigrams)
        repeated = sum(cnt - 1 for cnt in counts.values() if cnt > 1)
        return min(1.0, repeated / max(1, len(trigrams)))
