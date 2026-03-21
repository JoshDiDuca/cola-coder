"""Code Formatting Standardizer — Feature 91

Standardize code formatting in training data.  Normalizes:
- Indentation: tabs → spaces (4 for Python, 2 for TS/JS)
- Line endings: CRLF / CR → LF
- Trailing whitespace removed from every line
- Final newline ensured

Tracks formatting statistics across multiple files so callers can
understand how much cleanup the corpus needs.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the formatting standardizer is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class FormattingStats:
    """Cumulative statistics across all standardized files."""

    files_processed: int = 0
    files_changed: int = 0
    tab_fixes: int = 0
    trailing_ws_fixes: int = 0
    crlf_fixes: int = 0
    missing_final_newline_fixes: int = 0

    def merge(self, other: "FormattingStats") -> "FormattingStats":
        """Return a new stats object that sums self and other."""
        return FormattingStats(
            files_processed=self.files_processed + other.files_processed,
            files_changed=self.files_changed + other.files_changed,
            tab_fixes=self.tab_fixes + other.tab_fixes,
            trailing_ws_fixes=self.trailing_ws_fixes + other.trailing_ws_fixes,
            crlf_fixes=self.crlf_fixes + other.crlf_fixes,
            missing_final_newline_fixes=(
                self.missing_final_newline_fixes + other.missing_final_newline_fixes
            ),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "files_processed": self.files_processed,
            "files_changed": self.files_changed,
            "tab_fixes": self.tab_fixes,
            "trailing_ws_fixes": self.trailing_ws_fixes,
            "crlf_fixes": self.crlf_fixes,
            "missing_final_newline_fixes": self.missing_final_newline_fixes,
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class StandardizeResult:
    """Outcome of standardizing a single file / snippet."""

    code: str
    language: str
    changed: bool
    stats: FormattingStats = field(default_factory=FormattingStats)
    changes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Language defaults
# ---------------------------------------------------------------------------

_INDENT_SPACES: dict[str, int] = {
    "python": 4,
    "typescript": 2,
    "javascript": 2,
    "tsx": 2,
    "jsx": 2,
}

_DEFAULT_INDENT = 4


def _indent_width(language: str) -> int:
    return _INDENT_SPACES.get(language.lower(), _DEFAULT_INDENT)


# ---------------------------------------------------------------------------
# Core standardizer
# ---------------------------------------------------------------------------


class FormattingStandardizer:
    """Standardize code formatting for a given language."""

    def __init__(self) -> None:
        self._cumulative: FormattingStats = FormattingStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def standardize(self, code: str, language: str = "python") -> StandardizeResult:
        """Apply all formatting normalizations to *code*.

        Parameters
        ----------
        code:
            Raw source code to normalize.
        language:
            Programming language hint (``"python"``, ``"typescript"``, etc.).

        Returns
        -------
        StandardizeResult
            Contains the normalized code plus per-file statistics.
        """
        original = code
        stats = FormattingStats(files_processed=1)
        changes: list[str] = []
        width = _indent_width(language)

        # 1. Normalize CRLF / CR → LF
        fixed_le, count_crlf = self._fix_line_endings(code)
        if count_crlf:
            stats.crlf_fixes += count_crlf
            changes.append(f"fixed {count_crlf} CRLF/CR line endings")
        code = fixed_le

        # 2. Tabs → spaces
        fixed_tabs, count_tabs = self._fix_tabs(code, width)
        if count_tabs:
            stats.tab_fixes += count_tabs
            changes.append(f"replaced tabs with {width}-space indent on {count_tabs} lines")
        code = fixed_tabs

        # 3. Trailing whitespace
        fixed_ws, count_ws = self._fix_trailing_ws(code)
        if count_ws:
            stats.trailing_ws_fixes += count_ws
            changes.append(f"removed trailing whitespace from {count_ws} lines")
        code = fixed_ws

        # 4. Ensure final newline
        fixed_nl, added_nl = self._ensure_final_newline(code)
        if added_nl:
            stats.missing_final_newline_fixes += 1
            changes.append("added missing final newline")
        code = fixed_nl

        changed = code != original
        if changed:
            stats.files_changed = 1

        self._cumulative = self._cumulative.merge(stats)
        return StandardizeResult(
            code=code,
            language=language,
            changed=changed,
            stats=stats,
            changes=changes,
        )

    @property
    def cumulative_stats(self) -> FormattingStats:
        """Running totals across all calls to :meth:`standardize`."""
        return self._cumulative

    def reset_stats(self) -> None:
        """Reset the cumulative statistics counter."""
        self._cumulative = FormattingStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_line_endings(code: str) -> tuple[str, int]:
        """Normalize CRLF and bare CR to LF.  Returns (fixed, count)."""
        count = len(re.findall(r"\r\n|\r(?!\n)", code))
        fixed = code.replace("\r\n", "\n").replace("\r", "\n")
        return fixed, count

    @staticmethod
    def _fix_tabs(code: str, width: int) -> tuple[str, int]:
        """Replace leading tabs with *width* spaces.  Returns (fixed, lines_changed)."""
        lines = code.split("\n")
        out: list[str] = []
        changed = 0
        spaces = " " * width
        for line in lines:
            # Count leading tabs
            stripped = line.lstrip("\t")
            n_tabs = len(line) - len(stripped)
            if n_tabs:
                line = spaces * n_tabs + stripped
                changed += 1
            out.append(line)
        return "\n".join(out), changed

    @staticmethod
    def _fix_trailing_ws(code: str) -> tuple[str, int]:
        """Strip trailing whitespace from each line.  Returns (fixed, lines_changed)."""
        lines = code.split("\n")
        out: list[str] = []
        changed = 0
        for line in lines:
            stripped = line.rstrip()
            if stripped != line:
                changed += 1
            out.append(stripped)
        return "\n".join(out), changed

    @staticmethod
    def _ensure_final_newline(code: str) -> tuple[str, bool]:
        """Ensure the code ends with a single newline."""
        if not code:
            return code, False
        if not code.endswith("\n"):
            return code + "\n", True
        return code, False

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def standardize_batch(
        self, items: list[tuple[str, str]]
    ) -> list[StandardizeResult]:
        """Standardize a list of (code, language) pairs.

        Parameters
        ----------
        items:
            List of ``(code, language)`` tuples.

        Returns
        -------
        list[StandardizeResult]
        """
        return [self.standardize(code, lang) for code, lang in items]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_standardizer: Optional[FormattingStandardizer] = None


def get_standardizer() -> FormattingStandardizer:
    """Return (creating if needed) the module-level default standardizer."""
    global _default_standardizer
    if _default_standardizer is None:
        _default_standardizer = FormattingStandardizer()
    return _default_standardizer


def standardize_code(code: str, language: str = "python") -> StandardizeResult:
    """Module-level convenience wrapper around :class:`FormattingStandardizer`."""
    return get_standardizer().standardize(code, language)
