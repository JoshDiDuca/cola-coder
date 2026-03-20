"""Token Efficiency Metric: measure how efficiently the model uses tokens.

Quantifies "useful code per token" — useful for:
- Evaluating generated code quality (dense vs. verbose output)
- Filtering training data that's overly commented or padded
- Comparing model outputs across generations

For a TS dev: like a linter metric that scores signal-to-noise ratio in code.
Think of it as measuring how much actual logic you get per character/token spent.
"""

import re
import tokenize
import io
from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Boilerplate token patterns — tokens that add structure but little meaning
# ---------------------------------------------------------------------------
_BOILERPLATE_PATTERNS = re.compile(
    r"^(pass|return|self|cls|None|True|False|def|class|if|else|elif|for|while"
    r"|import|from|as|with|try|except|finally|raise|yield|lambda|and|or|not"
    r"|in|is|del|global|nonlocal|assert|break|continue|__init__|__str__"
    r"|__repr__|__len__|__eq__|print)$"
)


@dataclass
class TokenEfficiency:
    """Efficiency metrics for a piece of code.

    All ratios are in [0, 1].  Higher = more efficient / denser.
    """
    code_density: float = 0.0      # Non-whitespace, non-comment chars / total chars
    comment_ratio: float = 0.0     # Comment lines / total lines
    whitespace_ratio: float = 0.0  # Blank lines / total lines
    meaningful_ratio: float = 0.0  # Non-boilerplate tokens / total tokens
    overall: float = 0.0           # Weighted aggregate score


# ---------------------------------------------------------------------------
# EfficiencyAnalyzer
# ---------------------------------------------------------------------------

class EfficiencyAnalyzer:
    """Compute token efficiency metrics for Python source code."""

    # Weights for the overall score
    _WEIGHTS = {
        "code_density": 0.40,
        "comment_ratio": 0.20,   # inverted: fewer comments → higher score
        "whitespace_ratio": 0.15, # inverted: less blank space → higher score
        "meaningful_ratio": 0.25,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, code: str) -> TokenEfficiency:
        """Compute all metrics and return a TokenEfficiency dataclass."""
        cd = self.code_density(code)
        cr = self.comment_ratio(code)
        wr = self._whitespace_ratio(code)
        mr = self.meaningful_tokens(code)

        # Overall: reward density + meaningful tokens; penalise comment/whitespace bloat
        overall = (
            self._WEIGHTS["code_density"] * cd
            + self._WEIGHTS["comment_ratio"] * (1.0 - cr)
            + self._WEIGHTS["whitespace_ratio"] * (1.0 - wr)
            + self._WEIGHTS["meaningful_ratio"] * mr
        )
        overall = max(0.0, min(1.0, overall))

        return TokenEfficiency(
            code_density=cd,
            comment_ratio=cr,
            whitespace_ratio=wr,
            meaningful_ratio=mr,
            overall=overall,
        )

    def code_density(self, code: str) -> float:
        """Ratio of non-whitespace, non-comment characters to total characters.

        Strips comment text from the count so that comment-heavy code scores lower.
        Returns 0.0 for empty input.
        """
        if not code.strip():
            return 0.0

        # Remove comment text, keeping line structure intact
        code_only = self._strip_comments(code)
        total_chars = len(code)
        non_ws_chars = len(re.sub(r"\s", "", code_only))

        return non_ws_chars / total_chars if total_chars > 0 else 0.0

    def comment_ratio(self, code: str) -> float:
        """Fraction of non-empty lines that are purely comment lines.

        Returns 0.0 for empty input.
        """
        lines = code.splitlines()
        non_empty = [ln for ln in lines if ln.strip()]
        if not non_empty:
            return 0.0

        comment_lines = sum(
            1 for ln in non_empty if ln.strip().startswith("#")
        )
        return comment_lines / len(non_empty)

    def meaningful_tokens(self, code: str) -> float:
        """Ratio of non-boilerplate tokens to total tokens.

        Uses Python's tokenize module for accurate lexing.
        Falls back to a regex split if tokenization fails (e.g. syntax errors).
        Returns 0.0 for empty input.
        """
        tokens = self._get_name_tokens(code)
        if not tokens:
            return 0.0

        meaningful = [t for t in tokens if not _BOILERPLATE_PATTERNS.match(t)]
        return len(meaningful) / len(tokens)

    def compare(self, code_a: str, code_b: str) -> dict:
        """Compare the efficiency of two code snippets.

        Returns a dict with both analyses and a winner field.
        """
        e_a = self.analyze(code_a)
        e_b = self.analyze(code_b)

        if e_a.overall > e_b.overall:
            winner = "a"
        elif e_b.overall > e_a.overall:
            winner = "b"
        else:
            winner = "tie"

        return {
            "a": {
                "code_density": e_a.code_density,
                "comment_ratio": e_a.comment_ratio,
                "whitespace_ratio": e_a.whitespace_ratio,
                "meaningful_ratio": e_a.meaningful_ratio,
                "overall": e_a.overall,
            },
            "b": {
                "code_density": e_b.code_density,
                "comment_ratio": e_b.comment_ratio,
                "whitespace_ratio": e_b.whitespace_ratio,
                "meaningful_ratio": e_b.meaningful_ratio,
                "overall": e_b.overall,
            },
            "winner": winner,
            "delta": round(e_a.overall - e_b.overall, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _whitespace_ratio(self, code: str) -> float:
        """Fraction of total lines that are blank."""
        lines = code.splitlines()
        if not lines:
            return 0.0
        blank = sum(1 for ln in lines if not ln.strip())
        return blank / len(lines)

    def _strip_comments(self, code: str) -> str:
        """Return code with inline and full-line comment text replaced by spaces."""
        result = []
        for line in code.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # Full comment line — replace content with newline only
                result.append("\n")
            else:
                # Inline comment — remove from # onwards (naive but fast)
                in_str = False
                quote_char: Optional[str] = None
                out = []
                i = 0
                while i < len(line):
                    ch = line[i]
                    if in_str:
                        out.append(ch)
                        if ch == "\\" and i + 1 < len(line):
                            out.append(line[i + 1])
                            i += 2
                            continue
                        if ch == quote_char:
                            in_str = False
                    elif ch in ('"', "'"):
                        in_str = True
                        quote_char = ch
                        out.append(ch)
                    elif ch == "#":
                        break  # rest is comment
                    else:
                        out.append(ch)
                    i += 1
                result.append("".join(out))
        return "".join(result)

    def _get_name_tokens(self, code: str) -> list:
        """Return a list of NAME tokens from the code, using tokenize when possible."""
        try:
            tokens = []
            reader = io.StringIO(code).readline
            for tok_type, tok_str, _, _, _ in tokenize.generate_tokens(reader):
                if tok_type == tokenize.NAME:
                    tokens.append(tok_str)
            return tokens
        except tokenize.TokenError:
            # Fallback: split on non-identifier characters
            return re.findall(r"\b[A-Za-z_]\w*\b", code)


# ---------------------------------------------------------------------------
# EfficiencyTracker
# ---------------------------------------------------------------------------

class EfficiencyTracker:
    """Accumulate efficiency scores across multiple code snippets."""

    def __init__(self) -> None:
        self._analyzer = EfficiencyAnalyzer()
        self._records: list[TokenEfficiency] = []

    def record(self, code: str) -> None:
        """Analyze and store efficiency metrics for the given code snippet."""
        self._records.append(self._analyzer.analyze(code))

    def average_efficiency(self) -> float:
        """Mean overall efficiency across all recorded snippets.

        Returns 0.0 if no snippets have been recorded.
        """
        if not self._records:
            return 0.0
        return sum(r.overall for r in self._records) / len(self._records)

    def summary(self) -> dict:
        """Aggregate summary statistics across all recorded snippets."""
        if not self._records:
            return {
                "count": 0,
                "average_overall": 0.0,
                "average_code_density": 0.0,
                "average_comment_ratio": 0.0,
                "average_whitespace_ratio": 0.0,
                "average_meaningful_ratio": 0.0,
                "min_overall": 0.0,
                "max_overall": 0.0,
            }

        overalls = [r.overall for r in self._records]
        return {
            "count": len(self._records),
            "average_overall": round(sum(overalls) / len(overalls), 4),
            "average_code_density": round(
                sum(r.code_density for r in self._records) / len(self._records), 4
            ),
            "average_comment_ratio": round(
                sum(r.comment_ratio for r in self._records) / len(self._records), 4
            ),
            "average_whitespace_ratio": round(
                sum(r.whitespace_ratio for r in self._records) / len(self._records), 4
            ),
            "average_meaningful_ratio": round(
                sum(r.meaningful_ratio for r in self._records) / len(self._records), 4
            ),
            "min_overall": round(min(overalls), 4),
            "max_overall": round(max(overalls), 4),
        }
