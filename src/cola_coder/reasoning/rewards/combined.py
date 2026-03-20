"""Multi-signal reward combining type check + syntax + completeness.

Instead of relying on a single signal, this combines multiple lightweight
checks to produce a richer reward for GRPO training.

Signals and weights:
- Type check (0.4): Does it type-check with tsc --strict?
- Syntax (0.2): Does it parse cleanly as valid TypeScript?
- Style (0.1): Naming conventions, formatting heuristics
- Completeness (0.3): Does the code look complete (not truncated)?

If tsc is not available, the type check weight is redistributed to
syntax and completeness checks.

For a TS dev: think of this as a multi-check CI pipeline — lint, type-check,
and format all contribute to whether a PR is "good". Same idea, but as a
reward signal for the model to learn from.
"""

import logging
import re

from .type_check import TypeCheckReward

logger = logging.getLogger(__name__)


def _check_syntax(code: str) -> float:
    """Check if code has basic syntactic validity (bracket/brace matching).

    Returns:
        1.0 = balanced braces/brackets/parens
        0.5 = minor imbalance (1-2 off)
        0.0 = major imbalance or empty
    """
    if not code or not code.strip():
        return 0.0

    # Count bracket/brace/paren balance
    openers = {"(": ")", "[": "]", "{": "}"}
    closers = {v: k for k, v in openers.items()}
    stack = []
    in_string = False
    string_char = None
    prev_char = None

    for ch in code:
        # Simple string tracking (doesn't handle template literals perfectly)
        if ch in ('"', "'", "`") and prev_char != "\\":
            if in_string and ch == string_char:
                in_string = False
                string_char = None
            elif not in_string:
                in_string = True
                string_char = ch
        elif not in_string:
            if ch in openers:
                stack.append(ch)
            elif ch in closers:
                if stack and stack[-1] == closers[ch]:
                    stack.pop()
                else:
                    stack.append(ch)  # Mismatch
        prev_char = ch

    imbalance = len(stack)
    if imbalance == 0:
        return 1.0
    elif imbalance <= 2:
        return 0.5
    else:
        return 0.0


def _check_style(code: str) -> float:
    """Check basic TypeScript style conventions.

    Checks:
    - camelCase for variables/functions (not snake_case)
    - PascalCase for types/interfaces/classes
    - Consistent indentation (2 or 4 spaces, not mixed)
    - No excessively long lines

    Returns: 0.0 - 1.0
    """
    if not code or not code.strip():
        return 0.0

    score = 1.0
    lines = code.splitlines()

    # Check for snake_case function/variable declarations (TS convention is camelCase)
    snake_case_pattern = re.compile(
        r"\b(?:let|const|var|function)\s+[a-z]+_[a-z]"
    )
    snake_count = sum(
        1 for line in lines if snake_case_pattern.search(line)
    )
    if snake_count > 0:
        score -= min(0.3, snake_count * 0.05)

    # Check for excessively long lines (>120 chars)
    long_lines = sum(1 for line in lines if len(line) > 120)
    if long_lines > 0:
        score -= min(0.2, long_lines * 0.02)

    # Check for consistent indentation
    indent_sizes = set()
    for line in lines:
        if line and not line.isspace():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 0:
                indent_sizes.add(indent % 4 == 0 or indent % 2 == 0)
    # Mixed indentation is a small penalty
    if len(indent_sizes) > 1:
        score -= 0.1

    return max(0.0, score)


def _check_completeness(code: str) -> float:
    """Check if code looks complete (not truncated mid-statement).

    Signs of truncation:
    - Unbalanced braces (more opens than closes)
    - Ends mid-line without semicolon/brace
    - Very short code (< 10 chars)

    Returns: 0.0 - 1.0
    """
    if not code or not code.strip():
        return 0.0

    stripped = code.strip()

    # Very short code is likely truncated or placeholder
    if len(stripped) < 10:
        return 0.1

    score = 1.0

    # Check brace balance (more opens than closes = truncated)
    open_braces = stripped.count("{") - stripped.count("}")
    if open_braces > 0:
        score -= min(0.5, open_braces * 0.15)
    elif open_braces < 0:
        score -= 0.2  # Extra closing braces = malformed

    # Check paren balance
    open_parens = stripped.count("(") - stripped.count(")")
    if open_parens > 0:
        score -= min(0.3, open_parens * 0.1)

    # Check if code ends cleanly
    last_line = stripped.splitlines()[-1].strip() if stripped.splitlines() else ""
    clean_endings = ("}", ";", ")", "]", "*/", "//")
    if last_line and not any(last_line.endswith(e) for e in clean_endings):
        # Doesn't end on a clean boundary — might be truncated
        score -= 0.2

    return max(0.0, score)


class CombinedReward:
    """Multi-signal reward combining type check + syntax + style + completeness.

    Weights:
    - Type check: 0.4 (the primary signal)
    - Syntax:     0.2 (bracket/brace matching)
    - Style:      0.1 (naming conventions, formatting)
    - Complete:   0.3 (is the code truncated?)

    If tsc is not available, type check weight is redistributed:
    - Syntax becomes 0.35, completeness becomes 0.55, style stays 0.1
    """

    # Default weights
    W_TYPE = 0.4
    W_SYNTAX = 0.2
    W_STYLE = 0.1
    W_COMPLETE = 0.3

    # Fallback weights (no tsc)
    W_SYNTAX_FALLBACK = 0.35
    W_STYLE_FALLBACK = 0.1
    W_COMPLETE_FALLBACK = 0.55

    def __init__(self):
        """Initialize combined reward.

        If tsc is available, uses it for type checking. Otherwise,
        redistributes the weight to other signals.
        """
        if TypeCheckReward.is_available():
            self.type_checker = TypeCheckReward()
            self._has_tsc = True
            logger.info("CombinedReward: tsc available, using type check signal")
        else:
            self.type_checker = None
            self._has_tsc = False
            logger.warning(
                "CombinedReward: tsc not available, using fallback weights. "
                "Install TypeScript for better reward signal: npm install -g typescript"
            )

    @property
    def has_type_checker(self) -> bool:
        """Whether tsc is available for type checking."""
        return self._has_tsc

    def score(self, code: str, context: dict | None = None) -> float:
        """Combined multi-signal score.

        Args:
            code: TypeScript code to score.
            context: Optional context dict (reserved for future use,
                     e.g. passing test files for execution-based reward).

        Returns:
            Float score in approximately [-0.2, 1.0] range.
        """
        result = self.detailed_score(code, context)
        return result["combined_score"]

    def detailed_score(self, code: str, context: dict | None = None) -> dict:
        """Return detailed breakdown of all signal scores.

        Returns dict with:
            combined_score: weighted sum
            type_score: tsc score (or None if unavailable)
            syntax_score: bracket matching score
            style_score: naming/formatting score
            completeness_score: truncation check score
            weights: dict of signal weights used
        """
        syntax_score = _check_syntax(code)
        style_score = _check_style(code)
        completeness_score = _check_completeness(code)

        if self._has_tsc and self.type_checker is not None:
            type_score = self.type_checker.score(code)
            combined = (
                type_score * self.W_TYPE
                + syntax_score * self.W_SYNTAX
                + style_score * self.W_STYLE
                + completeness_score * self.W_COMPLETE
            )
            weights = {
                "type": self.W_TYPE,
                "syntax": self.W_SYNTAX,
                "style": self.W_STYLE,
                "completeness": self.W_COMPLETE,
            }
        else:
            type_score = None
            combined = (
                syntax_score * self.W_SYNTAX_FALLBACK
                + style_score * self.W_STYLE_FALLBACK
                + completeness_score * self.W_COMPLETE_FALLBACK
            )
            weights = {
                "type": 0.0,
                "syntax": self.W_SYNTAX_FALLBACK,
                "style": self.W_STYLE_FALLBACK,
                "completeness": self.W_COMPLETE_FALLBACK,
            }

        return {
            "combined_score": combined,
            "type_score": type_score,
            "syntax_score": syntax_score,
            "style_score": style_score,
            "completeness_score": completeness_score,
            "weights": weights,
        }

    def score_batch(self, codes: list[str]) -> list[float]:
        """Score a batch of code strings.

        For efficiency, consider using BatchTypeChecker directly if you
        only need type check scores. This method calls score() per item.
        """
        return [self.score(code) for code in codes]
