"""Syntax Error Classifier: classify common syntax errors in generated code.

Detects and categorizes errors such as:
- missing colons after if/for/while/def/class
- unmatched brackets/parentheses/braces
- indentation errors
- undefined names (simple heuristic)
- missing return type / incomplete expressions

Also tracks error category frequencies across multiple code samples,
useful for diagnosing what kinds of errors the model makes most often.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the syntax error classifier feature is active."""
    return FEATURE_ENABLED


class ErrorCategory(str, Enum):
    MISSING_COLON = "missing_colon"
    UNMATCHED_BRACKET = "unmatched_bracket"
    INDENTATION_ERROR = "indentation_error"
    UNDEFINED_NAME = "undefined_name"
    INCOMPLETE_EXPRESSION = "incomplete_expression"
    GENERAL_SYNTAX = "general_syntax"
    NONE = "none"  # Code is valid


@dataclass
class SyntaxErrorResult:
    """Result of classifying one code snippet."""

    is_valid: bool
    category: ErrorCategory
    message: str
    line_number: Optional[int] = None
    raw_error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "category": self.category.value,
            "message": self.message,
            "line_number": self.line_number,
            "raw_error": self.raw_error,
        }


@dataclass
class ErrorStats:
    """Aggregated error statistics across multiple samples."""

    total_samples: int = 0
    valid_count: int = 0
    error_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def validity_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.valid_count / self.total_samples

    def most_common_error(self) -> Optional[str]:
        if not self.error_counts:
            return None
        return max(self.error_counts, key=lambda k: self.error_counts[k])

    def summary(self) -> str:
        lines = [
            f"ErrorStats(total={self.total_samples}, valid={self.valid_count}, "
            f"validity_rate={self.validity_rate:.2%})"
        ]
        for cat, count in sorted(self.error_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {cat}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

_COLON_KEYWORDS = re.compile(
    r"^\s*(if|elif|else|for|while|def|class|try|except|finally|with|async\s+def|async\s+for|async\s+with)\b.*[^:]$",
    re.MULTILINE,
)

_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_PAIRS.items()}


def _check_unmatched_brackets(code: str) -> Optional[str]:
    """Return an error message if brackets are unmatched, else None."""
    stack = []
    in_str = False
    str_char = ""
    for i, ch in enumerate(code):
        if in_str:
            if ch == str_char and (i == 0 or code[i - 1] != "\\"):
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            str_char = ch
            continue
        if ch in _BRACKET_PAIRS:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            if not stack or stack[-1] != _CLOSE_TO_OPEN[ch]:
                return f"Unexpected '{ch}' — no matching '{_CLOSE_TO_OPEN[ch]}'"
            stack.pop()
    if stack:
        return f"Unclosed '{stack[-1]}'"
    return None


def _check_missing_colon(code: str) -> Optional[int]:
    """Return the 1-based line number of the first missing colon, or None."""
    for lineno, line in enumerate(code.splitlines(), start=1):
        stripped = line.rstrip()
        # Skip comment-only lines
        if re.match(r"^\s*#", stripped):
            continue
        if _COLON_KEYWORDS.match(stripped):
            # Make sure the line doesn't end with backslash (continuation)
            if not stripped.endswith("\\"):
                return lineno
    return None


def _check_indentation(code: str) -> Optional[tuple[int, str]]:
    """Return (lineno, msg) if there is an IndentationError, else None.

    Only IndentationError (a subclass of SyntaxError) is caught here;
    other SyntaxErrors are left for the caller to handle.
    """
    try:
        compile(code, "<string>", "exec")
    except IndentationError as exc:
        return (exc.lineno or 0, str(exc))
    except SyntaxError:
        pass
    return None


def _check_undefined_names(code: str) -> list[str]:
    """Heuristic: find names used before assignment in Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    defined: set[str] = set()
    undefined: list[str] = []

    # Very simple single-pass: collect all Name nodes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)

    # Builtins
    import builtins
    defined.update(dir(builtins))

    # Check Name Load nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined and node.id not in undefined:
                undefined.append(node.id)

    return undefined


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class SyntaxErrorClassifier:
    """Classify syntax errors in Python code snippets."""

    def classify(self, code: str) -> SyntaxErrorResult:
        """Classify the primary syntax error (if any) in *code*.

        Returns SyntaxErrorResult with is_valid=True if no error is found.
        """
        if not code.strip():
            return SyntaxErrorResult(
                is_valid=False,
                category=ErrorCategory.INCOMPLETE_EXPRESSION,
                message="Empty or whitespace-only code",
            )

        # 1. Check indentation first (Python-specific)
        indent_err = _check_indentation(code)
        if indent_err:
            lineno, msg = indent_err
            return SyntaxErrorResult(
                is_valid=False,
                category=ErrorCategory.INDENTATION_ERROR,
                message=f"Indentation error: {msg}",
                line_number=lineno,
                raw_error=msg,
            )

        # 2. Try to parse
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raw = str(exc)
            lineno = exc.lineno

            # Classify the SyntaxError
            if "expected ':'" in raw or _check_missing_colon(code) is not None:
                return SyntaxErrorResult(
                    is_valid=False,
                    category=ErrorCategory.MISSING_COLON,
                    message=f"Missing colon: {raw}",
                    line_number=lineno,
                    raw_error=raw,
                )

            bracket_err = _check_unmatched_brackets(code)
            if bracket_err:
                return SyntaxErrorResult(
                    is_valid=False,
                    category=ErrorCategory.UNMATCHED_BRACKET,
                    message=bracket_err,
                    line_number=lineno,
                    raw_error=raw,
                )

            if "EOF" in raw or "unexpected EOF" in raw or "was never closed" in raw:
                return SyntaxErrorResult(
                    is_valid=False,
                    category=ErrorCategory.INCOMPLETE_EXPRESSION,
                    message=f"Incomplete expression: {raw}",
                    line_number=lineno,
                    raw_error=raw,
                )

            return SyntaxErrorResult(
                is_valid=False,
                category=ErrorCategory.GENERAL_SYNTAX,
                message=f"Syntax error: {raw}",
                line_number=lineno,
                raw_error=raw,
            )

        # 3. Check bracket balance on the raw text
        bracket_err = _check_unmatched_brackets(code)
        if bracket_err:
            return SyntaxErrorResult(
                is_valid=False,
                category=ErrorCategory.UNMATCHED_BRACKET,
                message=bracket_err,
            )

        return SyntaxErrorResult(
            is_valid=True,
            category=ErrorCategory.NONE,
            message="No syntax errors detected",
        )

    def classify_many(self, code_samples: list[str]) -> list[SyntaxErrorResult]:
        """Classify multiple code samples and return a list of results."""
        return [self.classify(c) for c in code_samples]

    def track_errors(
        self,
        code_samples: list[str],
        stats: Optional[ErrorStats] = None,
    ) -> ErrorStats:
        """Classify samples and accumulate counts into an ErrorStats object."""
        if stats is None:
            stats = ErrorStats()
        for code in code_samples:
            result = self.classify(code)
            stats.total_samples += 1
            if result.is_valid:
                stats.valid_count += 1
            else:
                stats.error_counts[result.category.value] += 1
        return stats

    def find_undefined_names(self, code: str) -> list[str]:
        """Return potentially undefined names in valid Python code."""
        return _check_undefined_names(code)
