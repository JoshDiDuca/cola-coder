"""
syntax_validity_rate.py

Measures the percentage of generated code samples that are syntactically valid.
Supports Python (via ast.parse) and basic TypeScript/JavaScript (bracket/brace matching).
"""

import ast
from collections import defaultdict
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


class SyntaxChecker:
    """Checks syntactic validity of code in various languages."""

    def check_python(self, code: str) -> bool:
        """Return True if code parses successfully with ast.parse."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def check_javascript(self, code: str) -> bool:
        """
        Return True if code passes basic bracket/brace/paren matching checks
        and a few structural heuristics for JavaScript.
        """
        return self._bracket_balanced(code)

    def check_typescript(self, code: str) -> bool:
        """
        Same bracket/brace matching as JS plus awareness of type annotation
        angle brackets (generics). Falls back to JS check since angle brackets
        in generics are hard to distinguish from comparison operators without a
        full parser; we use the same balanced-bracket check and accept that as
        sufficient for basic validity.
        """
        return self._bracket_balanced(code)

    def check(self, code: str, language: str) -> bool:
        """Dispatch to the appropriate language checker."""
        lang = language.lower().strip()
        if lang == "python":
            return self.check_python(code)
        elif lang in ("javascript", "js"):
            return self.check_javascript(code)
        elif lang in ("typescript", "ts"):
            return self.check_typescript(code)
        else:
            # Unknown language: fall back to bracket matching as a proxy
            return self._bracket_balanced(code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bracket_balanced(self, code: str) -> bool:
        """
        Check that all bracket/brace/paren pairs are balanced, ignoring
        characters inside string literals and single-line comments.
        """
        openers = {"(": ")", "[": "]", "{": "}"}
        closers = set(openers.values())
        stack = []

        i = 0
        length = len(code)

        while i < length:
            ch = code[i]

            # Single-line comment: skip to end of line
            if ch == "/" and i + 1 < length and code[i + 1] == "/":
                while i < length and code[i] != "\n":
                    i += 1
                continue

            # Multi-line comment: skip until */
            if ch == "/" and i + 1 < length and code[i + 1] == "*":
                i += 2
                while i + 1 < length and not (code[i] == "*" and code[i + 1] == "/"):
                    i += 1
                i += 2  # skip past */
                continue

            # String literal (double or single quote, including template literals)
            if ch in ('"', "'", "`"):
                quote = ch
                i += 1
                while i < length:
                    c = code[i]
                    if c == "\\" and i + 1 < length:
                        i += 2  # skip escaped character
                        continue
                    if c == quote:
                        break
                    i += 1
                i += 1
                continue

            if ch in openers:
                stack.append(openers[ch])
            elif ch in closers:
                if not stack or stack[-1] != ch:
                    return False
                stack.pop()

            i += 1

        return len(stack) == 0


class SyntaxValidityTracker:
    """Records code samples and tracks their syntactic validity rates."""

    def __init__(self):
        self._checker = SyntaxChecker()
        # Maps language -> list of bools (True = valid)
        self._results: dict[str, list[bool]] = defaultdict(list)

    def record(self, code: str, language: str) -> None:
        """Check code and record the result."""
        valid = self._checker.check(code, language)
        self._results[language.lower().strip()].append(valid)

    def validity_rate(self, language: Optional[str] = None) -> float:
        """
        Return the fraction of valid samples.

        If language is provided, restrict to that language.
        Returns 0.0 if no samples have been recorded.
        """
        if language is not None:
            results = self._results.get(language.lower().strip(), [])
        else:
            results = [v for vals in self._results.values() for v in vals]

        if not results:
            return 0.0
        return sum(results) / len(results)

    def total_checked(self) -> int:
        """Return the total number of samples recorded across all languages."""
        return sum(len(v) for v in self._results.values())

    def summary(self) -> dict:
        """
        Return a dict with overall stats and per-language breakdown.

        Structure:
        {
            "total": int,
            "overall_validity_rate": float,
            "languages": {
                "<lang>": {
                    "total": int,
                    "valid": int,
                    "invalid": int,
                    "validity_rate": float,
                }
            }
        }
        """
        languages = {}
        for lang, results in self._results.items():
            valid = sum(results)
            total = len(results)
            languages[lang] = {
                "total": total,
                "valid": valid,
                "invalid": total - valid,
                "validity_rate": valid / total if total else 0.0,
            }

        return {
            "total": self.total_checked(),
            "overall_validity_rate": self.validity_rate(),
            "languages": languages,
        }

    def reset(self) -> None:
        """Clear all recorded samples."""
        self._results.clear()
