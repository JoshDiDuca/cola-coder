"""Docstring Quality Scorer: rate the quality of Python docstrings.

Scores generated docstrings on four axes:
  - parameter_coverage : fraction of function params documented
  - return_mentioned   : whether a return value is described
  - example_included   : whether there is a usage example
  - description_quality: length / completeness of the summary sentence

Final score is a weighted average in [0.0, 1.0].

For a TS dev: think of it like an ESLint plugin that counts how many JSDoc
fields are present and gives a 0–100 score.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DocstringScore:
    """Detailed docstring quality scores."""

    overall: float  # 0.0 – 1.0
    parameter_coverage: float  # fraction of params documented
    return_mentioned: bool
    example_included: bool
    description_quality: float  # 0.0 – 1.0

    # Raw counts (for debugging)
    params_found: int = 0
    params_documented: int = 0
    docstring_length: int = 0

    def __repr__(self) -> str:
        return (
            f"DocstringScore(overall={self.overall:.2f}, "
            f"param_cov={self.parameter_coverage:.2f}, "
            f"return={self.return_mentioned}, "
            f"example={self.example_included})"
        )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class DocstringScorer:
    """Score docstrings extracted from Python source code.

    Parameters
    ----------
    weights:
        Dict mapping dimension name -> weight.  Defaults to
        ``{"params": 0.35, "return": 0.25, "example": 0.20, "desc": 0.20}``.
    min_desc_words:
        Minimum words in the summary line to get full description score.
    """

    _DEFAULT_WEIGHTS = {
        "params": 0.35,
        "return": 0.25,
        "example": 0.20,
        "desc": 0.20,
    }

    # Patterns for various docstring styles (Google, NumPy, Sphinx)
    _PARAM_PATTERNS = [
        re.compile(r"^\s*:param\s+(\w+)\s*:", re.MULTILINE),  # Sphinx
        re.compile(r"^\s*(\w+)\s+\(.*?\)\s*:", re.MULTILINE),  # NumPy
        re.compile(r"^\s*(\w+)\s*:", re.MULTILINE),  # Google
        re.compile(r"@param\s+\{?\w*\}?\s+(\w+)", re.MULTILINE),  # JSDoc-style
    ]
    _RETURN_PATTERNS = [
        re.compile(r"^\s*:returns?:", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^\s*Returns\s*\n\s*[-─]+", re.MULTILINE),
        re.compile(r"^\s*Returns:\s*$", re.MULTILINE),
        re.compile(r"@returns?\b", re.MULTILINE | re.IGNORECASE),
        re.compile(r"\breturns?\b.*:", re.IGNORECASE),
    ]
    _EXAMPLE_PATTERNS = [
        re.compile(r"^\s*>>>", re.MULTILINE),  # doctest
        re.compile(r"^\s*Examples?\s*\n\s*[-─]+", re.MULTILINE),
        re.compile(r"^\s*Examples?:\s*$", re.MULTILINE),
        re.compile(r"Example\s+usage", re.IGNORECASE),
    ]

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_desc_words: int = 8,
    ) -> None:
        self.weights = weights if weights is not None else dict(self._DEFAULT_WEIGHTS)
        self.min_desc_words = min_desc_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, code: str) -> float:
        """Score the first docstring found in *code*, returning 0.0–1.0.

        If no docstring is found the score is 0.0.
        """
        return self.score_detailed(code).overall

    def score_detailed(self, code: str) -> DocstringScore:
        """Return a full DocstringScore for the first function/class in *code*."""
        docstring, param_names = self._extract_docstring_and_params(code)

        if not docstring:
            return DocstringScore(
                overall=0.0,
                parameter_coverage=0.0,
                return_mentioned=False,
                example_included=False,
                description_quality=0.0,
            )

        param_cov = self._score_param_coverage(docstring, param_names)
        return_ok = self._has_return(docstring)
        example_ok = self._has_example(docstring)
        desc_score = self._score_description(docstring)

        w = self.weights
        overall = (
            w.get("params", 0) * param_cov
            + w.get("return", 0) * float(return_ok)
            + w.get("example", 0) * float(example_ok)
            + w.get("desc", 0) * desc_score
        )
        overall = max(0.0, min(1.0, overall))

        return DocstringScore(
            overall=overall,
            parameter_coverage=param_cov,
            return_mentioned=return_ok,
            example_included=example_ok,
            description_quality=desc_score,
            params_found=len(param_names),
            params_documented=round(param_cov * len(param_names)),
            docstring_length=len(docstring),
        )

    def score_raw_docstring(self, docstring: str, param_names: list[str]) -> DocstringScore:
        """Score a docstring string directly (without parsing source code)."""
        if not docstring or not docstring.strip():
            return DocstringScore(
                overall=0.0,
                parameter_coverage=0.0,
                return_mentioned=False,
                example_included=False,
                description_quality=0.0,
                params_found=len(param_names),
                params_documented=0,
                docstring_length=0,
            )

        param_cov = self._score_param_coverage(docstring, param_names)
        return_ok = self._has_return(docstring)
        example_ok = self._has_example(docstring)
        desc_score = self._score_description(docstring)

        w = self.weights
        overall = (
            w.get("params", 0) * param_cov
            + w.get("return", 0) * float(return_ok)
            + w.get("example", 0) * float(example_ok)
            + w.get("desc", 0) * desc_score
        )
        overall = max(0.0, min(1.0, overall))

        return DocstringScore(
            overall=overall,
            parameter_coverage=param_cov,
            return_mentioned=return_ok,
            example_included=example_ok,
            description_quality=desc_score,
            params_found=len(param_names),
            params_documented=round(param_cov * len(param_names)),
            docstring_length=len(docstring),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_docstring_and_params(self, code: str) -> tuple[str, list[str]]:
        """Parse *code* and return (docstring, param_names) for the first func/class."""
        code = textwrap.dedent(code)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback: look for a triple-quoted string in the raw text
            m = re.search(r'"""(.*?)"""', code, re.DOTALL)
            if m:
                return m.group(1), []
            m = re.search(r"'''(.*?)'''", code, re.DOTALL)
            if m:
                return m.group(1), []
            return "", []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node) or ""
                param_names: list[str] = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = node.args
                    param_names = [
                        a.arg
                        for a in (args.args + args.posonlyargs + args.kwonlyargs)
                        if a.arg != "self" and a.arg != "cls"
                    ]
                    if args.vararg:
                        param_names.append(args.vararg.arg)
                    if args.kwarg:
                        param_names.append(args.kwarg.arg)
                return docstring, param_names

        # Module-level docstring
        docstring = ast.get_docstring(tree) or ""
        return docstring, []

    def _score_param_coverage(self, docstring: str, param_names: list[str]) -> float:
        """Return fraction of params mentioned in the docstring."""
        if not param_names:
            # If the function has no meaningful params, full credit
            return 1.0
        mentioned = sum(1 for p in param_names if re.search(rf"\b{re.escape(p)}\b", docstring))
        return mentioned / len(param_names)

    def _has_return(self, docstring: str) -> bool:
        return any(p.search(docstring) for p in self._RETURN_PATTERNS)

    def _has_example(self, docstring: str) -> bool:
        return any(p.search(docstring) for p in self._EXAMPLE_PATTERNS)

    def _score_description(self, docstring: str) -> float:
        """Score the quality of the summary line (0.0–1.0)."""
        # Take the first non-empty line as the summary
        lines = docstring.strip().splitlines()
        if not lines:
            return 0.0
        summary = lines[0].strip()
        if not summary:
            # Try second line
            for line in lines[1:]:
                if line.strip():
                    summary = line.strip()
                    break
        if not summary:
            return 0.0
        word_count = len(summary.split())
        # Scale linearly from 0 words -> 0.0 to min_desc_words -> 1.0
        return min(1.0, word_count / self.min_desc_words)
