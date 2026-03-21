"""Variable Name Quality: score variable naming quality in Python code.

Checks:
  - Single-character variable names (except loop counters i/j/k/x/y/z)
  - Common abbreviations that reduce readability (buf, tmp, val, etc.)
  - Misleading names (e.g. names that shadow builtins like list, dict, str)
  - Naming convention consistency: snake_case vs camelCase in local scope

Returns a VariableNameReport dataclass with per-check scores and details.

For a TS dev: think of this as a linter that checks variable naming hygiene
beyond syntax — similar to the 'id-length' and 'id-blacklist' ESLint rules.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


FEATURE_ENABLED = True

# Single-char names that are conventionally acceptable
_ALLOWED_SINGLE_CHAR = frozenset("ijkxyzntsfaebc")

# Common abbreviations considered low-quality
_COMMON_ABBREVS = frozenset(
    {
        "buf",
        "tmp",
        "temp",
        "val",
        "var",
        "ret",
        "res",
        "idx",
        "cnt",
        "num",
        "len",
        "ptr",
        "err",
        "msg",
        "arg",
        "obj",
        "src",
        "dst",
        "cfg",
        "ctx",
        "req",
        "resp",
        "cb",
        "fn",
        "func",
        "proc",
        "info",
        "data",
        "item",
    }
)

# Python builtin names that variables should not shadow
_BUILTIN_NAMES = frozenset(
    {
        "list",
        "dict",
        "set",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "type",
        "object",
        "input",
        "print",
        "id",
        "hash",
        "len",
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "open",
        "map",
        "filter",
        "zip",
        "range",
        "enumerate",
        "sorted",
        "reversed",
    }
)

_SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_CAMEL_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VariableNameReport:
    """Results of variable naming quality analysis."""

    total_names: int
    single_char_names: list[str] = field(default_factory=list)
    abbreviations: list[str] = field(default_factory=list)
    shadowed_builtins: list[str] = field(default_factory=list)
    # Names using camelCase in a predominantly snake_case file (or vice versa)
    convention_violations: list[str] = field(default_factory=list)
    dominant_convention: str = "snake_case"  # "snake_case" | "camelCase" | "mixed"

    @property
    def score(self) -> float:
        """0.0–1.0 naming quality score (higher is better)."""
        if self.total_names == 0:
            return 1.0
        penalties = (
            len(self.single_char_names) * 0.5
            + len(self.abbreviations) * 0.3
            + len(self.shadowed_builtins) * 0.8
            + len(self.convention_violations) * 0.4
        )
        raw = 1.0 - penalties / max(self.total_names, 1)
        return max(0.0, min(1.0, raw))

    @property
    def issues(self) -> list[str]:
        """Human-readable list of all detected issues."""
        out: list[str] = []
        if self.single_char_names:
            out.append(f"Single-char names: {self.single_char_names}")
        if self.abbreviations:
            out.append(f"Abbreviations: {self.abbreviations}")
        if self.shadowed_builtins:
            out.append(f"Shadowed builtins: {self.shadowed_builtins}")
        if self.convention_violations:
            out.append(
                f"Convention violations ({self.dominant_convention} expected): "
                f"{self.convention_violations}"
            )
        return out


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class VariableNameAnalyzer:
    """Analyse variable naming quality in a Python source string."""

    def __init__(self, allow_single_char: frozenset[str] = _ALLOWED_SINGLE_CHAR) -> None:
        self.allow_single_char = allow_single_char

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, source: str) -> VariableNameReport:
        """Return a :class:`VariableNameReport` for *source*."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return VariableNameReport(total_names=0)

        names = self._collect_local_names(tree)
        if not names:
            return VariableNameReport(total_names=0)

        dominant = self._dominant_convention(names)

        single_char = [
            n for n in names if len(n) == 1 and n.lower() not in self.allow_single_char
        ]
        abbrevs = sorted({n for n in names if n.lower() in _COMMON_ABBREVS})
        shadowed = sorted({n for n in names if n in _BUILTIN_NAMES})

        if dominant == "snake_case":
            violations = [n for n in names if _CAMEL_RE.match(n) and not _SNAKE_RE.match(n)]
        elif dominant == "camelCase":
            violations = [n for n in names if _SNAKE_RE.match(n) and "_" in n]
        else:
            violations = []

        return VariableNameReport(
            total_names=len(names),
            single_char_names=sorted(set(single_char)),
            abbreviations=abbrevs,
            shadowed_builtins=shadowed,
            convention_violations=sorted(set(violations)),
            dominant_convention=dominant,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_local_names(tree: ast.Module) -> list[str]:
        """Walk AST and collect all local variable / parameter names."""
        names: list[str] = []
        for node in ast.walk(tree):
            # Assignments: x = …, x: int = …
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                elif isinstance(node, ast.AnnAssign) and node.target:
                    targets = [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
                    elif isinstance(t, ast.Tuple):
                        for elt in ast.walk(t):
                            if isinstance(elt, ast.Name):
                                names.append(elt.id)
            # Function / lambda parameters
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    names.append(arg.arg)
                if node.args.vararg:
                    names.append(node.args.vararg.arg)
                if node.args.kwarg:
                    names.append(node.args.kwarg.arg)
            # For-loop variables
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    names.append(node.target.id)
        # Filter out dunders and class/function names
        return [n for n in names if not (n.startswith("__") and n.endswith("__"))]

    @staticmethod
    def _dominant_convention(names: list[str]) -> str:
        """Determine whether snake_case or camelCase dominates."""
        snake = sum(1 for n in names if "_" in n and _SNAKE_RE.match(n))
        camel = sum(
            1
            for n in names
            if _CAMEL_RE.match(n) and not _SNAKE_RE.match(n) and any(c.isupper() for c in n)
        )
        if snake == 0 and camel == 0:
            return "snake_case"
        if snake >= camel:
            return "snake_case"
        return "camelCase"
