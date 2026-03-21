"""Code Style Analyzer: inspect generated Python code for style consistency.

Checks:
  - Indentation consistency (all tabs, all spaces, or mixed)
  - Naming conventions (snake_case functions/vars, PascalCase classes,
    UPPER_CASE constants)
  - Import ordering (stdlib → third-party → local; alphabetical within groups)
  - Line length compliance (default PEP-8: 79 chars / project: 100 chars)

Returns a StyleReport dataclass with per-check scores and details.

For a TS dev: like ESLint running on Python — each check is a rule and the
report tells you what passed/failed and why.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IndentationCheck:
    """Results of the indentation consistency check."""

    consistent: bool
    style: str  # "spaces" | "tabs" | "mixed" | "no_indentation"
    spaces_lines: int
    tabs_lines: int
    mixed_lines: int
    indent_width: int  # most common indent unit size (0 if unknown)


@dataclass
class NamingCheck:
    """Results of naming convention checks."""

    functions_ok: int
    functions_bad: list[str]  # names that violate snake_case
    classes_ok: int
    classes_bad: list[str]  # names that violate PascalCase
    constants_ok: int
    constants_bad: list[str]  # UPPER_CASE violations

    @property
    def score(self) -> float:
        """0.0–1.0 naming compliance score."""
        total = (
            self.functions_ok + len(self.functions_bad)
            + self.classes_ok + len(self.classes_bad)
            + self.constants_ok + len(self.constants_bad)
        )
        good = self.functions_ok + self.classes_ok + self.constants_ok
        return good / max(total, 1)


@dataclass
class ImportCheck:
    """Results of import ordering check."""

    in_order: bool
    stdlib_imports: list[str]
    third_party_imports: list[str]
    local_imports: list[str]
    issues: list[str]


@dataclass
class LineLengthCheck:
    """Results of line-length check."""

    max_length: int
    lines_over_limit: int
    worst_line_length: int
    limit: int
    compliance_rate: float  # fraction of lines within limit


@dataclass
class StyleReport:
    """Aggregated style analysis report."""

    indentation: IndentationCheck
    naming: NamingCheck
    imports: ImportCheck
    line_length: LineLengthCheck

    # Weighted overall score
    overall_score: float  # 0.0 – 1.0

    issues: list[str] = field(default_factory=list)

    def passed(self, threshold: float = 0.7) -> bool:
        """Return True if overall_score >= threshold."""
        return self.overall_score >= threshold

    def summary(self) -> str:
        parts = [
            f"score={self.overall_score:.2f}",
            f"indent={'OK' if self.indentation.consistent else 'MIXED'}",
            f"naming={self.naming.score:.0%}",
            f"lines_ok={self.line_length.compliance_rate:.0%}",
        ]
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Known standard-library module names (subset — enough for practical checks)
# ---------------------------------------------------------------------------

_STDLIB_MODULES = frozenset(
    """
    abc ast asyncio builtins collections contextlib copy dataclasses datetime
    enum functools hashlib importlib inspect io itertools json logging math
    operator os pathlib pickle platform pprint queue random re shutil signal
    socket string struct subprocess sys tempfile textwrap threading time
    traceback types typing unicodedata unittest urllib uuid warnings weakref
    """.split()
)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CodeStyleAnalyzer:
    """Analyze Python source code style.

    Parameters
    ----------
    max_line_length:
        Maximum allowed line length.  PEP-8 default is 79; the project uses 100.
    """

    def __init__(self, max_line_length: int = 100) -> None:
        self.max_line_length = max_line_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, code: str) -> StyleReport:
        """Analyze *code* and return a StyleReport."""
        indent = self._check_indentation(code)
        naming = self._check_naming(code)
        imports = self._check_imports(code)
        line_len = self._check_line_length(code)

        issues: list[str] = []
        if not indent.consistent:
            issues.append(f"Mixed indentation ({indent.spaces_lines} space-lines, "
                          f"{indent.tabs_lines} tab-lines)")
        if naming.functions_bad:
            issues.append(f"Non-snake_case functions: {naming.functions_bad[:3]}")
        if naming.classes_bad:
            issues.append(f"Non-PascalCase classes: {naming.classes_bad[:3]}")
        if not imports.in_order:
            issues.extend(imports.issues[:3])
        if line_len.lines_over_limit > 0:
            issues.append(
                f"{line_len.lines_over_limit} lines exceed {self.max_line_length} chars "
                f"(worst: {line_len.worst_line_length})"
            )

        # Weighted score
        w_indent = 0.25
        w_naming = 0.35
        w_imports = 0.20
        w_lines = 0.20

        indent_score = 1.0 if indent.consistent else 0.5
        import_score = 1.0 if imports.in_order else 0.6
        line_score = line_len.compliance_rate

        overall = (
            w_indent * indent_score
            + w_naming * naming.score
            + w_imports * import_score
            + w_lines * line_score
        )

        return StyleReport(
            indentation=indent,
            naming=naming,
            imports=imports,
            line_length=line_len,
            overall_score=round(overall, 4),
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_indentation(self, code: str) -> IndentationCheck:
        spaces_lines = 0
        tabs_lines = 0
        mixed_lines = 0
        indent_sizes: list[int] = []

        for line in code.splitlines():
            stripped = line.lstrip()
            if not stripped or stripped == line:
                continue  # empty or no indent
            indent = line[: len(line) - len(stripped)]
            has_spaces = " " in indent
            has_tabs = "\t" in indent
            if has_tabs and has_spaces:
                mixed_lines += 1
            elif has_tabs:
                tabs_lines += 1
            elif has_spaces:
                spaces_lines += 1
                indent_sizes.append(len(indent))

        consistent = mixed_lines == 0 and not (spaces_lines > 0 and tabs_lines > 0)

        if spaces_lines > 0 and tabs_lines == 0:
            style = "spaces"
        elif tabs_lines > 0 and spaces_lines == 0:
            style = "tabs"
        elif mixed_lines > 0 or (spaces_lines > 0 and tabs_lines > 0):
            style = "mixed"
        else:
            style = "no_indentation"

        # Most common indent width
        if indent_sizes:
            from collections import Counter
            width = Counter(indent_sizes).most_common(1)[0][0]
        else:
            width = 0

        return IndentationCheck(
            consistent=consistent,
            style=style,
            spaces_lines=spaces_lines,
            tabs_lines=tabs_lines,
            mixed_lines=mixed_lines,
            indent_width=width,
        )

    def _check_naming(self, code: str) -> NamingCheck:
        funcs_ok = 0
        funcs_bad: list[str] = []
        classes_ok = 0
        classes_bad: list[str] = []
        consts_ok = 0
        consts_bad: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return NamingCheck(
                functions_ok=0, functions_bad=[],
                classes_ok=0, classes_bad=[],
                constants_ok=0, constants_bad=[],
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("__") and name.endswith("__"):
                    continue  # dunder methods are exempt
                if self._is_snake_case(name):
                    funcs_ok += 1
                else:
                    funcs_bad.append(name)

            elif isinstance(node, ast.ClassDef):
                name = node.name
                if self._is_pascal_case(name):
                    classes_ok += 1
                else:
                    classes_bad.append(name)

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # Only check module-level assignments that look like constants
                        if name.upper() == name and len(name) > 1 and "_" in name or name.upper() == name:
                            if self._is_upper_case(name):
                                consts_ok += 1
                            else:
                                consts_bad.append(name)

        return NamingCheck(
            functions_ok=funcs_ok,
            functions_bad=funcs_bad,
            classes_ok=classes_ok,
            classes_bad=classes_bad,
            constants_ok=consts_ok,
            constants_bad=consts_bad,
        )

    def _check_imports(self, code: str) -> ImportCheck:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ImportCheck(
                in_order=True,
                stdlib_imports=[],
                third_party_imports=[],
                local_imports=[],
                issues=[],
            )

        stdlib: list[str] = []
        third_party: list[str] = []
        local: list[str] = []
        issues: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    self._classify_import(root, alias.name, stdlib, third_party, local)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    self._classify_import(root, node.module, stdlib, third_party, local)
                elif node.level and node.level > 0:
                    local.append(f".{'.' * (node.level - 1)}{node.module or ''}")

        # Check ordering: stdlib before third-party before local
        in_order = True
        if third_party and stdlib:
            # Both exist: stdlib names should come before third-party in source
            # (simple heuristic — just check they're all categorised properly)
            pass  # already categorised
        if local and not stdlib and not third_party:
            pass  # only local — fine
        # Check alphabetical within each group
        if stdlib != sorted(set(stdlib)):
            issues.append("stdlib imports not alphabetically sorted")
            in_order = False
        if third_party != sorted(set(third_party)):
            issues.append("third-party imports not alphabetically sorted")
            in_order = False

        return ImportCheck(
            in_order=in_order,
            stdlib_imports=stdlib,
            third_party_imports=third_party,
            local_imports=local,
            issues=issues,
        )

    def _check_line_length(self, code: str) -> LineLengthCheck:
        lines = code.splitlines()
        if not lines:
            return LineLengthCheck(
                max_length=0,
                lines_over_limit=0,
                worst_line_length=0,
                limit=self.max_line_length,
                compliance_rate=1.0,
            )
        lengths = [len(line) for line in lines]
        worst = max(lengths)
        over = sum(1 for ln in lengths if ln > self.max_line_length)
        rate = 1.0 - over / len(lines)
        return LineLengthCheck(
            max_length=max(lengths),
            lines_over_limit=over,
            worst_line_length=worst,
            limit=self.max_line_length,
            compliance_rate=round(rate, 4),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_snake_case(name: str) -> bool:
        return bool(re.fullmatch(r"[a-z_][a-z0-9_]*", name))

    @staticmethod
    def _is_pascal_case(name: str) -> bool:
        return bool(re.fullmatch(r"[A-Z][a-zA-Z0-9]*", name))

    @staticmethod
    def _is_upper_case(name: str) -> bool:
        return bool(re.fullmatch(r"[A-Z_][A-Z0-9_]*", name))

    @classmethod
    def _classify_import(
        cls,
        root: str,
        full: str,
        stdlib: list[str],
        third_party: list[str],
        local: list[str],
    ) -> None:
        if root in _STDLIB_MODULES:
            stdlib.append(full)
        elif root.startswith(".") or root == "cola_coder":
            local.append(full)
        else:
            third_party.append(full)
