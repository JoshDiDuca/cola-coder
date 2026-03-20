"""Constitutional Coding: principle-based static analysis for generated code.

Implements a Constitutional AI-inspired checker (Bai et al., Anthropic 2022) that
evaluates code against a fixed set of coding principles without requiring a model
in the loop.  Each principle is a lightweight regex/AST heuristic that returns a
pass/fail decision and an optional line hint.

For a TS dev: think of this as ESLint rules (each principle = one rule), but the
"linter" is defined in pure Python so it can run inside the training loop to filter
or score generated code before it enters the dataset.

Two levels of integration:
  1. Offline data-generation filter — discard or flag bad generated samples.
  2. Training-time reward signal — penalise samples with 'error' violations.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CodingPrinciple:
    """A single constitutional rule for code quality.

    Attributes:
        name:        Short machine-readable identifier (e.g. 'no_eval').
        description: Human-readable explanation of what the rule checks.
        check_fn:    Callable[str, list[Violation]] — receives the full source
                     string and returns zero or more Violation objects.
        severity:    'error' | 'warning' | 'info'.  Errors should block the
                     sample from training; warnings are informational.
    """

    name: str
    description: str
    check_fn: Callable[[str], list["Violation"]]
    severity: str = "warning"  # 'error' | 'warning' | 'info'


@dataclass
class Violation:
    """A single principle violation found in a piece of code.

    Attributes:
        principle_name: Name of the CodingPrinciple that raised this violation.
        severity:       Inherited from the principle at check time.
        message:        Human-readable description of what was found.
        line_hint:      1-based line number if the violation can be localised,
                        or None for file-level checks.
    """

    principle_name: str
    severity: str
    message: str
    line_hint: Optional[int] = None


# ---------------------------------------------------------------------------
# Check functions (one per default principle)
# ---------------------------------------------------------------------------

# Regex for common secret patterns:
#   - Variable names that suggest a key/secret/token/password
#   - Assigned to a string literal that looks non-trivial (>= 8 chars)
_SECRET_VAR_RE = re.compile(
    r"""(?ix)
    (?:api[_-]?key | secret[_-]?key | auth[_-]?token | access[_-]?token
       | password | passwd | private[_-]?key | client[_-]?secret
       | bearer[_-]?token | db[_-]?pass)
    \s*=\s*
    ['"][^'"]{8,}['"]
    """,
)

# Patterns that look like real secrets (common prefixes used by APIs):
_SECRET_VALUE_RE = re.compile(
    r"""['"](?:sk-|ghp_|xox[bpoa]-|AKIA|AIza|ya29\.|Bearer\s)[A-Za-z0-9_\-/+]{8,}['"]"""
)


def _check_no_hardcoded_secrets(code: str) -> list[Violation]:
    violations: list[Violation] = []
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if _SECRET_VAR_RE.search(line) or _SECRET_VALUE_RE.search(line):
            violations.append(
                Violation(
                    principle_name="no_hardcoded_secrets",
                    severity="error",
                    message=(
                        "Possible hardcoded secret detected. "
                        "Use environment variables or a secrets manager instead."
                    ),
                    line_hint=i,
                )
            )
    return violations


def _check_no_eval(code: str) -> list[Violation]:
    """Detect eval() / exec() usage — both are security risks in generated code."""
    violations: list[Violation] = []
    # Match eval( or exec( not preceded by a dot (i.e. not obj.eval())
    pattern = re.compile(r"(?<!\.)(?:eval|exec)\s*\(")
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if pattern.search(line):
            violations.append(
                Violation(
                    principle_name="no_eval",
                    severity="error",
                    message=(
                        "Use of eval() or exec() detected. "
                        "These functions execute arbitrary code and are a security risk."
                    ),
                    line_hint=i,
                )
            )
    return violations


def _check_has_error_handling(code: str) -> list[Violation]:
    """Warn when a module-level function has no try/except block."""
    violations: list[Violation] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Only check top-level and class-method functions (depth <= 2)
        has_try = any(isinstance(child, ast.Try) for child in ast.walk(node))
        has_raise = any(isinstance(child, ast.Raise) for child in ast.walk(node))
        body_stmts = [s for s in node.body if not isinstance(s, (ast.Pass, ast.Expr))]
        if body_stmts and not has_try and not has_raise:
            violations.append(
                Violation(
                    principle_name="has_error_handling",
                    severity="warning",
                    message=(
                        f"Function '{node.name}' has no try/except or raise statement. "
                        "Consider adding error handling for robustness."
                    ),
                    line_hint=node.lineno,
                )
            )
    return violations


def _check_no_star_import(code: str) -> list[Violation]:
    """Disallow 'from module import *' — pollutes the namespace."""
    violations: list[Violation] = []
    pattern = re.compile(r"^\s*from\s+\S+\s+import\s+\*")
    for i, line in enumerate(code.splitlines(), 1):
        if pattern.match(line):
            violations.append(
                Violation(
                    principle_name="no_star_import",
                    severity="warning",
                    message=(
                        "Wildcard import 'from X import *' detected. "
                        "Import only the names you need to avoid namespace pollution."
                    ),
                    line_hint=i,
                )
            )
    return violations


def _check_no_print_debug(code: str) -> list[Violation]:
    """Warn about bare print() calls that look like debug output."""
    violations: list[Violation] = []
    # Match print( not inside a docstring — simple heuristic: not preceded by triple-quote
    pattern = re.compile(r"(?<!\w)print\s*\(")
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if pattern.search(line):
            violations.append(
                Violation(
                    principle_name="no_print_debug",
                    severity="info",
                    message=(
                        "print() call detected. "
                        "Use the logging module instead of print for library/production code."
                    ),
                    line_hint=i,
                )
            )
    return violations


def _check_has_docstring(code: str) -> list[Violation]:
    """Warn when public functions/classes lack a docstring."""
    violations: list[Violation] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Skip private names (_foo, __foo)
        if node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node)
        if not docstring:
            kind = "Function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "Class"
            violations.append(
                Violation(
                    principle_name="has_docstring",
                    severity="info",
                    message=(
                        f"{kind} '{node.name}' is missing a docstring. "
                        "Add a brief description of its purpose and parameters."
                    ),
                    line_hint=node.lineno,
                )
            )
    return violations


def _check_no_magic_numbers(code: str) -> list[Violation]:
    """Flag numeric literals used directly in expressions (magic numbers)."""
    violations: list[Violation] = []
    # Allow: 0, 1, -1, 2 (very common), and numbers in type annotations / default args
    _ALLOWED = {0, 1, -1, 2}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        # Only flag numbers used as operands in BinOp/Compare/Assign, not in defaults
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.n if hasattr(node, "n") else node.value, (int, float)):
            continue
        val = node.value if hasattr(node, "value") else node.n
        if isinstance(val, bool):
            continue
        if val in _ALLOWED:
            continue
        # Check the lineno is available
        lineno = getattr(node, "lineno", None)
        violations.append(
            Violation(
                principle_name="no_magic_numbers",
                severity="info",
                message=(
                    f"Magic number {val!r} detected. "
                    "Assign it to a named constant so the intent is clear."
                ),
                line_hint=lineno,
            )
        )
    return violations


def _check_no_global_state(code: str) -> list[Violation]:
    """Warn about mutable module-level state (lists, dicts, sets assigned at top level)."""
    violations: list[Violation] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in tree.body:
        # Only look at top-level assignments
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
            for target in node.targets:
                name = getattr(target, "id", None)
                if name and not name.isupper():  # ALL_CAPS constants are fine
                    violations.append(
                        Violation(
                            principle_name="no_global_state",
                            severity="warning",
                            message=(
                                f"Mutable global variable '{name}' detected. "
                                "Prefer constants (ALL_CAPS) or encapsulate state in a class."
                            ),
                            line_hint=node.lineno,
                        )
                    )
    return violations


def _check_proper_naming(code: str) -> list[Violation]:
    """Warn about single-character variable names outside of known conventional uses."""
    violations: list[Violation] = []
    # Common single-letter names that are accepted by convention
    _CONVENTIONAL = {"i", "j", "k", "n", "x", "y", "z", "e", "f", "v", "k"}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        name = node.id
        if len(name) == 1 and name.isalpha() and name.lower() not in _CONVENTIONAL:
            lineno = getattr(node, "lineno", None)
            violations.append(
                Violation(
                    principle_name="proper_naming",
                    severity="info",
                    message=(
                        f"Single-character variable '{name}' detected at line {lineno}. "
                        "Use descriptive names to improve readability."
                    ),
                    line_hint=lineno,
                )
            )
    return violations


def _check_no_bare_except(code: str) -> list[Violation]:
    """Flag bare 'except:' clauses that swallow all exceptions silently."""
    violations: list[Violation] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            # type is None → bare except:
            violations.append(
                Violation(
                    principle_name="no_bare_except",
                    severity="error",
                    message=(
                        "Bare 'except:' clause detected. "
                        "Catch specific exceptions (e.g. 'except ValueError:') "
                        "to avoid silently swallowing unexpected errors."
                    ),
                    line_hint=node.lineno,
                )
            )
    return violations


def _check_has_type_annotations(code: str) -> list[Violation]:
    """Info-level: encourage type annotations on function parameters and return values."""
    violations: list[Violation] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Check return annotation
        if node.returns is None and node.name != "__init__":
            violations.append(
                Violation(
                    principle_name="has_type_annotations",
                    severity="info",
                    message=(
                        f"Function '{node.name}' is missing a return type annotation. "
                        "Add '-> <type>' to improve IDE support and static analysis."
                    ),
                    line_hint=node.lineno,
                )
            )
        # Check argument annotations (skip 'self' and 'cls')
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is None:
                violations.append(
                    Violation(
                        principle_name="has_type_annotations",
                        severity="info",
                        message=(
                            f"Parameter '{arg.arg}' in function '{node.name}' "
                            "has no type annotation."
                        ),
                        line_hint=getattr(arg, "lineno", node.lineno),
                    )
                )
    return violations


def _check_no_unused_imports(code: str) -> list[Violation]:
    """Warn about import statements where the imported name never appears in the code body."""
    violations: list[Violation] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    imported_names: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                used_name = alias.asname if alias.asname else alias.name.split(".")[0]
                imported_names.append((used_name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                used_name = alias.asname if alias.asname else alias.name
                imported_names.append((used_name, node.lineno))

    # Collect all Name ids used outside import statements
    import_linenos = {lineno for _, lineno in imported_names}
    used_ids: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and getattr(node, "lineno", -1) not in import_linenos:
            used_ids.add(node.id)
        elif isinstance(node, ast.Attribute):
            # e.g. math.pi — the 'math' part is an ast.Name
            pass

    for name, lineno in imported_names:
        if name not in used_ids:
            violations.append(
                Violation(
                    principle_name="no_unused_imports",
                    severity="info",
                    message=(
                        f"Imported name '{name}' does not appear to be used. "
                        "Remove unused imports to keep the code clean."
                    ),
                    line_hint=lineno,
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Default principle registry
# ---------------------------------------------------------------------------

def _build_default_principles() -> list[CodingPrinciple]:
    return [
        CodingPrinciple(
            name="no_hardcoded_secrets",
            description="No API keys, passwords, or tokens hardcoded as string literals.",
            check_fn=_check_no_hardcoded_secrets,
            severity="error",
        ),
        CodingPrinciple(
            name="no_eval",
            description="Do not use eval() or exec() — they execute arbitrary code.",
            check_fn=_check_no_eval,
            severity="error",
        ),
        CodingPrinciple(
            name="has_error_handling",
            description="Functions should have try/except or raise statements for robustness.",
            check_fn=_check_has_error_handling,
            severity="warning",
        ),
        CodingPrinciple(
            name="no_star_import",
            description="Avoid 'from module import *' — it pollutes the namespace.",
            check_fn=_check_no_star_import,
            severity="warning",
        ),
        CodingPrinciple(
            name="no_print_debug",
            description="Use the logging module instead of print() in library code.",
            check_fn=_check_no_print_debug,
            severity="info",
        ),
        CodingPrinciple(
            name="has_docstring",
            description="Public functions and classes should have docstrings.",
            check_fn=_check_has_docstring,
            severity="info",
        ),
        CodingPrinciple(
            name="no_magic_numbers",
            description="Avoid unnamed numeric literals — assign them to named constants.",
            check_fn=_check_no_magic_numbers,
            severity="info",
        ),
        CodingPrinciple(
            name="no_global_state",
            description="Avoid mutable module-level variables; prefer constants or class state.",
            check_fn=_check_no_global_state,
            severity="warning",
        ),
        CodingPrinciple(
            name="proper_naming",
            description="Variables should have descriptive names, not single characters.",
            check_fn=_check_proper_naming,
            severity="info",
        ),
        CodingPrinciple(
            name="no_bare_except",
            description="Catch specific exceptions; bare 'except:' silently swallows errors.",
            check_fn=_check_no_bare_except,
            severity="error",
        ),
        CodingPrinciple(
            name="has_type_annotations",
            description="Functions should have type annotations on parameters and return values.",
            check_fn=_check_has_type_annotations,
            severity="info",
        ),
        CodingPrinciple(
            name="no_unused_imports",
            description="Remove imports that are never referenced in the code body.",
            check_fn=_check_no_unused_imports,
            severity="info",
        ),
    ]


# ---------------------------------------------------------------------------
# ConstitutionalChecker
# ---------------------------------------------------------------------------


class ConstitutionalChecker:
    """Run a set of CodingPrinciples against a source string.

    Usage::

        checker = ConstitutionalChecker()
        violations = checker.check(my_code)
        print(checker.summary(violations))

    You can extend the default constitution::

        checker.add_principle(CodingPrinciple(
            name="no_todo",
            description="No TODO comments in generated code.",
            check_fn=lambda c: [
                Violation("no_todo", "warning", f"TODO on line {i}", i)
                for i, l in enumerate(c.splitlines(), 1) if "TODO" in l
            ],
            severity="warning",
        ))
    """

    def __init__(self, principles: Optional[list[CodingPrinciple]] = None) -> None:
        if principles is None:
            self._principles: list[CodingPrinciple] = _build_default_principles()
        else:
            self._principles = list(principles)

    # ------------------------------------------------------------------

    def check(self, code: str) -> list[Violation]:
        """Check *code* against all registered principles.

        Returns a flat list of Violation objects.  Violations from each
        principle are appended in registration order.
        """
        violations: list[Violation] = []
        for principle in self._principles:
            try:
                found = principle.check_fn(code)
            except Exception:
                # A buggy check_fn must never crash the pipeline
                continue
            # Stamp each violation with the correct severity from the principle
            for v in found:
                if v.severity != principle.severity:
                    v = Violation(
                        principle_name=v.principle_name,
                        severity=principle.severity,
                        message=v.message,
                        line_hint=v.line_hint,
                    )
                violations.append(v)
        return violations

    # ------------------------------------------------------------------

    def add_principle(self, principle: CodingPrinciple) -> None:
        """Append a custom CodingPrinciple to the checker's constitution."""
        self._principles.append(principle)

    # ------------------------------------------------------------------

    def get_principles(self) -> list[CodingPrinciple]:
        """Return all registered CodingPrinciple objects (copy of internal list)."""
        return list(self._principles)

    # ------------------------------------------------------------------

    def summary(self, violations: list[Violation]) -> dict:
        """Return aggregate statistics for a list of Violation objects.

        The returned dict has the following keys:

        - ``total``           — total number of violations
        - ``by_severity``     — dict mapping severity → count
        - ``by_principle``    — dict mapping principle_name → count
        - ``error_count``     — convenience alias for by_severity['error']
        - ``warning_count``   — convenience alias for by_severity['warning']
        - ``info_count``      — convenience alias for by_severity['info']
        - ``passed``          — True when there are zero 'error' violations
        """
        by_severity: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        by_principle: dict[str, int] = {}

        for v in violations:
            by_severity[v.severity] = by_severity.get(v.severity, 0) + 1
            by_principle[v.principle_name] = by_principle.get(v.principle_name, 0) + 1

        return {
            "total": len(violations),
            "by_severity": by_severity,
            "by_principle": by_principle,
            "error_count": by_severity.get("error", 0),
            "warning_count": by_severity.get("warning", 0),
            "info_count": by_severity.get("info", 0),
            "passed": by_severity.get("error", 0) == 0,
        }
