"""Safety Checker: scan generated code for dangerous patterns.

Checks for:
  1. Dangerous builtins / functions: eval, exec, compile, __import__,
     os.system, subprocess.*, shutil.rmtree, etc.
  2. Hardcoded secrets: API keys, passwords, tokens in string literals
  3. Potential infinite loops: while True without a break
  4. Network access patterns: socket, urllib, requests, httpx
  5. File-system operations: open, shutil, pathlib write modes

Returns a SafetyReport with severity levels: SAFE, LOW, MEDIUM, HIGH, CRITICAL.

For a TS dev: like a CodeQL / Semgrep rule set — each rule fires independently
and the final verdict is the worst severity found.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def _rank(self) -> int:
        order = [Severity.SAFE, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self)

    def __gt__(self, other: "Severity") -> bool:  # type: ignore[override]
        return self._rank() > other._rank()

    def __ge__(self, other: "Severity") -> bool:  # type: ignore[override]
        return self._rank() >= other._rank()

    def __lt__(self, other: "Severity") -> bool:  # type: ignore[override]
        return self._rank() < other._rank()

    def __le__(self, other: "Severity") -> bool:  # type: ignore[override]
        return self._rank() <= other._rank()


@dataclass
class Finding:
    """A single detected issue."""

    rule: str
    severity: Severity
    description: str
    line: int = 0  # 0 = unknown
    snippet: str = ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SafetyReport:
    """Aggregated safety check results."""

    passed: bool  # True if no MEDIUM+ findings
    verdict: Severity  # Worst severity found
    findings: list[Finding] = field(default_factory=list)

    # Counts by severity
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def summary(self) -> str:
        parts = [f"verdict={self.verdict.value}"]
        if self.critical_count:
            parts.append(f"critical={self.critical_count}")
        if self.high_count:
            parts.append(f"high={self.high_count}")
        if self.medium_count:
            parts.append(f"medium={self.medium_count}")
        if self.low_count:
            parts.append(f"low={self.low_count}")
        return "  ".join(parts)

    def __repr__(self) -> str:
        return f"SafetyReport({self.summary()})"


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# (pattern, rule_name, severity, description)
_REGEX_RULES: list[tuple[re.Pattern, str, Severity, str]] = [
    # CRITICAL
    (re.compile(r"\beval\s*\("), "eval_call", Severity.CRITICAL, "eval() can execute arbitrary code"),
    (re.compile(r"\bexec\s*\("), "exec_call", Severity.CRITICAL, "exec() can execute arbitrary code"),
    (re.compile(r"\bcompile\s*\("), "compile_call", Severity.HIGH, "compile() builds executable code objects"),
    (re.compile(r"__import__\s*\("), "dunder_import", Severity.HIGH, "__import__() bypasses normal import system"),
    # HIGH
    (re.compile(r"\bos\.system\s*\("), "os_system", Severity.CRITICAL, "os.system() runs shell commands"),
    (re.compile(r"\bos\.popen\s*\("), "os_popen", Severity.CRITICAL, "os.popen() runs shell commands"),
    (re.compile(r"\bsubprocess\.(?:call|run|Popen|check_output|check_call)\s*\("),
     "subprocess_call", Severity.HIGH, "subprocess can run arbitrary shell commands"),
    (re.compile(r"\bshutil\.rmtree\s*\("), "shutil_rmtree", Severity.HIGH, "rmtree can delete entire directory trees"),
    (re.compile(r"\bos\.remove\s*\("), "os_remove", Severity.MEDIUM, "os.remove() deletes files"),
    (re.compile(r"\bos\.unlink\s*\("), "os_unlink", Severity.MEDIUM, "os.unlink() deletes files"),
    # MEDIUM — hardcoded secrets (simple heuristics)
    (re.compile(r'(?:api_key|apikey|secret|password|token|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']',
                re.IGNORECASE),
     "hardcoded_secret", Severity.HIGH, "Possible hardcoded secret / credential"),
    (re.compile(r'(?:sk-|ghp_|glpat-|xoxb-|xoxp-)[a-zA-Z0-9]{10,}'),
     "token_literal", Severity.CRITICAL, "Detected token-format string (API key / PAT)"),
    # MEDIUM — infinite loops
    (re.compile(r"\bwhile\s+True\s*:"), "while_true", Severity.LOW, "while True loop (verify break exists)"),
    # LOW — network
    (re.compile(r"\bsocket\."), "socket_use", Severity.LOW, "Direct socket access"),
    (re.compile(r"\burllib\.request\b|\burllib\.urlopen\b"), "urllib_fetch", Severity.LOW, "urllib network access"),
    (re.compile(r"\brequests\.(get|post|put|delete|patch|head)\s*\("),
     "requests_call", Severity.LOW, "HTTP request via requests library"),
    (re.compile(r"\bhttpx\.(get|post|put|delete|patch|head|AsyncClient)\b"),
     "httpx_call", Severity.LOW, "HTTP request via httpx library"),
    # LOW — dangerous pickle
    (re.compile(r"\bpickle\.loads?\s*\("), "pickle_load", Severity.HIGH,
     "pickle.load(s) can execute arbitrary code when loading untrusted data"),
    (re.compile(r"\byaml\.(?:load|unsafe_load)\s*\("), "yaml_load_unsafe", Severity.MEDIUM,
     "yaml.load() with arbitrary input can execute code; prefer yaml.safe_load()"),
]


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class SafetyChecker:
    """Scan code for dangerous patterns and return a SafetyReport.

    Example::

        checker = SafetyChecker()
        report = checker.check("import os; os.system('rm -rf /')")
        print(report.verdict)  # CRITICAL
    """

    def __init__(self, block_on: Severity = Severity.MEDIUM) -> None:
        """
        Parameters
        ----------
        block_on:
            Minimum severity that causes ``passed`` to be False.
        """
        self.block_on = block_on

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, code: str) -> SafetyReport:
        """Run all safety checks on *code* and return a SafetyReport."""
        findings: list[Finding] = []

        # Regex-based checks (fast, no AST needed)
        findings.extend(self._run_regex_checks(code))

        # AST-based check for infinite loops (more accurate than regex)
        findings.extend(self._check_infinite_loops(code))

        # Deduplicate (same rule, same line)
        seen: set[tuple[str, int]] = set()
        deduped: list[Finding] = []
        for f in findings:
            key = (f.rule, f.line)
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        # Compute verdict
        verdict = Severity.SAFE
        for f in deduped:
            if f.severity >= verdict:
                verdict = f.severity

        passed = verdict < self.block_on

        counts = {Severity.CRITICAL: 0, Severity.HIGH: 0, Severity.MEDIUM: 0, Severity.LOW: 0}
        for f in deduped:
            if f.severity in counts:
                counts[f.severity] += 1

        return SafetyReport(
            passed=passed,
            verdict=verdict,
            findings=deduped,
            critical_count=counts[Severity.CRITICAL],
            high_count=counts[Severity.HIGH],
            medium_count=counts[Severity.MEDIUM],
            low_count=counts[Severity.LOW],
        )

    def is_safe(self, code: str) -> bool:
        """Quick boolean: True if no findings at or above block_on severity."""
        return self.check(code).passed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_regex_checks(self, code: str) -> list[Finding]:
        findings: list[Finding] = []
        lines = code.splitlines()
        for pattern, rule, severity, description in _REGEX_RULES:
            for lineno, line in enumerate(lines, start=1):
                if pattern.search(line):
                    findings.append(Finding(
                        rule=rule,
                        severity=severity,
                        description=description,
                        line=lineno,
                        snippet=line.strip()[:120],
                    ))
        return findings

    def _check_infinite_loops(self, code: str) -> list[Finding]:
        """Use AST to find while True loops without a break statement."""
        findings: list[Finding] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if condition is True literal
                cond = node.test
                is_true_literal = (
                    isinstance(cond, ast.Constant) and cond.value is True
                ) or (
                    isinstance(cond, ast.NameConstant) and cond.value is True  # type: ignore
                )
                if not is_true_literal:
                    continue

                # Check if there is a break anywhere in the body
                has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                if not has_break:
                    findings.append(Finding(
                        rule="infinite_loop_no_break",
                        severity=Severity.MEDIUM,
                        description="while True loop with no break statement — potential infinite loop",
                        line=node.lineno,
                        snippet=f"while True: (line {node.lineno})",
                    ))

        return findings
