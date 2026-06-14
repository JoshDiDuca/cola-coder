"""Regression test suite for cola-coder model checkpoints.

Detects quality regressions between training checkpoints by comparing
generated outputs against known-good baselines.  Each baseline defines:
  - A prompt the model should complete
  - Patterns that MUST appear in the output  (expected_patterns)
  - Patterns that MUST NOT appear             (forbidden_patterns)
  - Acceptable output length bounds           (min_length / max_length)

This requires NO TypeScript / Node.js runtime — purely string-based checks.

For a TS dev: think of this like snapshot testing for your model.  If a new
checkpoint breaks a baseline that was previously passing it's a regression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .metrics import ProblemResult, compute_pass_at_k, paired_bootstrap_delta


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RegressionBaseline:
    """A single regression test baseline."""

    prompt: str
    expected_patterns: list[str]  # regex patterns that MUST match
    forbidden_patterns: list[str]  # regex patterns that MUST NOT match
    min_length: int
    max_length: int
    category: str
    description: str = ""


@dataclass
class RegressionResult:
    """Results from a full regression suite run."""

    total: int
    passed: int
    failed: int
    details: list[dict]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "REGRESSION TEST RESULTS",
            "=" * 60,
            f"  Pass rate: {self.pass_rate:.1%}  ({self.passed}/{self.total})",
            "",
        ]
        for d in self.details:
            status = "PASS" if d["passed"] else "FAIL"
            reasons = ""
            if not d["passed"]:
                reasons = "  <- " + "; ".join(d.get("failures", []))
            lines.append(f"  [{status}] {d['category']:<14} {d['description'][:40]}{reasons}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baselines (20 total)
# ---------------------------------------------------------------------------

BASELINES: list[RegressionBaseline] = [
    # --- Python function completion (4 baselines) --------------------------
    RegressionBaseline(
        description="Python add function",
        category="python",
        prompt="def add(a: int, b: int) -> int:\n    ",
        expected_patterns=[r"return\s+a\s*\+\s*b|return a \+ b"],
        forbidden_patterns=[r"TODO|pass\b"],
        min_length=10,
        max_length=200,
    ),
    RegressionBaseline(
        description="Python list comprehension",
        category="python",
        prompt="def squares(n: int) -> list[int]:\n    \"\"\"Return list of squares from 1 to n.\"\"\"\n    ",
        expected_patterns=[r"return\s+\[|return\s+list"],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=300,
    ),
    RegressionBaseline(
        description="Python class __init__",
        category="python",
        prompt="class Point:\n    def __init__(self, x: float, y: float) -> None:\n        ",
        expected_patterns=[r"self\.x\s*=\s*x", r"self\.y\s*=\s*y"],
        forbidden_patterns=[r"TODO|\.\.\."],
        min_length=15,
        max_length=300,
    ),
    RegressionBaseline(
        description="Python async function",
        category="python",
        prompt=(
            "import asyncio\n\n"
            "async def fetch_data(url: str) -> dict:\n"
            "    \"\"\"Fetch JSON data from the given URL.\"\"\"\n"
            "    "
        ),
        expected_patterns=[r"async\s+with|await\s+\w+|aiohttp|httpx"],
        forbidden_patterns=[r"TODO"],
        min_length=20,
        max_length=400,
    ),
    # --- TypeScript function completion (4 baselines) ----------------------
    RegressionBaseline(
        description="TS sum function",
        category="typescript",
        prompt="function sum(a: number, b: number): number {\n  ",
        expected_patterns=[r"return\s+a\s*\+\s*b"],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=150,
    ),
    RegressionBaseline(
        description="TS arrow function with types",
        category="typescript",
        prompt="const multiply = (a: number, b: number): number => ",
        expected_patterns=[r"a\s*\*\s*b"],
        forbidden_patterns=[r"TODO"],
        min_length=5,
        max_length=100,
    ),
    RegressionBaseline(
        description="TS generic function",
        category="typescript",
        prompt="function first<T>(arr: T[]): T | undefined {\n  ",
        expected_patterns=[r"return\s+arr\[0\]|arr\.length|arr\[0\]"],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=200,
    ),
    RegressionBaseline(
        description="TS async fetch wrapper",
        category="typescript",
        prompt=(
            "async function getJson<T>(url: string): Promise<T> {\n"
            "  "
        ),
        expected_patterns=[r"fetch\(|await\s+fetch"],
        forbidden_patterns=[r"TODO"],
        min_length=20,
        max_length=300,
    ),
    # --- Class definition completion (3 baselines) -------------------------
    RegressionBaseline(
        description="TS class with constructor",
        category="class",
        prompt=(
            "class Stack<T> {\n"
            "  private items: T[] = [];\n"
            "\n"
            "  push(item: T): void {\n"
            "    "
        ),
        expected_patterns=[r"this\.items\.push\(item\)|items\.push"],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=200,
    ),
    RegressionBaseline(
        description="Python dataclass",
        category="class",
        prompt=(
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class User:\n"
            "    "
        ),
        expected_patterns=[r"name\s*:\s*str|id\s*:\s*int|email\s*:\s*str"],
        forbidden_patterns=[r"TODO|pass\b"],
        min_length=10,
        max_length=300,
    ),
    RegressionBaseline(
        description="TS interface implementation",
        category="class",
        prompt=(
            "interface Serializable {\n"
            "  serialize(): string;\n"
            "}\n\n"
            "class Config implements Serializable {\n"
            "  constructor(private data: Record<string, unknown>) {}\n\n"
            "  serialize(): string {\n"
            "    "
        ),
        expected_patterns=[r"JSON\.stringify|return.*JSON"],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=200,
    ),
    # --- Import statement completion (2 baselines) -------------------------
    RegressionBaseline(
        description="React import",
        category="import",
        prompt="import React, { ",
        expected_patterns=[r"useState|useEffect|useCallback|useRef|useMemo"],
        forbidden_patterns=[],
        min_length=5,
        max_length=150,
    ),
    RegressionBaseline(
        description="Named imports from zod",
        category="import",
        prompt="import { z } from 'zod';\n\nconst schema = z.",
        expected_patterns=[r"object|string|number|array|enum"],
        forbidden_patterns=[],
        min_length=3,
        max_length=200,
    ),
    # --- Comment-to-code generation (3 baselines) --------------------------
    RegressionBaseline(
        description="JSDoc to TS implementation",
        category="comment_to_code",
        prompt=(
            "/**\n"
            " * Reverses a string.\n"
            " * @param s - the input string\n"
            " * @returns the reversed string\n"
            " */\n"
            "function reverseString(s: string): string {\n"
            "  "
        ),
        expected_patterns=[r"split|reverse|\.split\(''\)"],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=200,
    ),
    RegressionBaseline(
        description="Python docstring to implementation",
        category="comment_to_code",
        prompt=(
            "def is_even(n: int) -> bool:\n"
            "    \"\"\"Return True if n is even.\"\"\"\n"
            "    "
        ),
        expected_patterns=[r"return\s+n\s*%\s*2\s*==\s*0|n % 2"],
        forbidden_patterns=[r"TODO|pass\b"],
        min_length=5,
        max_length=150,
    ),
    RegressionBaseline(
        description="Inline comment to code",
        category="comment_to_code",
        prompt=(
            "// Sort an array of numbers in ascending order\n"
            "function sortNumbers(arr: number[]): number[] {\n"
            "  "
        ),
        expected_patterns=[r"\.sort\(|sort\("],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=200,
    ),
    # --- Multi-line completion (4 baselines) -------------------------------
    RegressionBaseline(
        description="Multi-line TS object literal",
        category="multiline",
        prompt=(
            "const config = {\n"
            "  host: 'localhost',\n"
            "  port: "
        ),
        expected_patterns=[r"\d{3,5}"],  # a port number
        forbidden_patterns=[],
        min_length=3,
        max_length=200,
    ),
    RegressionBaseline(
        description="Multi-line React component",
        category="multiline",
        prompt=(
            "import React from 'react';\n\n"
            "function App() {\n"
            "  return (\n"
            "    <div"
        ),
        expected_patterns=[r"className|style|>.*</div>|<div"],
        forbidden_patterns=[r"TODO"],
        min_length=5,
        max_length=400,
    ),
    RegressionBaseline(
        description="Multi-line try/catch",
        category="multiline",
        prompt=(
            "async function safeFetch(url: string) {\n"
            "  try {\n"
            "    const response = await fetch(url);\n"
            "    "
        ),
        expected_patterns=[r"await response\.json\(\)|response\.json\(\)|\.json\(\)"],
        forbidden_patterns=[r"TODO"],
        min_length=10,
        max_length=300,
    ),
    RegressionBaseline(
        description="Multi-line switch statement",
        category="multiline",
        prompt=(
            "function httpStatus(code: number): string {\n"
            "  switch (code) {\n"
            "    case 200:\n"
            "      "
        ),
        expected_patterns=[r"return|'OK'|\"OK\""],
        forbidden_patterns=[r"TODO"],
        min_length=5,
        max_length=400,
    ),
]

# Verify count
assert len(BASELINES) == 20, f"Expected 20 baselines, got {len(BASELINES)}"


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _check_baseline(output: str, baseline: RegressionBaseline) -> tuple[bool, list[str]]:
    """Evaluate a single generated output against a baseline.

    Returns (passed, list_of_failure_reasons).
    """
    failures: list[str] = []

    # Length bounds
    if len(output) < baseline.min_length:
        failures.append(f"too short ({len(output)} < {baseline.min_length})")
    if len(output) > baseline.max_length:
        failures.append(f"too long ({len(output)} > {baseline.max_length})")

    # Expected patterns
    for pat in baseline.expected_patterns:
        if not re.search(pat, output):
            failures.append(f"missing pattern: {pat!r}")

    # Forbidden patterns
    for pat in baseline.forbidden_patterns:
        if re.search(pat, output):
            failures.append(f"contains forbidden pattern: {pat!r}")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Statistical significance of a pass@k delta (EVAL-029)
# ---------------------------------------------------------------------------
#
# A raw point-delta in pass@k between two checkpoints is meaningless on its own:
# on a ~62-problem set the run-to-run noise easily swamps a +2pp move. A
# checkpoint is only honestly declared an IMPROVEMENT or a REGRESSION when the
# paired-bootstrap CI of the pass@k delta (B − A) EXCLUDES 0. Otherwise the move
# is "within noise / not significant". All the statistics live in
# ``metrics.paired_bootstrap_delta`` — this layer only turns its interval into a
# verdict and a human-readable line. (EVAL-029)


@dataclass
class SignificanceVerdict:
    """A statistically-assessed comparison of pass@k between two checkpoints.

    ``delta`` / ``ci_lo`` / ``ci_hi`` are in pass@k probability units (0.0-1.0),
    B − A (positive ⇒ B better). ``significant`` is True iff the CI excludes 0.
    ``direction`` is "improvement" / "regression" / "within noise". When the
    paired bootstrap could not run (no aggregate fallback either), the fields are
    None and ``assessed`` is False.
    """

    k: int
    delta: float | None
    ci_lo: float | None
    ci_hi: float | None
    ci: float
    significant: bool
    direction: str  # "improvement" | "regression" | "within noise"
    assessed: bool  # True if a paired-bootstrap CI was computed
    note: str = ""

    def render(self) -> str:
        """One-line verdict, e.g.

        ``pass@1: +6.1pp [95% CI +1.2 to +11.0] — significant improvement``
        ``pass@1: +1.3pp [95% CI -2.1 to +4.8] — within noise``
        """
        if self.delta is None:
            return f"pass@{self.k}: n/a — {self.note or 'not estimable'}"
        delta_pp = self.delta * 100.0
        head = f"pass@{self.k}: {delta_pp:+.1f}pp"
        if not self.assessed:
            tail = self.note or "raw delta only (significance not assessed)"
            return f"{head} — {tail}"
        lo_pp = self.ci_lo * 100.0
        hi_pp = self.ci_hi * 100.0
        ci_pct = f"{self.ci:.0%}"
        body = f"{head} [{ci_pct} CI {lo_pp:+.1f} to {hi_pp:+.1f}]"
        if self.direction == "within noise":
            return f"{body} — within noise"
        return f"{body} — significant {self.direction}"


def assess_pass_at_k_significance(
    results_a: list[ProblemResult],
    results_b: list[ProblemResult],
    k: int = 1,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> SignificanceVerdict:
    """Decide whether B differs from A on pass@k beyond noise (EVAL-029).

    Runs the paired bootstrap (``metrics.paired_bootstrap_delta``) on the shared
    problem set and bases the verdict on whether the CI excludes 0:

    - CI entirely above 0 → "improvement" (significant)
    - CI entirely below 0 → "regression" (significant)
    - CI spans 0          → "within noise"

    Reuses the bootstrap primitive — never reimplements it. Deterministic for a
    fixed ``seed``. Returns a verdict with ``delta=None`` when no paired problem
    qualifies (disjoint problem sets, or no problem with >= k samples in both).
    """
    boot = paired_bootstrap_delta(
        results_a, results_b, k, n_boot=n_boot, ci=ci, seed=seed
    )
    if boot is None:
        return SignificanceVerdict(
            k=k,
            delta=None,
            ci_lo=None,
            ci_hi=None,
            ci=ci,
            significant=False,
            direction="within noise",
            assessed=False,
            note="no shared problem with >= k samples in both models",
        )
    delta, lo, hi = boot
    if lo > 0.0:
        direction, significant = "improvement", True
    elif hi < 0.0:
        direction, significant = "regression", True
    else:
        direction, significant = "within noise", False
    return SignificanceVerdict(
        k=k,
        delta=delta,
        ci_lo=lo,
        ci_hi=hi,
        ci=ci,
        significant=significant,
        direction=direction,
        assessed=True,
    )


def assess_aggregate_delta(
    score_a: float | None,
    score_b: float | None,
    k: int = 1,
    ci: float = 0.95,
) -> SignificanceVerdict:
    """Back-compat fallback when only AGGREGATE pass@k scores are available.

    With no per-problem data the paired bootstrap can't run, so we report the
    raw point-delta and flag clearly that significance was NOT assessed — never
    silently presenting a raw delta as if it were a credible difference.
    """
    if score_a is None or score_b is None:
        return SignificanceVerdict(
            k=k,
            delta=None,
            ci_lo=None,
            ci_hi=None,
            ci=ci,
            significant=False,
            direction="within noise",
            assessed=False,
            note="aggregate pass@k not estimable",
        )
    return SignificanceVerdict(
        k=k,
        delta=score_b - score_a,
        ci_lo=None,
        ci_hi=None,
        ci=ci,
        significant=False,
        direction="within noise",
        assessed=False,
        note="raw delta only — no per-problem data, significance not assessed",
    )


# ---------------------------------------------------------------------------
# Main suite class
# ---------------------------------------------------------------------------


class RegressionSuite:
    """Detect model quality regressions between checkpoints.

    Usage::

        suite = RegressionSuite()
        result = suite.run(generator, tokenizer)
        print(result.summary())
    """

    BASELINES: list[RegressionBaseline] = BASELINES

    def run(
        self,
        generator,  # cola_coder.inference.generator.CodeGenerator (or mock)
        tokenizer,  # kept for API symmetry — not used directly
        max_new_tokens: int = 128,
        temperature: float = 0.2,
    ) -> RegressionResult:
        """Run all regression baselines against `generator`."""
        details: list[dict] = []
        passed_count = 0

        for baseline in self.BASELINES:
            try:
                output = generator.generate(
                    prompt=baseline.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.9,
                )
            except Exception as exc:
                details.append(
                    {
                        "description": baseline.description,
                        "category": baseline.category,
                        "passed": False,
                        "output": "",
                        "failures": [f"generator error: {exc}"],
                    }
                )
                continue

            passed, failures = _check_baseline(output, baseline)
            if passed:
                passed_count += 1

            details.append(
                {
                    "description": baseline.description,
                    "category": baseline.category,
                    "passed": passed,
                    "output": output,
                    "failures": failures,
                }
            )

        return RegressionResult(
            total=len(self.BASELINES),
            passed=passed_count,
            failed=len(self.BASELINES) - passed_count,
            details=details,
        )

    def compare_checkpoints(
        self,
        results_a: RegressionResult,
        results_b: RegressionResult,
        label_a: str = "checkpoint A",
        label_b: str = "checkpoint B",
    ) -> str:
        """Compare two regression results and highlight regressions.

        A *regression* is a baseline that passed in `results_a` but fails in
        `results_b`.  An *improvement* is the opposite direction.
        """
        # Build lookup maps: description -> passed
        map_a = {d["description"]: d["passed"] for d in results_a.details}
        map_b = {d["description"]: d["passed"] for d in results_b.details}

        regressions: list[str] = []
        improvements: list[str] = []
        unchanged: list[str] = []

        all_keys = sorted(set(map_a) | set(map_b))
        for key in all_keys:
            a_ok = map_a.get(key, False)
            b_ok = map_b.get(key, False)
            if a_ok and not b_ok:
                regressions.append(key)
            elif not a_ok and b_ok:
                improvements.append(key)
            else:
                unchanged.append(key)

        lines = [
            "=" * 60,
            f"CHECKPOINT COMPARISON: {label_a}  vs  {label_b}",
            "=" * 60,
            f"  {label_a}: {results_a.passed}/{results_a.total}  ({results_a.pass_rate:.1%})",
            f"  {label_b}: {results_b.passed}/{results_b.total}  ({results_b.pass_rate:.1%})",
            "",
        ]

        if regressions:
            lines.append(f"  REGRESSIONS ({len(regressions)}) — passed in A, failed in B:")
            for r in regressions:
                lines.append(f"    ✗ {r}")
        else:
            lines.append("  No regressions.")

        if improvements:
            lines.append(f"\n  IMPROVEMENTS ({len(improvements)}) — failed in A, passed in B:")
            for r in improvements:
                lines.append(f"    ✓ {r}")

        lines.append(f"\n  Unchanged: {len(unchanged)}")
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def compare_pass_at_k(
        results_a: list[ProblemResult],
        results_b: list[ProblemResult],
        k: int = 1,
        label_a: str = "checkpoint A",
        label_b: str = "checkpoint B",
        n_boot: int = 10_000,
        ci: float = 0.95,
        seed: int = 0,
    ) -> str:
        """Statistically-honest pass@k comparison of two checkpoints (EVAL-029).

        When per-problem ``ProblemResult`` lists are available for BOTH models on
        the same problem set, the verdict is driven by the paired-bootstrap CI of
        the pass@k delta — only a CI that EXCLUDES 0 declares an improvement or a
        regression; otherwise the move is reported as "within noise". Reuses
        ``paired_bootstrap_delta`` (never reimplements the bootstrap).
        """
        verdict = assess_pass_at_k_significance(
            results_a, results_b, k=k, n_boot=n_boot, ci=ci, seed=seed
        )
        score_a = compute_pass_at_k(results_a, [k]).get(f"pass@{k}")
        score_b = compute_pass_at_k(results_b, [k]).get(f"pass@{k}")
        a_str = f"{score_a:.1%}" if score_a is not None else "n/a"
        b_str = f"{score_b:.1%}" if score_b is not None else "n/a"

        lines = [
            "=" * 60,
            f"PASS@{k} COMPARISON: {label_a}  vs  {label_b}",
            "=" * 60,
            f"  {label_a}: pass@{k} = {a_str}",
            f"  {label_b}: pass@{k} = {b_str}",
            "",
            f"  {verdict.render()}",
            "=" * 60,
        ]
        return "\n".join(lines)
