"""Verifier-anchored FUNCTION-STEP process-credit profiler (EVAL-034).

A "poor-man's PRM" (process reward model). A real PRM scores each *step* of a
reasoning/solution trace; training one needs step-level labels we don't have.
This module fakes that signal for code by treating each FUNCTION in a candidate
as a "step" and grading every step with the sandbox verifier we already trust
(the same executor GRPO / HumanEval use). No learned reward model, no extra
labels — just the verifier we already have, re-pointed at function granularity.

Why bother when best-of-N already gives a single pass/fail per candidate? Because
that one bit hides *where* a candidate is weak. Two candidates that both fail the
overall tests look identical to best-of-N; this profiler shows that one has 3/4
functions working and one bad helper, while the other is wrong throughout. It
also surfaces FRAGILE functions — code that is dead or non-executable yet rides
along on a candidate whose top-level tests happen to pass (a latent bug the
verifier never exercised).

Pure analysis + an INJECTED ``execute_fn`` (signature matching
``evaluation.runner.execute_code``: ``(code, timeout) -> (success, output)``).
No GPU, no model, no training, no checkpoint writes — and tests pass a fake
executor so nothing is ever really run.

Reuse (per .claude/rules/dry-principle.md): ``split_test_cases`` for AST
assert-splitting, ``is_typescript`` / ``is_js_ts`` for language routing,
``scan_dangerous`` for the optional per-function security flag. We do NOT
reimplement sandboxing or AST splitting.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable

from ..data.scorers.language_detect import is_js_ts, is_typescript
from ..reasoning.rewards.partial_credit import split_test_cases
from ..security.code_patterns import scan_dangerous

# Same contract as evaluation.runner.execute_code: (code, timeout) -> (ok, output).
ExecuteFn = Callable[[str, float], tuple[bool, str]]

# TS/JS function-declaration fallback (no AST available for TS here): named
# `function f(...)`, `async function f(...)`, methods/arrows assigned to a name,
# and class methods. High-precision-ish; unparseable TS just yields fewer steps.
_TS_FUNCTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # function f(...) {   /   async function f(...) {
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(",
               re.MULTILINE),
    # const f = (...) => {   /   let f = async (...) =>   /   var f = function(...)
    re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?:=>|function)",
        re.MULTILINE,
    ),
)


@dataclass
class FunctionStep:
    """One function treated as a process "step"."""

    name: str
    src: str
    lineno: int
    end_lineno: int

    @property
    def n_lines(self) -> int:
        """Number of source lines this step spans (>= 1)."""
        return max(1, self.end_lineno - self.lineno + 1)


@dataclass
class StepProfile:
    """Per-candidate process-credit profile.

    Attributes:
        process_score: Length-normalized mean of the per-step scores, in [0, 1].
        steps: One dict per function — name, score, n_tests, executable, dangerous.
        fragile_functions: Names of functions that are dead / non-executable while
            the candidate's overall tests pass (latent bugs the verifier missed).
    """

    process_score: float
    steps: list[dict] = field(default_factory=list)
    fragile_functions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decomposition: code -> function steps
# ---------------------------------------------------------------------------


def _resolve_language(code: str, language: str) -> str:
    """Normalize the language hint, auto-detecting TS/JS vs Python when asked."""
    lang = (language or "python").lower()
    if lang == "auto":
        return "typescript" if (is_typescript(code) or is_js_ts(code)) else "python"
    return lang


def _decompose_python(code: str) -> list[FunctionStep]:
    """Decompose Python source into every (top-level and nested) function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    steps: list[FunctionStep] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            # end_lineno is set on py3.8+; fall back to start if somehow absent.
            end = getattr(node, "end_lineno", None) or start
            src = "\n".join(lines[start - 1:end])
            steps.append(FunctionStep(name=node.name, src=src, lineno=start, end_lineno=end))
    steps.sort(key=lambda s: s.lineno)
    return steps


def _decompose_ts(code: str) -> list[FunctionStep]:
    """Decompose TS/JS source into functions via regex (no TS AST available)."""
    lines = code.splitlines()
    seen: set[tuple[str, int]] = set()
    steps: list[FunctionStep] = []
    for pattern in _TS_FUNCTION_PATTERNS:
        for match in pattern.finditer(code):
            name = match.group("name")
            lineno = code.count("\n", 0, match.start()) + 1
            key = (name, lineno)
            if key in seen:
                continue
            seen.add(key)
            end = _ts_block_end(lines, lineno)
            src = "\n".join(lines[lineno - 1:end])
            steps.append(FunctionStep(name=name, src=src, lineno=lineno, end_lineno=end))
    steps.sort(key=lambda s: s.lineno)
    return steps


def _ts_block_end(lines: list[str], start_lineno: int) -> int:
    """Find the line where a brace-delimited TS/JS block starting at start_lineno ends.

    Brace-counts from the first ``{`` at/after the declaration line. If no brace is
    found (e.g. a one-line arrow ``const f = () => x``), the block is the start line.
    """
    depth = 0
    opened = False
    for idx in range(start_lineno - 1, len(lines)):
        for ch in lines[idx]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return idx + 1
    return start_lineno


def decompose_functions(code: str, language: str = "python") -> list[FunctionStep]:
    """Split code into function "steps".

    Python uses the ``ast`` module (every top-level AND nested ``def`` /
    ``async def``); TS/JS uses a regex fallback (routed via language_detect).
    Unparseable Python returns ``[]``.

    Args:
        code: Candidate source.
        language: "python", "typescript"/"javascript", or "auto" to detect.

    Returns:
        Function steps in source order ([] if none / unparseable).
    """
    if not code or not code.strip():
        return []
    lang = _resolve_language(code, language)
    if lang in ("typescript", "ts", "tsx", "javascript", "js", "jsx"):
        return _decompose_ts(code)
    return _decompose_python(code)


# ---------------------------------------------------------------------------
# Attribution: which asserts test which function
# ---------------------------------------------------------------------------


def _names_in_assert(assert_src: str) -> set[str]:
    """Extract the identifier names referenced by a single assert statement."""
    names: set[str] = set()
    try:
        tree = ast.parse(assert_src)
    except SyntaxError:
        # Couldn't parse the isolated assert — fall back to a token-ish scan.
        return set(re.findall(r"[A-Za-z_]\w*", assert_src))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _attribute_asserts(
    steps: list[FunctionStep], asserts: list[str]
) -> dict[str, list[str]]:
    """Map each function name -> the assert sources that reference it.

    A function is "tested" by an assert if its name appears as an AST identifier
    in that assert (covers ``f(...)`` calls and ``obj.f(...)`` method refs); a
    substring fallback catches names the AST scan misses (e.g. inside f-strings).
    An assert may be attributed to several functions (it references several).
    """
    attributed: dict[str, list[str]] = {s.name: [] for s in steps}
    names_by_step = {s.name: _names_in_assert_target(s) for s in steps}
    for assert_src in asserts:
        ref_names = _names_in_assert(assert_src)
        for step in steps:
            target = names_by_step[step.name]
            if target & ref_names or _substring_hit(target, assert_src):
                attributed[step.name].append(assert_src)
    return attributed


def _names_in_assert_target(step: FunctionStep) -> set[str]:
    """Names that, if referenced by an assert, attribute it to this step.

    Just the function's own name — nested helpers are separate steps with their
    own names, so each is attributed independently.
    """
    return {step.name}


def _substring_hit(target_names: set[str], assert_src: str) -> bool:
    """True if any target name appears as a word-boundaried substring of the assert."""
    return any(re.search(rf"\b{re.escape(n)}\b", assert_src) for n in target_names)


# ---------------------------------------------------------------------------
# Per-step scoring
# ---------------------------------------------------------------------------


def _score_attributed(
    code: str,
    setup: str,
    attributed: list[str],
    execute_fn: ExecuteFn,
    timeout: float,
) -> float:
    """Fraction of a function's attributed asserts that pass under the verifier."""
    if not attributed:
        return 0.0
    setup_block = f"\n{setup}\n" if setup else "\n"
    passed = 0
    for case in attributed:
        ok, _ = execute_fn(f"{code}{setup_block}{case}", timeout)
        passed += int(ok)
    return passed / len(attributed)


def _executability_probe(
    code: str,
    step: FunctionStep,
    execute_fn: ExecuteFn,
    setup: str,
    timeout: float,
) -> bool:
    """Probe whether a function with no attributed test even defines/imports cleanly.

    Runs the candidate (so the function is parsed and bound) under the verifier.
    A function that can't be defined (syntax error, missing import) fails the
    probe; one that defines but is never called still passes (it's executable,
    just untested — which is what makes it potentially *fragile*, not broken).
    """
    setup_block = f"\n{setup}\n" if setup else "\n"
    ok, _ = execute_fn(f"{code}{setup_block}pass", timeout)
    return ok


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _length_normalized_mean(steps: list[dict]) -> float:
    """Length-normalized mean of step scores — resists verbosity hacking.

    A naive mean lets a candidate pad its score with long, vacuous helpers that
    only clear the executability probe (score ~0.5) and never get tested. We
    weight each step so that:

    - A *tested* step (n_tests > 0) carries full weight 1.0 — real verifier
      signal is what we trust.
    - An *untested, probe-only* step is discounted by its length: weight
      ``1 / n_lines``. One short untested helper barely moves the mean; a long
      vacuous function (the verbosity hack) is discounted hard, so padding code
      can no longer inflate ``process_score``.

    With no steps the score is 0.0 (nothing verified).
    """
    if not steps:
        return 0.0
    total_w = 0.0
    acc = 0.0
    for s in steps:
        if s["n_tests"] > 0:
            weight = 1.0
        else:
            weight = 1.0 / max(1, int(s.get("n_lines", 1)))
        acc += weight * float(s["score"])
        total_w += weight
    return acc / total_w if total_w else 0.0


def function_step_scores(
    code: str,
    test_code: str,
    execute_fn: ExecuteFn,
    *,
    language: str = "python",
    timeout: float = 10.0,
    scan_security: bool = True,
) -> StepProfile:
    """Profile a candidate's functions as verifier-graded process steps.

    Pipeline:
      1. Decompose ``code`` into function steps (Python AST / TS regex).
      2. Split ``test_code`` into individual asserts + shared setup
         (reusing ``split_test_cases``).
      3. Attribute each assert to the function(s) it references.
      4. Per function: fraction of attributed asserts passing (run via
         ``execute_fn``); functions with no attributed test get an
         executability probe (define-and-no-op) instead.
      5. Aggregate ``process_score`` as a LENGTH-NORMALIZED mean (untested
         padding can't inflate it).
      6. ``fragile_functions`` = functions that are dead / non-executable while
         the candidate's overall tests pass.

    Args:
        code: Candidate solution source.
        test_code: The problem's test block.
        execute_fn: Sandboxed executor — (code, timeout) -> (ok, output).
        language: "python", "typescript"/"javascript", or "auto".
        timeout: Per-execution timeout in seconds.
        scan_security: If True, flag each step's source via ``scan_dangerous``.

    Returns:
        StepProfile with process_score, per-step dicts, and fragile_functions.
    """
    lang = _resolve_language(code, language)
    steps = decompose_functions(code, language=lang)

    # Python is the only language our injected execute_fn actually RUNS (the
    # sandbox runner pipes Python). For TS we can still decompose into steps and
    # flag security, but we can't grade via this executor — report executable as
    # unknown (None) and give every step the neutral probe score.
    python_executable = lang == "python"

    asserts, setup = ([], "")
    if python_executable:
        asserts, setup = split_test_cases(test_code)

    # Whole-candidate overall pass — used to decide fragility. Only meaningful
    # when we can actually execute (Python). Mirrors evaluate_solution's combine.
    overall_pass = False
    if python_executable and test_code and test_code.strip():
        overall_pass, _ = execute_fn(f"{code}\n\n{test_code}", timeout)

    attributed = _attribute_asserts(steps, asserts) if python_executable else {}

    step_dicts: list[dict] = []
    fragile: list[str] = []
    for step in steps:
        cases = attributed.get(step.name, [])
        n_tests = len(cases)
        dangerous = scan_dangerous(step.src) if scan_security else []

        if not python_executable:
            score = 0.5  # neutral: decomposed but ungradable by this executor
            executable: bool | None = None
        elif n_tests > 0:
            score = _score_attributed(code, setup, cases, execute_fn, timeout)
            executable = True
        else:
            executable = _executability_probe(code, step, execute_fn, setup, timeout)
            # Untested: neutral 0.5 if it at least defines cleanly, else 0.0.
            score = 0.5 if executable else 0.0

        step_dicts.append({
            "name": step.name,
            "score": round(score, 4),
            "n_tests": n_tests,
            "n_lines": step.n_lines,
            "executable": executable,
            "dangerous": dangerous,
        })

        # Fragile = the candidate's overall tests pass, yet this function is
        # untested (dead — never exercised by any assert) or outright
        # non-executable. A latent bug the top-level verifier never touched.
        if python_executable and overall_pass and (n_tests == 0 or executable is False):
            fragile.append(step.name)

    process_score = _length_normalized_mean(step_dicts)
    return StepProfile(
        process_score=round(process_score, 4),
        steps=step_dicts,
        fragile_functions=fragile,
    )


def profile_candidates(
    candidates: list[str],
    test_code: str,
    execute_fn: ExecuteFn,
    **kw: object,
) -> list[dict]:
    """Profile a list of candidate code strings.

    Args:
        candidates: Candidate solution sources.
        test_code: Shared problem test block.
        execute_fn: Sandboxed executor.
        **kw: Forwarded to ``function_step_scores`` (language, timeout, ...).

    Returns:
        One dict per candidate: index, process_score, fragile_functions, steps.
    """
    results: list[dict] = []
    for idx, code in enumerate(candidates):
        profile = function_step_scores(code, test_code, execute_fn, **kw)  # type: ignore[arg-type]
        results.append({
            "index": idx,
            "process_score": profile.process_score,
            "fragile_functions": profile.fragile_functions,
            "steps": profile.steps,
        })
    return results
