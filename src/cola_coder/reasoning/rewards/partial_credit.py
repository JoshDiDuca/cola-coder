"""Fractional (partial-credit) test reward for GRPO — finer execution-grounded credit.

The default ``python_exec`` reward runs the whole test block as ONE unit: pass all →
1.0, fail any → 0.0. That coarse, binary signal (the credit-assignment problem
Execution-Grounded Credit Assignment / EGCA flags, arXiv:2603.16158) wastes learning
signal — a solution passing 4 of 5 tests looks identical to one passing 0 — and makes
GRPO groups collapse to zero variance (all 0.0) far more often, starving the update.

This reward splits the test block into individual ``assert`` cases (via the AST, so
syntax is respected) and runs each against the candidate, returning the FRACTION
passed. Non-assert top-level statements (imports, helper defs, fixtures) are kept as
shared SETUP and prepended to every case so a split assert still sees what it needs.
Falls back to all-or-nothing when there are no top-level asserts (e.g. a custom check
harness) so behavior is never worse than binary.

Cost: N asserts → N sandbox executions per candidate (vs 1). It's opt-in
(``--reward python_partial``) and runs only during GRPO (offline), so the extra
sandbox time is acceptable for the denser signal. Untrusted code only ever runs
through the injected ``execute_fn`` (the sandboxed evaluation.runner.execute_code).
"""

from __future__ import annotations

import ast
from typing import Callable

# (code, timeout) -> (success, output) — the sandboxed executor contract.
ExecuteFn = Callable[[str, float], tuple[bool, str]]


def split_test_cases(test_code: str) -> tuple[list[str], str]:
    """Split a test block into (assert_cases, shared_setup).

    Returns the source of each top-level ``assert`` statement and the source of all
    OTHER top-level statements joined as setup. Returns ``([], "")`` when the block
    doesn't parse or has no top-level asserts (caller falls back to binary).
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return [], ""
    asserts: list[str] = []
    setup: list[str] = []
    for node in tree.body:
        src = ast.unparse(node)
        if isinstance(node, ast.Assert):
            asserts.append(src)
        else:
            setup.append(src)
    return asserts, "\n".join(setup)


def fractional_python_reward(
    code: str,
    test_code: str,
    execute_fn: ExecuteFn,
    timeout: float = 10.0,
) -> tuple[float, dict]:
    """Fraction of individual ``assert`` cases the candidate passes.

    Args:
        code: The candidate solution (thinking already stripped by the caller).
        test_code: The problem's test block (multiple asserts).
        execute_fn: Sandboxed executor — (code, timeout) -> (success, output).
        timeout: Per-case execution timeout.

    Returns:
        (reward in [0,1], info). info carries num_tests / num_passed / mode.
    """
    asserts, setup = split_test_cases(test_code)
    if not asserts:
        # No splittable asserts → all-or-nothing on the whole block (binary).
        ok, output = execute_fn(code + "\n\n" + test_code, timeout)
        return (1.0 if ok else 0.0), {
            "num_tests": 1,
            "num_passed": int(ok),
            "mode": "binary_fallback",
            "execution_output": output[-200:],
        }

    setup_block = f"\n{setup}\n" if setup else "\n"
    passed = 0
    for case in asserts:
        full = f"{code}{setup_block}{case}"
        ok, _ = execute_fn(full, timeout)
        passed += int(ok)
    frac = passed / len(asserts)
    return frac, {
        "num_tests": len(asserts),
        "num_passed": passed,
        "mode": "fractional",
    }
