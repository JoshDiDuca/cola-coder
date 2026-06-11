"""EVAL-006: evaluate_solution must not run non-Python problems as Python.

`evaluate_solution` (evaluation/runner.py) executes Python via execute_code. It
ignored `problem.language`, so a TypeScript CodingProblem (TYPESCRIPT_PROBLEMS,
reachable via ProblemSet.filter_by_language) would have its TS test_code piped
through the Python interpreter and always fail with a misleading SyntaxError —
silently deflating pass@k for this TS-primary project. The guard fails loud with
a clear pointer to the tsc-based ts_benchmark instead.

execute_code is stubbed so these tests never spawn a subprocess and can assert
whether the Python execution path was reached.
"""

import cola_coder.evaluation.runner as runner_mod
from cola_coder.evaluation.runner import evaluate_solution
from cola_coder.evaluation.humaneval import CodingProblem


def test_typescript_problem_is_not_executed_as_python(monkeypatch):
    called = []
    monkeypatch.setattr(
        runner_mod, "execute_code",
        lambda code, timeout=10.0: (called.append(code) or (True, "")),
    )
    p = CodingProblem(
        task_id="ts_demo", prompt="", test_code="const r: boolean = true;",
        entry_point="x", language="typescript",
    )
    passed, msg = evaluate_solution(p, "function x(): void {}")
    assert passed is False
    assert "LANGUAGE NOT SUPPORTED" in msg
    assert "ts_demo" in msg and "typescript" in msg
    # The Python executor must never run on a TS problem.
    assert called == []


def test_python_problem_still_reaches_execution(monkeypatch):
    called = []
    monkeypatch.setattr(
        runner_mod, "execute_code",
        lambda code, timeout=10.0: (called.append(code) or (True, "ok")),
    )
    p = CodingProblem(
        task_id="py_demo", prompt="", test_code="assert x() == 1",
        entry_point="x", language="python",
    )
    passed, _ = evaluate_solution(p, "def x():\n    return 1")
    assert passed is True
    assert called  # Python path was reached and given the combined code


def test_empty_test_code_guard_still_fires_for_python(monkeypatch):
    monkeypatch.setattr(runner_mod, "execute_code", lambda code, timeout=10.0: (True, ""))
    p = CodingProblem(
        task_id="py_empty", prompt="", test_code="   ",
        entry_point="x", language="python",
    )
    passed, msg = evaluate_solution(p, "def x(): pass")
    assert passed is False
    assert "NO TESTS" in msg


def test_default_language_is_python_and_executes(monkeypatch):
    # language defaults to "python" — a problem constructed without it must still
    # run (the guard must not reject the common case).
    called = []
    monkeypatch.setattr(
        runner_mod, "execute_code",
        lambda code, timeout=10.0: (called.append(code) or (True, "")),
    )
    p = CodingProblem(task_id="d", prompt="", test_code="assert True", entry_point="x")
    evaluate_solution(p, "pass")
    assert called
