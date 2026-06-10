"""Tests for sandboxed execution of model-generated code.

Regression coverage for routing evaluation/runner.py::execute_code through
the shared SandboxedRunner (previously a raw subprocess.run on the host with
only a timeout + empty PATH).
"""

from __future__ import annotations

from cola_coder.data.scorers.sandbox import SandboxedRunner
from cola_coder.evaluation.runner import (
    evaluate_solution,
    execute_code,
    get_execution_runner,
)


def _native_runner(timeout: int = 10) -> SandboxedRunner:
    return SandboxedRunner(use_docker=False, timeout=timeout)


class TestExecuteCode:
    def test_successful_code(self):
        ok, out = execute_code("print('hello')", runner=_native_runner())
        assert ok is True
        assert "hello" in out

    def test_failing_code(self):
        ok, out = execute_code("raise ValueError('boom')", runner=_native_runner())
        assert ok is False
        assert "boom" in out

    def test_assertion_failure(self):
        ok, _ = execute_code("assert 1 == 2", runner=_native_runner())
        assert ok is False

    def test_timeout_kills_infinite_loop(self):
        ok, out = execute_code(
            "while True:\n    pass", timeout=2, runner=_native_runner(),
        )
        assert ok is False
        assert "TIMEOUT" in out

    def test_timeout_message_format_preserved(self):
        """Callers (rewards) may match on the TIMEOUT prefix — keep it stable."""
        ok, out = execute_code(
            "import time; time.sleep(30)", timeout=1, runner=_native_runner(),
        )
        assert ok is False
        assert out.startswith("TIMEOUT:")

    def test_restricted_path_blocks_system_commands(self):
        """Empty PATH means os.system can't resolve commands by name."""
        ok, out = execute_code(
            "import subprocess\n"
            "subprocess.run(['definitely-not-a-real-command'], check=True)\n",
            runner=_native_runner(),
        )
        assert ok is False

    def test_runs_through_sandboxed_runner(self):
        """Execution must be attributed to the runner (counters move)."""
        runner = _native_runner()
        before = runner.get_run_summary()["total_runs"]
        execute_code("print(1)", runner=runner)
        assert runner.get_run_summary()["total_runs"] == before + 1

    def test_default_runner_is_cached(self):
        get_execution_runner.cache_clear()
        a = get_execution_runner()
        b = get_execution_runner()
        assert a is b
        get_execution_runner.cache_clear()


class TestEvaluateSolution:
    def test_passing_solution(self):
        from cola_coder.evaluation.humaneval import CodingProblem

        problem = CodingProblem(
            task_id="t/0",
            prompt="def add(a, b):\n",
            entry_point="add",
            test_code="assert add(1, 2) == 3",
            canonical_solution="    return a + b",
        )
        passed, _ = evaluate_solution(problem, "def add(a, b):\n    return a + b")
        assert passed is True

    def test_failing_solution(self):
        from cola_coder.evaluation.humaneval import CodingProblem

        problem = CodingProblem(
            task_id="t/1",
            prompt="def add(a, b):\n",
            entry_point="add",
            test_code="assert add(1, 2) == 3",
            canonical_solution="    return a + b",
        )
        passed, _ = evaluate_solution(problem, "def add(a, b):\n    return a - b")
        assert passed is False
