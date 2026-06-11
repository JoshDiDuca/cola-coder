"""INFER-005 + EVAL-004 correctness fixes.

- INFER-005: CodeGenerator.generate_batch must forward min_p /
  repetition_penalty / stop_tokens (it silently dropped them before).
- EVAL-004: evaluate_solution with empty test_code must NOT count as a pass
  (it would run with no assertions, exit 0, and inflate pass@k).
"""

from cola_coder.evaluation.humaneval import CodingProblem


# ---------------------------------------------------------------------------
# INFER-005 — generate_batch forwards sampling params
# ---------------------------------------------------------------------------


class _RecordingGenerator:
    """Stands in for CodeGenerator: records kwargs passed to generate()."""

    def __init__(self):
        self.calls = []

    # Bind the real generate_batch onto this fake so we test the actual code.
    from cola_coder.inference.generator import CodeGenerator
    generate_batch = CodeGenerator.generate_batch

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["prompt"] + " out"


class TestGenerateBatchForwardsParams:
    def test_forwards_min_p_and_penalty_and_stops(self):
        gen = _RecordingGenerator()
        gen.generate_batch(
            ["a", "b"], max_new_tokens=32, temperature=0.5, top_k=10,
            top_p=0.8, min_p=0.07, repetition_penalty=1.3, stop_tokens=["\n\n"],
        )
        assert len(gen.calls) == 2
        for call in gen.calls:
            assert call["min_p"] == 0.07
            assert call["repetition_penalty"] == 1.3
            assert call["stop_tokens"] == ["\n\n"]
            assert call["top_p"] == 0.8

    def test_defaults_preserved(self):
        gen = _RecordingGenerator()
        gen.generate_batch(["x"])
        call = gen.calls[0]
        assert call["min_p"] == 0.0
        assert call["repetition_penalty"] == 1.1
        assert call["stop_tokens"] is None


# ---------------------------------------------------------------------------
# EVAL-004 — empty test_code is not a pass
# ---------------------------------------------------------------------------


def _problem(test_code: str) -> CodingProblem:
    return CodingProblem(
        task_id="t1", prompt="def f():\n", test_code=test_code, entry_point="f",
    )


class TestEmptyTestCode:
    def test_empty_test_code_is_not_passed(self):
        from cola_coder.evaluation.runner import evaluate_solution

        passed, msg = evaluate_solution(_problem(""), "def f():\n    return 1")
        assert passed is False
        assert "NO TESTS" in msg

    def test_whitespace_test_code_is_not_passed(self):
        from cola_coder.evaluation.runner import evaluate_solution

        passed, msg = evaluate_solution(_problem("   \n  \n"), "def f(): pass")
        assert passed is False
        assert "NO TESTS" in msg

    def test_empty_test_does_not_execute_code(self, monkeypatch):
        # Guard must short-circuit BEFORE execute_code (no sandbox spin-up).
        import cola_coder.evaluation.runner as runner

        called = {"n": 0}
        monkeypatch.setattr(
            runner, "execute_code",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1) or (True, ""),
        )
        runner.evaluate_solution(_problem(""), "whatever")
        assert called["n"] == 0

    def test_nonempty_test_still_runs(self, monkeypatch):
        import cola_coder.evaluation.runner as runner

        seen = {}
        monkeypatch.setattr(
            runner, "execute_code",
            lambda code, **k: (seen.update(code=code) or (True, "ok")),
        )
        passed, _ = runner.evaluate_solution(
            _problem("assert f() == 1"), "def f():\n    return 1"
        )
        assert passed is True
        assert "assert f() == 1" in seen["code"]
