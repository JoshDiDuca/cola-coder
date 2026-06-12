"""EVAL-009: the HumanEval orchestrator must distinguish HARNESS errors from
genuine model failures.

scripts/evaluate.py previously did `except Exception: pass` around the whole
generate→extract→grade pipeline, so a generator crash / OOM / sandbox misconfig
was silently counted as "model got it wrong". A 0% pass@k then meant either a
weak model OR a broken harness, with no way to tell. `_evaluate_problem` now
returns (num_correct, harness_errors) separately.
"""

import importlib.util
import types
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "evaluate.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem():
    return types.SimpleNamespace(
        task_id="t1", prompt="def f():\n", entry_point="f", language="python",
    )


class _Gen:
    def __init__(self, behavior):
        self.behavior = behavior  # "ok" | "raise"

    def generate(self, **kwargs):
        if self.behavior == "raise":
            raise RuntimeError("CUDA OOM")
        return "def f(): return 1"


def _extract_ok(text, entry_point):
    return text


class TestEvaluateProblem:
    def test_all_pass_no_harness_errors(self):
        m = _load()
        nc, he = m._evaluate_problem(
            _Gen("ok"), _problem(), num_samples=4, temperature=0.2,
            extract_fn=_extract_ok, evaluate_fn=lambda p, c: (True, "ok"),
        )
        assert nc == 4 and he == 0

    def test_test_failures_are_not_harness_errors(self):
        # evaluate_fn returns (False, ...) WITHOUT raising → genuine model fail,
        # not a harness error.
        m = _load()
        nc, he = m._evaluate_problem(
            _Gen("ok"), _problem(), num_samples=3, temperature=0.2,
            extract_fn=_extract_ok, evaluate_fn=lambda p, c: (False, "assertion failed"),
        )
        assert nc == 0 and he == 0

    def test_generator_crash_counted_as_harness_error(self):
        m = _load()
        nc, he = m._evaluate_problem(
            _Gen("raise"), _problem(), num_samples=5, temperature=0.2,
            extract_fn=_extract_ok, evaluate_fn=lambda p, c: (True, "ok"),
        )
        assert nc == 0 and he == 5  # every sample errored, NONE counted as a pass

    def test_grader_exception_is_harness_error(self):
        # An unexpected raise inside grading (e.g. sandbox misconfig) is a harness
        # error, distinct from a (False, ...) return.
        m = _load()

        def boom(problem, code):
            raise OSError("sandbox unavailable")

        nc, he = m._evaluate_problem(
            _Gen("ok"), _problem(), num_samples=2, temperature=0.2,
            extract_fn=_extract_ok, evaluate_fn=boom,
        )
        assert nc == 0 and he == 2

    def test_mixed_outcomes(self):
        # 2 pass, then grader fails the rest via a counter.
        m = _load()
        state = {"calls": 0}

        def sometimes(problem, code):
            state["calls"] += 1
            return (state["calls"] <= 2, "")

        nc, he = m._evaluate_problem(
            _Gen("ok"), _problem(), num_samples=5, temperature=0.2,
            extract_fn=_extract_ok, evaluate_fn=sometimes,
        )
        assert nc == 2 and he == 0  # the rest are honest test failures, not errors
