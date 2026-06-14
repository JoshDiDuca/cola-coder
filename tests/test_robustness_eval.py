"""Tests for verifier-graded robustness evaluation (EVAL-030).

Hermetic. The model is replaced by a deterministic ``generate_fn`` stub. A couple
of tests do let the real sandbox verifier run TRIVIAL Python (a one-line addition)
— that is safe in CI and exercises the real DRY verifier path. No GPU, no model,
no checkpoint loads.
"""

from __future__ import annotations

from cola_coder.evaluation.humaneval import CodingProblem
from cola_coder.evaluation.robustness_eval import RobustnessReport, evaluate_robustness

_PROMPT = '''def add_two(a: int, b: int) -> int:
    """Return the sum of the given two numbers.
    >>> add_two(1, 2)
    3
    >>> add_two(5, 5)
    10
    """
'''

# A correct, complete function definition the verifier can execute.
_SOLUTION = "def add_two(a: int, b: int) -> int:\n    return a + b\n"
# A wrong solution that fails the asserts.
_WRONG = "def add_two(a: int, b: int) -> int:\n    return a - b\n"


def _problem() -> CodingProblem:
    return CodingProblem(
        task_id="add_two",
        prompt=_PROMPT,
        test_code="assert add_two(1, 2) == 3\nassert add_two(5, 5) == 10\n",
        entry_point="add_two",
    )


def test_always_correct_stub_is_fully_robust() -> None:
    """A stub returning the canonical solution for ANY prompt → robust=1, consistent=1."""
    report = evaluate_robustness(lambda prompt: _SOLUTION, [_problem()])
    assert isinstance(report, RobustnessReport)
    assert report.robust_pass_at_1 == 1.0
    assert report.consistency_rate == 1.0
    assert report.fragile_task_ids == []
    assert report.num_problems == 1


def test_always_wrong_stub_is_consistent_but_not_robust() -> None:
    """Wrong everywhere → robust=0 but verdict is invariant, so consistency=1, not fragile."""
    report = evaluate_robustness(lambda prompt: _WRONG, [_problem()])
    assert report.robust_pass_at_1 == 0.0
    assert report.consistency_rate == 1.0
    assert report.fragile_task_ids == []  # never solved clean → not "fragile"


def test_fragility_detected_when_only_clean_prompt_passes() -> None:
    """Stub that solves ONLY the exact clean prompt → fragile + low robustness."""
    clean_prompt = _PROMPT

    def fragile_fn(prompt: str) -> str:
        return _SOLUTION if prompt == clean_prompt else _WRONG

    report = evaluate_robustness(fragile_fn, [_problem()])
    assert report.fragile_task_ids == ["add_two"]
    assert report.robust_pass_at_1 == 0.0  # worst-case fails
    assert report.consistency_rate == 0.0  # verdict varies across variants
    row = report.per_problem[0]
    assert row["clean_pass"] is True
    assert row["robust_pass"] is False
    assert row["fragile"] is True


def test_per_problem_records_verdict_for_every_variant() -> None:
    report = evaluate_robustness(lambda prompt: _SOLUTION, [_problem()])
    row = report.per_problem[0]
    assert "clean" in row["verdicts"]
    assert row["num_variants"] == len(row["verdicts"])
    assert all(v is True for v in row["verdicts"].values())


def test_bootstrap_ci_attached_when_requested() -> None:
    problems = [_problem()]
    report = evaluate_robustness(
        lambda prompt: _SOLUTION, problems, compute_ci=True, n_boot=200
    )
    assert report.robust_pass_at_1_ci is not None
    point, lo, hi = report.robust_pass_at_1_ci
    assert point == 1.0
    assert lo <= point <= hi


def test_harness_exception_counts_as_fail_not_crash() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("generation blew up")

    # Must not raise; a crash is graded as a fail for that variant.
    report = evaluate_robustness(boom, [_problem()])
    assert report.robust_pass_at_1 == 0.0
    assert report.num_problems == 1


def test_kinds_subset_limits_variants() -> None:
    report = evaluate_robustness(
        lambda prompt: _SOLUTION, [_problem()], kinds=["paraphrase"]
    )
    row = report.per_problem[0]
    # clean + paraphrase only.
    assert set(row["verdicts"].keys()) <= {"clean", "paraphrase"}
    assert "clean" in row["verdicts"]


def test_empty_problem_set_is_safe() -> None:
    report = evaluate_robustness(lambda prompt: _SOLUTION, [])
    assert report.robust_pass_at_1 == 0.0
    assert report.consistency_rate == 0.0
    assert report.num_problems == 0
    assert report.fragile_task_ids == []
