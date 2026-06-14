"""Tests for difficulty-stratified robustness evaluation (EVAL-031).

Hermetic. The model is a deterministic ``generate_fn`` stub; difficulty tiers are
injected as a ``task_id -> tier`` mapping so no best-of-N / GPU / checkpoint is
needed. Verifies per-tier robust_pass@1 / consistency / n, that each tier's
bootstrap CI brackets its point estimate, the overall CI, the unknown-tier bucket,
and back-compat (no mapping → empty ``by_tier``, existing fields untouched).
"""

from __future__ import annotations

from cola_coder.evaluation.humaneval import CodingProblem
from cola_coder.evaluation.robustness_eval import (
    UNKNOWN_TIER,
    RobustnessReport,
    evaluate_robustness,
)

# Correct / wrong solutions for a trivial addition the real sandbox can execute.
_SOLUTION = "def add_two(a: int, b: int) -> int:\n    return a + b\n"
_WRONG = "def add_two(a: int, b: int) -> int:\n    return a - b\n"


_PROMPT = '''def add_two(a: int, b: int) -> int:
    """Return the sum of the given two numbers.
    >>> add_two(1, 2)
    3
    >>> add_two(5, 5)
    10
    """
'''


def _problem(task_id: str) -> CodingProblem:
    return CodingProblem(
        task_id=task_id,
        prompt=_PROMPT,
        test_code="assert add_two(1, 2) == 3\nassert add_two(5, 5) == 10\n",
        entry_point="add_two",
    )


def _always_correct(prompt: str) -> str:
    return _SOLUTION


def _always_wrong(prompt: str) -> str:
    return _WRONG


def test_no_mapping_leaves_by_tier_empty() -> None:
    """Back-compat: without a tier mapping, ``by_tier`` is empty (EVAL-030 intact)."""
    report = evaluate_robustness(_always_correct, [_problem("p1")])
    assert isinstance(report, RobustnessReport)
    assert report.by_tier == {}
    assert report.robust_pass_at_1 == 1.0


def test_per_tier_metrics_are_correct() -> None:
    """robust_pass@1, consistency, and n are computed per tier and grouped right."""
    problems = [_problem("e1"), _problem("e2"), _problem("h1")]
    tiers = {"e1": "easy", "e2": "easy", "h1": "hard"}
    report = evaluate_robustness(_always_correct, problems, difficulty_tiers=tiers)

    assert set(report.by_tier.keys()) == {"easy", "hard"}
    assert report.by_tier["easy"]["n"] == 2
    assert report.by_tier["hard"]["n"] == 1
    # All solved → robust_pass@1 == 1.0 and fully consistent in every tier.
    assert report.by_tier["easy"]["robust_pass_at_1"] == 1.0
    assert report.by_tier["hard"]["robust_pass_at_1"] == 1.0
    assert report.by_tier["easy"]["consistency_rate"] == 1.0


def test_failing_tier_has_zero_robust_pass() -> None:
    """A tier whose problems all fail reports robust_pass@1 == 0.0."""
    problems = [_problem("h1"), _problem("h2")]
    tiers = {"h1": "hard", "h2": "hard"}
    report = evaluate_robustness(_always_wrong, problems, difficulty_tiers=tiers)
    assert set(report.by_tier.keys()) == {"hard"}
    assert report.by_tier["hard"]["n"] == 2
    assert report.by_tier["hard"]["robust_pass_at_1"] == 0.0
    # Wrong everywhere → invariant verdict → fully consistent.
    assert report.by_tier["hard"]["consistency_rate"] == 1.0


def test_distinct_robust_pass_per_tier() -> None:
    """A solving stub vs a failing stub yield robust_pass@1 of 1.0 vs 0.0 per tier.

    The stub is global (prompts are identical across problems), so we assert each
    tier from its own run: solving → 1.0, failing → 0.0.
    """
    report_easy = evaluate_robustness(
        _always_correct, [_problem("e1")], difficulty_tiers={"e1": "easy"}
    )
    report_med = evaluate_robustness(
        _always_wrong, [_problem("m1")], difficulty_tiers={"m1": "medium"}
    )
    assert report_easy.by_tier["easy"]["robust_pass_at_1"] == 1.0
    assert report_med.by_tier["medium"]["robust_pass_at_1"] == 0.0


def test_tier_emission_order_follows_tiers_constant() -> None:
    """Tiers are emitted easy→medium→hard→unsolved, with unknown last."""
    problems = [_problem("h"), _problem("e"), _problem("u"), _problem("x")]
    tiers = {"h": "hard", "e": "easy", "u": "unsolved"}  # 'x' → unknown
    report = evaluate_robustness(_always_correct, problems, difficulty_tiers=tiers)
    assert list(report.by_tier.keys()) == ["easy", "hard", "unsolved", UNKNOWN_TIER]


def test_per_tier_ci_brackets_point_estimate() -> None:
    """When compute_ci, each tier's bootstrap CI brackets its point estimate."""
    problems = [_problem("e1"), _problem("e2"), _problem("e3")]
    tiers = {"e1": "easy", "e2": "easy", "e3": "easy"}
    report = evaluate_robustness(
        _always_correct, problems, difficulty_tiers=tiers, compute_ci=True, n_boot=200
    )
    stats = report.by_tier["easy"]
    assert stats["ci"] is not None
    point, lo, hi = stats["ci"]
    assert point == stats["robust_pass_at_1"]
    assert lo <= point <= hi


def test_overall_ci_still_present_with_stratification() -> None:
    """Overall robust_pass@1 CI is attached alongside per-tier CIs."""
    problems = [_problem("e1"), _problem("h1")]
    tiers = {"e1": "easy", "h1": "hard"}
    report = evaluate_robustness(
        _always_correct, problems, difficulty_tiers=tiers, compute_ci=True, n_boot=200
    )
    assert report.robust_pass_at_1_ci is not None
    point, lo, hi = report.robust_pass_at_1_ci
    assert lo <= point <= hi
    # Every tier also carries a CI.
    for stats in report.by_tier.values():
        assert stats["ci"] is not None


def test_no_ci_when_not_requested() -> None:
    """Per-tier ``ci`` is None when compute_ci is False."""
    problems = [_problem("e1")]
    report = evaluate_robustness(
        _always_correct, problems, difficulty_tiers={"e1": "easy"}
    )
    assert report.by_tier["easy"]["ci"] is None


def test_unknown_tier_bucket_for_missing_and_invalid() -> None:
    """Unmapped task_ids and unrecognized tier labels fall into UNKNOWN_TIER."""
    problems = [_problem("a"), _problem("b"), _problem("c")]
    tiers = {"a": "easy", "b": "bogus_tier"}  # 'c' missing, 'b' invalid
    report = evaluate_robustness(_always_correct, problems, difficulty_tiers=tiers)
    assert "easy" in report.by_tier
    assert UNKNOWN_TIER in report.by_tier
    assert report.by_tier["easy"]["n"] == 1
    # 'b' (invalid) + 'c' (missing) both land in unknown.
    assert report.by_tier[UNKNOWN_TIER]["n"] == 2


def test_tier_n_sums_to_num_problems() -> None:
    """Per-tier counts partition the full problem set exactly."""
    problems = [_problem(f"p{i}") for i in range(5)]
    tiers = {"p0": "easy", "p1": "easy", "p2": "medium", "p3": "hard"}  # p4 unmapped
    report = evaluate_robustness(_always_correct, problems, difficulty_tiers=tiers)
    assert sum(s["n"] for s in report.by_tier.values()) == report.num_problems == 5
    assert report.by_tier[UNKNOWN_TIER]["n"] == 1


def test_empty_mapping_dict_buckets_everything_unknown() -> None:
    """An empty (but non-None) mapping still stratifies — all into unknown."""
    problems = [_problem("a"), _problem("b")]
    report = evaluate_robustness(_always_correct, problems, difficulty_tiers={})
    # Empty dict is falsy: by_tier is computed (mapping is not None) with all unknown.
    assert report.by_tier == {UNKNOWN_TIER: report.by_tier[UNKNOWN_TIER]}
    assert report.by_tier[UNKNOWN_TIER]["n"] == 2
