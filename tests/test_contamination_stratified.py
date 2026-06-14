"""Tests for contamination-trust-stratified pass@k (EVAL-036).

All CPU-only and model-free: pure text containment + statistics over already-
collected ``ProblemResult`` records.
"""

from __future__ import annotations

import pytest

from cola_coder.evaluation.contamination_stratified import (
    DEFAULT_CONTAMINATED_THRESHOLD,
    DEFAULT_SUSPECT_THRESHOLD,
    TIERS,
    ProblemContamination,
    StratifiedPassAtKReport,
    build_contamination_detector,
    contamination_tier,
    score_problem_contamination,
    stratified_pass_at_k,
)
from cola_coder.evaluation.humaneval import CodingProblem
from cola_coder.evaluation.metrics import ProblemResult


# --------------------------------------------------------------------------- #
# contamination_tier
# --------------------------------------------------------------------------- #


def test_tier_boundaries() -> None:
    assert contamination_tier(0.0) == "clean"
    assert contamination_tier(0.49) == "clean"
    assert contamination_tier(DEFAULT_SUSPECT_THRESHOLD) == "suspect"
    assert contamination_tier(0.79) == "suspect"
    assert contamination_tier(DEFAULT_CONTAMINATED_THRESHOLD) == "contaminated"
    assert contamination_tier(1.0) == "contaminated"


def test_tier_custom_thresholds() -> None:
    assert contamination_tier(0.4, suspect_threshold=0.3, contaminated_threshold=0.6) == "suspect"
    assert contamination_tier(0.2, suspect_threshold=0.3, contaminated_threshold=0.6) == "clean"
    assert contamination_tier(0.7, suspect_threshold=0.3, contaminated_threshold=0.6) == "contaminated"


def test_tier_rejects_misordered_thresholds() -> None:
    with pytest.raises(ValueError):
        contamination_tier(0.5, suspect_threshold=0.9, contaminated_threshold=0.8)
    with pytest.raises(ValueError):
        contamination_tier(0.5, suspect_threshold=-0.1, contaminated_threshold=0.8)
    with pytest.raises(ValueError):
        contamination_tier(0.5, suspect_threshold=0.5, contaminated_threshold=1.5)


def test_tiers_constant_shape() -> None:
    assert TIERS == ("clean", "suspect", "contaminated")


# --------------------------------------------------------------------------- #
# score_problem_contamination
# --------------------------------------------------------------------------- #


def _problem(task_id: str, prompt: str, solution: str = "") -> CodingProblem:
    return CodingProblem(
        task_id=task_id,
        prompt=prompt,
        test_code="assert True",
        entry_point=task_id,
        canonical_solution=solution,
    )


def test_score_detects_prompt_leak() -> None:
    prompt = "def add(a, b):\n    '''return the sum of a and b'''\n"
    problems = [_problem("add", prompt)]
    # Training doc fully embeds the prompt → containment ~1.0 → contaminated.
    train = ["# some file\n" + prompt + "\n    return a + b\n# more code"]
    diag = score_problem_contamination(problems, train)[0]
    assert diag.task_id == "add"
    assert diag.score >= DEFAULT_CONTAMINATED_THRESHOLD
    assert diag.tier == "contaminated"
    assert diag.matched_unit == "prompt"


def test_score_detects_solution_leak_over_prompt() -> None:
    prompt = "Implement a function that reverses a string."
    solution = "def reverse(s):\n    return s[::-1]\n"
    problems = [_problem("reverse", prompt, solution)]
    # Only the SOLUTION is embedded in training, not the prose prompt.
    train = ["random unrelated text", "helper:\n" + solution + "\nx = reverse('hi')"]
    diag = score_problem_contamination(problems, train)[0]
    assert diag.matched_unit == "solution"
    assert diag.tier == "contaminated"


def test_score_clean_when_unrelated() -> None:
    problems = [_problem("novel", "def parse_iso_timestamp(s): ...", "def parse_iso_timestamp(s): return s")]
    train = ["completely different content about cooking recipes and gardening"]
    diag = score_problem_contamination(problems, train)[0]
    assert diag.score < DEFAULT_SUSPECT_THRESHOLD
    assert diag.tier == "clean"
    # Some low-but-nonzero overlap exists (shared tokens like "def"); the unit is
    # whichever scored highest, or "none" only when containment is exactly 0.
    assert diag.matched_unit in ("prompt", "solution", "none")


def test_score_empty_train_is_clean() -> None:
    problems = [_problem("p", "def f(): ...", "def f(): return 1")]
    diags = score_problem_contamination(problems, [])
    assert len(diags) == 1
    assert diags[0].score == 0.0
    assert diags[0].tier == "clean"


def test_score_preserves_input_order() -> None:
    problems = [_problem(f"p{i}", f"def f{i}(): ...") for i in range(5)]
    diags = score_problem_contamination(problems, ["x"])
    assert [d.task_id for d in diags] == [f"p{i}" for i in range(5)]


def test_score_rejects_bad_shingle_size() -> None:
    with pytest.raises(ValueError):
        score_problem_contamination([_problem("p", "abc")], ["abc"], shingle_size=0)


def test_score_matches_detector_containment_definition() -> None:
    # The continuous score must equal the binary detector's containment for the
    # same shingle size — they share _containment/_shingles, so a contaminated
    # problem here is exactly one the detector flags.
    prompt = "def gcd(a, b): return a if b == 0 else gcd(b, a % b)"
    train = ["module:\n" + prompt + "\nprint(gcd(12, 8))"]
    diag = score_problem_contamination([_problem("gcd", prompt)], train, shingle_size=5)[0]
    detector = build_contamination_detector(train, shingle_size=5, threshold=diag.score)
    report = detector.check_eval([prompt], metric="containment")
    assert report.has_leakage()  # detector flags it at the same score → consistent


# --------------------------------------------------------------------------- #
# stratified_pass_at_k
# --------------------------------------------------------------------------- #


def _result(task_id: str, n: int, c: int) -> ProblemResult:
    return ProblemResult(task_id=task_id, num_samples=n, num_correct=c)


def _diag(task_id: str, tier: str) -> ProblemContamination:
    score = {"clean": 0.1, "suspect": 0.6, "contaminated": 0.95}[tier]
    return ProblemContamination(task_id=task_id, score=score, tier=tier, matched_unit="prompt")


def test_stratified_separates_tiers_and_computes_delta() -> None:
    # Contaminated problems all pass; clean problems all fail → big NEGATIVE delta
    # (clean − contaminated), the memorisation signature.
    results = [
        _result("c1", 4, 4), _result("c2", 4, 4),   # contaminated, solved
        _result("k1", 4, 0), _result("k2", 4, 0),   # clean, unsolved
    ]
    contamination = [
        _diag("c1", "contaminated"), _diag("c2", "contaminated"),
        _diag("k1", "clean"), _diag("k2", "clean"),
    ]
    report = stratified_pass_at_k(results, contamination, k_values=(1,))
    assert isinstance(report, StratifiedPassAtKReport)
    assert report.by_tier["contaminated"].num_problems == 2
    assert report.by_tier["clean"].num_problems == 2
    assert report.by_tier["contaminated"].pass_at_k[1] == pytest.approx(1.0)
    assert report.by_tier["clean"].pass_at_k[1] == pytest.approx(0.0)
    # headline mixes both → 0.5; decontaminated (clean) → 0.0
    assert report.overall[1] == pytest.approx(0.5)
    assert report.trusted_pass_at_k(1) == pytest.approx(0.0)
    assert report.trust_delta[1] == pytest.approx(-1.0)


def test_stratified_trust_delta_none_when_a_tier_empty() -> None:
    results = [_result("k1", 4, 2), _result("k2", 4, 3)]
    contamination = [_diag("k1", "clean"), _diag("k2", "clean")]
    report = stratified_pass_at_k(results, contamination, k_values=(1,))
    assert report.by_tier["contaminated"].num_problems == 0
    assert report.trust_delta[1] is None  # no contaminated tier to compare
    assert report.trusted_pass_at_k(1) is not None


def test_stratified_records_unmatched() -> None:
    results = [_result("k1", 4, 1)]
    contamination = [_diag("k1", "clean"), _diag("ghost", "contaminated")]
    report = stratified_pass_at_k(results, contamination, k_values=(1,))
    assert report.unmatched_task_ids == ["ghost"]
    # the ghost diagnosis does not leak into any tier's results
    assert report.by_tier["contaminated"].num_problems == 0


def test_stratified_handles_insufficient_samples_for_k() -> None:
    # n < k → pass@k not estimable → omitted from the dict (not a spurious 0/1).
    results = [_result("k1", 2, 1), _result("c1", 2, 2)]
    contamination = [_diag("k1", "clean"), _diag("c1", "contaminated")]
    report = stratified_pass_at_k(results, contamination, k_values=(5,))
    assert 5 not in report.overall
    assert 5 not in report.by_tier["clean"].pass_at_k
    assert report.trust_delta[5] is None


def test_stratified_rejects_empty_k_values() -> None:
    with pytest.raises(ValueError):
        stratified_pass_at_k([], [], k_values=())


def test_stratified_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        stratified_pass_at_k([], [], k_values=(0,))
    with pytest.raises(ValueError):
        stratified_pass_at_k([], [], k_values=(-1,))


def test_report_summary_renders_all_k() -> None:
    results = [_result("k1", 5, 2), _result("c1", 5, 5)]
    contamination = [_diag("k1", "clean"), _diag("c1", "contaminated")]
    report = stratified_pass_at_k(results, contamination, k_values=(1, 5))
    text = report.summary()
    assert "pass@1" in text
    assert "pass@5" in text
    assert "trust_delta" in text


def test_problem_contamination_summary() -> None:
    diag = ProblemContamination(task_id="t", score=0.83, tier="contaminated", matched_unit="solution")
    s = diag.summary()
    assert "t" in s and "contaminated" in s and "0.83" in s and "solution" in s


# --------------------------------------------------------------------------- #
# End-to-end: score real built-in problems against a planted leak
# --------------------------------------------------------------------------- #


def test_end_to_end_planted_leak_lowers_trust() -> None:
    from cola_coder.evaluation.problem_loader import ProblemSet

    ps = ProblemSet().add_builtin(extended=False)
    problems = list(ps)[:6]
    assert len(problems) == 6

    # Plant the first problem's canonical solution (or prompt) verbatim in training.
    leaked = problems[0]
    leak_text = leaked.canonical_solution or leaked.prompt
    train = ["unrelated docs about weather", "memo:\n" + leak_text + "\n# end"]

    diags = score_problem_contamination(problems, train)
    by_id = {d.task_id: d for d in diags}
    assert by_id[leaked.task_id].tier == "contaminated"
    # the other five should not be flagged as contaminated by an unrelated leak
    assert sum(1 for d in diags if d.tier == "contaminated") == 1

    # The contaminated problem "passes" (memorised), the rest fail → trust delta < 0.
    results = []
    for p in problems:
        if p.task_id == leaked.task_id:
            results.append(_result(p.task_id, 5, 5))
        else:
            results.append(_result(p.task_id, 5, 0))
    report = stratified_pass_at_k(results, diags, k_values=(1,))
    assert report.overall[1] == pytest.approx(1.0 / 6)
    assert report.trusted_pass_at_k(1) == pytest.approx(0.0)
    assert report.trust_delta[1] is not None and report.trust_delta[1] < 0
