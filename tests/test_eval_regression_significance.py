"""Tests for statistically-honest pass@k comparison/regression verdicts (EVAL-029).

A checkpoint is only declared an IMPROVEMENT or a REGRESSION when the
paired-bootstrap CI of the pass@k delta EXCLUDES 0; a raw point-delta inside the
CI is "within noise". These tests are hermetic — they construct ``ProblemResult``
lists directly and never load a model or touch a GPU.
"""

from __future__ import annotations

from cola_coder.evaluation.metrics import ProblemResult
from cola_coder.evaluation.model_comparison import ComparisonResult
from cola_coder.evaluation.regression import (
    RegressionSuite,
    SignificanceVerdict,
    assess_aggregate_delta,
    assess_pass_at_k_significance,
)


def _uniform(prefix: str, n: int, correct: int, samples: int = 10):
    """n problems, each with the same (correct, samples) counts."""
    return [
        ProblemResult(task_id=f"{prefix}{i}", num_samples=samples, num_correct=correct)
        for i in range(n)
    ]


class TestSignificanceVerdict:
    def test_identical_models_within_noise(self):
        # Same per-problem results -> delta exactly 0, CI spans 0.
        a = [
            ProblemResult(task_id=f"p{i}", num_samples=10, num_correct=c)
            for i, c in enumerate([3, 6, 9, 1, 7, 4])
        ]
        b = [
            ProblemResult(task_id=f"p{i}", num_samples=10, num_correct=c)
            for i, c in enumerate([3, 6, 9, 1, 7, 4])
        ]
        v = assess_pass_at_k_significance(a, b, k=1, n_boot=1500, seed=0)
        assert v.assessed is True
        assert v.significant is False
        assert v.direction == "within noise"
        assert v.delta == 0.0
        assert v.ci_lo <= 0.0 <= v.ci_hi

    def test_b_dominates_significant_improvement(self):
        # B strictly better on every problem -> CI excludes 0, positive.
        a = _uniform("p", 8, correct=2)
        b = _uniform("p", 8, correct=8)
        v = assess_pass_at_k_significance(a, b, k=1, n_boot=1500, seed=0)
        assert v.significant is True
        assert v.direction == "improvement"
        assert v.delta > 0.0
        assert v.ci_lo > 0.0

    def test_a_dominates_significant_regression(self):
        # A better than B everywhere -> negative delta, CI excludes 0 below.
        a = _uniform("p", 8, correct=9)
        b = _uniform("p", 8, correct=1)
        v = assess_pass_at_k_significance(a, b, k=1, n_boot=1500, seed=0)
        assert v.significant is True
        assert v.direction == "regression"
        assert v.delta < 0.0
        assert v.ci_hi < 0.0

    def test_small_mixed_delta_within_noise(self):
        # Tiny, inconsistent per-problem differences -> CI spans 0.
        a = [
            ProblemResult(task_id=f"p{i}", num_samples=10, num_correct=c)
            for i, c in enumerate([5, 5, 5, 5, 5, 5])
        ]
        b = [
            ProblemResult(task_id=f"p{i}", num_samples=10, num_correct=c)
            for i, c in enumerate([6, 4, 6, 4, 6, 4])
        ]
        v = assess_pass_at_k_significance(a, b, k=1, n_boot=2000, seed=0)
        assert v.assessed is True
        assert v.direction == "within noise"
        assert v.significant is False
        assert v.ci_lo <= 0.0 <= v.ci_hi

    def test_none_when_problem_sets_disjoint(self):
        a = [ProblemResult(task_id="a", num_samples=10, num_correct=5)]
        b = [ProblemResult(task_id="z", num_samples=10, num_correct=5)]
        v = assess_pass_at_k_significance(a, b, k=1)
        assert v.assessed is False
        assert v.delta is None
        assert "n/a" in v.render()

    def test_deterministic_for_fixed_seed(self):
        a = _uniform("p", 8, correct=3)
        b = _uniform("p", 8, correct=6)
        v1 = assess_pass_at_k_significance(a, b, k=1, n_boot=1200, seed=7)
        v2 = assess_pass_at_k_significance(a, b, k=1, n_boot=1200, seed=7)
        assert (v1.delta, v1.ci_lo, v1.ci_hi) == (v2.delta, v2.ci_lo, v2.ci_hi)


class TestRendering:
    def test_render_significant_improvement_format(self):
        v = SignificanceVerdict(
            k=1, delta=0.061, ci_lo=0.012, ci_hi=0.110, ci=0.95,
            significant=True, direction="improvement", assessed=True,
        )
        s = v.render()
        assert s == "pass@1: +6.1pp [95% CI +1.2 to +11.0] — significant improvement"

    def test_render_within_noise_format(self):
        v = SignificanceVerdict(
            k=1, delta=0.013, ci_lo=-0.021, ci_hi=0.048, ci=0.95,
            significant=False, direction="within noise", assessed=True,
        )
        s = v.render()
        assert s == "pass@1: +1.3pp [95% CI -2.1 to +4.8] — within noise"

    def test_render_regression_includes_word(self):
        v = SignificanceVerdict(
            k=1, delta=-0.08, ci_lo=-0.13, ci_hi=-0.02, ci=0.95,
            significant=True, direction="regression", assessed=True,
        )
        assert "significant regression" in v.render()


class TestAggregateFallback:
    def test_aggregate_delta_reports_not_assessed(self):
        # Only aggregate scores -> raw delta, flagged as not significance-assessed.
        v = assess_aggregate_delta(0.40, 0.46, k=1)
        assert v.assessed is False
        assert abs(v.delta - 0.06) < 1e-9
        rendered = v.render()
        assert "+6.0pp" in rendered
        assert "significance not assessed" in rendered

    def test_aggregate_none_when_unscorable(self):
        v = assess_aggregate_delta(None, 0.5)
        assert v.delta is None
        assert v.assessed is False


class TestSuiteComparePassAtK:
    def test_compare_pass_at_k_significant(self):
        a = _uniform("p", 8, correct=2)
        b = _uniform("p", 8, correct=8)
        out = RegressionSuite.compare_pass_at_k(
            a, b, k=1, label_a="step_1000", label_b="step_2000", n_boot=1500
        )
        assert "PASS@1 COMPARISON" in out
        assert "significant improvement" in out
        assert "step_1000" in out and "step_2000" in out

    def test_compare_pass_at_k_within_noise(self):
        a = _uniform("p", 6, correct=5)
        b = _uniform("p", 6, correct=5)
        out = RegressionSuite.compare_pass_at_k(a, b, k=1, n_boot=1500)
        assert "within noise" in out


class TestComparisonResultSignificance:
    def _model(self, name):
        return {"name": name, "checkpoint": name, "params": 0, "step": 0}

    def test_significance_report_with_problem_data(self):
        a = _uniform("p", 8, correct=2)
        b = _uniform("p", 8, correct=8)
        result = ComparisonResult(
            models=[self._model("A"), self._model("B")],
            prompts=[],
            outputs=[[], []],
            metrics=[{}, {}],
            problem_results=[a, b],
        )
        report = result.significance_report(k=1, n_boot=1500)
        assert report is not None
        assert "A -> B" in report
        assert "significant improvement" in report
        # And it flows into the markdown.
        md = result.to_markdown()
        assert "Statistical Significance" in md

    def test_significance_report_none_without_problem_data(self):
        # Back-compat: prompt-only comparison carries no problem_results.
        result = ComparisonResult(
            models=[self._model("A"), self._model("B")],
            prompts=["x"],
            outputs=[["o"], ["o"]],
            metrics=[{}, {}],
        )
        assert result.significance_report() is None
        # Markdown omits the significance section.
        assert "Statistical Significance" not in result.to_markdown()
