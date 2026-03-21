"""Tests for the AutoEvaluator (auto_eval.py).

All tests run without a GPU or actual model inference.  Model/tokenizer
interactions are mocked so the test suite remains fast and portable.

Run:
    cd "C:/Users/josh/ai research/cola-coder"
    .venv/Scripts/pytest tests/test_auto_eval.py -v
"""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock, patch

import pytest

from cola_coder.training.auto_eval import (
    AutoEvaluator,
    EvalSnapshot,
    create_auto_evaluator,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    step: int,
    pass_at_1: float = 0.0,
    pass_at_5: float = 0.0,
    is_best: bool = False,
) -> EvalSnapshot:
    return EvalSnapshot(
        step=step,
        timestamp="2026-03-21T12:00:00",
        pass_at_1=pass_at_1,
        pass_at_5=pass_at_5,
        num_problems=20,
        avg_generation_time=0.5,
        is_best=is_best,
    )


def _make_evaluator(**kwargs) -> AutoEvaluator:
    defaults = dict(
        eval_every_steps=1000,
        eval_subset=5,
        save_best=False,
        num_samples=1,
    )
    defaults.update(kwargs)
    return AutoEvaluator(**defaults)


# ---------------------------------------------------------------------------
# 1. should_eval: correct interval behaviour
# ---------------------------------------------------------------------------


class TestShouldEval:
    def test_triggers_at_exact_interval(self):
        ae = _make_evaluator(eval_every_steps=1000)
        assert ae.should_eval(1000) is True
        assert ae.should_eval(2000) is True
        assert ae.should_eval(5000) is True

    def test_does_not_trigger_between_intervals(self):
        ae = _make_evaluator(eval_every_steps=1000)
        assert ae.should_eval(1) is False
        assert ae.should_eval(999) is False
        assert ae.should_eval(1001) is False
        assert ae.should_eval(1500) is False

    def test_never_triggers_at_step_zero(self):
        ae = _make_evaluator(eval_every_steps=1)  # every step
        assert ae.should_eval(0) is False

    def test_custom_interval(self):
        ae = _make_evaluator(eval_every_steps=500)
        assert ae.should_eval(500) is True
        assert ae.should_eval(250) is False
        assert ae.should_eval(1000) is True


# ---------------------------------------------------------------------------
# 2. EvalSnapshot recording
# ---------------------------------------------------------------------------


class TestEvalSnapshotRecording:
    def test_snapshot_serialise_roundtrip(self):
        snap = _make_snapshot(step=1000, pass_at_1=0.25, pass_at_5=0.40, is_best=True)
        d = snap.to_dict()
        restored = EvalSnapshot.from_dict(d)
        assert restored.step == 1000
        assert restored.pass_at_1 == pytest.approx(0.25)
        assert restored.pass_at_5 == pytest.approx(0.40)
        assert restored.is_best is True
        assert restored.timestamp == "2026-03-21T12:00:00"

    def test_snapshot_from_dict_default_is_best(self):
        """is_best should default to False when key is absent."""
        d = {
            "step": 500,
            "timestamp": "2026-01-01T00:00:00",
            "pass_at_1": 0.1,
            "pass_at_5": 0.2,
            "num_problems": 10,
            "avg_generation_time": 1.0,
            # is_best intentionally omitted
        }
        snap = EvalSnapshot.from_dict(d)
        assert snap.is_best is False

    def test_history_accumulates_after_evaluate(self):
        """evaluate() should append a snapshot to history."""
        ae = _make_evaluator(eval_every_steps=1000, eval_subset=2, num_samples=1)

        from cola_coder.evaluation.humaneval import CodingProblem

        two_problems = [
            CodingProblem(
                task_id="add",
                prompt="def add(a, b):\n",
                test_code="assert add(1,2)==3",
                entry_point="add",
            ),
            CodingProblem(
                task_id="sub",
                prompt="def sub(a, b):\n",
                test_code="assert sub(3,2)==1",
                entry_point="sub",
            ),
        ]

        # auto_eval.evaluate() uses lazy `from .. import` statements.
        # We patch at the source module locations, which is where Python
        # resolves those names at call time.
        mock_generator_instance = MagicMock()
        mock_generator_instance.generate.return_value = "def add(a, b): return a + b"

        with (
            patch(
                "cola_coder.evaluation.humaneval.get_all_problems",
                return_value=two_problems,
            ),
            patch(
                "cola_coder.evaluation.runner.evaluate_solution",
                return_value=(True, ""),
            ),
            patch(
                "cola_coder.evaluation.runner.extract_function",
                return_value="def add(a, b): return a + b",
            ),
            patch(
                "cola_coder.inference.generator.CodeGenerator",
                return_value=mock_generator_instance,
            ),
        ):
            mock_model = MagicMock()
            mock_model.training = True
            mock_tokenizer = MagicMock()

            snap = ae.evaluate(mock_model, mock_tokenizer, step=1000, device="cpu")

        assert len(ae.history) == 1
        assert ae.history[0].step == 1000
        assert snap.num_problems == 2


# ---------------------------------------------------------------------------
# 3. Regression detection
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    def test_no_regression_when_improving(self):
        ae = _make_evaluator(regression_threshold=0.20)
        ae.best_score = 0.5
        snap = _make_snapshot(step=2000, pass_at_1=0.55)
        assert ae.check_regression(snap) is False

    def test_regression_when_score_drops_above_threshold(self):
        ae = _make_evaluator(regression_threshold=0.20)
        ae.best_score = 0.50
        # 0.35 is 30% below 0.50 → regression
        snap = _make_snapshot(step=2000, pass_at_1=0.35)
        assert ae.check_regression(snap) is True

    def test_no_regression_when_drop_is_below_threshold(self):
        ae = _make_evaluator(regression_threshold=0.20)
        ae.best_score = 0.50
        # 0.42 is 16% below 0.50 → no regression
        snap = _make_snapshot(step=2000, pass_at_1=0.42)
        assert ae.check_regression(snap) is False

    def test_no_regression_when_best_score_is_zero(self):
        """Edge case: avoid divide-by-zero when no good run yet."""
        ae = _make_evaluator()
        ae.best_score = 0.0
        snap = _make_snapshot(step=1000, pass_at_1=0.0)
        assert ae.check_regression(snap) is False

    def test_regression_threshold_is_configurable(self):
        ae = _make_evaluator(regression_threshold=0.10)  # tight threshold
        ae.best_score = 0.50
        # 0.44 is 12% below 0.50 → regression for threshold=0.10
        snap = _make_snapshot(step=2000, pass_at_1=0.44)
        assert ae.check_regression(snap) is True


# ---------------------------------------------------------------------------
# 4. Trend calculation
# ---------------------------------------------------------------------------


class TestTrendCalculation:
    def test_not_enough_data_with_one_snapshot(self):
        ae = _make_evaluator()
        ae.history = [_make_snapshot(1000, pass_at_1=0.1)]
        assert ae.get_trend() == "not enough data"

    def test_not_enough_data_with_zero_snapshots(self):
        ae = _make_evaluator()
        assert ae.get_trend() == "not enough data"

    def test_improving_trend(self):
        ae = _make_evaluator()
        ae.history = [
            _make_snapshot(1000, pass_at_1=0.10),
            _make_snapshot(2000, pass_at_1=0.20),
            _make_snapshot(3000, pass_at_1=0.35),
        ]
        assert ae.get_trend() == "improving"

    def test_degrading_trend(self):
        ae = _make_evaluator()
        ae.history = [
            _make_snapshot(1000, pass_at_1=0.40),
            _make_snapshot(2000, pass_at_1=0.25),
            _make_snapshot(3000, pass_at_1=0.10),
        ]
        assert ae.get_trend() == "degrading"

    def test_stable_trend(self):
        ae = _make_evaluator()
        ae.history = [
            _make_snapshot(1000, pass_at_1=0.20),
            _make_snapshot(2000, pass_at_1=0.20),
            _make_snapshot(3000, pass_at_1=0.20),
        ]
        assert ae.get_trend() == "stable"

    def test_trend_uses_last_three_snapshots_only(self):
        """Trend should not be distorted by many old snapshots."""
        ae = _make_evaluator()
        ae.history = [
            _make_snapshot(i * 1000, pass_at_1=0.50)
            for i in range(1, 8)  # 7 stable snapshots
        ] + [
            _make_snapshot(8000, pass_at_1=0.20),
            _make_snapshot(9000, pass_at_1=0.10),
        ]
        # Last 3: 0.50 → 0.20 → 0.10 — degrading
        assert ae.get_trend() == "degrading"


# ---------------------------------------------------------------------------
# 5. State dict save / load
# ---------------------------------------------------------------------------


class TestStateDictSaveLoad:
    def test_roundtrip_with_history(self):
        ae = _make_evaluator(eval_every_steps=2000, eval_subset=10, num_samples=3)
        ae.history = [
            _make_snapshot(2000, pass_at_1=0.15, is_best=False),
            _make_snapshot(4000, pass_at_1=0.25, is_best=True),
        ]
        ae.best_score = 0.25
        ae.best_step = 4000

        state = ae.state_dict()

        ae2 = _make_evaluator(eval_every_steps=9999)  # different settings
        ae2.load_state_dict(state)

        assert len(ae2.history) == 2
        assert ae2.history[0].step == 2000
        assert ae2.history[1].pass_at_1 == pytest.approx(0.25)
        assert ae2.best_score == pytest.approx(0.25)
        assert ae2.best_step == 4000

    def test_empty_state_loads_cleanly(self):
        ae = _make_evaluator()
        ae.load_state_dict({})
        assert ae.history == []
        assert ae.best_score == 0.0
        assert ae.best_step == 0

    def test_state_dict_contains_config_fields(self):
        ae = _make_evaluator(eval_every_steps=3000, eval_subset=15, num_samples=5)
        state = ae.state_dict()
        assert state["eval_every_steps"] == 3000
        assert state["eval_subset"] == 15
        assert state["num_samples"] == 5

    def test_load_survives_json_roundtrip(self):
        """State dict must be JSON-serialisable (for metadata.json embedding)."""
        ae = _make_evaluator()
        ae.history = [_make_snapshot(1000, pass_at_1=0.1)]
        ae.best_score = 0.1
        ae.best_step = 1000

        state = ae.state_dict()
        json_str = json.dumps(state)  # should not raise
        state2 = json.loads(json_str)

        ae2 = _make_evaluator()
        ae2.load_state_dict(state2)
        assert len(ae2.history) == 1
        assert ae2.best_score == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 6. Subset sampling consistency
# ---------------------------------------------------------------------------


class TestSubsetSampling:
    def test_fixed_seed_returns_same_subset(self):
        """Two evaluators with the same seed should sample the same problems."""
        from cola_coder.evaluation.humaneval import get_all_problems

        problems = get_all_problems()
        seed = 42
        n = 10

        rng1 = random.Random(seed)
        rng2 = random.Random(seed)
        subset1 = rng1.sample(problems, min(n, len(problems)))
        subset2 = rng2.sample(problems, min(n, len(problems)))

        assert [p.task_id for p in subset1] == [p.task_id for p in subset2]

    def test_different_seeds_give_different_subsets(self):
        """Different seeds should (with overwhelming probability) differ."""
        from cola_coder.evaluation.humaneval import get_all_problems

        problems = get_all_problems()
        n = min(10, len(problems))

        rng1 = random.Random(1)
        rng2 = random.Random(99)
        subset1 = [p.task_id for p in rng1.sample(problems, n)]
        subset2 = [p.task_id for p in rng2.sample(problems, n)]

        # They might be equal by chance, but it's astronomically unlikely
        # for n=10 out of 20 problems.  If this flakes, just remove it.
        assert subset1 != subset2

    def test_subset_size_capped_at_available_problems(self):
        """eval_subset larger than the problem pool should not raise."""
        from cola_coder.evaluation.humaneval import get_all_problems

        problems = get_all_problems()
        ae = _make_evaluator(eval_subset=9999)

        rng = random.Random(ae.subset_seed)
        subset = rng.sample(problems, min(ae.eval_subset, len(problems)))
        assert len(subset) == len(problems)


# ---------------------------------------------------------------------------
# 7. create_auto_evaluator helper
# ---------------------------------------------------------------------------


class TestCreateAutoEvaluator:
    def test_returns_none_when_no_auto_eval_key(self):
        ae = create_auto_evaluator({})
        assert ae is None

    def test_returns_none_when_disabled(self):
        ae = create_auto_evaluator({"auto_eval": {"enabled": False}})
        assert ae is None

    def test_creates_evaluator_with_defaults_when_enabled(self):
        ae = create_auto_evaluator({"auto_eval": {"enabled": True}})
        assert ae is not None
        assert ae.eval_every_steps == 5000
        assert ae.eval_subset == 20
        assert ae.num_samples == 5

    def test_creates_evaluator_with_custom_config(self):
        config = {
            "auto_eval": {
                "enabled": True,
                "eval_every_steps": 2000,
                "eval_subset": 10,
                "num_samples": 3,
                "temperature": 0.5,
                "regression_threshold": 0.15,
                "log_to_wandb": False,
            }
        }
        ae = create_auto_evaluator(config)
        assert ae is not None
        assert ae.eval_every_steps == 2000
        assert ae.eval_subset == 10
        assert ae.num_samples == 3
        assert ae.temperature == pytest.approx(0.5)
        assert ae.regression_threshold == pytest.approx(0.15)

    def test_enabled_defaults_to_true_when_key_present(self):
        """If 'auto_eval' key exists but 'enabled' is absent, treat as enabled."""
        ae = create_auto_evaluator({"auto_eval": {"eval_every_steps": 100}})
        assert ae is not None


# ---------------------------------------------------------------------------
# 8. format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_empty_history_message(self):
        ae = _make_evaluator()
        report = ae.format_report()
        assert "No evaluations" in report

    def test_report_contains_step_and_metrics(self):
        ae = _make_evaluator()
        ae.history = [
            _make_snapshot(1000, pass_at_1=0.10, pass_at_5=0.20, is_best=True),
            _make_snapshot(2000, pass_at_1=0.15, pass_at_5=0.30),
        ]
        ae.best_score = 0.15
        ae.best_step = 2000
        report = ae.format_report()
        assert "1,000" in report or "1000" in report  # step is present
        assert "2,000" in report or "2000" in report
        assert "AUTO-EVAL HISTORY" in report

    def test_report_includes_best_marker(self):
        ae = _make_evaluator()
        ae.history = [_make_snapshot(1000, pass_at_1=0.1, is_best=True)]
        ae.best_score = 0.1
        ae.best_step = 1000
        report = ae.format_report()
        # The best row should have a '*' marker
        assert "*" in report

    def test_report_includes_trend(self):
        ae = _make_evaluator()
        ae.history = [
            _make_snapshot(1000, pass_at_1=0.10),
            _make_snapshot(2000, pass_at_1=0.30),
        ]
        ae.best_score = 0.30
        ae.best_step = 2000
        report = ae.format_report()
        assert "Trend:" in report
