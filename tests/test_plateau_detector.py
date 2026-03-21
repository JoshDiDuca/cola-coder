"""Tests for PlateauDetector (features/plateau_detector.py)."""

from __future__ import annotations



from cola_coder.features.plateau_detector import (
    FEATURE_ENABLED,
    PlateauAction,
    PlateauDetector,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def steadily_improving(start: float = 5.0, steps: int = 100, rate: float = 0.03) -> list[float]:
    return [max(0.1, start - i * rate) for i in range(steps)]


def flat_loss(value: float = 2.5, steps: int = 100) -> list[float]:
    return [value] * steps


def oscillating_loss(base: float = 2.5, amplitude: float = 0.1, steps: int = 60) -> list[float]:
    return [base + amplitude * ((-1) ** i) for i in range(steps)]


def improving_then_flat(steps: int = 100) -> list[float]:
    improving = [5.0 - i * 0.04 for i in range(50)]
    flat = [improving[-1]] * 50
    return improving + flat


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestImprovingLoss:
    def test_steadily_improving_not_plateau(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(steadily_improving(steps=80))
        assert report.is_plateau is False

    def test_improving_suggests_none_or_wait(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(steadily_improving(steps=80))
        # Should not suggest aggressive interventions
        assert PlateauAction.NONE in report.suggested_actions or PlateauAction.WAIT in report.suggested_actions

    def test_negative_trend_for_improving(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(steadily_improving(steps=80))
        assert report.current_trend < 0  # negative slope = improving


class TestFlatLoss:
    def test_flat_loss_is_plateau(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(flat_loss(steps=80))
        assert report.is_plateau is True

    def test_flat_loss_has_plateau_regions(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(flat_loss(steps=80))
        assert len(report.plateaus) > 0

    def test_flat_loss_suggests_reduce_lr(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(flat_loss(steps=80))
        assert PlateauAction.REDUCE_LR in report.suggested_actions

    def test_plateau_region_properties(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(flat_loss(2.5, steps=80))
        assert report.longest_plateau is not None
        region = report.longest_plateau
        assert region.length > 0
        assert abs(region.mean_loss - 2.5) < 0.1


class TestOscillation:
    def test_oscillating_detected(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(oscillating_loss(steps=80))
        assert report.is_oscillating is True

    def test_oscillating_suggests_reduce_lr(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(oscillating_loss(steps=80))
        assert PlateauAction.REDUCE_LR in report.suggested_actions


class TestImprovingThenFlat:
    def test_detects_plateau_after_improvement(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(improving_then_flat())
        assert report.is_plateau is True

    def test_recent_trend_near_zero_in_flat_region(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(improving_then_flat())
        # Trend in flat region should be close to 0
        assert abs(report.current_trend) < 0.01


class TestEdgeCases:
    def test_too_short_history_returns_wait(self):
        detector = PlateauDetector(window=20, min_plateau_len=5)
        report = detector.detect([2.5, 2.4, 2.3])
        assert report.is_plateau is False
        assert PlateauAction.WAIT in report.suggested_actions

    def test_empty_history_returns_wait(self):
        detector = PlateauDetector()
        report = detector.detect([])
        assert PlateauAction.WAIT in report.suggested_actions

    def test_summary_is_string(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(flat_loss(steps=80))
        assert isinstance(report.summary, str)
        assert "PLATEAU" in report.summary or "improving" in report.summary

    def test_longest_plateau_none_when_no_plateaus(self):
        detector = PlateauDetector(window=10, min_plateau_len=5)
        report = detector.detect(steadily_improving(steps=80))
        assert report.longest_plateau is None
