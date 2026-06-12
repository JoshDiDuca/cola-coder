"""MODEL-009: GRPO curriculum temperature must SCALE the base temperature, not
replace it.

The curriculum used absolute per-difficulty temps {0.7, 0.8, 0.9} that replaced
the run's temperature, so --temperature was silently ignored. Now the values are
multipliers of the base, so a user's temperature is honored — while the default
base (0.8) still reproduces the old absolute values exactly.
"""

import pytest

from cola_coder.reasoning.grpo import _CURRICULUM_TEMP_MULT, _step_temperature


class TestStepTemperature:
    def test_off_returns_base(self):
        for d in ("easy", "medium", "hard", "unknown"):
            assert _step_temperature(0.8, d, curriculum=False) == 0.8
            assert _step_temperature(1.3, d, curriculum=False) == 1.3

    def test_default_base_reproduces_legacy_absolutes(self):
        # base 0.8 → the old {easy:0.7, medium:0.8, hard:0.9} (backward compatible).
        assert _step_temperature(0.8, "easy", True) == pytest.approx(0.7)
        assert _step_temperature(0.8, "medium", True) == pytest.approx(0.8)
        assert _step_temperature(0.8, "hard", True) == pytest.approx(0.9)

    def test_base_temperature_is_honored(self):
        # The whole point: a different base shifts every tier (was ignored before).
        assert _step_temperature(1.0, "easy", True) == pytest.approx(0.875)
        assert _step_temperature(1.0, "medium", True) == pytest.approx(1.0)
        assert _step_temperature(1.0, "hard", True) == pytest.approx(1.125)

    def test_easy_tighter_than_hard(self):
        base = 0.9
        assert (_step_temperature(base, "easy", True)
                < _step_temperature(base, "medium", True)
                < _step_temperature(base, "hard", True))

    def test_unknown_difficulty_uses_base(self):
        assert _step_temperature(0.8, "legendary", True) == pytest.approx(0.8)

    def test_multipliers_are_factors_around_one(self):
        assert _CURRICULUM_TEMP_MULT["medium"] == 1.0
        assert _CURRICULUM_TEMP_MULT["easy"] < 1.0 < _CURRICULUM_TEMP_MULT["hard"]
