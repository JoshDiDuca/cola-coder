"""IDEA-013: entropy-gated clip controller.

Raises the DAPO clip-higher bound when policy entropy collapses below a floor, but
only when the verifier (pass-rate) is unsatisfied; relaxes to base otherwise.
"""

import pytest

from cola_coder.reasoning.entropy_controller import EntropyClipController


class TestEntropyClipController:
    def test_healthy_entropy_stays_at_base(self):
        c = EntropyClipController(target_entropy=1.0, clip_high=0.28)
        low, high = c.update(measured_entropy=1.5)  # above target -> no deficit
        assert low == 0.2
        assert high == 0.28

    def test_collapsing_entropy_raises_clip_high(self):
        # base 0.28 + gain*deficit (0.5*0.4=0.2) = 0.48, but default max caps at 0.40.
        c = EntropyClipController(target_entropy=1.0, clip_high=0.28, gain=0.5)
        _, high = c.update(measured_entropy=0.6)      # deficit 0.4
        assert high > 0.28                             # exploration injected
        assert high == pytest.approx(0.40)             # capped at max_clip_high

    def test_uncapped_raise_is_exactly_proportional(self):
        # With headroom, the raise equals base + gain*deficit exactly.
        c = EntropyClipController(target_entropy=1.0, clip_high=0.28,
                                  max_clip_high=1.0, gain=0.5)
        _, high = c.update(measured_entropy=0.6)      # deficit 0.4
        assert high == pytest.approx(0.28 + 0.5 * 0.4)  # 0.48, under the 1.0 cap

    def test_raise_is_proportional_to_deficit(self):
        c = EntropyClipController(target_entropy=2.0, clip_high=0.2,
                                  max_clip_high=1.0, gain=0.5)
        _, small = c.update(measured_entropy=1.8)     # deficit 0.2 -> +0.1
        _, large = c.update(measured_entropy=1.0)     # deficit 1.0 -> +0.5
        assert small == pytest.approx(0.3)
        assert large == pytest.approx(0.7)
        assert large > small

    def test_clip_high_is_capped(self):
        c = EntropyClipController(target_entropy=5.0, clip_high=0.2,
                                  max_clip_high=0.35, gain=1.0)
        _, high = c.update(measured_entropy=0.0)      # huge deficit
        assert high == 0.35                            # capped

    def test_verifier_satisfied_suppresses_exploration(self):
        # Even with collapsed entropy, a high pass-rate must NOT inject exploration.
        c = EntropyClipController(target_entropy=2.0, clip_high=0.28,
                                  pass_rate_ceiling=0.9)
        _, high = c.update(measured_entropy=0.1, pass_rate=0.95)
        assert high == 0.28                            # relaxed to base

    def test_verifier_unsatisfied_allows_exploration(self):
        c = EntropyClipController(target_entropy=2.0, clip_high=0.28,
                                  max_clip_high=2.0, gain=0.5, pass_rate_ceiling=0.9)
        _, high = c.update(measured_entropy=0.0, pass_rate=0.3)  # deficit 2.0
        assert high == pytest.approx(0.28 + 0.5 * 2.0)           # 1.28, under cap

    def test_relaxes_back_after_recovery(self):
        c = EntropyClipController(target_entropy=1.0, clip_high=0.28,
                                  max_clip_high=1.0, gain=0.5)
        _, raised = c.update(measured_entropy=0.0, pass_rate=0.0)   # raised
        assert raised > 0.28
        _, relaxed = c.update(measured_entropy=1.2)                # recovered
        assert relaxed == 0.28
        assert c.current_clip_high == 0.28

    def test_clip_low_is_never_modulated(self):
        c = EntropyClipController(target_entropy=2.0, clip_low=0.2, clip_high=0.28)
        for e in (0.0, 0.5, 1.0, 3.0):
            low, _ = c.update(measured_entropy=e)
            assert low == 0.2

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError, match="target_entropy"):
            EntropyClipController(target_entropy=-1.0)
        with pytest.raises(ValueError, match="max_clip_high"):
            EntropyClipController(target_entropy=1.0, clip_high=0.5, max_clip_high=0.3)
        with pytest.raises(ValueError, match="gain"):
            EntropyClipController(target_entropy=1.0, gain=-0.1)
