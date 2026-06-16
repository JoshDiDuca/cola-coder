"""ZClip adaptive gradient-norm spike mitigation (MODEL-054).

Pins the warmup behavior, the z-score spike test, stat-poisoning protection, the
config validation, and state_dict round-trip. Pure-numeric — no torch.
"""

import math

import pytest

from cola_coder.training.zclip import ZClipper, ZClipResult


class TestConfigValidation:
    def test_rejects_bad_alpha(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                ZClipper(alpha=bad)

    def test_rejects_nonpositive_z_thresh(self) -> None:
        with pytest.raises(ValueError):
            ZClipper(z_thresh=0.0)

    def test_rejects_bad_warmup(self) -> None:
        with pytest.raises(ValueError):
            ZClipper(warmup=0)


class TestWarmup:
    def test_no_clipping_during_warmup(self) -> None:
        clip = ZClipper(warmup=10)
        for _ in range(10):
            r = clip.observe(5.0)
            assert r.clipped is False
            assert r.max_norm == 5.0
            assert r.z is None  # no z-score reported pre-warmup
        assert clip.threshold is None or clip.count >= clip.warmup

    def test_threshold_available_after_warmup(self) -> None:
        clip = ZClipper(warmup=5)
        for _ in range(6):
            clip.observe(2.0)
        assert clip.threshold is not None


class TestSpikeClipping:
    def _warm(self, clip: ZClipper, value: float, n: int) -> None:
        for _ in range(n):
            clip.observe(value)

    def test_stable_norms_are_not_clipped(self) -> None:
        clip = ZClipper(alpha=0.9, z_thresh=2.5, warmup=20)
        # Mild jitter around 1.0 — none should be flagged as a spike.
        seq = [1.0, 1.05, 0.95, 1.02, 0.98] * 8
        clipped_any = False
        for v in seq:
            if clip.observe(v).clipped:
                clipped_any = True
        assert clipped_any is False

    def test_clear_spike_is_clipped(self) -> None:
        clip = ZClipper(alpha=0.9, z_thresh=2.5, warmup=20)
        # Build stable stats with small variance, then a big spike.
        for i in range(40):
            clip.observe(1.0 + 0.01 * (i % 3))
        result = clip.observe(50.0)
        assert result.clipped is True
        assert result.z is not None and result.z > 2.5
        # Clipped down to mean + z_thresh*std, far below the raw 50.0.
        assert result.max_norm < 50.0
        assert result.max_norm == pytest.approx(clip.threshold, rel=0.5)

    def test_clip_does_not_poison_statistics(self) -> None:
        clip = ZClipper(alpha=0.9, z_thresh=2.5, warmup=20)
        for _ in range(40):
            clip.observe(1.0)
        mean_before = clip.mean
        clip.observe(1000.0)  # huge spike
        # The EMA must move only modestly (it updated with the CLIPPED value, not 1000).
        assert clip.mean < mean_before + 5.0

    def test_clip_to_mean_false_uses_mean(self) -> None:
        clip = ZClipper(alpha=0.9, z_thresh=2.0, warmup=15, clip_to_mean=False)
        for _ in range(30):
            clip.observe(1.0)
        result = clip.observe(20.0)
        assert result.clipped is True
        assert result.max_norm == pytest.approx(clip.mean, rel=1e-6)


class TestRobustness:
    def test_nonfinite_norm_treated_as_zero(self) -> None:
        clip = ZClipper(warmup=3)
        r = clip.observe(float("nan"))
        assert r.max_norm == 0.0
        assert math.isfinite(r.mean)
        r2 = clip.observe(float("inf"))
        assert r2.max_norm == 0.0

    def test_negative_norm_treated_as_zero(self) -> None:
        assert ZClipper(warmup=3).observe(-7.0).max_norm == 0.0

    def test_observe_returns_result_type(self) -> None:
        assert isinstance(ZClipper().observe(1.0), ZClipResult)


class TestStateDict:
    def test_round_trip_preserves_statistics(self) -> None:
        clip = ZClipper(alpha=0.95, z_thresh=3.0, warmup=10)
        for i in range(25):
            clip.observe(1.0 + 0.1 * (i % 5))
        state = clip.state_dict()

        restored = ZClipper()
        restored.load_state_dict(state)
        assert restored.mean == pytest.approx(clip.mean)
        assert restored.std == pytest.approx(clip.std)
        assert restored.count == clip.count
        assert restored.z_thresh == clip.z_thresh
        # A subsequent identical observation behaves the same on both.
        assert restored.observe(2.0).clipped == clip.observe(2.0).clipped
