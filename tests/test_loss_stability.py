"""Tests for the training loss-stability meter (``ui/loss_stability_view.py``).

Covers the pure ``compute_loss_stability`` classifier (trend, EMA, spike
detection, verdict) and the ``loss_stability`` log-reading wrapper. No network,
no GPU, no model — every test is a pure computation or a ``tmp_path`` log file.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.loss_stability_view import compute_loss_stability, loss_stability

_VALID_TRENDS = {"improving", "flat", "worsening", "unknown"}
_VALID_VERDICTS = {"stable", "watch", "spiking", "insufficient_data"}


def _make_log(tmp_path: Path, losses: list[float]) -> str:
    """Write a real-format training log with one step line per loss; return path."""
    lines: list[str] = []
    for i, loss in enumerate(losses):
        step = (i + 1) * 1000
        lines.append(
            f"01:00:{i:02d} step {step:,} ( 0.7%) loss {loss:.4f} ppl 7.0 lr 6.00e-04   100 "
        )
    log = tmp_path / "train.log"
    log.write_text("\n".join(lines), encoding="utf-8")
    return str(log)


class TestComputeLossStability:
    """Unit tests for the pure ``compute_loss_stability`` classifier."""

    def test_insufficient_data_fewer_than_six(self) -> None:
        """Fewer than 6 non-None losses -> insufficient_data / unknown trend."""
        losses = [3.0, 2.8, 2.6, 2.4, 2.2]
        result = compute_loss_stability(losses)

        assert result["verdict"] == "insufficient_data"
        assert result["trend"] == "unknown"
        assert result["points_used"] == len(losses)
        assert result["current_loss"] is None
        assert result["ema_loss"] is None
        assert result["recent_max_z"] is None
        assert result["spike_count"] == 0

    def test_steadily_decreasing_is_improving_stable(self) -> None:
        """A monotonically decreasing series improves, is stable, no spikes."""
        losses = [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
        result = compute_loss_stability(losses)

        assert result["trend"] == "improving"
        assert result["verdict"] == "stable"
        assert result["spike_count"] == 0
        assert result["current_loss"] == 1.0
        assert result["points_used"] == len(losses)

    def test_steadily_increasing_is_worsening_watch(self) -> None:
        """A monotonically increasing series worsens -> watch (no spikes)."""
        losses = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        result = compute_loss_stability(losses)

        assert result["trend"] == "worsening"
        assert result["verdict"] == "watch"
        assert result["spike_count"] == 0
        assert result["current_loss"] == 3.0

    def test_flat_jitter_is_flat_stable(self) -> None:
        """Small jitter around a constant level reads as flat / stable."""
        losses = [1.5, 1.51, 1.49, 1.5, 1.52, 1.48, 1.5, 1.51, 1.49, 1.5]
        result = compute_loss_stability(losses)

        assert result["trend"] == "flat"
        assert result["verdict"] == "stable"
        assert result["spike_count"] == 0

    def test_recent_spike_dominates_verdict(self) -> None:
        """A stable run with a big recent upward jump -> spiking, spike_count >= 1."""
        losses = [1.5] * 30 + [1.51, 1.49, 1.50, 9.9]
        result = compute_loss_stability(losses)

        assert result["spike_count"] >= 1
        assert result["verdict"] == "spiking"
        # The spike is the last delta, so the recent z-score is large.
        assert result["recent_max_z"] is not None
        assert result["recent_max_z"] > 3.0

    def test_none_values_are_skipped(self) -> None:
        """None entries are skipped, don't crash, and don't count toward points_used."""
        losses = [1.5, None, 1.4, None, 1.3, None, 1.2, 1.1, 1.0]
        non_none = [x for x in losses if x is not None]
        result = compute_loss_stability(losses)  # type: ignore[arg-type]

        assert result["points_used"] == len(non_none)
        assert result["verdict"] in _VALID_VERDICTS
        assert result["trend"] in _VALID_TRENDS
        assert result["current_loss"] == 1.0

    def test_rounded_floats_and_recent_max_z_type(self) -> None:
        """current/ema losses are rounded floats; recent_max_z is a float when used."""
        losses = [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6]
        result = compute_loss_stability(losses)

        assert isinstance(result["current_loss"], float)
        assert isinstance(result["ema_loss"], float)
        assert isinstance(result["recent_max_z"], float)
        # Rounding: 4 decimal places for losses, 3 for z-score.
        assert result["current_loss"] == round(result["current_loss"], 4)
        assert result["ema_loss"] == round(result["ema_loss"], 4)
        assert result["recent_max_z"] == round(result["recent_max_z"], 3)


class TestLossStability:
    """Tests for the ``loss_stability`` log-reading wrapper."""

    def test_reads_real_format_log(self, tmp_path: Path) -> None:
        """A log with >= 6 gently-decreasing step lines yields a valid, non-empty verdict."""
        losses = [2.0, 1.95, 1.9, 1.85, 1.8, 1.78, 1.75]
        log_path = _make_log(tmp_path, losses)

        result = loss_stability(log_path)

        assert "error" not in result
        assert result["points_used"] > 0
        assert result["verdict"] != "insufficient_data"
        assert result["verdict"] in _VALID_VERDICTS
        assert result["trend"] in {"improving", "flat"}

    def test_nonexistent_log_returns_error(self, tmp_path: Path) -> None:
        """A missing log path returns an {'error': ...} dict, never raises."""
        missing = str(tmp_path / "does_not_exist.log")
        result = loss_stability(missing)

        assert "error" in result
        assert isinstance(result["error"], str)
        assert result["error"]
