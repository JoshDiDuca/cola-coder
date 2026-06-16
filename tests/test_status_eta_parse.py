"""Regression guard for BUG-138: the trainer's pretty log line gained a trailing
``| ETA …`` suffix, which silently broke the dashboard status parser and the
metrics-history parser because both anchored the tok/s capture on end-of-line.

Both parsers now anchor on the literal ``tok/s``. These tests exercise the REAL
regexes and public functions (no mocking) against representative NEW (ETA-suffixed)
and LEGACY (no-ETA) lines to ensure the tok/s count is captured correctly and the
metrics chart is no longer empty for ETA-format logs.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.metrics_history import _LOG_LINE_RE as METRICS_RE
from cola_coder.ui.metrics_history import training_history
from cola_coder.ui.status import (
    _LOG_LINE_RE,
    _extract_eta,
    _parse_log,
    get_training_status,
)

# Representative trainer log lines.
NEW_LINE: str = (
    "08:38:21 step  16,200 (10.8%) loss 1.2492 ppl      3.5 lr 6.00e-04    "
    "11,738 tok/s | ETA 338h 58m 51s (11:37)"
)
LEGACY_LINE: str = (
    "03:12:20 step   2,500 ( 1.7%) loss 1.6057 ppl      5.0 lr 6.00e-04     "
    "1,813 tok/s"
)


class TestStatusLogLineRegex:
    """The status ``_LOG_LINE_RE`` must match both line formats."""

    def test_matches_new_eta_line(self) -> None:
        """NEW (ETA-suffixed) line matches with correctly captured groups."""
        m = _LOG_LINE_RE.search(NEW_LINE)
        assert m is not None
        assert m.group(1) == "16,200"
        assert m.group(2) == "10.8"
        assert m.group(3) == "1.2492"
        assert m.group(4) == "3.5"
        # The bug-prone capture: the 5th group MUST be the tok/s count, NOT the
        # "6" (or any fragment) of the lr "6.00e-04".
        assert m.group(5) == "11,738"
        assert m.group(5) != "6"

    def test_matches_legacy_line(self) -> None:
        """LEGACY (no-ETA) line still matches with the tok/s count."""
        m = _LOG_LINE_RE.search(LEGACY_LINE)
        assert m is not None
        assert m.group(1) == "2,500"
        assert m.group(5) == "1,813"


class TestMetricsLogLineRegex:
    """The metrics ``_LOG_LINE_RE`` must match both formats and capture lr + tok/s."""

    def test_matches_new_eta_line(self) -> None:
        """NEW line captures lr (group 5) and tok/s (group 6)."""
        m = METRICS_RE.search(NEW_LINE)
        assert m is not None
        assert m.group(1) == "16,200"
        assert m.group(5) == "6.00e-04"
        assert m.group(6) == "11,738"

    def test_matches_legacy_line(self) -> None:
        """LEGACY line captures lr and tok/s identically."""
        m = METRICS_RE.search(LEGACY_LINE)
        assert m is not None
        assert m.group(5) == "6.00e-04"
        assert m.group(6) == "1,813"


class TestExtractEta:
    """``_extract_eta`` returns the human-readable remaining time, or None."""

    def test_extracts_from_new_line(self) -> None:
        """The ETA string is parsed without the trailing wall-clock paren."""
        assert _extract_eta(NEW_LINE) == "338h 58m 51s"

    def test_none_for_legacy_line(self) -> None:
        """A line without an ETA suffix yields None."""
        assert _extract_eta(LEGACY_LINE) is None


class TestParseLog:
    """``_parse_log`` returns the parsed dict for the most recent step line."""

    def test_parses_new_line_fields(self) -> None:
        """All numeric fields + ETA are extracted from a NEW-format line."""
        parsed = _parse_log(NEW_LINE)
        assert parsed is not None
        assert parsed["step"] == 16200
        assert parsed["loss"] == 1.2492
        assert parsed["ppl"] == 3.5
        assert parsed["tok_per_s"] == 11738.0
        assert parsed["eta"] == "338h 58m 51s"

    def test_picks_last_matching_line(self) -> None:
        """Given multiple step lines, the LAST one is returned."""
        text = "\n".join(
            [
                "03:12:20 step   2,500 ( 1.7%) loss 1.6057 ppl 5.0 lr 6.00e-04 1,813 tok/s",
                "05:00:00 step   9,000 ( 6.0%) loss 1.4000 ppl 4.1 lr 6.00e-04 5,000 tok/s",
                NEW_LINE,
            ]
        )
        parsed = _parse_log(text)
        assert parsed is not None
        assert parsed["step"] == 16200
        assert parsed["tok_per_s"] == 11738.0


class TestGetTrainingStatusRoundTrip:
    """Round-trip the NEW line through the public ``get_training_status`` API."""

    def test_reads_new_line_from_file(self, tmp_path: Path) -> None:
        """Writing a NEW line to a .log yields the correct parsed status."""
        log_file = tmp_path / "train.log"
        log_file.write_text(NEW_LINE + "\n", encoding="utf-8")
        err_path = str(tmp_path / "none.err")  # definitely nonexistent

        status = get_training_status(log_path=str(log_file), err_path=err_path)
        assert status["step"] == 16200
        assert status["loss"] == 1.2492
        assert status["tok_per_s"] == 11738.0
        assert status["eta"] == "338h 58m 51s"


class TestTrainingHistory:
    """``training_history`` must produce a non-empty series for ETA-format logs."""

    def test_eta_format_log_yields_points(self, tmp_path: Path) -> None:
        """Several NEW ETA-suffixed lines parse into a populated points list."""
        lines = [
            "03:12:20 step   2,500 ( 1.7%) loss 1.6057 ppl 5.0 lr 6.00e-04 "
            "1,813 tok/s | ETA 400h 00m 00s (11:00)",
            "05:00:00 step   9,000 ( 6.0%) loss 1.4000 ppl 4.1 lr 6.00e-04 "
            "5,000 tok/s | ETA 360h 30m 00s (11:20)",
            NEW_LINE,
        ]
        log_file = tmp_path / "train.log"
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        history = training_history(log_path=str(log_file))
        assert "error" not in history
        assert history["count"] == 3
        points = history["points"]
        assert isinstance(points, list)
        assert len(points) > 0
        last = points[-1]
        assert last["step"] == 16200
        assert last["loss"] == 1.2492
        assert last["lr"] == 6.00e-04
        assert last["tok_s"] == 11738.0
