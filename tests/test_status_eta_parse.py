"""Log-parse format regression guards (BUG-138 + its correction).

BUG-138 history (kept as a cautionary tale): the trainer's pretty step line was
believed to end with "… tok/s | ETA …" on ONE line, and the parsers were
"fixed" to anchor on a literal "tok/s". That was WRONG — the real trainer writes
the step line ending at the token COUNT (no inline "tok/s"), and emits
"tok/s | ETA …" on a SEPARATE line. Requiring "tok/s" froze the dashboard on the
real format. The parsers were reverted to an end-of-line anchor with an OPTIONAL
"tok/s", which matches BOTH the old inline-"tok/s" format and the current split
format; the ETA is scanned from the FULL log text, not the step line.

These tests pin the REAL formats so the mistake can't recur in either direction.
"""

from cola_coder.ui.status import _LOG_LINE_RE, _extract_eta, _parse_log, get_training_status
from cola_coder.ui.metrics_history import _LOG_LINE_RE as METRICS_RE, training_history

# The CURRENT real step line: ends at the token count, NO inline "tok/s".
REAL_STEP = "08:53:16 step  16,300 (10.9%) loss 1.2487 ppl      3.5 lr 6.00e-04    10,985 "
# The ETA lives on its OWN line (a continuation), separate from the step line.
REAL_ETA_LINE = "tok/s | ETA 337h 31m 47s (11:37)"
# The OLDER format: step line ends inline with "… tok/s".
LEGACY_STEP = "03:12:20 step   2,500 ( 1.7%) loss 1.6057 ppl      5.0 lr 6.00e-04     1,813 tok/s"


class TestStatusLogLineRegex:
    def test_matches_real_split_line(self) -> None:
        m = _LOG_LINE_RE.search(REAL_STEP)
        assert m is not None
        # group 5 is the token count — must be "10,985", NOT the lr digits.
        assert m.group(5) == "10,985"
        assert m.group(1) == "16,300"

    def test_matches_legacy_inline_tok_s_line(self) -> None:
        m = _LOG_LINE_RE.search(LEGACY_STEP)
        assert m is not None
        assert m.group(5) == "1,813"


class TestMetricsLogLineRegex:
    def test_matches_real_split_line(self) -> None:
        m = METRICS_RE.search(REAL_STEP)
        assert m is not None
        assert m.group(5) == "6.00e-04"   # lr
        assert m.group(6) == "10,985"     # tok count

    def test_matches_legacy_inline_line(self) -> None:
        assert METRICS_RE.search(LEGACY_STEP) is not None


class TestExtractEta:
    def test_extracts_from_separate_eta_line_in_full_text(self) -> None:
        text = f"{REAL_STEP}\n{REAL_ETA_LINE}\n"
        assert _extract_eta(text) == "337h 31m 47s"

    def test_none_when_no_eta(self) -> None:
        assert _extract_eta(f"{LEGACY_STEP}\n") is None

    def test_returns_last_eta_when_multiple(self) -> None:
        text = "x | ETA 999h 00m 00s (01:00)\n" + REAL_STEP + "\n" + REAL_ETA_LINE + "\n"
        assert _extract_eta(text) == "337h 31m 47s"


class TestParseLog:
    def test_parses_real_format_fields_and_eta(self) -> None:
        text = f"{REAL_STEP}\n{REAL_ETA_LINE}\n"
        parsed = _parse_log(text)
        assert parsed is not None
        assert parsed["step"] == 16300
        assert parsed["loss"] == 1.2487
        assert parsed["ppl"] == 3.5
        assert parsed["tok_per_s"] == 10985.0
        assert parsed["eta"] == "337h 31m 47s"

    def test_picks_last_matching_step_line(self) -> None:
        text = (
            "01:00:00 step   1,000 ( 0.7%) loss 2.0000 ppl 7.0 lr 6.00e-04   100 \n"
            "02:00:00 step   2,000 ( 1.3%) loss 1.8000 ppl 6.0 lr 6.00e-04   200 \n"
        )
        parsed = _parse_log(text)
        assert parsed is not None and parsed["step"] == 2000


class TestGetTrainingStatusRoundTrip:
    def test_reads_real_format_from_file(self, tmp_path) -> None:
        log = tmp_path / "train.log"
        log.write_text(f"{REAL_STEP}\n{REAL_ETA_LINE}\n", encoding="utf-8")
        status = get_training_status(log_path=str(log), err_path=str(tmp_path / "none.err"))
        assert status["step"] == 16300
        assert status["loss"] == 1.2487
        assert status["tok_per_s"] == 10985.0
        assert status["eta"] == "337h 31m 47s"


class TestTrainingHistory:
    def test_real_format_log_yields_points(self, tmp_path) -> None:
        log = tmp_path / "train.log"
        lines = [
            "01:00:00 step   1,000 ( 0.7%) loss 2.0000 ppl 7.0 lr 6.00e-04   100 ",
            "02:00:00 step   2,000 ( 1.3%) loss 1.8000 ppl 6.0 lr 6.00e-04   200 ",
            "03:00:00 step   3,000 ( 2.0%) loss 1.7000 ppl 5.0 lr 6.00e-04   300 ",
        ]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        history = training_history(str(log))
        assert "error" not in history
        assert history["count"] == 3
        last = history["points"][-1]
        assert last["step"] == 3000
        assert last["loss"] == 1.7
        assert last["lr"] == 0.0006
        assert last["tok_s"] == 300.0
