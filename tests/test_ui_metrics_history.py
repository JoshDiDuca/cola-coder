"""Tests for the UI metrics_history time-series parser.

Hermetic: every test writes its own fake .log into tmp_path. The real, live
training log is never opened or written.
"""

from __future__ import annotations

from cola_coder.ui.metrics_history import training_history


def _line(time: str, step: str, pct: str, loss: str, ppl: str, lr: str, toks: str) -> str:
    return (
        f"{time} step {step:>7} ({pct:>5}%) loss {loss} "
        f"ppl {ppl:>8} lr {lr}     {toks}"
    )


def _write_log(path, lines: list[str]) -> str:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _good_lines(n: int) -> list[str]:
    lines = []
    for i in range(n):
        step = f"{(i + 1) * 100:,}"
        loss = f"{2.0 - i * 0.01:.4f}"
        ppl = f"{4.0 + i * 0.1:.1f}"
        toks = f"{1000 + i * 10:,}"
        lines.append(_line("11:42:57", step, f"{i:.1f}", loss, ppl, "6.00e-04", toks))
    return lines


def test_missing_file_returns_error(tmp_path):
    result = training_history(str(tmp_path / "does_not_exist.log"))
    assert "error" in result
    assert "points" not in result


def test_basic_parse_count_and_order(tmp_path):
    log = _write_log(tmp_path / "t.log", _good_lines(10))
    result = training_history(log, max_points=500)
    assert "error" not in result
    assert result["count"] == 10
    assert len(result["points"]) == 10
    steps = [p["step"] for p in result["points"]]
    assert steps == sorted(steps)
    assert steps[0] == 100
    assert steps[-1] == 1000


def test_field_values_and_comma_stripping(tmp_path):
    line = _line("11:42:57", "5,200", " 3.5", "1.3845", "4.0", "6.00e-04", "9,144")
    log = _write_log(tmp_path / "t.log", [line])
    result = training_history(log)
    assert result["count"] == 1
    p = result["points"][0]
    assert p["step"] == 5200
    assert p["loss"] == 1.3845
    assert p["ppl"] == 4.0
    assert p["lr"] == 6.00e-04
    assert p["tok_s"] == 9144.0


def test_point_keys_exact(tmp_path):
    log = _write_log(tmp_path / "t.log", _good_lines(1))
    p = training_history(log)["points"][0]
    assert set(p.keys()) == {"step", "loss", "ppl", "lr", "tok_s"}


def test_malformed_lines_skipped(tmp_path):
    lines = _good_lines(10)
    lines.insert(3, "this is not a step line at all")
    lines.insert(7, "11:42:57 step  loss garbled ppl")  # partial / malformed
    log = _write_log(tmp_path / "t.log", lines)
    result = training_history(log)
    assert result["count"] == 10  # only valid lines counted
    assert all(p["step"] is not None for p in result["points"])


def test_empty_log_returns_zero_count(tmp_path):
    log = _write_log(tmp_path / "t.log", [])
    result = training_history(log)
    assert result["count"] == 0
    assert result["points"] == []


def test_downsampling_respects_max_points(tmp_path):
    log = _write_log(tmp_path / "t.log", _good_lines(50))
    result = training_history(log, max_points=10)
    assert result["count"] == 50  # count is BEFORE downsampling
    assert len(result["points"]) <= 10


def test_downsampling_keeps_first_and_last(tmp_path):
    lines = _good_lines(50)
    log = _write_log(tmp_path / "t.log", lines)
    result = training_history(log, max_points=10)
    points = result["points"]
    assert points[0]["step"] == 100  # first valid line
    assert points[-1]["step"] == 5000  # last valid line (50 * 100)


def test_no_downsampling_when_under_limit(tmp_path):
    log = _write_log(tmp_path / "t.log", _good_lines(5))
    result = training_history(log, max_points=10)
    assert len(result["points"]) == 5


def test_carriage_return_separated_lines(tmp_path):
    # Defensively handle \r separators (tqdm style) even though pretty lines
    # are newline-separated.
    lines = _good_lines(3)
    (tmp_path / "t.log").write_text("\r".join(lines) + "\r", encoding="utf-8")
    result = training_history(str(tmp_path / "t.log"))
    assert result["count"] == 3


def test_last_point_is_last_valid_line_with_trailing_malformed(tmp_path):
    lines = _good_lines(20)
    lines.append("11:42:57 step partial broken tail")  # trailing malformed
    log = _write_log(tmp_path / "t.log", lines)
    result = training_history(log, max_points=5)
    assert result["count"] == 20
    assert result["points"][-1]["step"] == 2000  # 20 * 100, last VALID line


def test_values_agree_with_status_style_line(tmp_path):
    # The canonical example line from the task / status.py docstring.
    line = "11:42:57 step   5,200 ( 3.5%) loss 1.3845 ppl      4.0 lr 6.00e-04     9,144"
    log = _write_log(tmp_path / "t.log", [line])
    p = training_history(log)["points"][0]
    assert p["step"] == 5200
    assert p["loss"] == 1.3845
    assert p["ppl"] == 4.0
    assert p["tok_s"] == 9144.0
