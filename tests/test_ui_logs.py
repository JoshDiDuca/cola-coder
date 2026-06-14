"""Tests for the UI log/artifact browser (src/cola_coder/ui/logs.py)."""

from __future__ import annotations

import os
import time

from cola_coder.ui.logs import list_logs, tail_log


def _write(path, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def test_list_logs_missing_root_returns_empty(tmp_path):
    assert list_logs(str(tmp_path / "does_not_exist")) == []


def test_list_logs_finds_log_err_and_job_logs(tmp_path):
    root = tmp_path
    _write(str(root / "train.log"), "step 1\nstep 2\n")
    _write(str(root / "train.err"), "Training: 2%|x| 2/100\r")
    _write(str(root / "ui_jobs" / "job123.log"), "job started\n")

    entries = list_logs(str(root))
    names = {e["name"] for e in entries}
    assert names == {"train.log", "train.err", "job123.log"}

    by_name = {e["name"]: e for e in entries}
    # Every entry carries the fixed contract keys.
    for entry in entries:
        assert set(entry.keys()) == {"name", "path", "size_bytes", "mtime"}
        assert isinstance(entry["size_bytes"], int)
        assert isinstance(entry["mtime"], float)
        assert os.path.isfile(entry["path"])

    assert by_name["train.log"]["size_bytes"] == os.stat(by_name["train.log"]["path"]).st_size


def test_list_logs_ignores_unrelated_top_level_files(tmp_path):
    root = tmp_path
    _write(str(root / "keep.log"), "x")
    _write(str(root / "readme.md"), "ignore me")
    _write(str(root / "data.npy"), "ignore me")

    names = {e["name"] for e in list_logs(str(root))}
    assert names == {"keep.log"}


def test_list_logs_newest_first_by_mtime(tmp_path):
    root = tmp_path
    old = str(root / "old.log")
    new = str(root / "new.log")
    _write(old, "old")
    _write(new, "new")

    # Force a clear mtime ordering regardless of FS resolution.
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))

    entries = list_logs(str(root))
    assert [e["name"] for e in entries] == ["new.log", "old.log"]


def test_list_logs_no_ui_jobs_dir_is_fine(tmp_path):
    root = tmp_path
    _write(str(root / "solo.log"), "hello")
    entries = list_logs(str(root))
    assert [e["name"] for e in entries] == ["solo.log"]


def test_tail_log_returns_last_n_lines(tmp_path):
    path = str(tmp_path / "big.log")
    _write(path, "\n".join(f"line {i}" for i in range(1000)) + "\n")

    result = tail_log(path, lines=10)
    assert "error" not in result
    assert result["path"] == path
    assert result["lines"] == [f"line {i}" for i in range(990, 1000)]
    assert result["truncated"] is True
    assert result["size_bytes"] == os.stat(path).st_size


def test_tail_log_not_truncated_when_fewer_lines(tmp_path):
    path = str(tmp_path / "small.log")
    _write(path, "a\nb\nc\n")

    result = tail_log(path, lines=200)
    assert result["lines"] == ["a", "b", "c"]
    assert result["truncated"] is False


def test_tail_log_splits_on_carriage_returns(tmp_path):
    # tqdm-style progress bar: carriage-return separated, no newlines.
    path = str(tmp_path / "progress.err")
    bar = "Training:  1%\rTraining:  2%\rTraining:  3%\rTraining:  4%"
    _write(path, bar)

    result = tail_log(path, lines=2)
    assert result["lines"] == ["Training:  3%", "Training:  4%"]
    assert result["truncated"] is True


def test_tail_log_mixed_cr_and_lf(tmp_path):
    path = str(tmp_path / "mixed.err")
    _write(path, "log start\nTraining: 1%\rTraining: 2%\nepoch done\n")

    result = tail_log(path, lines=10)
    assert result["lines"] == [
        "log start",
        "Training: 1%",
        "Training: 2%",
        "epoch done",
    ]
    assert result["truncated"] is False


def test_tail_log_missing_file_returns_error(tmp_path):
    result = tail_log(str(tmp_path / "nope.log"))
    assert "error" in result
    assert "lines" not in result


def test_tail_log_zero_lines(tmp_path):
    path = str(tmp_path / "z.log")
    _write(path, "a\nb\nc\n")
    result = tail_log(path, lines=0)
    assert result["lines"] == []
    assert result["truncated"] is True


def test_tail_log_empty_file(tmp_path):
    path = str(tmp_path / "empty.log")
    _write(path, "")
    result = tail_log(path, lines=50)
    assert result["lines"] == []
    assert result["truncated"] is False
    assert result["size_bytes"] == 0
