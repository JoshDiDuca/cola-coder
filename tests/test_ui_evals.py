"""Tests for the UI eval-artifact browser (``cola_coder.ui.evals``)."""

from __future__ import annotations

import json

from cola_coder.ui.evals import list_eval_results, read_eval_result


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_artifacts(tmp_path):
    """Create a representative set of artifacts under tmp_path. Returns paths."""
    reports = tmp_path / "reports"
    eval_results = tmp_path / "eval_results"

    # HumanEval-shaped JSON with pass@k metrics.
    humaneval = eval_results / "humaneval_results.json"
    _write(
        humaneval,
        json.dumps(
            {
                "checkpoint": "checkpoints/tiny/latest",
                "pass@1": 0.34,
                "pass@10": 0.51,
            }
        ),
    )

    # Regression .jsonl (one object per line).
    regression = reports / "regression_v1.jsonl"
    _write(
        regression,
        "\n".join(
            [
                json.dumps({"task": "a", "passed": True}),
                "   ",  # blank line, should be skipped
                json.dumps({"task": "b", "passed": False}),
            ]
        ),
    )

    # Quality report markdown.
    quality_md = reports / "quality_report_tiny_step_0000100.md"
    _write(quality_md, "# Cola-Coder Quality Report\n- HumanEval pass@1: 34.0%\n")

    # Quality report JSON (QualityReport.to_dict shape).
    quality_json = reports / "quality_report_tiny_step_0000100.json"
    _write(
        quality_json,
        json.dumps(
            {
                "humaneval_pass_at_1": 0.34,
                "smoke_test_passed": True,
                "training_step": 100,
                "training_loss": 2.345,
            }
        ),
    )

    return {
        "humaneval": humaneval,
        "regression": regression,
        "quality_md": quality_md,
        "quality_json": quality_json,
    }


def test_missing_root_returns_empty(tmp_path):
    assert list_eval_results(str(tmp_path / "nope")) == []


def test_empty_root_returns_empty(tmp_path):
    assert list_eval_results(str(tmp_path)) == []


def test_discovers_all_artifacts(tmp_path):
    _make_artifacts(tmp_path)
    results = list_eval_results(str(tmp_path))
    names = {entry["name"] for entry in results}
    assert "humaneval_results.json" in names
    assert "regression_v1.jsonl" in names
    assert "quality_report_tiny_step_0000100.md" in names
    assert "quality_report_tiny_step_0000100.json" in names


def test_classification(tmp_path):
    _make_artifacts(tmp_path)
    by_name = {entry["name"]: entry for entry in list_eval_results(str(tmp_path))}
    assert by_name["humaneval_results.json"]["kind"] == "humaneval"
    assert by_name["regression_v1.jsonl"]["kind"] == "regression"
    assert by_name["quality_report_tiny_step_0000100.md"]["kind"] == "quality_report"
    assert by_name["quality_report_tiny_step_0000100.json"]["kind"] == "quality_report"


def test_completion_benchmark_classification(tmp_path):
    _write(
        tmp_path / "eval_results" / "completion_benchmark.json",
        json.dumps({"score": 0.5}),
    )
    by_name = {entry["name"]: entry for entry in list_eval_results(str(tmp_path))}
    assert by_name["completion_benchmark.json"]["kind"] == "completion_benchmark"


def test_pass_at_k_summary(tmp_path):
    arts = _make_artifacts(tmp_path)
    by_name = {entry["name"]: entry for entry in list_eval_results(str(tmp_path))}
    summary = by_name[arts["humaneval"].name]["summary"]
    assert "pass@1 0.34" in summary
    assert "pass@10 0.51" in summary


def test_quality_json_summary(tmp_path):
    arts = _make_artifacts(tmp_path)
    by_name = {entry["name"]: entry for entry in list_eval_results(str(tmp_path))}
    summary = by_name[arts["quality_json"].name]["summary"]
    assert "pass@1 0.34" in summary
    assert "smoke pass" in summary
    assert "step 100" in summary


def test_markdown_summary_is_empty(tmp_path):
    arts = _make_artifacts(tmp_path)
    by_name = {entry["name"]: entry for entry in list_eval_results(str(tmp_path))}
    assert by_name[arts["quality_md"].name]["summary"] == ""


def test_results_sorted_newest_first(tmp_path):
    arts = _make_artifacts(tmp_path)
    import os
    import time

    # Force a clear mtime ordering: regression oldest, humaneval newest.
    now = time.time()
    os.utime(arts["regression"], (now - 100, now - 100))
    os.utime(arts["quality_md"], (now - 50, now - 50))
    os.utime(arts["quality_json"], (now - 40, now - 40))
    os.utime(arts["humaneval"], (now, now))

    results = list_eval_results(str(tmp_path))
    mtimes = [entry["mtime"] for entry in results]
    assert mtimes == sorted(mtimes, reverse=True)
    assert results[0]["name"] == "humaneval_results.json"


def test_read_json_artifact(tmp_path):
    arts = _make_artifacts(tmp_path)
    result = read_eval_result(str(arts["humaneval"]))
    assert "error" not in result
    assert result["kind"] == "humaneval"
    assert result["content"] is None
    assert result["truncated"] is False
    assert result["parsed"]["pass@1"] == 0.34


def test_read_jsonl_artifact(tmp_path):
    arts = _make_artifacts(tmp_path)
    result = read_eval_result(str(arts["regression"]))
    assert "error" not in result
    assert result["kind"] == "regression"
    assert result["content"] is None
    # Blank line skipped -> two parsed objects.
    assert isinstance(result["parsed"], list)
    assert len(result["parsed"]) == 2
    assert result["parsed"][0]["task"] == "a"


def test_read_markdown_artifact(tmp_path):
    arts = _make_artifacts(tmp_path)
    result = read_eval_result(str(arts["quality_md"]))
    assert "error" not in result
    assert result["kind"] == "quality_report"
    assert result["parsed"] is None
    assert "Cola-Coder Quality Report" in result["content"]
    assert result["truncated"] is False


def test_read_markdown_truncation(tmp_path):
    big = tmp_path / "reports" / "quality_report_big.md"
    _write(big, "x" * 50000)
    result = read_eval_result(str(big))
    assert result["truncated"] is True
    assert len(result["content"]) == 40000


def test_read_missing_path_returns_error(tmp_path):
    result = read_eval_result(str(tmp_path / "does_not_exist.json"))
    assert "error" in result


def test_read_invalid_json_returns_error(tmp_path):
    bad = tmp_path / "eval_results" / "humaneval_bad.json"
    _write(bad, "{not valid json")
    result = read_eval_result(str(bad))
    assert "error" in result
