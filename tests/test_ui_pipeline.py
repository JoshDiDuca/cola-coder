"""Tests for the UI pipeline-run browsing helpers."""

from __future__ import annotations

import json

from cola_coder.ui.pipeline import list_pipeline_runs, read_pipeline_run


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_list_missing_dir(tmp_path):
    assert list_pipeline_runs(str(tmp_path / "nope")) == []


def test_list_empty_dir(tmp_path):
    assert list_pipeline_runs(str(tmp_path)) == []


def _seed_runs(tmp_path):
    # A well-formed run with a stages list (mixed statuses).
    _write(
        tmp_path / "good.json",
        {
            "name": "good",
            "config_path": "configs/small.yaml",
            "stages": [
                {"name": "collect", "status": "completed"},
                {"name": "prepare", "status": "completed"},
                {"name": "pretrain", "status": "running"},
                {"name": "evaluate", "status": "pending"},
            ],
        },
    )
    # A minimal / odd-shaped run: no stages field at all.
    _write(tmp_path / "odd.json", {"name": "odd", "notes": "no stages here"})
    # Invalid JSON.
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")


def test_list_sorted_with_core_fields(tmp_path):
    _seed_runs(tmp_path)
    runs = list_pipeline_runs(str(tmp_path))

    assert [r["name"] for r in runs] == ["broken", "good", "odd"]
    for run in runs:
        assert "name" in run
        assert "path" in run
        assert "mtime" in run
        assert run["path"].endswith(".json")


def test_list_computes_stage_summary(tmp_path):
    _seed_runs(tmp_path)
    runs = {r["name"]: r for r in list_pipeline_runs(str(tmp_path))}

    good = runs["good"]
    assert good["num_stages"] == 4
    assert good["completed"] == 2
    # A running stage present -> overall running.
    assert good["status"] == "running"
    assert "error" not in good


def test_list_tolerates_odd_shape(tmp_path):
    _seed_runs(tmp_path)
    runs = {r["name"]: r for r in list_pipeline_runs(str(tmp_path))}

    odd = runs["odd"]
    assert odd["num_stages"] is None
    assert odd["completed"] is None
    assert odd["status"] is None
    assert "error" not in odd


def test_list_tolerates_invalid_json(tmp_path):
    _seed_runs(tmp_path)
    runs = {r["name"]: r for r in list_pipeline_runs(str(tmp_path))}

    broken = runs["broken"]
    assert "error" in broken
    assert broken["name"] == "broken"
    assert broken["mtime"] is not None


def test_list_stages_as_dict(tmp_path):
    # stages as a name->status mapping; all completed -> overall completed.
    _write(
        tmp_path / "mapping.json",
        {"stages": {"a": "completed", "b": "completed", "c": "completed"}},
    )
    runs = {r["name"]: r for r in list_pipeline_runs(str(tmp_path))}
    mapping = runs["mapping"]
    assert mapping["num_stages"] == 3
    assert mapping["completed"] == 3
    assert mapping["status"] == "completed"


def test_list_failed_takes_precedence(tmp_path):
    _write(
        tmp_path / "fail.json",
        {"stages": [{"status": "completed"}, {"status": "failed"}, {"status": "running"}]},
    )
    runs = {r["name"]: r for r in list_pipeline_runs(str(tmp_path))}
    assert runs["fail"]["status"] == "failed"
    assert runs["fail"]["completed"] == 1


def test_read_good_returns_full_dict(tmp_path):
    payload = {"name": "good", "config_path": "configs/small.yaml", "stages": []}
    _write(tmp_path / "good.json", payload)

    result = read_pipeline_run(str(tmp_path / "good.json"))
    assert result == payload


def test_read_missing_returns_error(tmp_path):
    result = read_pipeline_run(str(tmp_path / "missing.json"))
    assert "error" in result


def test_read_invalid_returns_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    result = read_pipeline_run(str(bad))
    assert "error" in result


def test_read_non_dict_returns_error(tmp_path):
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    result = read_pipeline_run(str(arr))
    assert "error" in result
