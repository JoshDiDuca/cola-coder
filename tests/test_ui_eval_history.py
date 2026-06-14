"""Tests for the UI eval_history auto-eval-over-training series collector.

Hermetic: every test synthesizes its own checkpoint tree under tmp_path. The
real checkpoints directory is never opened or written.

The on-disk format mirrors what ``scripts/training_eval_history.py`` reads:
- ``checkpoints/<model>/step_*/metadata.json`` carrying an ``auto_eval`` state
  dict with a ``history`` list of EvalSnapshot dicts, and
- a standalone ``checkpoints/<model>/auto_eval_history.json`` array.
"""

from __future__ import annotations

import json

from cola_coder.ui.eval_history import eval_history


def _snapshot(step: int, p1: float, p5: float, *, is_best: bool = False) -> dict:
    """Build an EvalSnapshot-shaped dict (matches auto_eval.EvalSnapshot.to_dict)."""
    return {
        "step": step,
        "timestamp": f"2026-06-14T00:0{step % 10}:00",
        "pass_at_1": p1,
        "pass_at_5": p5,
        "num_problems": 20,
        "avg_generation_time": 1.5,
        "is_best": is_best,
    }


def _write_step_meta(root, model: str, step: int, history: list[dict]) -> None:
    step_dir = root / "checkpoints" / model / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "metadata.json").write_text(
        json.dumps({"step": step, "auto_eval": {"history": history}}),
        encoding="utf-8",
    )


def _write_standalone(root, model: str, history: list[dict]) -> None:
    model_dir = root / "checkpoints" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "auto_eval_history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )


# ── empty / discovery ─────────────────────────────────────────────────────


def test_empty_tree_is_not_error(tmp_path):
    result = eval_history(str(tmp_path))
    assert result == {"snapshots": [], "count": 0, "metric_keys": []}


def test_no_checkpoints_dir_is_empty(tmp_path):
    (tmp_path / "unrelated").mkdir()
    result = eval_history(str(tmp_path))
    assert result["count"] == 0
    assert "error" not in result


def test_nonexistent_root_returns_error(tmp_path):
    result = eval_history(str(tmp_path / "nope"))
    assert "error" in result
    assert "snapshots" not in result


# ── parsing from step metadata ────────────────────────────────────────────


def test_parses_metadata_history(tmp_path):
    _write_step_meta(
        tmp_path,
        "tiny",
        5000,
        [_snapshot(1000, 0.05, 0.10), _snapshot(5000, 0.12, 0.20, is_best=True)],
    )
    result = eval_history(str(tmp_path))
    assert result["count"] == 2
    steps = [s["step"] for s in result["snapshots"]]
    assert steps == [1000, 5000]
    first = result["snapshots"][0]
    assert first["metrics"]["pass_at_1"] == 0.05
    assert first["metrics"]["pass_at_5"] == 0.10
    assert "step" not in first["metrics"]
    assert "timestamp" not in first["metrics"]


def test_metric_keys_union(tmp_path):
    _write_step_meta(tmp_path, "tiny", 1000, [_snapshot(1000, 0.05, 0.10)])
    result = eval_history(str(tmp_path))
    keys = set(result["metric_keys"])
    assert {"pass_at_1", "pass_at_5", "num_problems", "avg_generation_time", "is_best"} <= keys
    assert "step" not in keys
    assert "timestamp" not in keys


def test_chronological_order_by_step(tmp_path):
    # Two separate step dirs, written out of order.
    _write_step_meta(tmp_path, "tiny", 9000, [_snapshot(9000, 0.30, 0.40)])
    _write_step_meta(tmp_path, "tiny", 2000, [_snapshot(2000, 0.10, 0.15)])
    result = eval_history(str(tmp_path))
    assert [s["step"] for s in result["snapshots"]] == [2000, 9000]


# ── parsing from standalone json array ────────────────────────────────────


def test_parses_standalone_array(tmp_path):
    _write_standalone(
        tmp_path,
        "small",
        [_snapshot(1000, 0.06, 0.11), _snapshot(2000, 0.09, 0.14)],
    )
    result = eval_history(str(tmp_path))
    assert result["count"] == 2
    assert [s["step"] for s in result["snapshots"]] == [1000, 2000]


def test_combines_metadata_and_standalone_across_models(tmp_path):
    _write_step_meta(tmp_path, "tiny", 3000, [_snapshot(3000, 0.15, 0.25)])
    _write_standalone(tmp_path, "small", [_snapshot(1000, 0.06, 0.11)])
    result = eval_history(str(tmp_path))
    assert result["count"] == 2
    # Sorted by step across both sources.
    assert [s["step"] for s in result["snapshots"]] == [1000, 3000]


# ── robustness ────────────────────────────────────────────────────────────


def test_metadata_without_auto_eval_is_skipped(tmp_path):
    step_dir = tmp_path / "checkpoints" / "tiny" / "step_00001000"
    step_dir.mkdir(parents=True)
    (step_dir / "metadata.json").write_text(
        json.dumps({"step": 1000, "loss": 1.5}), encoding="utf-8"
    )
    result = eval_history(str(tmp_path))
    assert result == {"snapshots": [], "count": 0, "metric_keys": []}


def test_garbage_json_file_is_skipped_not_error(tmp_path):
    model_dir = tmp_path / "checkpoints" / "tiny"
    model_dir.mkdir(parents=True)
    (model_dir / "auto_eval_history.json").write_text("{not valid json", encoding="utf-8")
    # A good step alongside the garbage standalone file.
    _write_step_meta(tmp_path, "tiny", 1000, [_snapshot(1000, 0.05, 0.10)])
    result = eval_history(str(tmp_path))
    assert "error" not in result
    assert result["count"] == 1
    assert result["snapshots"][0]["step"] == 1000


def test_snapshot_without_step_sorted_last(tmp_path):
    no_step = {"pass_at_1": 0.5, "pass_at_5": 0.6}
    _write_standalone(tmp_path, "tiny", [no_step, _snapshot(1000, 0.05, 0.10)])
    result = eval_history(str(tmp_path))
    assert result["count"] == 2
    steps = [s["step"] for s in result["snapshots"]]
    assert steps == [1000, None]


def test_path_and_mtime_present(tmp_path):
    _write_step_meta(tmp_path, "tiny", 1000, [_snapshot(1000, 0.05, 0.10)])
    result = eval_history(str(tmp_path))
    snap = result["snapshots"][0]
    assert snap["path"].endswith("metadata.json")
    assert isinstance(snap["mtime"], float)


def test_result_is_json_serializable(tmp_path):
    _write_step_meta(tmp_path, "tiny", 1000, [_snapshot(1000, 0.05, 0.10, is_best=True)])
    result = eval_history(str(tmp_path))
    # Round-trips cleanly — proves no non-serializable objects leak through.
    assert json.loads(json.dumps(result))["count"] == 1
