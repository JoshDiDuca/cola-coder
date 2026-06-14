"""Tests for the UI status helpers (cola_coder.ui.status)."""

from __future__ import annotations

import json

from cola_coder.ui.status import (
    get_system_status,
    get_training_status,
    list_checkpoints,
)

# A realistic pretty log block (note comma-grouped numbers and \r line endings).
_SAMPLE_LOG = (
    "09:21:57 step   4,500 ( 3.0%) loss 1.4185 ppl      4.1 lr 6.00e-04    "
    "10,460 tok/s\r"
    "Checkpoint saved: checkpoints\\small_react_best\\step_00004500\r"
    "03:12:20 step   2,500 ( 1.7%) loss 1.6057 ppl      5.0 lr 6.00e-04     "
    "1,813 tok/s\r\n"
)

# A realistic tqdm err block (\r separated, last bar is the latest step).
_SAMPLE_ERR = (
    "Training:   2%|x| 2515/150000 [04:40<700:27:01, 17.10s/it]\r"
    "Training:   3%|x| 4792/150000 [7:13:45<1906:10:21, 47.26s/it]\r"
)


def test_parse_log_takes_last_line_and_strips_commas(tmp_path):
    log = tmp_path / "train.log"
    err = tmp_path / "train.err"
    log.write_text(_SAMPLE_LOG, encoding="utf-8")
    err.write_text(_SAMPLE_ERR, encoding="utf-8")

    status = get_training_status(str(log), str(err))

    # Last parseable pretty line wins (step 2,500).
    assert status["step"] == 2500
    assert status["loss"] == 1.6057
    assert status["ppl"] == 5.0
    assert status["tok_per_s"] == 1813.0
    assert status["progress_pct"] == 1.7
    assert status["s_per_it"] is None
    assert status["total_steps"] is None
    assert "step" in status["last_log_line"]
    assert isinstance(status["alive"], bool)


def test_falls_back_to_err_when_log_has_no_step_line(tmp_path):
    log = tmp_path / "train.log"
    err = tmp_path / "train.err"
    log.write_text("starting up...\nno step lines here\n", encoding="utf-8")
    err.write_text(_SAMPLE_ERR, encoding="utf-8")

    status = get_training_status(str(log), str(err))

    # Last tqdm bar wins (step 4792 / 150000).
    assert status["step"] == 4792
    assert status["total_steps"] == 150000
    assert status["s_per_it"] == 47.26
    assert status["progress_pct"] is not None
    assert abs(status["progress_pct"] - (4792 / 150000 * 100)) < 1e-3
    assert status["loss"] is None
    assert status["ppl"] is None


def test_falls_back_to_err_when_log_missing(tmp_path):
    err = tmp_path / "train.err"
    err.write_text(_SAMPLE_ERR, encoding="utf-8")

    status = get_training_status(str(tmp_path / "missing.log"), str(err))

    assert status["step"] == 4792
    assert status["total_steps"] == 150000


def test_all_none_when_both_files_missing(tmp_path):
    status = get_training_status(
        str(tmp_path / "missing.log"), str(tmp_path / "missing.err")
    )
    for key in (
        "step",
        "total_steps",
        "progress_pct",
        "loss",
        "ppl",
        "tok_per_s",
        "s_per_it",
        "last_log_line",
    ):
        assert status[key] is None
    assert isinstance(status["alive"], bool)


def test_empty_files_return_none_fields(tmp_path):
    log = tmp_path / "train.log"
    err = tmp_path / "train.err"
    log.write_text("", encoding="utf-8")
    err.write_text("", encoding="utf-8")

    status = get_training_status(str(log), str(err))
    assert status["step"] is None
    assert status["last_log_line"] is None


def test_alive_returns_bool():
    status = get_training_status("nope.log", "nope.err")
    assert isinstance(status["alive"], bool)


def test_get_system_status_shape():
    status = get_system_status()
    assert isinstance(status, dict)
    for key in (
        "gpu_name",
        "gpu_util_pct",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "gpu_power_w",
    ):
        assert key in status


def _make_step(model_dir, step_name, loss=None):
    step_dir = model_dir / step_name
    step_dir.mkdir(parents=True)
    if loss is not None:
        meta = {"step": int(step_name.split("_")[1]), "loss": loss, "config": {}}
        (step_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
    return step_dir


def test_list_checkpoints_parses_step_and_loss(tmp_path):
    root = tmp_path / "checkpoints"
    small = root / "small"
    _make_step(small, "step_00000100", loss=1.4185)
    _make_step(small, "step_00004500", loss=1.401)

    ckpts = list_checkpoints(str(root))

    assert len(ckpts) == 2
    assert [c["step"] for c in ckpts] == [100, 4500]
    assert ckpts[0]["model"] == "small"
    assert ckpts[0]["name"] == "step_00000100"
    assert ckpts[0]["loss"] == 1.4185
    assert ckpts[1]["loss"] == 1.401
    assert ckpts[0]["path"].endswith("step_00000100")
    assert isinstance(ckpts[0]["mtime"], float)


def test_list_checkpoints_sorted_by_model_then_step(tmp_path):
    root = tmp_path / "checkpoints"
    _make_step(root / "zeta", "step_00000050")
    _make_step(root / "alpha", "step_00000200")
    _make_step(root / "alpha", "step_00000010")

    ckpts = list_checkpoints(str(root))
    keys = [(c["model"], c["step"]) for c in ckpts]
    assert keys == [("alpha", 10), ("alpha", 200), ("zeta", 50)]


def test_list_checkpoints_loss_none_when_no_metadata(tmp_path):
    root = tmp_path / "checkpoints"
    _make_step(root / "small", "step_00000100")
    ckpts = list_checkpoints(str(root))
    assert ckpts[0]["loss"] is None


def test_list_checkpoints_missing_root_returns_empty(tmp_path):
    assert list_checkpoints(str(tmp_path / "nope")) == []


def test_list_checkpoints_ignores_non_step_dirs(tmp_path):
    root = tmp_path / "checkpoints"
    model = root / "small"
    model.mkdir(parents=True)
    (model / "latest").write_text("step_00000100", encoding="utf-8")
    (model / "notes").mkdir()
    _make_step(model, "step_00000100")

    ckpts = list_checkpoints(str(root))
    assert len(ckpts) == 1
    assert ckpts[0]["step"] == 100
