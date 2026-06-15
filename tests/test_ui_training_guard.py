"""Guard tests: training-active detection must survive OPS-001 (elevated trainer).

The live trainer is often launched at higher OS integrity than the UI process,
so the psutil cmdline scan cannot see it. The dashboard ``alive`` flag and the
``/api/generate`` gate therefore also trust the per-step tqdm ``.err`` mtime
(rewritten every step). Without this, a UI generation could load the model on
the GPU and contend with the live training run. These tests lock that fix.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from cola_coder.ui import status as st
from cola_coder.ui.app import create_app


def _write_fresh(path: Path, age_s: float) -> None:
    """Create ``path`` and set its mtime ``age_s`` seconds in the past."""
    path.write_text("Training:   8%| 12000/150000 [11:00<700:00:00, 20s/it]\n", encoding="utf-8")
    when = time.time() - age_s
    os.utime(path, (when, when))


def test_progress_fresh_true_for_recent_err(tmp_path: Path) -> None:
    err = tmp_path / "train.err"
    _write_fresh(err, age_s=10.0)
    assert st._training_progress_fresh(str(err)) is True


def test_progress_fresh_false_for_stale_err(tmp_path: Path) -> None:
    err = tmp_path / "train.err"
    _write_fresh(err, age_s=st._TRAIN_FRESH_MAX_AGE_S + 120.0)
    assert st._training_progress_fresh(str(err)) is False


def test_progress_fresh_false_for_missing_err(tmp_path: Path) -> None:
    assert st._training_progress_fresh(str(tmp_path / "nope.err")) is False


def test_progress_fresh_tolerates_slow_dataloader_gap(tmp_path: Path) -> None:
    """BUG-136: this dataloader-bound run can go 22-30 min between writes — a NORMAL
    slow patch, not a stall. The window must keep treating it as active (a 10-min
    window false-negatived, breaking the inference gate)."""
    err = tmp_path / "train.err"
    _write_fresh(err, age_s=22 * 60)  # 22 minutes — within the 45-min cadence window
    assert st._TRAIN_FRESH_MAX_AGE_S >= 22 * 60
    assert st._training_progress_fresh(str(err)) is True


def test_progress_fresh_uses_freshest_of_err_and_log(tmp_path: Path) -> None:
    """Either the per-step .err OR the 100-step .log being fresh ⇒ active."""
    err = tmp_path / "train.err"
    log = tmp_path / "train.log"
    _write_fresh(err, age_s=st._TRAIN_FRESH_MAX_AGE_S + 600.0)  # err stale
    _write_fresh(log, age_s=30.0)  # but log just written
    assert st._training_progress_fresh(str(err), str(log)) is True


def test_is_training_active_true_on_fresh_progress(tmp_path: Path) -> None:
    """Even with no detectable train.py process, fresh progress ⇒ active."""
    err = tmp_path / "train.err"
    log = tmp_path / "train.log"
    _write_fresh(err, age_s=5.0)
    assert st.is_training_active(str(log), str(err)) is True


def test_generate_endpoint_refused_while_training_active(tmp_path: Path) -> None:
    """/api/generate must return 409 (never load a model) when training is live."""
    err = tmp_path / "train.err"
    log = tmp_path / "train.log"
    _write_fresh(err, age_s=5.0)
    app = create_app(log_path=str(log), err_path=str(err))
    client = TestClient(app)
    resp = client.post(
        "/api/generate",
        json={
            "prompt": "def add(a, b):",
            "checkpoint": "checkpoints/small/latest",
            "config": "configs/small.yaml",
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 409
    assert "training is running" in resp.json()["error"]


def test_generate_endpoint_not_gated_when_idle(tmp_path: Path) -> None:
    """With stale/absent progress and no trainer process, the gate does NOT fire.

    We assert the response is NOT the 409 training-guard (it will be a 400/500
    model-load error since the checkpoint path is bogus — proving the request got
    PAST the guard rather than being refused by it).
    """
    err = tmp_path / "train.err"
    log = tmp_path / "train.log"
    _write_fresh(err, age_s=st._TRAIN_FRESH_MAX_AGE_S + 300.0)
    app = create_app(log_path=str(log), err_path=str(err))
    client = TestClient(app)
    resp = client.post(
        "/api/generate",
        json={
            "prompt": "x",
            "checkpoint": str(tmp_path / "does-not-exist"),
            "config": str(tmp_path / "nope.yaml"),
            "max_tokens": 4,
        },
    )
    assert resp.status_code != 409
