"""Tests for the UI pipeline-run lifecycle ops + their FastAPI routes.

These exercise pure state operations on ``pipeline_runs/{name}.json`` — no stage
is ever executed, so they are safe to run alongside a live trainer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cola_coder.ui import pipeline_ops as po
from cola_coder.ui.app import create_app
from cola_coder.ui.jobs import JobManager
from cola_coder.ui.schemas import PipelineDeleteResult, PipelineRunDetail


@pytest.fixture()
def config_file(tmp_path: Path) -> str:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  dim: 64\n", encoding="utf-8")
    return str(cfg)


@pytest.fixture()
def runs_dir(tmp_path: Path) -> str:
    return str(tmp_path / "pipeline_runs")


def test_create_run_returns_valid_detail(config_file: str, runs_dir: str) -> None:
    result = po.create_run("my-run", config_file, runs_dir=runs_dir)
    detail = PipelineRunDetail.model_validate(result)  # schema-valid, no extras
    assert detail.name == "my-run"
    assert detail.num_stages == 10
    assert len(detail.stages) == 10
    assert detail.status == "pending"
    assert detail.completed == 0
    # Optional stages (4 extend-context, 7 upcycle-moe) are flagged.
    optional = {s.num for s in detail.stages if s.optional}
    assert optional == {4, 7}


def test_create_with_skip_stages_marks_them_skipped(config_file: str, runs_dir: str) -> None:
    result = po.create_run("skip-run", config_file, skip_stages=[4, 7], runs_dir=runs_dir)
    detail = PipelineRunDetail.model_validate(result)
    skipped = {s.num for s in detail.stages if s.status == "skipped"}
    assert skipped == {4, 7}
    assert detail.active_stages == 8


def test_create_rejects_bad_name(config_file: str, runs_dir: str) -> None:
    for bad in ["../escape", "with/slash", "has space", "", "a" * 65]:
        result = po.create_run(bad, config_file, runs_dir=runs_dir)
        assert "error" in result, f"expected rejection for {bad!r}"


def test_create_rejects_missing_config(runs_dir: str) -> None:
    result = po.create_run("ok-name", "does/not/exist.yaml", runs_dir=runs_dir)
    assert "error" in result and "config not found" in result["error"]


def test_create_refuses_to_clobber(config_file: str, runs_dir: str) -> None:
    po.create_run("dup", config_file, runs_dir=runs_dir)
    result = po.create_run("dup", config_file, runs_dir=runs_dir)
    assert "error" in result and "already exists" in result["error"]


def test_create_rejects_unknown_skip_stage(config_file: str, runs_dir: str) -> None:
    result = po.create_run("bad-skip", config_file, skip_stages=[99], runs_dir=runs_dir)
    assert "error" in result and "unknown stage" in result["error"]


def test_reset_then_override_roundtrip(config_file: str, runs_dir: str) -> None:
    po.create_run("rt", config_file, runs_dir=runs_dir)
    reset = po.reset_run("rt", 3, runs_dir=runs_dir)
    PipelineRunDetail.model_validate(reset)
    ov = po.set_override("rt", 2, "/data/processed/train.npy", runs_dir=runs_dir)
    detail = PipelineRunDetail.model_validate(ov)
    stage2 = next(s for s in detail.stages if s.num == 2)
    assert stage2.override == "/data/processed/train.npy"


def test_reset_unknown_stage_errors(config_file: str, runs_dir: str) -> None:
    po.create_run("re", config_file, runs_dir=runs_dir)
    assert "error" in po.reset_run("re", 99, runs_dir=runs_dir)


def test_ops_on_missing_run_error(runs_dir: str) -> None:
    assert "error" in po.reset_run("ghost", 1, runs_dir=runs_dir)
    assert "error" in po.set_override("ghost", 1, "x", runs_dir=runs_dir)
    assert "error" in po.get_run_detail("ghost", runs_dir=runs_dir)
    assert "error" in po.delete_run("ghost", runs_dir=runs_dir)


def test_delete_run(config_file: str, runs_dir: str) -> None:
    po.create_run("die", config_file, runs_dir=runs_dir)
    result = po.delete_run("die", runs_dir=runs_dir)
    deleted = PipelineDeleteResult.model_validate(result)
    assert deleted.ok and deleted.name == "die"
    assert "error" in po.get_run_detail("die", runs_dir=runs_dir)


# ── HTTP layer ──────────────────────────────────────────────────────────────

def test_endpoints_full_lifecycle(tmp_path: Path) -> None:
    cfg = tmp_path / "configs" / "small.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("model:\n  dim: 64\n", encoding="utf-8")
    app = create_app(job_manager=JobManager(), project_root=str(tmp_path))
    c = TestClient(app)

    # create
    r = c.post("/api/pipeline/create", json={"name": "e2e", "config_path": str(cfg)})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "e2e"

    # detail
    r = c.get("/api/pipeline/detail", params={"name": "e2e"})
    assert r.status_code == 200 and r.json()["num_stages"] == 10

    # reset + override
    assert c.post("/api/pipeline/reset", json={"name": "e2e", "stage_num": 5}).status_code == 200
    r = c.post("/api/pipeline/override", json={"name": "e2e", "stage_num": 2, "path": "/tmp/x"})
    assert r.status_code == 200
    assert any(s["override"] == "/tmp/x" for s in r.json()["stages"])

    # error path returns a typed {error} body (200, union-validated)
    r = c.post("/api/pipeline/create", json={"name": "../bad", "config_path": str(cfg)})
    assert r.status_code == 200 and "error" in r.json()

    # delete
    r = c.post("/api/pipeline/delete", json={"name": "e2e"})
    assert r.status_code == 200 and r.json()["ok"] is True
