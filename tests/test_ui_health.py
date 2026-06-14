"""Hermetic tests for the UI project-health checklist.

Each test builds a fake project tree under ``tmp_path`` so the checks reflect
exactly which dirs/files are present. No real project state is touched.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.health import project_health


def _build_full_project(root: Path) -> None:
    """Create a tree where every check should pass."""
    (root / ".venv").mkdir()
    (root / "src" / "cola_coder").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs" / "small.yaml").write_text("model: {}\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    step_dir = root / "checkpoints" / "small" / "step_00001000"
    step_dir.mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    (root / "train.log").write_text("step 1 loss 2.0\n", encoding="utf-8")


def _checks_by_name(result: dict) -> dict:
    return {c["name"]: c for c in result["checks"]}


def test_returns_valid_shape() -> None:
    result = project_health(".")
    assert "error" not in result
    assert isinstance(result["score"], int)
    assert 0 <= result["score"] <= 100
    assert isinstance(result["checks"], list)
    assert isinstance(result["summary"], str)
    for c in result["checks"]:
        assert set(c) == {"name", "ok", "detail"}
        assert isinstance(c["ok"], bool)


def test_full_project_all_pass(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    result = project_health(str(tmp_path))
    assert "error" not in result
    assert result["score"] == 100
    checks = _checks_by_name(result)
    assert all(c["ok"] for c in checks.values())
    n = len(result["checks"])
    assert result["summary"] == f"{n}/{n} checks OK"


def test_empty_root_low_score(tmp_path: Path) -> None:
    result = project_health(str(tmp_path))
    assert "error" not in result
    assert result["score"] == 0
    assert all(not c["ok"] for c in result["checks"])
    n = len(result["checks"])
    assert result["summary"] == f"0/{n} checks OK"


def test_missing_root_never_raises() -> None:
    missing = "this/path/does/not/exist/anywhere"
    result = project_health(missing)
    assert "error" not in result
    assert result["score"] == 0
    assert isinstance(result["checks"], list)


def test_missing_venv_detected(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    # remove .venv only
    (tmp_path / ".venv").rmdir()
    result = project_health(str(tmp_path))
    checks = _checks_by_name(result)
    assert checks["venv"]["ok"] is False
    assert result["score"] < 100


def test_missing_key_dir_detected(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    (tmp_path / "tests").rmdir()
    result = project_health(str(tmp_path))
    checks = _checks_by_name(result)
    assert checks["tests"]["ok"] is False
    assert checks["scripts"]["ok"] is True


def test_empty_configs_dir_fails_yaml_check(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    (tmp_path / "configs" / "small.yaml").unlink()
    result = project_health(str(tmp_path))
    checks = _checks_by_name(result)
    # configs/ dir still present, but no yaml inside
    assert checks["configs"]["ok"] is True
    assert checks["configs_has_yaml"]["ok"] is False


def test_no_tokenizer_detected(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    result = project_health(str(tmp_path))
    checks = _checks_by_name(result)
    assert checks["tokenizer"]["ok"] is False


def test_tokenizer_under_data_subdir(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    ds = tmp_path / "data" / "mydataset"
    ds.mkdir(parents=True)
    (ds / "tokenizer.json").write_text("{}", encoding="utf-8")
    result = project_health(str(tmp_path))
    checks = _checks_by_name(result)
    assert checks["tokenizer"]["ok"] is True


def test_no_checkpoint_when_no_step_dir(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    # remove the step_* dir but keep checkpoints/small
    step_dir = tmp_path / "checkpoints" / "small" / "step_00001000"
    step_dir.rmdir()
    result = project_health(str(tmp_path))
    checks = _checks_by_name(result)
    assert checks["checkpoint"]["ok"] is False


def test_non_step_dir_not_counted_as_checkpoint(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    step_dir = tmp_path / "checkpoints" / "small" / "step_00001000"
    step_dir.rmdir()
    (tmp_path / "checkpoints" / "small" / "latest_misc").mkdir()
    result = project_health(str(tmp_path))
    checks = _checks_by_name(result)
    assert checks["checkpoint"]["ok"] is False


def test_score_is_passing_fraction(tmp_path: Path) -> None:
    _build_full_project(tmp_path)
    # break exactly one check
    (tmp_path / "train.log").unlink()
    result = project_health(str(tmp_path))
    n = len(result["checks"])
    n_ok = sum(1 for c in result["checks"] if c["ok"])
    assert n_ok == n - 1
    assert result["score"] == int(round(100 * n_ok / n))
    assert result["summary"] == f"{n_ok}/{n} checks OK"
    checks = _checks_by_name(result)
    assert checks["training_log"]["ok"] is False
