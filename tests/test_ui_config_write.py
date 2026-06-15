"""R11: config write endpoint — validate YAML + path containment + atomic write.

A bad edit (invalid YAML or a path outside configs/) must be refused BEFORE touching
disk, so a config can never be corrupted from the UI. Editing a file does not affect
an already-running trainer (it read its config at launch).
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui import configs as cfg


def test_writes_valid_yaml(tmp_path: Path) -> None:
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    target = cfgs / "small.yaml"
    target.write_text("model:\n  dim: 256\n", encoding="utf-8")

    result = cfg.write_config(str(target), "model:\n  dim: 512\n", configs_dir=str(cfgs))

    assert result.get("ok") is True
    assert result["bytes_written"] > 0
    assert target.read_text(encoding="utf-8") == "model:\n  dim: 512\n"


def test_creates_nested_file_inside_configs(tmp_path: Path) -> None:
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    target = cfgs / "auto" / "derived.yaml"

    result = cfg.write_config(str(target), "a: 1\n", configs_dir=str(cfgs))

    assert result.get("ok") is True
    assert target.exists()


def test_rejects_invalid_yaml_without_writing(tmp_path: Path) -> None:
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    target = cfgs / "small.yaml"
    target.write_text("model:\n  dim: 256\n", encoding="utf-8")

    result = cfg.write_config(str(target), "model:\n  dim: : : [unbalanced\n", configs_dir=str(cfgs))

    assert "error" in result
    assert "YAML" in result["error"]
    # Original content is untouched.
    assert target.read_text(encoding="utf-8") == "model:\n  dim: 256\n"


def test_rejects_path_traversal(tmp_path: Path) -> None:
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    outside = tmp_path / "evil.yaml"

    result = cfg.write_config(str(outside), "a: 1\n", configs_dir=str(cfgs))

    assert "error" in result
    assert "outside" in result["error"]
    assert not outside.exists()


def test_rejects_non_yaml_suffix(tmp_path: Path) -> None:
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    target = cfgs / "notes.txt"

    result = cfg.write_config(str(target), "a: 1\n", configs_dir=str(cfgs))

    assert "error" in result
    assert not target.exists()
