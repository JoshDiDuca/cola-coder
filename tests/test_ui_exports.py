"""Tests for the read-only export-overview UI helper."""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.exports import export_overview


def _make_checkpoint(root: Path, model: str, step: int) -> Path:
    name = f"step_{step:08d}"
    ckpt = root / "checkpoints" / model / name
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.safetensors").write_text("x", encoding="utf-8")
    return ckpt


def test_returns_three_keys(tmp_path: Path) -> None:
    result = export_overview(str(tmp_path))
    assert set(result) == {"checkpoints", "formats", "existing"}


def test_checkpoint_discovered(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, "small", 1000)
    result = export_overview(str(tmp_path))
    ckpts = result["checkpoints"]
    assert len(ckpts) == 1
    entry = ckpts[0]
    assert entry["model"] == "small"
    assert entry["step"] == 1000
    assert entry["name"] == "step_00001000"
    assert Path(entry["path"]).is_dir()


def test_multiple_models_and_steps(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, "small", 1000)
    _make_checkpoint(tmp_path, "small", 2000)
    _make_checkpoint(tmp_path, "small_sft", 500)
    _make_checkpoint(tmp_path, "moe", 100)
    result = export_overview(str(tmp_path))
    models = {c["model"] for c in result["checkpoints"]}
    assert models == {"small", "small_sft", "moe"}
    assert len(result["checkpoints"]) == 4


def test_checkpoints_newest_first_within_model(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, "small", 1000)
    _make_checkpoint(tmp_path, "small", 3000)
    _make_checkpoint(tmp_path, "small", 2000)
    result = export_overview(str(tmp_path))
    small_steps = [c["step"] for c in result["checkpoints"] if c["model"] == "small"]
    assert small_steps == [3000, 2000, 1000]


def test_non_step_dirs_ignored(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, "small", 1000)
    (tmp_path / "checkpoints" / "small" / "latest").write_text(
        "step_00001000", encoding="utf-8"
    )
    (tmp_path / "checkpoints" / "small" / "exports").mkdir()
    (tmp_path / "checkpoints" / "small" / "step_notanint").mkdir()
    result = export_overview(str(tmp_path))
    steps = [c["step"] for c in result["checkpoints"]]
    assert steps == [1000]


def test_formats_non_empty_shape(tmp_path: Path) -> None:
    result = export_overview(str(tmp_path))
    formats = result["formats"]
    assert isinstance(formats, list)
    assert len(formats) > 0
    for fmt in formats:
        assert set(fmt) == {"key", "label", "desc"}
        assert isinstance(fmt["key"], str) and fmt["key"]
        assert isinstance(fmt["label"], str) and fmt["label"]
        assert isinstance(fmt["desc"], str) and fmt["desc"]


def test_formats_keys_match_export_script(tmp_path: Path) -> None:
    result = export_overview(str(tmp_path))
    keys = {f["key"] for f in result["formats"]}
    # These mirror scripts/export_model.py _ACTION_MAP — real supported actions.
    assert {"gguf-f16", "gguf-q8", "gguf-q4", "ollama", "quantize"} <= keys


def test_existing_empty_when_nothing_exported(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, "small", 1000)
    result = export_overview(str(tmp_path))
    assert result["existing"] == []


def test_existing_gguf_discovered(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    gguf = exports / "model.gguf"
    gguf.write_bytes(b"GGUF" * 100)
    result = export_overview(str(tmp_path))
    existing = result["existing"]
    assert len(existing) == 1
    entry = existing[0]
    assert entry["format"] == "gguf"
    assert entry["size_bytes"] == 400
    assert Path(entry["path"]).name == "model.gguf"
    assert isinstance(entry["mtime"], float)


def test_existing_modelfile_and_int8(tmp_path: Path) -> None:
    exports = tmp_path / "checkpoints" / "small" / "exports"
    exports.mkdir(parents=True)
    (exports / "Modelfile").write_text("FROM cola-coder-f16.gguf\n", encoding="utf-8")
    (exports / "cola-coder-int8.pt").write_bytes(b"\x00" * 50)
    result = export_overview(str(tmp_path))
    formats = {e["format"] for e in result["existing"]}
    assert formats == {"ollama", "int8"}


def test_existing_newest_first(tmp_path: Path) -> None:
    import os
    import time

    exports = tmp_path / "exports"
    exports.mkdir()
    older = exports / "old.gguf"
    newer = exports / "new.gguf"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))
    result = export_overview(str(tmp_path))
    paths = [Path(e["path"]).name for e in result["existing"]]
    assert paths == ["new.gguf", "old.gguf"]


def test_missing_root_no_crash(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    result = export_overview(str(missing))
    assert result["checkpoints"] == []
    assert result["existing"] == []
    assert len(result["formats"]) > 0


def test_error_only_on_broken_input() -> None:
    # A non-str/path that breaks Path() construction triggers the error branch.
    result = export_overview(root=12345)  # type: ignore[arg-type]
    assert "error" in result
