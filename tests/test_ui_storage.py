"""Hermetic tests for cola_coder.ui.storage_view.read_storage.

These build a fake project root under tmp_path with a real-ish storage.yaml,
a populated data dir and checkpoints dir, then assert the summary surfaces the
right keys, existence flags, and byte sizes. One test runs against the real
configs/storage.yaml in the repo to confirm it parses without error.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cola_coder.ui.storage_view import _dir_size, read_storage


def _write_storage_yaml(root: Path, storage: dict) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "storage.yaml").write_text(
        yaml.safe_dump({"storage": storage}), encoding="utf-8"
    )


def _make_project(tmp_path: Path) -> Path:
    """Build a fake project root with relative storage paths + content."""
    _write_storage_yaml(
        tmp_path,
        {
            "data_dir": "./data",
            "checkpoints_dir": "./checkpoints",
            "tokenizer_path": "./tokenizer.json",
            "cache_dir": "./cache",
            "hf_cache_dir": "",
        },
    )

    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "raw.txt").write_text("hello world", encoding="utf-8")
    (data_dir / "processed" / "train_data.npy").write_text("x" * 100, encoding="utf-8")

    ckpt = tmp_path / "checkpoints" / "small" / "step_00000100"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_text("y" * 50, encoding="utf-8")

    (tmp_path / "tokenizer.json").write_text('{"model": {}}', encoding="utf-8")
    return tmp_path


def test_returns_resolved_yaml_path(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    result = read_storage(str(root))
    assert "error" not in result
    assert result["path"] == str(root / "configs" / "storage.yaml")


def test_raw_is_full_safe_loaded_yaml(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    result = read_storage(str(root))
    assert result["raw"]["storage"]["cache_dir"] == "./cache"
    assert result["raw"]["storage"]["hf_cache_dir"] == ""


def test_surfaces_key_paths_resolved_relative_to_root(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    result = read_storage(str(root))
    assert result["tokenizer_path"] == str(root / "tokenizer.json")
    assert result["data_dir"] == str(root / "data")
    # YAML key is "checkpoints_dir" but contract field is "checkpoint_dir".
    assert result["checkpoint_dir"] == str(root / "checkpoints")


def test_entries_cover_expected_locations(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    result = read_storage(str(root))
    names = {e["name"] for e in result["entries"]}
    assert names == {"tokenizer", "data_dir", "checkpoints_dir", "data_processed"}


def test_populated_dir_has_positive_size(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    result = read_storage(str(root))
    entries = {e["name"]: e for e in result["entries"]}

    data_entry = entries["data_dir"]
    assert data_entry["exists"] is True
    assert isinstance(data_entry["size_bytes"], int)
    assert data_entry["size_bytes"] > 0

    tok_entry = entries["tokenizer"]
    assert tok_entry["exists"] is True
    assert tok_entry["size_bytes"] > 0  # file size


def test_missing_dir_is_none_size(tmp_path: Path) -> None:
    # storage.yaml points at dirs that do not exist on disk.
    _write_storage_yaml(
        tmp_path,
        {
            "data_dir": "./nope_data",
            "checkpoints_dir": "./nope_ckpts",
            "tokenizer_path": "./nope_tok.json",
        },
    )
    result = read_storage(str(tmp_path))
    entries = {e["name"]: e for e in result["entries"]}
    for name in ("data_dir", "checkpoints_dir", "tokenizer", "data_processed"):
        assert entries[name]["exists"] is False
        assert entries[name]["size_bytes"] is None


def test_blank_keys_yield_none(tmp_path: Path) -> None:
    _write_storage_yaml(tmp_path, {"data_dir": "", "checkpoints_dir": "  "})
    result = read_storage(str(tmp_path))
    assert result["data_dir"] is None
    assert result["checkpoint_dir"] is None
    assert result["tokenizer_path"] is None


def test_absolute_paths_kept(tmp_path: Path) -> None:
    abs_data = tmp_path / "elsewhere" / "data"
    abs_data.mkdir(parents=True)
    (abs_data / "f.bin").write_text("z" * 10, encoding="utf-8")
    _write_storage_yaml(tmp_path, {"data_dir": str(abs_data)})
    result = read_storage(str(tmp_path))
    assert result["data_dir"] == str(abs_data)


def test_walk_cap_respected(tmp_path: Path) -> None:
    # Create more files than a tiny cap; size must reflect only capped count.
    big = tmp_path / "big"
    big.mkdir()
    n_files = 50
    for i in range(n_files):
        (big / f"f{i}.txt").write_text("a", encoding="utf-8")  # 1 byte each

    full = _dir_size(str(big), cap=10000)
    capped = _dir_size(str(big), cap=5)
    assert full == n_files  # 50 files x 1 byte
    assert capped is not None
    assert capped <= 5  # bailed after counting at most 5 files


def test_missing_storage_yaml_returns_error(tmp_path: Path) -> None:
    result = read_storage(str(tmp_path))  # no configs/storage.yaml
    assert "error" in result


def test_garbage_yaml_returns_error(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "storage.yaml").write_text(
        "this: : : not valid\n  - [unbalanced", encoding="utf-8"
    )
    result = read_storage(str(tmp_path))
    assert "error" in result


def test_yaml_without_storage_block_returns_error(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "storage.yaml").write_text(
        "something_else: 1\n", encoding="utf-8"
    )
    result = read_storage(str(tmp_path))
    assert "error" in result


def test_real_storage_yaml_parses(tmp_path: Path) -> None:
    # Resolve the repo root from this test file: tests/ -> repo root.
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "configs" / "storage.yaml").is_file()
    result = read_storage(str(repo_root))
    assert "error" not in result
    assert result["data_dir"] is not None
    assert isinstance(result["raw"], dict)
    assert {e["name"] for e in result["entries"]} == {
        "tokenizer",
        "data_dir",
        "checkpoints_dir",
        "data_processed",
    }
