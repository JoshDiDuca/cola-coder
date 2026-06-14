"""Tests for the UI config-browsing helpers (cola_coder.ui.configs)."""

from __future__ import annotations

from cola_coder.ui.configs import list_configs, read_config


def _make_configs(tmp_path):
    """Create a tmp configs dir with one valid and one invalid YAML file."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    good = configs_dir / "a.yaml"
    good.write_text("model:\n  dim: 512\n", encoding="utf-8")
    bad = configs_dir / "bad.yaml"
    bad.write_text(":\n  - [", encoding="utf-8")
    return configs_dir, good, bad


def test_list_configs_finds_both_sorted(tmp_path):
    configs_dir, _good, _bad = _make_configs(tmp_path)
    entries = list_configs(str(configs_dir))

    assert len(entries) == 2
    rels = [entry["rel"] for entry in entries]
    assert rels == sorted(rels)
    assert {entry["rel"] for entry in entries} == {"a.yaml", "bad.yaml"}

    for entry in entries:
        assert set(entry.keys()) == {"name", "path", "rel", "size_bytes", "mtime"}
        assert entry["size_bytes"] > 0


def test_list_configs_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert list_configs(str(missing)) == []


def test_read_config_valid(tmp_path):
    _configs_dir, good, _bad = _make_configs(tmp_path)
    result = read_config(str(good))

    assert result["path"] == str(good)
    assert "model" in result["content"]
    assert result["truncated"] is False
    assert result["parsed"]["model"]["dim"] == 512


def test_read_config_invalid_no_raise(tmp_path):
    _configs_dir, _good, bad = _make_configs(tmp_path)
    result = read_config(str(bad))

    assert result["parsed"] is None
    assert "content" in result
    assert "error" not in result


def test_read_config_missing_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    result = read_config(str(missing))

    assert "error" in result


def test_read_config_truncation(tmp_path):
    _configs_dir, good, _bad = _make_configs(tmp_path)
    result = read_config(str(good), max_chars=5)

    assert result["truncated"] is True
    assert len(result["content"]) == 5
