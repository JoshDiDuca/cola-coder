"""Tests for DatasetResolver.

Covers path resolution, name derivation, metadata persistence, and
edge cases (missing YAML, unsafe characters, empty config).

Run:
    cd "C:/Users/josh/ai research/cola-coder"
    .venv/Scripts/pytest tests/test_dataset_resolver.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from cola_coder.data.dataset_resolver import DatasetResolver


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_sources(tmp_path: Path, sources: dict) -> Path:
    """Write a minimal data_sources.yaml and return its path."""
    cfg_path = tmp_path / "data_sources.yaml"
    cfg_path.write_text(yaml.dump({"sources": sources}), encoding="utf-8")
    return cfg_path


def _mock_storage(data_dir: Path) -> MagicMock:
    """Return a mock storage config with the given data_dir."""
    mock = MagicMock()
    mock.data_dir = str(data_dir)
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# get_dataset_name
# ─────────────────────────────────────────────────────────────────────────────


class TestGetDatasetName:
    def test_typical_full_config(self, tmp_path: Path) -> None:
        """Languages sorted + enabled sources give correct name."""
        cfg = _write_sources(
            tmp_path,
            {
                "code": {
                    "enabled": True,
                    "languages": ["typescript", "javascript", "python"],
                },
                "text": {"enabled": True},
                "math": {"enabled": True},
            },
        )
        name = DatasetResolver.get_dataset_name(cfg)
        # yaml.dump sorts dict keys alphabetically: math < text
        assert name == "javascript-python-typescript-math-text"

    def test_languages_sorted_alphabetically(self, tmp_path: Path) -> None:
        """Code languages are sorted regardless of YAML order."""
        cfg = _write_sources(
            tmp_path,
            {
                "code": {
                    "enabled": True,
                    "languages": ["rust", "go", "zig"],
                },
            },
        )
        name = DatasetResolver.get_dataset_name(cfg)
        assert name == "go-rust-zig"

    def test_disabled_sources_excluded(self, tmp_path: Path) -> None:
        """Disabled sources do not appear in the name."""
        cfg = _write_sources(
            tmp_path,
            {
                "code": {"enabled": True, "languages": ["python"]},
                "text": {"enabled": False},
                "math": {"enabled": True},
            },
        )
        name = DatasetResolver.get_dataset_name(cfg)
        assert name == "python-math"

    def test_code_source_disabled(self, tmp_path: Path) -> None:
        """If code source is disabled, no languages appear."""
        cfg = _write_sources(
            tmp_path,
            {
                "code": {"enabled": False, "languages": ["python"]},
                "text": {"enabled": True},
            },
        )
        name = DatasetResolver.get_dataset_name(cfg)
        assert name == "text"

    def test_unsafe_characters_sanitized(self, tmp_path: Path) -> None:
        """Language names like c# / c++ are sanitized with underscores."""
        cfg = _write_sources(
            tmp_path,
            {
                "code": {
                    "enabled": True,
                    "languages": ["c#", "c++"],
                },
            },
        )
        name = DatasetResolver.get_dataset_name(cfg)
        # c# → c_, c++ → c__ ; sorted alphabetically
        assert "#" not in name
        assert "+" not in name

    def test_missing_yaml_returns_default(self, tmp_path: Path) -> None:
        """Non-existent YAML file returns 'default'."""
        name = DatasetResolver.get_dataset_name(tmp_path / "nonexistent.yaml")
        assert name == "default"

    def test_empty_sources_returns_default(self, tmp_path: Path) -> None:
        """All sources disabled → 'default'."""
        cfg = _write_sources(
            tmp_path,
            {
                "code": {"enabled": False},
                "text": {"enabled": False},
            },
        )
        name = DatasetResolver.get_dataset_name(cfg)
        assert name == "default"

    def test_non_code_sources_preserve_definition_order(self, tmp_path: Path) -> None:
        """Non-code sources appear in YAML definition order (not sorted)."""
        cfg_path = tmp_path / "data_sources.yaml"
        # Use explicit ordering: math before text
        raw = (
            "sources:\n"
            "  math:\n    enabled: true\n"
            "  text:\n    enabled: true\n"
        )
        cfg_path.write_text(raw, encoding="utf-8")
        name = DatasetResolver.get_dataset_name(cfg_path)
        assert name == "math-text"

    def test_config_path_overrides_code_languages(self, tmp_path: Path) -> None:
        """Model config data.languages overrides data_sources.yaml languages."""
        cfg = _write_sources(
            tmp_path,
            {
                "code": {
                    "enabled": True,
                    "languages": ["python", "javascript", "typescript"],
                },
                "text": {"enabled": True},
            },
        )
        model_cfg = tmp_path / "model.yaml"
        model_cfg.write_text(
            "data:\n  languages:\n    - typescript\n", encoding="utf-8"
        )
        name = DatasetResolver.get_dataset_name(cfg, config_path=model_cfg)
        assert name == "typescript-text"

    def test_config_path_missing_languages_falls_back(self, tmp_path: Path) -> None:
        """Model config without data.languages falls back to data_sources.yaml."""
        cfg = _write_sources(
            tmp_path,
            {"code": {"enabled": True, "languages": ["go"]}},
        )
        model_cfg = tmp_path / "model.yaml"
        model_cfg.write_text("model:\n  layers: 12\n", encoding="utf-8")
        name = DatasetResolver.get_dataset_name(cfg, config_path=model_cfg)
        assert name == "go"

    def test_config_path_nonexistent_falls_back(self, tmp_path: Path) -> None:
        """Non-existent model config falls back to data_sources.yaml languages."""
        cfg = _write_sources(
            tmp_path,
            {"code": {"enabled": True, "languages": ["rust"]}},
        )
        name = DatasetResolver.get_dataset_name(cfg, config_path=tmp_path / "missing.yaml")
        assert name == "rust"


# ─────────────────────────────────────────────────────────────────────────────
# get_dataset_dir
# ─────────────────────────────────────────────────────────────────────────────


class TestGetDatasetDir:
    def test_returns_dataset_subdir(self, tmp_path: Path) -> None:
        """Returns storage.data_dir / dataset_name."""
        cfg = _write_sources(
            tmp_path,
            {"code": {"enabled": True, "languages": ["python"]}},
        )
        data_root = tmp_path / "data"
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            result = DatasetResolver.get_dataset_dir(cfg)

        assert result == data_root / "python"

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Directory is created on first call."""
        cfg = _write_sources(
            tmp_path,
            {"code": {"enabled": True, "languages": ["go"]}},
        )
        data_root = tmp_path / "data"
        assert not data_root.exists()

        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            result = DatasetResolver.get_dataset_dir(cfg)

        assert result.is_dir()

    def test_missing_yaml_uses_default_subdir(self, tmp_path: Path) -> None:
        """Missing YAML → dataset name 'default' → data_dir/default."""
        data_root = tmp_path / "data"
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            result = DatasetResolver.get_dataset_dir(tmp_path / "missing.yaml")

        assert result.name == "default"
        assert result.is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# find_dataset_npys (BUG-117 / DATA-048 — single source of "where's the data")
# ─────────────────────────────────────────────────────────────────────────────


class TestFindDatasetNpys:
    def test_finds_in_per_dataset_dir_excluding_sidecars(self, tmp_path: Path) -> None:
        cfg = _write_sources(
            tmp_path, {"code": {"enabled": True, "languages": ["python"]}}
        )
        data_root = tmp_path / "data"
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            ds = DatasetResolver.get_dataset_dir(cfg)  # data_root/python (created)
            (ds / "code_data.npy").write_bytes(b"\x00")
            (ds / "code_data.weights.npy").write_bytes(b"\x00")  # sidecar
            (ds / "code_data.scores.npy").write_bytes(b"\x00")   # sidecar
            found = DatasetResolver.find_dataset_npys(cfg)

        assert [p.name for p in found] == ["code_data.npy"]

    def test_empty_when_nothing_prepared(self, tmp_path: Path) -> None:
        cfg = _write_sources(
            tmp_path, {"code": {"enabled": True, "languages": ["python"]}}
        )
        data_root = tmp_path / "data"
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            assert DatasetResolver.find_dataset_npys(cfg) == []

    def test_falls_back_to_legacy_processed_dir(self, tmp_path: Path) -> None:
        cfg = _write_sources(
            tmp_path, {"code": {"enabled": True, "languages": ["python"]}}
        )
        data_root = tmp_path / "data"
        legacy = data_root / "processed"
        legacy.mkdir(parents=True)
        (legacy / "train_data.npy").write_bytes(b"\x00")
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            # Per-dataset dir is empty → fall back to processed/.
            found = DatasetResolver.find_dataset_npys(cfg)

        assert [p.name for p in found] == ["train_data.npy"]


# ─────────────────────────────────────────────────────────────────────────────
# get_tokenizer_path / tokenizer_exists
# ─────────────────────────────────────────────────────────────────────────────


class TestGetTokenizerPath:
    def test_tokenizer_path_inside_dataset_dir(self, tmp_path: Path) -> None:
        """Tokenizer path is dataset_dir / 'tokenizer.json'."""
        cfg = _write_sources(
            tmp_path,
            {"code": {"enabled": True, "languages": ["typescript"]}},
        )
        data_root = tmp_path / "data"
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            tok = DatasetResolver.get_tokenizer_path(cfg)

        assert tok.name == "tokenizer.json"
        assert tok.parent.name == "typescript"

    def test_tokenizer_exists_false_when_missing(self, tmp_path: Path) -> None:
        """tokenizer_exists returns False when file is absent."""
        cfg = _write_sources(
            tmp_path,
            {"code": {"enabled": True, "languages": ["python"]}},
        )
        data_root = tmp_path / "data"
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            exists = DatasetResolver.tokenizer_exists(cfg)

        assert exists is False

    def test_tokenizer_exists_true_when_present(self, tmp_path: Path) -> None:
        """tokenizer_exists returns True when file is present."""
        cfg = _write_sources(
            tmp_path,
            {"code": {"enabled": True, "languages": ["python"]}},
        )
        data_root = tmp_path / "data"
        with patch(
            "cola_coder.data.dataset_resolver.get_storage_config",
            return_value=_mock_storage(data_root),
        ):
            tok_path = DatasetResolver.get_tokenizer_path(cfg)
            tok_path.write_text("{}", encoding="utf-8")
            exists = DatasetResolver.tokenizer_exists(cfg)

        assert exists is True


# ─────────────────────────────────────────────────────────────────────────────
# save_tokenizer_meta / get_tokenizer_meta
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenizerMeta:
    def _setup_tok_path(self, tmp_path: Path) -> Path:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        return dataset_dir / "tokenizer.json"

    def test_save_writes_meta_json(self, tmp_path: Path) -> None:
        tok_path = self._setup_tok_path(tmp_path)
        DatasetResolver.save_tokenizer_meta(
            tok_path,
            vocab_size=32768,
            sources=["code", "text"],
            num_samples=50000,
        )
        meta_path = tok_path.parent / "tokenizer_meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["vocab_size"] == 32768
        assert data["sources"] == ["code", "text"]
        assert data["num_samples"] == 50000
        assert "trained_at" in data

    def test_save_trained_at_is_iso_utc(self, tmp_path: Path) -> None:
        tok_path = self._setup_tok_path(tmp_path)
        DatasetResolver.save_tokenizer_meta(
            tok_path, vocab_size=1000, sources=[], num_samples=0
        )
        data = json.loads((tok_path.parent / "tokenizer_meta.json").read_text())
        # Should be parseable as ISO datetime with timezone
        from datetime import datetime
        dt = datetime.fromisoformat(data["trained_at"])
        assert dt.tzinfo is not None

    def test_get_returns_meta(self, tmp_path: Path) -> None:
        tok_path = self._setup_tok_path(tmp_path)
        DatasetResolver.save_tokenizer_meta(
            tok_path, vocab_size=8192, sources=["math"], num_samples=100
        )
        meta = DatasetResolver.get_tokenizer_meta(tok_path)
        assert meta["vocab_size"] == 8192
        assert meta["sources"] == ["math"]

    def test_get_returns_empty_when_missing(self, tmp_path: Path) -> None:
        tok_path = self._setup_tok_path(tmp_path)
        # No meta file written
        meta = DatasetResolver.get_tokenizer_meta(tok_path)
        assert meta == {}

    def test_get_returns_empty_on_corrupt_json(self, tmp_path: Path) -> None:
        tok_path = self._setup_tok_path(tmp_path)
        (tok_path.parent / "tokenizer_meta.json").write_text(
            "not valid json {{{{", encoding="utf-8"
        )
        meta = DatasetResolver.get_tokenizer_meta(tok_path)
        assert meta == {}

    def test_get_returns_empty_when_non_dict(self, tmp_path: Path) -> None:
        tok_path = self._setup_tok_path(tmp_path)
        (tok_path.parent / "tokenizer_meta.json").write_text(
            "[1, 2, 3]", encoding="utf-8"
        )
        meta = DatasetResolver.get_tokenizer_meta(tok_path)
        assert meta == {}
