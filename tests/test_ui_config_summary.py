"""Tests for ``config_summary`` (UI-103) — grouped YAML hyperparameter summary.

Covers the contract in ``src/cola_coder/ui/config_summary_view.py``:
group emission (Model / Training / Data / Checkpoint, only when present), the
Derived effective-batch item, boundary string coercion of all values, missing
files (``exists=False``, not an error), and malformed YAML (``{"error": str}``).
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.config_summary_view import config_summary

_FULL_CONFIG = """\
model:
  dim: 768
  n_layers: 12
  n_heads: 12
  n_kv_heads: 4
  rope_theta: 500000
  dropout: 0.0
  qk_norm: true
training:
  batch_size: 24
  gradient_accumulation: 4
  learning_rate: 6.0e-4
  warmup_steps: 2000
  precision: bf16
  gradient_checkpointing: false
data:
  dataset: bigcode/starcoderdata
  languages:
    - typescript
    - python
  fim_rate: 0.1
checkpoint:
  save_every: 1000
  output_dir: checkpoints/small
"""


def _write(tmp_path: Path, name: str, text: str) -> str:
    """Write ``text`` to ``tmp_path/name`` and return its path string."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestFullConfig:
    """A complete config exercising every group plus Derived."""

    def test_exists_and_group_order(self, tmp_path: Path) -> None:
        """All four sections present → exists True; Derived appended last."""
        result = config_summary(_write(tmp_path, "small.yaml", _FULL_CONFIG))

        assert result["exists"] is True
        assert "error" not in result
        titles = [group["title"] for group in result["groups"]]
        assert titles == ["Model", "Training", "Data", "Checkpoint", "Derived"]

    def test_derived_effective_batch(self, tmp_path: Path) -> None:
        """Derived group holds one item: effective batch = 24 × 4 = 96."""
        result = config_summary(_write(tmp_path, "small.yaml", _FULL_CONFIG))

        derived = next(g for g in result["groups"] if g["title"] == "Derived")
        assert derived["items"] == [{"label": "effective batch", "value": "96"}]

    def test_path_and_name_round_trip(self, tmp_path: Path) -> None:
        """``path`` round-trips and ``name`` is the file basename."""
        path = _write(tmp_path, "small.yaml", _FULL_CONFIG)
        result = config_summary(path)

        assert result["path"] == str(Path(path))
        assert result["name"] == "small.yaml"


class TestValueCoercion:
    """Every emitted value must be a (non-empty) string at the boundary."""

    def test_all_values_are_strings(self, tmp_path: Path) -> None:
        """Across all groups, ``value`` is always a ``str``."""
        result = config_summary(_write(tmp_path, "small.yaml", _FULL_CONFIG))

        for group in result["groups"]:
            for item in group["items"]:
                assert isinstance(item["value"], str)
                assert isinstance(item["label"], str)

    def test_numeric_dim_coerced(self, tmp_path: Path) -> None:
        """An integer model.dim=768 renders as the string ``"768"``."""
        result = config_summary(_write(tmp_path, "small.yaml", _FULL_CONFIG))

        model = next(g for g in result["groups"] if g["title"] == "Model")
        dim_item = next(i for i in model["items"] if i["label"] == "dim")
        assert dim_item["value"] == "768"

    def test_learning_rate_item_present_nonempty(self, tmp_path: Path) -> None:
        """The learning-rate item exists with a non-empty string value."""
        result = config_summary(_write(tmp_path, "small.yaml", _FULL_CONFIG))

        training = next(g for g in result["groups"] if g["title"] == "Training")
        lr_item = next(i for i in training["items"] if i["label"] == "learning rate")
        assert isinstance(lr_item["value"], str)
        assert lr_item["value"] != ""


class TestBooleanRendering:
    """Booleans render as lowercase ``true`` / ``false``."""

    def test_true_and_false(self, tmp_path: Path) -> None:
        """qk_norm:true → "true"; gradient_checkpointing:false → "false"."""
        result = config_summary(_write(tmp_path, "small.yaml", _FULL_CONFIG))

        model = next(g for g in result["groups"] if g["title"] == "Model")
        qk = next(i for i in model["items"] if i["label"] == "QK-norm")
        assert qk["value"] == "true"

        training = next(g for g in result["groups"] if g["title"] == "Training")
        gc = next(i for i in training["items"] if i["label"] == "grad checkpointing")
        assert gc["value"] == "false"


class TestListRendering:
    """Lists render joined with ``", "``."""

    def test_languages_joined(self, tmp_path: Path) -> None:
        """data.languages:[ts, py] → "typescript, python"."""
        result = config_summary(_write(tmp_path, "small.yaml", _FULL_CONFIG))

        data = next(g for g in result["groups"] if g["title"] == "Data")
        langs = next(i for i in data["items"] if i["label"] == "languages")
        assert langs["value"] == "typescript, python"


class TestPartialConfig:
    """Sections that are absent produce no group; Derived needs training."""

    def test_model_only_no_derived(self, tmp_path: Path) -> None:
        """A model-only config → just the Model group, no Derived."""
        result = config_summary(_write(tmp_path, "m.yaml", "model:\n  dim: 768\n"))

        titles = [g["title"] for g in result["groups"]]
        assert titles == ["Model"]
        assert "Derived" not in titles

    def test_derived_omitted_without_grad_accum(self, tmp_path: Path) -> None:
        """Missing gradient_accumulation → no Derived group."""
        text = "training:\n  batch_size: 24\n  learning_rate: 6.0e-4\n"
        result = config_summary(_write(tmp_path, "t.yaml", text))

        titles = [g["title"] for g in result["groups"]]
        assert "Training" in titles
        assert "Derived" not in titles

    def test_derived_omitted_when_non_int(self, tmp_path: Path) -> None:
        """Non-int batch_size (string) → no Derived group."""
        text = 'training:\n  batch_size: "auto"\n  gradient_accumulation: 4\n'
        result = config_summary(_write(tmp_path, "t.yaml", text))

        titles = [g["title"] for g in result["groups"]]
        assert "Derived" not in titles


class TestMissingFile:
    """A missing path is not an error — it is ``exists=False``."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """Nonexistent path → exists False, empty groups, no error, basename name."""
        missing = str(tmp_path / "does_not_exist.yaml")
        result = config_summary(missing)

        assert result["exists"] is False
        assert result["groups"] == []
        assert "error" not in result
        assert result["name"] == "does_not_exist.yaml"
        assert result["path"] == str(Path(missing))


class TestMalformedYaml:
    """Unparseable or non-mapping YAML returns ``{"error": str}``."""

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Garbage YAML → result has a string ``error``."""
        result = config_summary(_write(tmp_path, "bad.yaml", "::: not yaml :::"))

        assert "error" in result
        assert isinstance(result["error"], str)

    def test_top_level_list(self, tmp_path: Path) -> None:
        """A top-level list (non-mapping) → result has a string ``error``."""
        result = config_summary(_write(tmp_path, "list.yaml", "- a\n- b\n"))

        assert "error" in result
        assert isinstance(result["error"], str)
