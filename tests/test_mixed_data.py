"""Tests for the Prepare Mixed Data fix.

The menu item used to pass --mix-code/--mix-text/--mix-math to
prepare_data.py, which has no such args, so it always errored with
"unrecognized arguments". It now writes a derived data_sources config and
runs the real multi-source collector (collect_data.py). The dead
MixedDataset class (training-time sampler, never wired) was removed.
"""

from pathlib import Path

import yaml

from cola_coder.data.mixing import write_weighted_data_sources

ROOT = Path(__file__).parent.parent

_BASE = {
    "sources": {
        "code": {"dataset": "bigcode/x", "weight": 0.7, "enabled": True,
                 "languages": ["python"]},
        "text": {"dataset": "hf/text", "weight": 0.2, "enabled": True,
                 "min_length": 100},
        "math": {"dataset": "hf/math", "weight": 0.1, "enabled": True},
    },
    "github": {"min_stars": 50},
}


class TestWriteWeightedDataSources:
    def _base(self, tmp_path):
        p = tmp_path / "base.yaml"
        p.write_text(yaml.safe_dump(_BASE), encoding="utf-8")
        return p

    def test_overrides_weights(self, tmp_path):
        base = self._base(tmp_path)
        out = write_weighted_data_sources(
            {"code": 0.5, "text": 0.3, "math": 0.2},
            tmp_path / "auto" / "mixed.yaml", base_path=base,
        )
        cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert cfg["sources"]["code"]["weight"] == 0.5
        assert cfg["sources"]["text"]["weight"] == 0.3
        assert cfg["sources"]["math"]["weight"] == 0.2

    def test_preserves_other_fields(self, tmp_path):
        base = self._base(tmp_path)
        out = write_weighted_data_sources(
            {"code": 1.0, "text": 0.0, "math": 0.0},
            tmp_path / "mixed.yaml", base_path=base,
        )
        cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
        # Non-weight fields copied through
        assert cfg["sources"]["code"]["dataset"] == "bigcode/x"
        assert cfg["sources"]["code"]["languages"] == ["python"]
        assert cfg["github"]["min_stars"] == 50

    def test_zero_weight_disables_source(self, tmp_path):
        base = self._base(tmp_path)
        out = write_weighted_data_sources(
            {"code": 1.0, "text": 0.0, "math": 0.0},
            tmp_path / "mixed.yaml", base_path=base,
        )
        cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert cfg["sources"]["code"]["enabled"] is True
        assert cfg["sources"]["text"]["enabled"] is False
        assert cfg["sources"]["math"]["enabled"] is False

    def test_creates_parent_dirs(self, tmp_path):
        base = self._base(tmp_path)
        out = write_weighted_data_sources(
            {"code": 0.7, "text": 0.2, "math": 0.1},
            tmp_path / "deep" / "nested" / "mixed.yaml", base_path=base,
        )
        assert out.exists()


class TestMenuWiring:
    def test_menu_runs_collect_data_not_broken_prepare(self):
        text = (ROOT / "src" / "cola_coder" / "features" / "menus"
                / "data_menu.py").read_text(encoding="utf-8")
        # Runs the real collector with a derived data-sources config...
        assert "write_weighted_data_sources" in text
        assert '_run_script("collect_data.py"' in text
        assert '"--data-sources"' in text
        # ...and the broken mix flags are no longer PASSED as args (quoted-arg
        # form). A historical mention in a comment is fine, so check the literal
        # argument form, not any occurrence of the string.
        assert '"--mix-code"' not in text
        assert '"--mix-text"' not in text
        assert '"--mix-math"' not in text


class TestMixedDatasetRemoved:
    def test_class_gone_from_dataset_module(self):
        text = (ROOT / "src" / "cola_coder" / "data"
                / "dataset.py").read_text(encoding="utf-8")
        assert "class MixedDataset" not in text

    def test_not_importable(self):
        import cola_coder.data.dataset as ds
        assert not hasattr(ds, "MixedDataset")
