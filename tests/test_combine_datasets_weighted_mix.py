"""BUG-105: combine_datasets.py non-interactive --datasets weighted-mix path.

`_run_weighted_mix` (reachable via `combine_datasets.py --datasets a:0.8 b:0.2`)
had three defects in an otherwise untested CLI path:
  1. non-2D inputs were `continue`-skipped, desyncing paths/arrays/weights so
     later weights applied to the wrong dataset;
  2. `row_counts[-1] = total - sum(rest)` could go NEGATIVE when many tiny
     datasets were each clamped to >=1, crashing np.random.choice(size<0);
  3. no chunk_size compatibility check (np.concatenate fails opaquely).

These import the script module and drive the function directly.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "combine_datasets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("combine_datasets_wmix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path: Path, n_rows: int, cols: int = 2, fill: int = 1) -> str:
    arr = np.full((n_rows, cols), fill, dtype=np.uint16)
    np.save(str(path), arr)
    return str(path) + ".npy"


class TestParseWeightedDatasets:
    def test_default_weight(self):
        mod = _load_module()
        paths, weights = mod._parse_weighted_datasets(["a.npy", "b.npy"])
        assert paths == ["a.npy", "b.npy"]
        assert weights == [1.0, 1.0]

    def test_explicit_weight(self):
        mod = _load_module()
        paths, weights = mod._parse_weighted_datasets(["a.npy:0.8", "b.npy:0.2"])
        assert weights == [0.8, 0.2]

    def test_windows_path_splits_on_last_colon(self):
        mod = _load_module()
        paths, weights = mod._parse_weighted_datasets([r"C:\data\a.npy:0.3"])
        assert paths == [r"C:\data\a.npy"]
        assert weights == [0.3]

    def test_invalid_weight_defaults_to_one(self):
        mod = _load_module()
        _, weights = mod._parse_weighted_datasets(["a.npy:abc", "b.npy:-1"])
        assert weights == [1.0, 1.0]


class TestRunWeightedMix:
    def test_basic_mix_row_count_and_shape(self, tmp_path):
        mod = _load_module()
        a = _save(tmp_path / "a", 3, fill=1)
        b = _save(tmp_path / "b", 3, fill=2)
        out = tmp_path / "mixed.npy"
        mod._run_weighted_mix([a, b], [1.0, 1.0], str(out))
        result = np.load(str(out))
        # Output total == sum of input rows.
        assert result.shape == (6, 2)

    def test_many_tiny_datasets_no_negative_crash(self, tmp_path):
        # One large-weight dataset + 9 tiny ones. The old code forced the LAST
        # bucket to total - sum(rest), which went negative here and crashed
        # np.random.choice(size<0).
        mod = _load_module()
        paths = [_save(tmp_path / "big", 2, fill=9)]
        weights = [100.0]
        for i in range(9):
            paths.append(_save(tmp_path / f"t{i}", 1, fill=i))
            weights.append(1.0)
        out = tmp_path / "mix_many.npy"
        mod._run_weighted_mix(paths, weights, str(out))  # must not raise
        result = np.load(str(out))
        total_rows = 2 + 9  # sum of input rows
        assert result.shape == (total_rows, 2)

    def test_chunk_size_mismatch_aborts(self, tmp_path):
        mod = _load_module()
        a = _save(tmp_path / "a", 3, cols=2)
        b = _save(tmp_path / "b", 3, cols=4)  # different chunk_size
        out = tmp_path / "bad.npy"
        with pytest.raises(SystemExit):
            mod._run_weighted_mix([a, b], [1.0, 1.0], str(out))

    def test_skewed_weights_sums_to_total(self, tmp_path):
        # 90/10 split over datasets with plenty of rows: total preserved.
        mod = _load_module()
        a = _save(tmp_path / "a", 20, fill=1)
        b = _save(tmp_path / "b", 20, fill=2)
        out = tmp_path / "skew.npy"
        mod._run_weighted_mix([a, b], [0.9, 0.1], str(out))
        result = np.load(str(out))
        assert result.shape == (40, 2)
