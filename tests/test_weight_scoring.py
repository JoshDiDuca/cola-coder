"""DATA-044: shared quality-weight scoring for tokenized .npy datasets.

compute_chunk_weights must align 1:1 with chunks (so the .weights.npy lines up
with the data after dedup), and score_npy_to_weights must honor the code_scorer
feature gate (returning (None, None) rather than writing a meaningless sidecar).
"""

import numpy as np
import pytest

from cola_coder.data.weight_scoring import compute_chunk_weights, score_npy_to_weights

CHUNK = 8


class _FakeTokenizer:
    """decode() echoes the chunk's first token id so a test can recover it."""

    def decode(self, ids):
        return f"v{int(ids[0])}"


class _FakeScorer:
    def score(self, text):
        return text

    def score_to_weight(self, s):
        # Recover the encoded first-token id → a deterministic per-chunk weight.
        return float(int(s[1:])) * 0.001


def _make_npy(tmp_path, first_tokens):
    """Build a .npy where chunk k starts with first_tokens[k]."""
    data = np.array(
        [[v] + [0] * (CHUNK - 1) for v in first_tokens], dtype=np.uint16
    )
    p = tmp_path / "data.npy"
    np.save(str(p), data)
    return str(p)


class TestComputeChunkWeights:
    def test_weights_align_one_to_one(self, tmp_path):
        first = [10, 250, 7, 99, 33]
        path = _make_npy(tmp_path, first)
        weights = compute_chunk_weights(path, _FakeTokenizer(), _FakeScorer())
        assert weights.shape == (len(first),)
        assert weights.dtype == np.float32
        for i, v in enumerate(first):
            assert weights[i] == pytest.approx(v * 0.001, abs=1e-6)


class TestScoreNpyToWeights:
    def test_writes_aligned_sidecar(self, tmp_path, monkeypatch):
        import cola_coder.features.code_scorer as cs

        monkeypatch.setattr(cs, "is_enabled", lambda: True)
        monkeypatch.setattr(cs, "CodeScorer", _FakeScorer)
        first = [5, 120, 64]
        path = _make_npy(tmp_path, first)
        wpath, weights = score_npy_to_weights(path, _FakeTokenizer())
        assert wpath is not None and wpath.endswith(".weights.npy")
        on_disk = np.load(wpath)
        assert np.allclose(on_disk, weights)
        assert len(on_disk) == len(first)

    def test_feature_disabled_returns_none(self, tmp_path, monkeypatch):
        import cola_coder.features.code_scorer as cs

        monkeypatch.setattr(cs, "is_enabled", lambda: False)
        path = _make_npy(tmp_path, [1, 2, 3])
        wpath, weights = score_npy_to_weights(path, _FakeTokenizer())
        assert wpath is None and weights is None
        # No meaningless sidecar written.
        assert not (tmp_path / "data.weights.npy").exists()
