"""DATA-057: SoftDedup reweighting — down-weight near-duplicates instead of dropping."""

import numpy as np
import pytest

from cola_coder.data.dedup import CrossDatasetDeduplicator

datasketch = pytest.importorskip("datasketch")  # skip if not installed


def _dedup():
    return CrossDatasetDeduplicator(method="minhash", threshold=0.8, num_perm=128, ngram_size=5)


def test_identical_rows_share_weight():
    # 3 identical chunks + 1 distinct. The identical trio should each get ~1/3;
    # the distinct one should get ~1.0.
    base = np.arange(100, 140, dtype=np.int64)          # 40 distinct tokens
    other = np.arange(500, 540, dtype=np.int64)         # clearly different
    data = np.stack([base, base.copy(), base.copy(), other])
    w = _dedup().compute_soft_weights(data)
    assert w.shape == (4,)
    # trio down-weighted to ~1/3
    assert all(abs(w[i] - 1 / 3) < 1e-6 for i in range(3)), w
    # distinct row keeps full weight
    assert abs(w[3] - 1.0) < 1e-6, w


def test_all_unique_rows_full_weight():
    data = np.stack([np.arange(i * 50, i * 50 + 40, dtype=np.int64) for i in range(4)])
    w = _dedup().compute_soft_weights(data)
    assert np.allclose(w, 1.0)


def test_empty_data():
    w = _dedup().compute_soft_weights(np.empty((0, 40), dtype=np.int64))
    assert w.shape == (0,)


def test_weights_reduce_effective_count():
    # A cluster of k identical chunks contributes ~1 sample's worth of weight total.
    base = np.arange(200, 250, dtype=np.int64)
    data = np.stack([base.copy() for _ in range(5)])
    w = _dedup().compute_soft_weights(data)
    assert abs(w.sum() - 1.0) < 1e-5  # 5 * (1/5) == 1
