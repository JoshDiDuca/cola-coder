"""TOOL-014: estimate_unique_tokens must never report an IMPOSSIBLE count.

The unique-token estimator extrapolated a sample's distinct count by
sqrt(data/sample) and capped at 2**20 — but .npy token data is uint16, so there
can be at most 65536 distinct ids. The estimate could therefore report far more
unique tokens than the dtype can represent. Now it caps at the dtype's range and
never falls below what the sample actually observed.

The estimator now lives in the shared ``cola_coder.data.stats`` library (used by
both scripts/data_stats.py and the web UI).
"""

import numpy as np

from cola_coder.data.stats import estimate_unique_tokens


class TestEstimateUniqueTokens:
    def test_small_array_is_exact(self):
        arr = np.array([1, 1, 2, 3, 3, 3], dtype=np.uint16)
        assert estimate_unique_tokens(arr, sample_size=1000) == 3

    def test_estimate_never_exceeds_dtype_range(self):
        # uint8 → at most 256 distinct values. Force the sqrt-scale to over-shoot
        # (small sample, large array) and confirm it's clamped to <= 256.
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=20_000, dtype=np.uint8)
        est = estimate_unique_tokens(arr, sample_size=200)
        assert est <= 256, f"reported {est} distinct uint8 values (max possible 256)"

    def test_uint16_capped_at_65536(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 65536, size=2_000_000, dtype=np.uint16)
        est = estimate_unique_tokens(arr, sample_size=10_000)
        assert est <= 65536

    def test_estimate_not_below_sample_observed(self):
        # An array of all-distinct values: the sample alone sees `sample_size`
        # distinct, so the estimate must be at least that.
        arr = np.arange(50_000, dtype=np.uint16)
        est = estimate_unique_tokens(arr, sample_size=5_000)
        assert est >= 5_000  # never under-reports below the observed sample
        assert est <= 65536  # ...and never impossible
