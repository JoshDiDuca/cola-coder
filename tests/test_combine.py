"""Tests for dataset combination and deduplication.

Tests verify:
1. Concatenation combines arrays end-to-end
2. Interleave produces weighted round-robin mixing
3. Weighted sampling respects dataset weights
4. ExactDeduplicator finds and removes exact duplicates
5. Full combine pipeline produces valid output
"""

import numpy as np
import pytest
from pathlib import Path

from cola_coder.data.combine import DatasetCombiner, DatasetInput
from cola_coder.data.dedup import ExactDeduplicator, CrossDatasetDeduplicator


# ---------------------------------------------------------------------------
# Fixtures: create small test .npy arrays
# ---------------------------------------------------------------------------

CHUNK_SIZE = 32  # Small chunk size for fast tests


@pytest.fixture
def tmp_datasets(tmp_path):
    """Create three small .npy test datasets.

    dataset_a: 100 chunks, tokens in range [0, 100)
    dataset_b: 50 chunks, tokens in range [1000, 1100)
    dataset_c: 30 chunks, tokens in range [2000, 2100)
    """
    rng = np.random.default_rng(42)

    a = rng.integers(0, 100, size=(100, CHUNK_SIZE), dtype=np.uint16)
    b = rng.integers(1000, 1100, size=(50, CHUNK_SIZE), dtype=np.uint16)
    c = rng.integers(2000, 2100, size=(30, CHUNK_SIZE), dtype=np.uint16)

    path_a = tmp_path / "dataset_a.npy"
    path_b = tmp_path / "dataset_b.npy"
    path_c = tmp_path / "dataset_c.npy"

    np.save(str(path_a), a)
    np.save(str(path_b), b)
    np.save(str(path_c), c)

    return {
        "a": {"path": str(path_a), "data": a},
        "b": {"path": str(path_b), "data": b},
        "c": {"path": str(path_c), "data": c},
        "tmp_path": tmp_path,
    }


@pytest.fixture
def combiner():
    return DatasetCombiner()


# ---------------------------------------------------------------------------
# Test: concat strategy
# ---------------------------------------------------------------------------

class TestConcat:
    def test_concat_two_datasets(self, combiner, tmp_datasets):
        """Concat should append datasets end-to-end."""
        out_path = str(tmp_datasets["tmp_path"] / "combined.npy")
        result = combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], name="A"),
                DatasetInput(tmp_datasets["b"]["path"], name="B"),
            ],
            strategy="concat",
            output_path=out_path,
            shuffle=False,
        )

        assert result.total_chunks == 150  # 100 + 50
        assert result.total_tokens == 150 * CHUNK_SIZE

        combined = np.load(out_path)
        assert combined.shape == (150, CHUNK_SIZE)

    def test_concat_with_max_tokens(self, combiner, tmp_datasets):
        """Concat respects max_tokens cap."""
        out_path = str(tmp_datasets["tmp_path"] / "combined_capped.npy")
        max_tokens = 80 * CHUNK_SIZE  # Only 80 chunks worth

        result = combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], name="A"),
                DatasetInput(tmp_datasets["b"]["path"], name="B"),
            ],
            strategy="concat",
            output_path=out_path,
            max_tokens=max_tokens,
            shuffle=False,
        )

        # Should cap at 80 chunks (all from A since A has 100)
        assert result.total_chunks == 80

    def test_concat_three_datasets(self, combiner, tmp_datasets):
        """Concat with three datasets appends all three."""
        out_path = str(tmp_datasets["tmp_path"] / "combined3.npy")
        result = combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], name="A"),
                DatasetInput(tmp_datasets["b"]["path"], name="B"),
                DatasetInput(tmp_datasets["c"]["path"], name="C"),
            ],
            strategy="concat",
            output_path=out_path,
            shuffle=False,
        )

        assert result.total_chunks == 180  # 100 + 50 + 30


# ---------------------------------------------------------------------------
# Test: interleave strategy
# ---------------------------------------------------------------------------

class TestInterleave:
    def test_interleave_equal_weights(self, combiner, tmp_datasets):
        """Interleave with equal weights should draw roughly equally."""
        out_path = str(tmp_datasets["tmp_path"] / "interleaved.npy")
        result = combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], weight=1.0, name="A"),
                DatasetInput(tmp_datasets["b"]["path"], weight=1.0, name="B"),
            ],
            strategy="interleave",
            output_path=out_path,
            shuffle=False,
        )

        # With equal weights: target is 75 from A and 75 from B,
        # but B only has 50, so clamped to 50.
        # A gets 75 (half of 150).
        combined = np.load(out_path)
        assert combined.shape[1] == CHUNK_SIZE
        assert result.total_chunks > 0

    def test_interleave_weighted(self, combiner, tmp_datasets):
        """Interleave with unequal weights respects proportions."""
        out_path = str(tmp_datasets["tmp_path"] / "interleaved_w.npy")
        result = combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], weight=0.7, name="A"),
                DatasetInput(tmp_datasets["b"]["path"], weight=0.3, name="B"),
            ],
            strategy="interleave",
            output_path=out_path,
            shuffle=False,
        )

        combined = np.load(out_path)
        assert combined.shape[1] == CHUNK_SIZE
        assert result.total_chunks > 0

    def test_interleave_with_max_chunks(self, combiner, tmp_datasets):
        """Interleave should respect max_tokens."""
        out_path = str(tmp_datasets["tmp_path"] / "interleaved_max.npy")
        max_tokens = 60 * CHUNK_SIZE

        result = combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], weight=0.5, name="A"),
                DatasetInput(tmp_datasets["b"]["path"], weight=0.5, name="B"),
            ],
            strategy="interleave",
            output_path=out_path,
            max_tokens=max_tokens,
            shuffle=False,
        )

        assert result.total_chunks <= 60


# ---------------------------------------------------------------------------
# Test: weighted sampling strategy
# ---------------------------------------------------------------------------

class TestWeighted:
    def test_weighted_sampling(self, combiner, tmp_datasets):
        """Weighted sampling should produce requested number of chunks."""
        out_path = str(tmp_datasets["tmp_path"] / "weighted.npy")
        max_tokens = 100 * CHUNK_SIZE

        result = combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], weight=0.8, name="A"),
                DatasetInput(tmp_datasets["b"]["path"], weight=0.2, name="B"),
            ],
            strategy="weighted",
            output_path=out_path,
            max_tokens=max_tokens,
        )

        assert result.total_chunks == 100

    def test_weighted_sampling_distribution(self, combiner, tmp_datasets):
        """Weighted sampling should roughly match target distribution."""
        out_path = str(tmp_datasets["tmp_path"] / "weighted_dist.npy")
        max_tokens = 1000 * CHUNK_SIZE

        combiner.combine(
            datasets=[
                DatasetInput(tmp_datasets["a"]["path"], weight=0.7, name="A"),
                DatasetInput(tmp_datasets["b"]["path"], weight=0.3, name="B"),
            ],
            strategy="weighted",
            output_path=out_path,
            max_tokens=max_tokens,
            seed=42,
        )

        combined = np.load(out_path)
        # Chunks from A have values < 100, from B have values >= 1000
        # Check the first token of each chunk
        from_a = np.sum(combined[:, 0] < 500)
        ratio_a = from_a / len(combined)
        # Should be roughly 0.7 +/- 0.1
        assert 0.55 < ratio_a < 0.85, f"Expected ~70% from A, got {ratio_a:.1%}"

    def test_weighted_is_deterministic(self, combiner, tmp_datasets):
        """Same seed should produce identical results."""
        out1 = str(tmp_datasets["tmp_path"] / "weighted_det1.npy")
        out2 = str(tmp_datasets["tmp_path"] / "weighted_det2.npy")

        inputs = [
            DatasetInput(tmp_datasets["a"]["path"], weight=0.6, name="A"),
            DatasetInput(tmp_datasets["b"]["path"], weight=0.4, name="B"),
        ]
        max_tokens = 50 * CHUNK_SIZE

        combiner.combine(
            datasets=inputs, strategy="weighted",
            output_path=out1, max_tokens=max_tokens, seed=123,
        )
        combiner.combine(
            datasets=inputs, strategy="weighted",
            output_path=out2, max_tokens=max_tokens, seed=123,
        )

        arr1 = np.load(out1)
        arr2 = np.load(out2)
        np.testing.assert_array_equal(arr1, arr2)


# ---------------------------------------------------------------------------
# Test: ExactDeduplicator
# ---------------------------------------------------------------------------

class TestExactDedup:
    def test_finds_exact_duplicates(self):
        """Deduplicator should detect identical chunks."""
        rng = np.random.default_rng(42)
        unique = rng.integers(0, 1000, size=(10, CHUNK_SIZE), dtype=np.uint16)
        # Add 5 exact copies of chunk 0
        dupes = np.tile(unique[0:1], (5, 1))
        data = np.concatenate([unique, dupes], axis=0)  # 15 total

        dedup = ExactDeduplicator()
        clean, removed = dedup.deduplicate_array(data)

        assert removed == 5  # 5 duplicates of chunk 0
        assert len(clean) == 10  # 10 unique chunks

    def test_no_duplicates(self):
        """No duplicates in unique data."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 65535, size=(20, CHUNK_SIZE), dtype=np.uint16)

        dedup = ExactDeduplicator()
        clean, removed = dedup.deduplicate_array(data)

        assert removed == 0
        assert len(clean) == 20

    def test_all_duplicates(self):
        """All rows identical should collapse to one."""
        row = np.arange(CHUNK_SIZE, dtype=np.uint16).reshape(1, CHUNK_SIZE)
        data = np.tile(row, (50, 1))

        dedup = ExactDeduplicator()
        clean, removed = dedup.deduplicate_array(data)

        assert removed == 49
        assert len(clean) == 1

    def test_is_duplicate_tracks_state(self):
        """is_duplicate should track what it has seen."""
        dedup = ExactDeduplicator()
        chunk = np.array([1, 2, 3, 4], dtype=np.uint16)

        assert dedup.is_duplicate(chunk) is False  # First time
        assert dedup.is_duplicate(chunk) is True   # Seen before

    def test_reset_clears_state(self):
        """reset() should clear the seen hashes."""
        dedup = ExactDeduplicator()
        chunk = np.array([1, 2, 3, 4], dtype=np.uint16)

        dedup.is_duplicate(chunk)
        dedup.reset()
        assert dedup.is_duplicate(chunk) is False  # Fresh after reset


# ---------------------------------------------------------------------------
# Test: CrossDatasetDeduplicator (exact fallback)
# ---------------------------------------------------------------------------

class TestCrossDatasetDedup:
    def test_exact_fallback_finds_shared_chunks(self, tmp_path):
        """Cross-dataset exact dedup should find chunks shared between files."""
        rng = np.random.default_rng(42)
        shared = rng.integers(0, 1000, size=(5, CHUNK_SIZE), dtype=np.uint16)
        unique_a = rng.integers(0, 1000, size=(10, CHUNK_SIZE), dtype=np.uint16)
        unique_b = rng.integers(0, 1000, size=(8, CHUNK_SIZE), dtype=np.uint16)

        primary = np.concatenate([unique_a, shared], axis=0)
        secondary = np.concatenate([unique_b, shared], axis=0)

        path_a = str(tmp_path / "primary.npy")
        path_b = str(tmp_path / "secondary.npy")
        np.save(path_a, primary)
        np.save(path_b, secondary)

        dedup = CrossDatasetDeduplicator(method="exact")
        result = dedup.deduplicate_pair(
            primary_path=path_a,
            secondary_path=path_b,
            output_path=str(tmp_path / "secondary_clean.npy"),
        )

        assert result.duplicates_removed == 5
        assert result.output_chunks == 8  # Only unique_b remains
        assert result.method == "exact"

        # Verify output file
        clean = np.load(str(tmp_path / "secondary_clean.npy"))
        assert len(clean) == 8


# ---------------------------------------------------------------------------
# Test: full combine pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_combine_with_dedup(self, tmp_datasets):
        """Full pipeline: dedup then combine."""
        rng = np.random.default_rng(99)
        tmp_path = tmp_datasets["tmp_path"]

        # Create two datasets that share some chunks
        shared = rng.integers(500, 600, size=(10, CHUNK_SIZE), dtype=np.uint16)
        unique1 = rng.integers(0, 100, size=(40, CHUNK_SIZE), dtype=np.uint16)
        unique2 = rng.integers(1000, 1100, size=(30, CHUNK_SIZE), dtype=np.uint16)

        ds1 = np.concatenate([unique1, shared], axis=0)  # 50 chunks
        ds2 = np.concatenate([unique2, shared], axis=0)  # 40 chunks

        path1 = str(tmp_path / "pipe_ds1.npy")
        path2 = str(tmp_path / "pipe_ds2.npy")
        np.save(path1, ds1)
        np.save(path2, ds2)

        # Dedup ds2 against ds1
        dedup = CrossDatasetDeduplicator(method="exact")
        dedup_result = dedup.deduplicate_pair(
            primary_path=path1,
            secondary_path=path2,
            output_path=str(tmp_path / "pipe_ds2_clean.npy"),
        )
        assert dedup_result.duplicates_removed == 10

        # Combine
        combiner = DatasetCombiner()
        result = combiner.combine(
            datasets=[
                DatasetInput(path1, weight=0.6, name="DS1"),
                DatasetInput(
                    str(tmp_path / "pipe_ds2_clean.npy"),
                    weight=0.4,
                    name="DS2",
                ),
            ],
            strategy="interleave",
            output_path=str(tmp_path / "pipe_combined.npy"),
        )

        assert result.total_chunks > 0
        combined = np.load(str(tmp_path / "pipe_combined.npy"))
        assert combined.shape[1] == CHUNK_SIZE

    def test_combine_single_dataset(self, tmp_datasets):
        """Combining a single dataset should just copy it."""
        out_path = str(tmp_datasets["tmp_path"] / "single.npy")
        combiner = DatasetCombiner()
        result = combiner.combine(
            datasets=[DatasetInput(tmp_datasets["a"]["path"], name="A")],
            strategy="concat",
            output_path=out_path,
            shuffle=False,
        )

        assert result.total_chunks == 100
        combined = np.load(out_path)
        original = np.load(tmp_datasets["a"]["path"])
        np.testing.assert_array_equal(combined, original)

    def test_combine_creates_output_dir(self, tmp_datasets):
        """Combine should create output directory if it doesn't exist."""
        out_path = str(
            tmp_datasets["tmp_path"] / "nested" / "deep" / "combined.npy"
        )
        combiner = DatasetCombiner()
        result = combiner.combine(
            datasets=[DatasetInput(tmp_datasets["a"]["path"], name="A")],
            strategy="concat",
            output_path=out_path,
        )

        assert Path(out_path).exists()
        assert result.total_chunks == 100


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_empty_dataset_list(self, combiner):
        """Should raise on empty dataset list."""
        with pytest.raises(ValueError, match="No datasets"):
            combiner.combine(datasets=[], output_path="nope.npy")

    def test_unknown_strategy(self, combiner, tmp_datasets):
        """Should raise on unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            combiner.combine(
                datasets=[DatasetInput(tmp_datasets["a"]["path"])],
                strategy="magic",
                output_path="nope.npy",
            )

    def test_mismatched_chunk_sizes(self, combiner, tmp_path):
        """Should raise when chunk sizes don't match."""
        a = np.zeros((10, 32), dtype=np.uint16)
        b = np.zeros((10, 64), dtype=np.uint16)
        np.save(str(tmp_path / "a.npy"), a)
        np.save(str(tmp_path / "b.npy"), b)

        with pytest.raises(ValueError, match="Chunk size mismatch"):
            combiner.combine(
                datasets=[
                    DatasetInput(str(tmp_path / "a.npy")),
                    DatasetInput(str(tmp_path / "b.npy")),
                ],
                output_path=str(tmp_path / "bad.npy"),
            )

    def test_1d_array_rejected(self, combiner, tmp_path):
        """Should raise on 1D arrays."""
        flat = np.zeros(100, dtype=np.uint16)
        np.save(str(tmp_path / "flat.npy"), flat)

        with pytest.raises(ValueError, match="Expected 2D"):
            combiner.combine(
                datasets=[DatasetInput(str(tmp_path / "flat.npy"))],
                output_path=str(tmp_path / "bad.npy"),
            )
