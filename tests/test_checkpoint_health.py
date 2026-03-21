"""Tests for CheckpointHealthChecker (features/checkpoint_health.py)."""

from __future__ import annotations

import json


from cola_coder.features.checkpoint_health import (
    FEATURE_ENABLED,
    CheckpointHealthChecker,
    CheckpointHealthReport,
    is_enabled,
    validate_metadata,
)

# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# validate_metadata
# ---------------------------------------------------------------------------


class TestValidateMetadata:
    def test_valid_metadata_no_issues(self):
        meta = {"step": 1000, "loss": 2.3}
        assert validate_metadata(meta) == []

    def test_missing_step(self):
        issues = validate_metadata({"loss": 2.3})
        assert any("step" in i for i in issues)

    def test_missing_loss(self):
        issues = validate_metadata({"step": 100})
        assert any("loss" in i for i in issues)

    def test_nan_loss(self):
        issues = validate_metadata({"step": 10, "loss": float("nan")})
        assert any("NaN" in i for i in issues)

    def test_negative_loss(self):
        issues = validate_metadata({"step": 10, "loss": -1.5})
        assert any("negative" in i for i in issues)

    def test_invalid_step(self):
        issues = validate_metadata({"step": -5, "loss": 2.3})
        assert any("step" in i for i in issues)


# ---------------------------------------------------------------------------
# check_from_dicts — healthy checkpoint
# ---------------------------------------------------------------------------


class TestCheckFromDictsHealthy:
    def _healthy_tensors(self):
        return {
            "tok_emb.weight": {"shape": (32000, 512), "dtype": "float32", "values": [0.1, 0.2, 0.3]},
            "lm_head.weight": {"shape": (512, 512), "dtype": "float32", "values": [0.01, -0.01]},
        }

    def test_healthy_is_true(self):
        checker = CheckpointHealthChecker()
        r = checker.check_from_dicts(
            metadata={"step": 500, "loss": 2.5},
            tensors=self._healthy_tensors(),
        )
        assert r.is_healthy

    def test_returns_report_type(self):
        checker = CheckpointHealthChecker()
        r = checker.check_from_dicts({"step": 1, "loss": 3.0}, {})
        assert isinstance(r, CheckpointHealthReport)

    def test_healthy_tensors_count(self):
        checker = CheckpointHealthChecker()
        r = checker.check_from_dicts(
            {"step": 1, "loss": 2.0},
            self._healthy_tensors(),
        )
        assert r.healthy_tensors == 2


# ---------------------------------------------------------------------------
# check_from_dicts — unhealthy checkpoint
# ---------------------------------------------------------------------------


class TestCheckFromDictsUnhealthy:
    def test_nan_tensor_detected(self):
        checker = CheckpointHealthChecker()
        r = checker.check_from_dicts(
            {"step": 1, "loss": 2.0},
            {"bad": {"shape": (2,), "dtype": "float32", "values": [float("nan"), 1.0]}},
        )
        assert not r.is_healthy
        assert "bad" in r.nan_tensors

    def test_inf_tensor_detected(self):
        checker = CheckpointHealthChecker()
        r = checker.check_from_dicts(
            {"step": 1, "loss": 2.0},
            {"inf_tensor": {"shape": (1,), "dtype": "float32", "values": [float("inf")]}},
        )
        assert not r.is_healthy
        assert "inf_tensor" in r.inf_tensors

    def test_large_norm_flagged(self):
        checker = CheckpointHealthChecker(norm_threshold=10.0)
        r = checker.check_from_dicts(
            {"step": 1, "loss": 2.0},
            {"big": {"shape": (3,), "dtype": "float32", "values": [100.0, 100.0, 100.0]}},
        )
        assert "big" in r.suspicious_tensors

    def test_missing_file_reported(self):
        checker = CheckpointHealthChecker()
        r = checker.check_from_dicts(
            {"step": 1, "loss": 2.0},
            {},
            files_present=["metadata.json"],  # missing model.safetensors
        )
        assert not r.is_healthy
        assert any("model.safetensors" in issue for issue in r.issues)


# ---------------------------------------------------------------------------
# Shape consistency
# ---------------------------------------------------------------------------


class TestShapeConsistency:
    def test_matching_shapes_no_issues(self):
        checker = CheckpointHealthChecker()
        tensors = {"w": {"shape": (128, 256)}}
        issues = checker.shape_consistency_check(tensors, {"w": (128, 256)})
        assert issues == []

    def test_shape_mismatch_detected(self):
        checker = CheckpointHealthChecker()
        tensors = {"w": {"shape": (128, 256)}}
        issues = checker.shape_consistency_check(tensors, {"w": (64, 256)})
        assert len(issues) == 1
        assert "mismatch" in issues[0]

    def test_missing_tensor_detected(self):
        checker = CheckpointHealthChecker()
        issues = checker.shape_consistency_check({}, {"w": (128, 256)})
        assert len(issues) == 1
        assert "not found" in issues[0]


# ---------------------------------------------------------------------------
# Directory check (uses tmp_path)
# ---------------------------------------------------------------------------


class TestDirectoryCheck:
    def test_missing_directory(self, tmp_path):
        checker = CheckpointHealthChecker()
        r = checker.check_directory(tmp_path / "nonexistent")
        assert not r.is_healthy
        assert r.files_missing

    def test_valid_directory_with_files(self, tmp_path):
        # Write required files
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        (tmp_path / "metadata.json").write_text(
            json.dumps({"step": 100, "loss": 2.5})
        )
        checker = CheckpointHealthChecker()
        r = checker.check_directory(tmp_path)
        assert r.is_healthy
        assert "metadata.json" in r.files_present

    def test_bad_json_metadata(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        (tmp_path / "metadata.json").write_text("{ invalid json }")
        checker = CheckpointHealthChecker()
        r = checker.check_directory(tmp_path)
        assert not r.is_healthy
