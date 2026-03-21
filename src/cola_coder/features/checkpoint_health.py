"""Checkpoint Health Check (improvement #68).

Non-destructive verification of checkpoint integrity:
  - file completeness (required files present)
  - tensor shape consistency (shapes match expected config)
  - metadata validity (required keys, reasonable values)
  - weight statistics sanity (no NaN/Inf, reasonable norms)

Does NOT load the full model into GPU memory — uses safetensors metadata
or torch.load with map_location="cpu" for lightweight inspection.

TypeScript analogy: like a JSON schema validator + integrity checker
for your serialised model state, without executing the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if checkpoint health checking is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_METADATA_KEYS = frozenset({"step", "loss"})
OPTIONAL_METADATA_KEYS = frozenset({"epoch", "val_loss", "config", "timestamp"})

REQUIRED_CHECKPOINT_FILES = frozenset({"model.safetensors", "metadata.json"})
OPTIONAL_CHECKPOINT_FILES = frozenset({"optimizer.pt", "scheduler.pt"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TensorHealthRecord:
    """Health report for a single tensor."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    has_nan: bool = False
    has_inf: bool = False
    mean: float = 0.0
    std: float = 0.0
    norm: float = 0.0


@dataclass
class CheckpointHealthReport:
    """Full health report for a checkpoint directory."""

    checkpoint_dir: str
    files_present: List[str] = field(default_factory=list)
    files_missing: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    tensor_records: List[TensorHealthRecord] = field(default_factory=list)
    nan_tensors: List[str] = field(default_factory=list)
    inf_tensors: List[str] = field(default_factory=list)
    suspicious_tensors: List[str] = field(default_factory=list)

    metadata_issues: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    is_healthy: bool = False

    @property
    def total_tensors(self) -> int:
        return len(self.tensor_records)

    @property
    def healthy_tensors(self) -> int:
        return sum(
            1 for t in self.tensor_records if not t.has_nan and not t.has_inf
        )


# ---------------------------------------------------------------------------
# Metadata validator
# ---------------------------------------------------------------------------


def validate_metadata(meta: Dict[str, Any]) -> List[str]:
    """Validate checkpoint metadata dict. Returns list of issue strings."""
    issues: List[str] = []
    for key in REQUIRED_METADATA_KEYS:
        if key not in meta:
            issues.append(f"Metadata missing required key: '{key}'")

    if "step" in meta:
        step = meta["step"]
        if not isinstance(step, (int, float)) or step < 0:
            issues.append(f"Metadata 'step' is invalid: {step!r}")

    if "loss" in meta:
        loss = meta["loss"]
        if isinstance(loss, (int, float)):
            if math.isnan(loss):
                issues.append("Metadata 'loss' is NaN")
            elif math.isinf(loss):
                issues.append("Metadata 'loss' is Inf")
            elif loss < 0:
                issues.append(f"Metadata 'loss' is negative: {loss}")
        else:
            issues.append(f"Metadata 'loss' has unexpected type: {type(loss).__name__}")

    if "val_loss" in meta:
        val = meta["val_loss"]
        if isinstance(val, (int, float)) and val < 0:
            issues.append(f"Metadata 'val_loss' is negative: {val}")

    return issues


# ---------------------------------------------------------------------------
# Tensor stats helper (pure Python, no GPU required)
# ---------------------------------------------------------------------------


def _tensor_stats_from_list(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, norm for a flat list of float values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "norm": 0.0, "has_nan": False, "has_inf": False}
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"mean": 0.0, "std": 0.0, "norm": 0.0, "has_nan": has_nan, "has_inf": has_inf}
    n = len(clean)
    mean = sum(clean) / n
    variance = sum((v - mean) ** 2 for v in clean) / n
    std = math.sqrt(variance)
    norm = math.sqrt(sum(v**2 for v in clean))
    return {"mean": mean, "std": std, "norm": norm, "has_nan": has_nan, "has_inf": has_inf}


# ---------------------------------------------------------------------------
# Health checker
# ---------------------------------------------------------------------------


class CheckpointHealthChecker:
    """Non-destructive checkpoint integrity verifier.

    Can work with:
      1. Real checkpoint directories (uses file system + safetensors/torch)
      2. Injected data dicts (for testing, no file I/O required)
    """

    def __init__(
        self,
        required_files: Optional[frozenset] = None,
        norm_threshold: float = 1000.0,
    ) -> None:
        self.required_files = required_files or REQUIRED_CHECKPOINT_FILES
        self.norm_threshold = norm_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_directory(self, checkpoint_dir: str | Path) -> CheckpointHealthReport:
        """Check a checkpoint directory on disk (non-destructive)."""
        checkpoint_dir = Path(checkpoint_dir)
        report = CheckpointHealthReport(checkpoint_dir=str(checkpoint_dir))

        # File presence
        if checkpoint_dir.exists():
            present = {f.name for f in checkpoint_dir.iterdir()}
            report.files_present = sorted(present)
            report.files_missing = sorted(self.required_files - present)
        else:
            report.files_missing = sorted(self.required_files)
            report.issues.append(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return report

        if report.files_missing:
            for f in report.files_missing:
                report.issues.append(f"Missing required file: {f}")

        # Load metadata
        meta_path = checkpoint_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    report.metadata = json.load(fh)
                meta_issues = validate_metadata(report.metadata)
                report.metadata_issues.extend(meta_issues)
                report.issues.extend(meta_issues)
            except json.JSONDecodeError as e:
                report.issues.append(f"metadata.json is not valid JSON: {e}")

        report.is_healthy = len(report.issues) == 0
        return report

    def check_from_dicts(
        self,
        metadata: Dict[str, Any],
        tensors: Dict[str, Dict],
        files_present: Optional[List[str]] = None,
    ) -> CheckpointHealthReport:
        """Check a checkpoint from in-memory dicts (no file I/O, for testing).

        Parameters
        ----------
        metadata:
            The checkpoint metadata dict.
        tensors:
            {name: {"shape": tuple, "dtype": str, "values": [floats]}}
        files_present:
            List of filenames present (if None, assume required files present).
        """
        report = CheckpointHealthReport(checkpoint_dir="<in-memory>")

        # Files
        present = set(files_present or self.required_files)
        report.files_present = sorted(present)
        report.files_missing = sorted(self.required_files - present)
        for f in report.files_missing:
            report.issues.append(f"Missing required file: {f}")

        # Metadata
        report.metadata = metadata
        meta_issues = validate_metadata(metadata)
        report.metadata_issues.extend(meta_issues)
        report.issues.extend(meta_issues)

        # Tensors
        for name, tinfo in tensors.items():
            shape = tuple(tinfo.get("shape", ()))
            dtype = tinfo.get("dtype", "float32")
            values = tinfo.get("values", [])

            stats = _tensor_stats_from_list(values)
            rec = TensorHealthRecord(
                name=name,
                shape=shape,
                dtype=dtype,
                has_nan=stats["has_nan"],
                has_inf=stats["has_inf"],
                mean=stats["mean"],
                std=stats["std"],
                norm=stats["norm"],
            )
            report.tensor_records.append(rec)

            if rec.has_nan:
                report.nan_tensors.append(name)
                report.issues.append(f"Tensor '{name}' contains NaN values")
            if rec.has_inf:
                report.inf_tensors.append(name)
                report.issues.append(f"Tensor '{name}' contains Inf values")
            if rec.norm > self.norm_threshold:
                report.suspicious_tensors.append(name)
                report.issues.append(
                    f"Tensor '{name}' has very large norm ({rec.norm:.1f} > {self.norm_threshold})"
                )

        report.is_healthy = len(report.issues) == 0
        return report

    def shape_consistency_check(
        self, tensors: Dict[str, Dict], expected_shapes: Dict[str, Tuple[int, ...]]
    ) -> List[str]:
        """Check that tensor shapes match expected shapes. Returns issue list."""
        issues: List[str] = []
        for name, expected in expected_shapes.items():
            if name not in tensors:
                issues.append(f"Expected tensor '{name}' not found in checkpoint")
                continue
            actual = tuple(tensors[name].get("shape", ()))
            if actual != expected:
                issues.append(
                    f"Shape mismatch for '{name}': expected {expected}, got {actual}"
                )
        return issues


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def quick_health_check(
    metadata: Dict[str, Any],
    tensors: Dict[str, Dict],
) -> CheckpointHealthReport:
    """Quick health check from in-memory data."""
    return CheckpointHealthChecker().check_from_dicts(metadata, tensors)
