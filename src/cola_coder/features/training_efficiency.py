"""Training Efficiency Tracker: monitor compute utilization during training.

Tracks tokens per second, GPU utilization, memory usage, and
Model FLOPs Utilization (MFU) — the fraction of peak GPU FLOP/s actually used.

All metrics can be updated via event records; no GPU or subprocess required.
The tracker operates on numeric snapshots supplied by the caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the training efficiency tracker feature is active."""
    return FEATURE_ENABLED


@dataclass
class EfficiencySnapshot:
    """A single measurement point during training."""

    step: int
    elapsed_seconds: float
    tokens_processed: int
    gpu_utilization_pct: float  # 0-100
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.tokens_processed / self.elapsed_seconds

    @property
    def memory_utilization_pct(self) -> float:
        if self.gpu_memory_total_mb <= 0:
            return 0.0
        return 100.0 * self.gpu_memory_used_mb / self.gpu_memory_total_mb


@dataclass
class MFUReport:
    """Model FLOPs Utilization report."""

    mfu_pct: float  # Actual MFU percentage (0-100)
    achieved_tflops: float  # FLOPs/s actually achieved (in TFLOPs)
    peak_tflops: float  # Theoretical peak FLOPs/s of the hardware
    model_flops_per_token: float  # Estimated FLOPs per forward+backward token
    tokens_per_second: float

    def summary(self) -> str:
        return (
            f"MFU={self.mfu_pct:.2f}%, "
            f"achieved={self.achieved_tflops:.2f} TFLOPs, "
            f"peak={self.peak_tflops:.2f} TFLOPs, "
            f"tok/s={self.tokens_per_second:.0f}"
        )


@dataclass
class EfficiencySummary:
    """Aggregated efficiency metrics over a training run."""

    n_snapshots: int
    mean_tokens_per_second: float
    mean_gpu_utilization_pct: float
    mean_memory_utilization_pct: float
    peak_tokens_per_second: float
    total_tokens: int
    total_elapsed_seconds: float
    mfu_report: Optional[MFUReport] = None

    def summary(self) -> str:
        lines = [
            f"EfficiencySummary({self.n_snapshots} snapshots)",
            f"  tok/s mean={self.mean_tokens_per_second:.0f}, peak={self.peak_tokens_per_second:.0f}",
            f"  GPU util={self.mean_gpu_utilization_pct:.1f}%",
            f"  VRAM util={self.mean_memory_utilization_pct:.1f}%",
            f"  total_tokens={self.total_tokens}, elapsed={self.total_elapsed_seconds:.1f}s",
        ]
        if self.mfu_report:
            lines.append(f"  {self.mfu_report.summary()}")
        return "\n".join(lines)


def _estimate_model_flops_per_token(n_params: int, seq_len: int) -> float:
    """Estimate FLOPs per token for a transformer (forward + backward).

    Rough formula: 6 × n_params per token for forward+backward
    (Kaplan et al., "Scaling Laws for Neural Language Models").
    For attention: +2 × seq_len × n_layers × d_model FLOPs, but we use the
    simpler dominant term.
    """
    return 6.0 * n_params


class TrainingEfficiencyTracker:
    """Collect and analyze training efficiency snapshots."""

    def __init__(
        self,
        n_params: Optional[int] = None,
        seq_len: int = 512,
        peak_gpu_tflops: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_params:
            Model parameter count (used for MFU computation).
        seq_len:
            Sequence length (used for MFU computation).
        peak_gpu_tflops:
            Theoretical peak FLOPs/s of the GPU in TFLOPs.
            e.g. RTX 3080 ≈ 29.8, RTX 4080 ≈ 48.7 (FP16).
        """
        self.n_params = n_params
        self.seq_len = seq_len
        self.peak_gpu_tflops = peak_gpu_tflops
        self.snapshots: list[EfficiencySnapshot] = []

    def record(
        self,
        step: int,
        elapsed_seconds: float,
        tokens_processed: int,
        gpu_utilization_pct: float = 0.0,
        gpu_memory_used_mb: float = 0.0,
        gpu_memory_total_mb: float = 0.0,
    ) -> EfficiencySnapshot:
        """Record one efficiency snapshot and return it."""
        snap = EfficiencySnapshot(
            step=step,
            elapsed_seconds=elapsed_seconds,
            tokens_processed=tokens_processed,
            gpu_utilization_pct=gpu_utilization_pct,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
        )
        self.snapshots.append(snap)
        return snap

    def compute_mfu(self, tokens_per_second: float) -> Optional[MFUReport]:
        """Compute MFU given a tokens/s figure.

        Returns None if n_params or peak_gpu_tflops are not set.
        """
        if self.n_params is None or self.peak_gpu_tflops is None:
            return None
        flops_per_token = _estimate_model_flops_per_token(self.n_params, self.seq_len)
        achieved_flops_per_sec = flops_per_token * tokens_per_second
        achieved_tflops = achieved_flops_per_sec / 1e12
        peak_tflops = self.peak_gpu_tflops
        mfu_pct = 100.0 * achieved_tflops / peak_tflops if peak_tflops > 0 else 0.0
        return MFUReport(
            mfu_pct=mfu_pct,
            achieved_tflops=achieved_tflops,
            peak_tflops=peak_tflops,
            model_flops_per_token=flops_per_token,
            tokens_per_second=tokens_per_second,
        )

    def summarize(self) -> EfficiencySummary:
        """Aggregate all recorded snapshots into a summary."""
        if not self.snapshots:
            return EfficiencySummary(
                n_snapshots=0,
                mean_tokens_per_second=0.0,
                mean_gpu_utilization_pct=0.0,
                mean_memory_utilization_pct=0.0,
                peak_tokens_per_second=0.0,
                total_tokens=0,
                total_elapsed_seconds=0.0,
            )

        tps_values = [s.tokens_per_second for s in self.snapshots]
        gpu_utils = [s.gpu_utilization_pct for s in self.snapshots]
        mem_utils = [s.memory_utilization_pct for s in self.snapshots]

        mean_tps = sum(tps_values) / len(tps_values)
        mfu_report = self.compute_mfu(mean_tps)

        return EfficiencySummary(
            n_snapshots=len(self.snapshots),
            mean_tokens_per_second=mean_tps,
            mean_gpu_utilization_pct=sum(gpu_utils) / len(gpu_utils),
            mean_memory_utilization_pct=sum(mem_utils) / len(mem_utils),
            peak_tokens_per_second=max(tps_values),
            total_tokens=max(s.tokens_processed for s in self.snapshots),
            total_elapsed_seconds=max(s.elapsed_seconds for s in self.snapshots),
            mfu_report=mfu_report,
        )

    def tokens_per_second_history(self) -> list[tuple[int, float]]:
        """Return [(step, tok/s)] for all snapshots."""
        return [(s.step, s.tokens_per_second) for s in self.snapshots]

    def efficiency_trend(self) -> str:
        """Describe whether efficiency is improving, declining, or stable."""
        if len(self.snapshots) < 3:
            return "insufficient_data"
        tps = [s.tokens_per_second for s in self.snapshots]
        first_half = tps[: len(tps) // 2]
        second_half = tps[len(tps) // 2 :]
        m1 = sum(first_half) / len(first_half)
        m2 = sum(second_half) / len(second_half)
        rel_change = (m2 - m1) / max(m1, 1e-6)
        if rel_change > 0.05:
            return "improving"
        elif rel_change < -0.05:
            return "declining"
        return "stable"

    def detect_stalls(self, threshold_pct: float = 50.0) -> list[int]:
        """Return step numbers where tokens/s dropped below threshold_pct of peak."""
        if not self.snapshots:
            return []
        peak = max(s.tokens_per_second for s in self.snapshots)
        if peak <= 0:
            return []
        cutoff = peak * (threshold_pct / 100.0)
        return [s.step for s in self.snapshots if s.tokens_per_second < cutoff]
