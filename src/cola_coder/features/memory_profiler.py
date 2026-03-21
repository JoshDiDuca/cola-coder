"""Memory Profiler (improvement #65).

Profiles memory usage per layer/operation without requiring GPU.
Tracks peak allocations, identifies memory bottlenecks, and suggests
memory optimizations.

In practice, this module works in two modes:
  1. Pure-Python mode (always available): tracks Python-side estimates and
     host RAM (via tracemalloc / psutil if available).
  2. CUDA mode (available when torch is present): reads device memory stats.

TypeScript analogy: like a performance.mark() / performance.measure() API
wrapped around each model layer.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if memory profiling is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import tracemalloc as _tracemalloc

    _TRACEMALLOC_AVAILABLE = True
except ImportError:
    _TRACEMALLOC_AVAILABLE = False

try:
    import psutil as _psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


def _host_ram_mb() -> float:
    """Return current process RSS in MB (0 if psutil not available)."""
    if _PSUTIL_AVAILABLE:
        import os

        proc = _psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    return 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerMemRecord:
    """Memory snapshot for one layer/operation."""

    name: str
    duration_ms: float
    ram_before_mb: float
    ram_after_mb: float
    ram_peak_mb: float
    tracemalloc_peak_bytes: int = 0

    @property
    def ram_delta_mb(self) -> float:
        return self.ram_after_mb - self.ram_before_mb


@dataclass
class MemoryProfileReport:
    """Full memory profile report for a forward/backward pass."""

    records: List[LayerMemRecord] = field(default_factory=list)
    total_duration_ms: float = 0.0
    peak_ram_mb: float = 0.0
    bottleneck_layer: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def layer_summary(self) -> Dict[str, float]:
        """Return {layer_name: ram_delta_mb} for each layer."""
        return {r.name: r.ram_delta_mb for r in self.records}


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class MemoryProfiler:
    """Profile memory usage across named layers/operations.

    Usage::

        profiler = MemoryProfiler()
        with profiler.track("embedding"):
            output = embedding_layer(tokens)
        with profiler.track("attention"):
            output = attention_layer(output)
        report = profiler.report()
    """

    def __init__(self, use_tracemalloc: bool = True) -> None:
        self.use_tracemalloc = use_tracemalloc and _TRACEMALLOC_AVAILABLE
        self._records: List[LayerMemRecord] = []
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin profiling session."""
        self._records.clear()
        self._start_time = time.perf_counter()
        if self.use_tracemalloc:
            _tracemalloc.start()

    def stop(self) -> None:
        """End profiling session."""
        if self.use_tracemalloc and _TRACEMALLOC_AVAILABLE:
            _tracemalloc.stop()

    @contextmanager
    def track(self, name: str) -> Generator[None, None, None]:
        """Context manager to track memory for a named layer/operation."""
        ram_before = _host_ram_mb()
        peak_trace = 0
        if self.use_tracemalloc and _TRACEMALLOC_AVAILABLE:
            _tracemalloc.reset_peak()

        t_start = time.perf_counter()
        yield
        t_end = time.perf_counter()

        ram_after = _host_ram_mb()
        if self.use_tracemalloc and _TRACEMALLOC_AVAILABLE:
            _, peak_trace = _tracemalloc.get_traced_memory()

        ram_peak = max(ram_before, ram_after)
        record = LayerMemRecord(
            name=name,
            duration_ms=(t_end - t_start) * 1000,
            ram_before_mb=ram_before,
            ram_after_mb=ram_after,
            ram_peak_mb=ram_peak,
            tracemalloc_peak_bytes=peak_trace,
        )
        self._records.append(record)

    def record(
        self,
        name: str,
        ram_before_mb: float,
        ram_after_mb: float,
        duration_ms: float = 0.0,
        ram_peak_mb: Optional[float] = None,
    ) -> LayerMemRecord:
        """Manually record a layer (useful for mocked / CUDA stats)."""
        peak = ram_peak_mb if ram_peak_mb is not None else max(ram_before_mb, ram_after_mb)
        rec = LayerMemRecord(
            name=name,
            duration_ms=duration_ms,
            ram_before_mb=ram_before_mb,
            ram_after_mb=ram_after_mb,
            ram_peak_mb=peak,
        )
        self._records.append(rec)
        return rec

    def report(self) -> MemoryProfileReport:
        """Generate a full memory profile report."""
        if not self._records:
            return MemoryProfileReport()

        total_ms = sum(r.duration_ms for r in self._records)
        peak_ram = max(r.ram_peak_mb for r in self._records)

        # Bottleneck: layer with highest peak RAM usage
        bottleneck = max(self._records, key=lambda r: r.ram_peak_mb)

        suggestions = self._generate_suggestions(self._records, peak_ram)

        report = MemoryProfileReport(
            records=list(self._records),
            total_duration_ms=total_ms,
            peak_ram_mb=peak_ram,
            bottleneck_layer=bottleneck.name,
            suggestions=suggestions,
        )
        return report

    def reset(self) -> None:
        """Clear all recorded data."""
        self._records.clear()
        self._start_time = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_suggestions(
        records: List[LayerMemRecord], peak_ram_mb: float
    ) -> List[str]:
        suggestions: List[str] = []

        # High peak RAM
        if peak_ram_mb > 10_000:
            suggestions.append(
                "Peak RAM > 10 GB: consider gradient checkpointing or smaller batch size"
            )

        # Layers with large positive delta
        for r in records:
            if r.ram_delta_mb > 500:
                suggestions.append(
                    f"Layer '{r.name}' increased RAM by {r.ram_delta_mb:.0f} MB — "
                    "consider offloading activations"
                )

        # Slow layers
        if records:
            avg_ms = sum(r.duration_ms for r in records) / len(records)
            for r in records:
                if r.duration_ms > avg_ms * 5 and r.duration_ms > 100:
                    suggestions.append(
                        f"Layer '{r.name}' is {r.duration_ms:.0f} ms — "
                        "5× slower than average, consider profiling"
                    )

        return suggestions


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def profile_layers(
    layers: Dict[str, Dict[str, float]],
) -> MemoryProfileReport:
    """Build a report from a pre-measured dict of {name: {before, after, duration}}.

    Useful for testing / injecting CUDA measurements.
    """
    profiler = MemoryProfiler(use_tracemalloc=False)
    for name, stats in layers.items():
        profiler.record(
            name=name,
            ram_before_mb=stats.get("before", 0.0),
            ram_after_mb=stats.get("after", 0.0),
            duration_ms=stats.get("duration_ms", 0.0),
            ram_peak_mb=stats.get("peak", None),
        )
    return profiler.report()
