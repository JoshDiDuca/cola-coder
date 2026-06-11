"""Extensible data pipeline with pluggable sources, filters, and transforms.

This is the core of the pipeline architecture. It defines:

  - DataRecord: A single code file flowing through the pipeline
  - DataSource: Base class for all data sources (HF, GitHub, local, etc.)
  - FilterPlugin: Base class for all filters (quality, length, dedup, etc.)
  - Transform: Base class for data transforms (whitespace, metadata, etc.)
  - PipelineConfig: Pipeline configuration (loadable from YAML)
  - DataPipeline: Composes sources + filters + transforms into a pipeline

Think of it like Express middleware for data:
  source.stream() → filter.check() → transform.apply() → yield record

The existing download.py and quality_filter.py are NOT modified.
They are wrapped by HuggingFaceSource and QualityFilterPlugin respectively.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class DataRecord:
    """A single code file flowing through the pipeline.

    Like a Request object in Express — carries the data plus metadata
    that gets enriched as it flows through middleware (transforms).
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class DataSource(ABC):
    """Base class for all data sources.

    Subclasses must implement name() and stream().
    Optionally override estimate_size() for progress tracking.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @abstractmethod
    def stream(self) -> Iterator[DataRecord]:
        """Yield DataRecord objects from this source."""
        ...

    def estimate_size(self) -> int | None:
        """Optional: estimated number of records. Used for progress bars."""
        return None


class FilterPlugin(ABC):
    """Base class for all filters.

    Filters decide whether to keep or reject a record.
    Like Express middleware that can short-circuit the request.
    """

    @abstractmethod
    def name(self) -> str:
        """Filter name for stats tracking."""
        ...

    @abstractmethod
    def check(self, record: DataRecord) -> tuple[bool, str]:
        """Check whether to keep this record.

        Returns:
            (keep, reason) — keep=True means the record passes.
            reason is only used when keep=False (for stats).
        """
        ...

    def setup(self, config: dict[str, Any]) -> None:
        """Optional: configure the filter from YAML config."""
        pass


class Transform(ABC):
    """Base class for data transforms (modify records, don't filter).

    Like Express middleware that enriches the request object
    without blocking it.
    """

    @abstractmethod
    def name(self) -> str:
        """Transform name for logging."""
        ...

    @abstractmethod
    def apply(self, record: DataRecord) -> DataRecord:
        """Transform a record. Return the modified record."""
        ...


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    """Tracks records kept, rejected, and rejection reasons across filters."""

    total: int = 0
    kept: int = 0
    rejected: int = 0
    filter_rejections: dict[str, dict[str, int]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def record_kept(self) -> None:
        self.total += 1
        self.kept += 1

    def record_rejection(self, filter_name: str, reason: str) -> None:
        self.total += 1
        self.rejected += 1
        if filter_name not in self.filter_rejections:
            self.filter_rejections[filter_name] = {}
        reasons = self.filter_rejections[filter_name]
        reasons[reason] = reasons.get(reason, 0) + 1

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        rate = self.total / max(elapsed, 0.001)
        lines = [
            f"Pipeline results ({elapsed:.1f}s, {rate:.0f} records/sec):",
            f"  Total:    {self.total:,}",
            f"  Kept:     {self.kept:,} ({self.kept / max(self.total, 1) * 100:.1f}%)",
            f"  Rejected: {self.rejected:,} ({self.rejected / max(self.total, 1) * 100:.1f}%)",
        ]
        if self.filter_rejections:
            lines.append("  Rejections by filter:")
            for fname, reasons in self.filter_rejections.items():
                total_rej = sum(reasons.values())
                lines.append(f"    {fname}: {total_rej:,}")
                for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"      {reason}: {count:,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Pipeline configuration, loadable from YAML.

    Like a tsconfig.json but for the data pipeline — declares what
    sources to pull from, what filters to apply, and what transforms
    to run.
    """
    sources: list[dict[str, Any]] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    transforms: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load pipeline config from a YAML file."""
        import yaml

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        pipeline = raw.get("pipeline", raw)
        return cls(
            sources=pipeline.get("sources", []),
            filters=pipeline.get("filters", []),
            transforms=pipeline.get("transforms", []),
            output=pipeline.get("output", {}),
        )


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """Compose sources, filters, and transforms into a processing pipeline.

    This is the main orchestrator. Think of it like an Express app:
      app.use(source)    → where the data comes from
      app.use(filter)    → middleware that can reject requests
      app.use(transform) → middleware that enriches requests

    Usage:
        config = PipelineConfig.from_yaml("configs/pipeline.yaml")
        pipeline = DataPipeline(config)
        for record in pipeline.stream():
            process(record.content)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        sources: list[DataSource] | None = None,
        filters: list[FilterPlugin] | None = None,
        transforms: list[Transform] | None = None,
    ):
        if config is not None:
            self.sources = self._build_sources(config.sources)
            self.filters = self._build_filters(config.filters)
            self.transforms = self._build_transforms(config.transforms)
        else:
            self.sources = sources or []
            self.filters = filters or []
            self.transforms = transforms or []
        self.stats = PipelineStats()

    def _build_sources(self, source_configs: list[dict[str, Any]]) -> list[DataSource]:
        """Instantiate sources from config dicts using the registry."""
        from cola_coder.data.registry import get_source

        sources = []
        for cfg in source_configs:
            cfg = dict(cfg)  # copy so we don't mutate
            source_type = cfg.pop("type")
            # weight is used by MixedSource, not the source itself
            cfg.pop("weight", None)
            source_cls = get_source(source_type)
            sources.append(source_cls(**cfg))
        return sources

    def _build_filters(self, filter_configs: list[dict[str, Any]]) -> list[FilterPlugin]:
        """Instantiate filters from config dicts using the registry."""
        from cola_coder.data.registry import get_filter

        filters = []
        for cfg in filter_configs:
            cfg = dict(cfg)
            filter_type = cfg.pop("type")
            filter_cls = get_filter(filter_type)
            instance = filter_cls()
            if cfg:
                instance.setup(cfg)
            filters.append(instance)
        return filters

    def _build_transforms(
        self, transform_configs: list[dict[str, Any]]
    ) -> list[Transform]:
        """Instantiate transforms from config dicts using the registry."""
        from cola_coder.data.registry import get_transform

        transforms = []
        for cfg in transform_configs:
            cfg = dict(cfg)
            transform_type = cfg.pop("type")
            transform_cls = get_transform(transform_type)
            transforms.append(transform_cls(**cfg) if cfg else transform_cls())
        return transforms

    def stream(self) -> Iterator[DataRecord]:
        """Yield processed records through the full pipeline.

        Records flow: sources → filters → transforms → yield
        """
        self.stats = PipelineStats()

        for source in self.sources:
            for record in source.stream():
                # Run all filters (short-circuit on first rejection)
                keep = True
                for f in self.filters:
                    passed, reason = f.check(record)
                    if not passed:
                        self.stats.record_rejection(f.name(), reason)
                        keep = False
                        break
                if not keep:
                    continue

                # Run all transforms
                for t in self.transforms:
                    record = t.apply(record)

                self.stats.record_kept()
                yield record

    def content_stream(self) -> Iterator[str]:
        """Convenience: yield just content strings (for tokenizer compatibility)."""
        for record in self.stream():
            yield record.content
