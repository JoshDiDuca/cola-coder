"""Real-time statistics tracking during data processing and training.

Tracks token counts, language distribution, sample lengths, and processing speed
as samples flow through the pipeline. Optionally renders a live Rich table;
falls back to plain text if Rich is not installed.
"""

import time
from collections import defaultdict
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# DataStats
# ---------------------------------------------------------------------------

class DataStats:
    """Accumulates per-sample statistics in real time."""

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, sample: str, language: str = "unknown") -> None:
        """Record a single sample."""
        self._total_samples += 1
        tokens = len(sample.split())
        self._total_tokens += tokens
        self._total_length += len(sample)
        self._language_counts[language] += 1
        # Record wall-clock time of first and most-recent update for speed calc
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        self._last_time = now

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._total_samples: int = 0
        self._total_tokens: int = 0
        self._total_length: int = 0
        self._language_counts: dict[str, int] = defaultdict(int)
        self._start_time: Optional[float] = None
        self._last_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def total_samples(self) -> int:
        """Total number of samples recorded."""
        return self._total_samples

    def total_tokens(self) -> int:
        """Approximate total tokens (whitespace-split)."""
        return self._total_tokens

    def language_distribution(self) -> dict[str, int]:
        """Mapping of language -> sample count."""
        return dict(self._language_counts)

    def avg_sample_length(self) -> float:
        """Mean character length of recorded samples."""
        if self._total_samples == 0:
            return 0.0
        return self._total_length / self._total_samples

    def samples_per_second(self) -> float:
        """Throughput in samples/second since the first update."""
        if self._start_time is None or self._last_time is None:
            return 0.0
        elapsed = self._last_time - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_samples / elapsed

    def summary(self) -> dict:
        """Return a dict snapshot of all current statistics."""
        return {
            "total_samples": self.total_samples(),
            "total_tokens": self.total_tokens(),
            "avg_sample_length": self.avg_sample_length(),
            "samples_per_second": self.samples_per_second(),
            "language_distribution": self.language_distribution(),
        }


# ---------------------------------------------------------------------------
# LiveStatsDisplay
# ---------------------------------------------------------------------------

def _try_import_rich():
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live
        return Console, Table, Live
    except ImportError:
        return None, None, None


_Console, _Table, _Live = _try_import_rich()
_RICH_AVAILABLE = _Console is not None


class LiveStatsDisplay:
    """Renders DataStats as a live table (Rich if available, else plain text)."""

    def __init__(self) -> None:
        self._console = _Console() if _RICH_AVAILABLE else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, stats: DataStats) -> None:
        """Refresh the display with the latest stats."""
        if _RICH_AVAILABLE and self._console is not None:
            self._console.print(self._build_rich_table(stats))
        else:
            print(self.format_table(stats))

    def format_table(self, stats: DataStats) -> str:
        """Return a plain-text table string summarising *stats*."""
        s = stats.summary()
        lang_lines = "\n".join(
            f"    {lang:<20} {count:>8}"
            for lang, count in sorted(s["language_distribution"].items(), key=lambda x: -x[1])
        ) or "    (none)"

        return (
            "=" * 50 + "\n"
            "  Data Processing Statistics\n"
            "=" * 50 + "\n"
            f"  Total samples       : {s['total_samples']:>10,}\n"
            f"  Total tokens        : {s['total_tokens']:>10,}\n"
            f"  Avg sample length   : {s['avg_sample_length']:>10.1f} chars\n"
            f"  Processing speed    : {s['samples_per_second']:>10.1f} samples/s\n"
            "  Language distribution:\n"
            f"{lang_lines}\n"
            "=" * 50
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rich_table(self, stats: DataStats):
        """Build a Rich Table object from the current stats."""
        s = stats.summary()
        table = _Table(title="Data Processing Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total samples", f"{s['total_samples']:,}")
        table.add_row("Total tokens", f"{s['total_tokens']:,}")
        table.add_row("Avg sample length", f"{s['avg_sample_length']:.1f} chars")
        table.add_row("Processing speed", f"{s['samples_per_second']:.1f} samples/s")
        table.add_section()

        for lang, count in sorted(s["language_distribution"].items(), key=lambda x: -x[1]):
            table.add_row(f"  {lang}", str(count))

        return table
