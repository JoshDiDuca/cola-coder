"""Checkpoint leaderboard: rank and compare training checkpoints by evaluation metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class CheckpointEntry:
    name: str
    step: int
    metrics: dict[str, float]
    timestamp: str
    path: str


class Leaderboard:
    def __init__(self, primary_metric: str = "loss", lower_is_better: bool = True) -> None:
        self.primary_metric = primary_metric
        self.lower_is_better = lower_is_better
        self._entries: list[CheckpointEntry] = []

    def add_entry(self, entry: CheckpointEntry) -> None:
        """Add a checkpoint entry to the leaderboard."""
        self._entries.append(entry)

    def _sort_key(self, entry: CheckpointEntry) -> float:
        value = entry.metrics.get(self.primary_metric, float("inf"))
        return value if self.lower_is_better else -value

    def rank(self, top_k: int = 10) -> list[CheckpointEntry]:
        """Return entries sorted by primary metric, best first, limited to top_k."""
        sorted_entries = sorted(self._entries, key=self._sort_key)
        return sorted_entries[:top_k]

    def best(self) -> CheckpointEntry:
        """Return the entry with the best primary metric value."""
        if not self._entries:
            raise ValueError("Leaderboard is empty")
        return sorted(self._entries, key=self._sort_key)[0]

    def worst(self) -> CheckpointEntry:
        """Return the entry with the worst primary metric value."""
        if not self._entries:
            raise ValueError("Leaderboard is empty")
        return sorted(self._entries, key=self._sort_key)[-1]

    def get_by_name(self, name: str) -> Optional[CheckpointEntry]:
        """Look up an entry by checkpoint name."""
        for entry in self._entries:
            if entry.name == name:
                return entry
        return None

    def format_table(self) -> str:
        """Return a formatted text table of all entries ranked by primary metric."""
        if not self._entries:
            return f"Leaderboard (primary_metric={self.primary_metric}): empty\n"

        ranked = self.rank(top_k=len(self._entries))

        # Collect all metric keys across entries for column headers
        all_metric_keys: list[str] = []
        seen: set[str] = set()
        for entry in ranked:
            for key in entry.metrics:
                if key not in seen:
                    all_metric_keys.append(key)
                    seen.add(key)

        # Column widths
        rank_w = max(4, len("Rank"))
        name_w = max(max(len(e.name) for e in ranked), len("Name"))
        step_w = max(max(len(str(e.step)) for e in ranked), len("Step"))
        ts_w = max(max(len(e.timestamp) for e in ranked), len("Timestamp"))

        metric_widths: dict[str, int] = {}
        for key in all_metric_keys:
            vals = [entry.metrics.get(key) for entry in ranked]
            val_strs = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
            metric_widths[key] = max(len(key), max(len(s) for s in val_strs))

        # Build header
        parts = [
            f"{'Rank':<{rank_w}}",
            f"{'Name':<{name_w}}",
            f"{'Step':<{step_w}}",
            f"{'Timestamp':<{ts_w}}",
        ]
        for key in all_metric_keys:
            w = metric_widths[key]
            marker = "*" if key == self.primary_metric else ""
            header = f"{marker}{key}"
            parts.append(f"{header:<{w}}")

        header_line = "  ".join(parts)
        sep = "-" * len(header_line)

        lines = [
            f"Leaderboard (primary_metric={self.primary_metric}, "
            f"lower_is_better={self.lower_is_better})",
            sep,
            header_line,
            sep,
        ]

        for i, entry in enumerate(ranked, start=1):
            row_parts = [
                f"{i:<{rank_w}}",
                f"{entry.name:<{name_w}}",
                f"{entry.step:<{step_w}}",
                f"{entry.timestamp:<{ts_w}}",
            ]
            for key in all_metric_keys:
                w = metric_widths[key]
                v = entry.metrics.get(key)
                val_str = f"{v:.4f}" if v is not None else "N/A"
                row_parts.append(f"{val_str:<{w}}")
            lines.append("  ".join(row_parts))

        lines.append(sep)
        return "\n".join(lines) + "\n"

    def save(self, path: str) -> None:
        """Save the leaderboard to a JSON file."""
        data = {
            "primary_metric": self.primary_metric,
            "lower_is_better": self.lower_is_better,
            "entries": [asdict(e) for e in self._entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Leaderboard":
        """Load a leaderboard from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lb = cls(
            primary_metric=data["primary_metric"],
            lower_is_better=data["lower_is_better"],
        )
        for entry_dict in data["entries"]:
            lb.add_entry(CheckpointEntry(**entry_dict))
        return lb

    def summary(self) -> dict:
        """Return a summary dict of leaderboard statistics."""
        if not self._entries:
            return {
                "primary_metric": self.primary_metric,
                "lower_is_better": self.lower_is_better,
                "total_entries": 0,
                "best": None,
                "worst": None,
            }

        best_entry = self.best()
        worst_entry = self.worst()

        return {
            "primary_metric": self.primary_metric,
            "lower_is_better": self.lower_is_better,
            "total_entries": len(self._entries),
            "best": {
                "name": best_entry.name,
                "step": best_entry.step,
                "metric_value": best_entry.metrics.get(self.primary_metric),
            },
            "worst": {
                "name": worst_entry.name,
                "step": worst_entry.step,
                "metric_value": worst_entry.metrics.get(self.primary_metric),
            },
        }
