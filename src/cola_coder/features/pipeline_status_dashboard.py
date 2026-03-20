"""Pipeline Status Dashboard — tracks and displays ML pipeline stage progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class PipelineStage:
    name: str
    status: str = "pending"  # 'pending'|'running'|'completed'|'failed'|'skipped'
    progress: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    details: dict = field(default_factory=dict)


class PipelineStatusDashboard:
    DEFAULT_STAGES = [
        "data_prep",
        "tokenization",
        "training",
        "evaluation",
        "export",
        "deploy",
    ]

    def __init__(self, stages: list[str] = None):
        stage_names = stages if stages is not None else self.DEFAULT_STAGES
        self._stages: dict[str, PipelineStage] = {
            name: PipelineStage(name=name) for name in stage_names
        }
        self._order: list[str] = list(stage_names)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update_stage(
        self,
        name: str,
        status: str,
        progress: float = 0.0,
        details: dict = None,
    ) -> None:
        if name not in self._stages:
            self._stages[name] = PipelineStage(name=name)
            self._order.append(name)

        stage = self._stages[name]
        now = datetime.now().isoformat(timespec="seconds")

        # Track start/end timestamps automatically
        if status == "running" and stage.status != "running":
            stage.start_time = now
        elif status in ("completed", "failed", "skipped"):
            if stage.end_time is None:
                stage.end_time = now
            if stage.start_time is None:
                stage.start_time = now

        stage.status = status
        stage.progress = max(0.0, min(1.0, progress))
        if details is not None:
            stage.details = details

    def reset(self) -> None:
        for name in self._order:
            self._stages[name] = PipelineStage(name=name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_stage(self, name: str) -> PipelineStage:
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not found")
        return self._stages[name]

    def overall_progress(self) -> float:
        if not self._order:
            return 0.0
        total = 0.0
        for name in self._order:
            stage = self._stages[name]
            if stage.status == "completed":
                total += 1.0
            elif stage.status == "skipped":
                total += 1.0
            elif stage.status == "running":
                total += stage.progress
            # pending / failed contribute 0
        return total / len(self._order)

    def current_stage(self) -> Optional[PipelineStage]:
        for name in self._order:
            if self._stages[name].status == "running":
                return self._stages[name]
        return None

    def is_complete(self) -> bool:
        return all(
            self._stages[name].status in ("completed", "skipped")
            for name in self._order
        )

    def failed_stages(self) -> list[PipelineStage]:
        return [
            self._stages[name]
            for name in self._order
            if self._stages[name].status == "failed"
        ]

    def summary(self) -> dict:
        counts: dict[str, int] = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
        }
        for name in self._order:
            s = self._stages[name].status
            counts[s] = counts.get(s, 0) + 1

        return {
            "stages": {
                name: {
                    "status": self._stages[name].status,
                    "progress": self._stages[name].progress,
                    "start_time": self._stages[name].start_time,
                    "end_time": self._stages[name].end_time,
                    "details": self._stages[name].details,
                }
                for name in self._order
            },
            "overall_progress": self.overall_progress(),
            "is_complete": self.is_complete(),
            "status_counts": counts,
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def format_dashboard(self) -> str:
        WIDTH = 50
        BAR_LEN = 20

        STATUS_ICONS = {
            "pending": "·",
            "running": "▶",
            "completed": "✓",
            "failed": "✗",
            "skipped": "⊘",
        }

        lines: list[str] = []
        lines.append("=" * WIDTH)
        lines.append(" ML PIPELINE STATUS DASHBOARD")
        lines.append("=" * WIDTH)

        overall = self.overall_progress()
        filled = int(overall * BAR_LEN)
        overall_bar = "[" + "█" * filled + "░" * (BAR_LEN - filled) + "]"
        lines.append(f" Overall  {overall_bar} {overall * 100:.1f}%")
        lines.append("-" * WIDTH)

        for name in self._order:
            stage = self._stages[name]
            icon = STATUS_ICONS.get(stage.status, "?")
            filled_s = int(stage.progress * BAR_LEN)
            bar = "[" + "█" * filled_s + "░" * (BAR_LEN - filled_s) + "]"
            status_label = stage.status.upper().ljust(9)
            lines.append(f" {icon} {name.ljust(14)} {status_label} {bar} {stage.progress * 100:.0f}%")

            if stage.details:
                detail_items = ", ".join(
                    f"{k}={v}" for k, v in list(stage.details.items())[:4]
                )
                lines.append(f"   └─ {detail_items}")

        lines.append("=" * WIDTH)

        failed = self.failed_stages()
        if failed:
            lines.append(f" FAILED STAGES: {', '.join(s.name for s in failed)}")
            lines.append("=" * WIDTH)

        return "\n".join(lines)
