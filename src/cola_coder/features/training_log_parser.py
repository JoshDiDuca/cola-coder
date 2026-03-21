"""Training Log Parser: extract training metrics from log files.

Parses plain-text or structured training logs produced by scripts/train.py
and the TrainingMonitor feature.  Extracts loss curves, learning rates,
gradient norms, and step timing without touching the live training process.

For a TS dev: like parsing a JSON server log — pull out the numbers, give you
a typed data structure you can query or export.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the training log parser feature is active."""
    return FEATURE_ENABLED


@dataclass
class StepRecord:
    """Metrics captured at a single training step."""

    step: int
    loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    epoch: int | None = None
    elapsed_sec: float | None = None


@dataclass
class TrainingLog:
    """Parsed result of a training log file."""

    source_path: str = ""
    records: list[StepRecord] = field(default_factory=list)
    # Aggregates (computed after parsing)
    min_loss: float | None = None
    max_loss: float | None = None
    final_loss: float | None = None
    min_lr: float | None = None
    max_lr: float | None = None
    total_steps: int = 0
    total_epochs: int = 0

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    def loss_curve(self) -> list[tuple[int, float]]:
        """Return (step, loss) pairs for all records with a loss value."""
        return [(r.step, r.loss) for r in self.records if r.loss is not None]

    def lr_curve(self) -> list[tuple[int, float]]:
        """Return (step, lr) pairs for all records with an LR value."""
        return [(r.step, r.lr) for r in self.records if r.lr is not None]

    def grad_norm_curve(self) -> list[tuple[int, float]]:
        """Return (step, grad_norm) pairs."""
        return [(r.step, r.grad_norm) for r in self.records if r.grad_norm is not None]

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the full log to a JSON string."""
        return json.dumps(
            {
                "source_path": self.source_path,
                "total_steps": self.total_steps,
                "total_epochs": self.total_epochs,
                "min_loss": self.min_loss,
                "max_loss": self.max_loss,
                "final_loss": self.final_loss,
                "records": [asdict(r) for r in self.records],
            },
            indent=2,
        )

    def to_csv(self, file: IO[str] | None = None) -> str:
        """Serialize records to CSV.  Returns CSV string (or writes to *file*)."""
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["step", "loss", "lr", "grad_norm", "tokens_per_sec", "epoch", "elapsed_sec"]
        )
        for r in self.records:
            writer.writerow(
                [r.step, r.loss, r.lr, r.grad_norm, r.tokens_per_sec, r.epoch, r.elapsed_sec]
            )
        result = buf.getvalue()
        if file is not None:
            file.write(result)
        return result

    def plot_ascii(self, metric: str = "loss", width: int = 60, height: int = 12) -> str:
        """Render an ASCII line chart of *metric* vs step.

        Args:
            metric: One of ``"loss"``, ``"lr"``, ``"grad_norm"``.
            width: Chart width in characters.
            height: Chart height in lines.

        Returns:
            Multi-line string with the ASCII chart.
        """
        if metric == "loss":
            curve = self.loss_curve()
        elif metric == "lr":
            curve = self.lr_curve()
        elif metric == "grad_norm":
            curve = self.grad_norm_curve()
        else:
            return f"Unknown metric: {metric}"

        if not curve:
            return f"No data for metric '{metric}'"

        steps = [s for s, _ in curve]
        values = [v for _, v in curve]

        v_min = min(values)
        v_max = max(values)
        v_range = v_max - v_min or 1.0

        # Sample down to width data points
        step_count = len(steps)
        if step_count > width:
            indices = [int(i * step_count / width) for i in range(width)]
        else:
            indices = list(range(step_count))

        sampled = [values[i] for i in indices]

        # Build grid
        grid = [[" "] * len(sampled) for _ in range(height)]
        for col, val in enumerate(sampled):
            row = int((val - v_min) / v_range * (height - 1))
            row = height - 1 - row  # flip: top = high
            grid[row][col] = "▪"

        lines: list[str] = []
        for r_idx, row in enumerate(grid):
            val_at_row = v_max - (r_idx / (height - 1)) * v_range if height > 1 else v_max
            lines.append(f"{val_at_row:8.4f} |{''.join(row)}")

        step_label = f"{'':9s}+{'-' * len(sampled)}"
        lines.append(step_label)
        lines.append(
            f"{'':10s}{steps[0]}{'':>{max(1, len(sampled)-len(str(steps[0]))-len(str(steps[-1])))}}{steps[-1]}"
        )
        header = f"\n  {metric.upper()} over steps ({len(curve)} points)\n"
        return header + "\n".join(lines)


# ---------------------------------------------------------------------------
# Log patterns
# ---------------------------------------------------------------------------

# Common pattern:  step=1000  loss=2.3456  lr=3.00e-04  grad_norm=1.234
_STEP_RE = re.compile(
    r"step[=:\s]+(?P<step>\d+)"
    r"(?:.*?loss[=:\s]+(?P<loss>[0-9]+\.[0-9]+(?:[eE][+-]?\d+)?))?"
    r"(?:.*?lr[=:\s]+(?P<lr>[0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?))?"
    r"(?:.*?grad[_\s]?norm[=:\s]+(?P<grad_norm>[0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?))?"
    r"(?:.*?(?:tok/s|tokens_per_sec|tps)[=:\s]+(?P<tps>[0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?))?"
    r"(?:.*?epoch[=:\s]+(?P<epoch>\d+))?",
    re.IGNORECASE,
)

# JSON-per-line logs: {"step": 1000, "loss": 2.34, ...}
_JSON_LINE_RE = re.compile(r"^\s*\{.*\}\s*$")


class TrainingLogParser:
    """Parse training log files into structured TrainingLog objects.

    Supports:
    - Plain-text logs with key=value or key: value patterns
    - JSON-per-line logs
    - CSV logs with header row

    Usage::

        parser = TrainingLogParser()
        log = parser.parse("runs/train_2024.log")
        print(log.plot_ascii("loss"))
    """

    def parse(self, log_path: str | Path) -> TrainingLog:
        """Parse a training log file.

        Args:
            log_path: Path to the log file.

        Returns:
            TrainingLog with all extracted records.
        """
        path = Path(log_path)
        log = TrainingLog(source_path=str(path))

        if not path.exists():
            return log

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        # Detect format
        if lines and _JSON_LINE_RE.match(lines[0]):
            records = self._parse_jsonl(lines)
        elif lines and re.match(r"step\s*,", lines[0], re.IGNORECASE):
            records = self._parse_csv(lines)
        else:
            records = self._parse_text(lines)

        log.records = records
        self._compute_aggregates(log)
        return log

    def parse_text(self, text: str) -> TrainingLog:
        """Parse log content from a string (useful for testing).

        Args:
            text: Raw log text.

        Returns:
            TrainingLog with extracted records.
        """
        log = TrainingLog(source_path="<string>")
        lines = text.splitlines()
        if lines and _JSON_LINE_RE.match(lines[0]):
            log.records = self._parse_jsonl(lines)
        else:
            log.records = self._parse_text(lines)
        self._compute_aggregates(log)
        return log

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_text(lines: list[str]) -> list[StepRecord]:
        records: list[StepRecord] = []
        seen_steps: set[int] = set()
        for line in lines:
            m = _STEP_RE.search(line)
            if not m or m.group("step") is None:
                continue
            step = int(m.group("step"))
            if step in seen_steps:
                continue
            seen_steps.add(step)
            record = StepRecord(step=step)
            if m.group("loss"):
                record.loss = float(m.group("loss"))
            if m.group("lr"):
                record.lr = float(m.group("lr"))
            if m.group("grad_norm"):
                record.grad_norm = float(m.group("grad_norm"))
            if m.group("tps"):
                record.tokens_per_sec = float(m.group("tps"))
            if m.group("epoch"):
                record.epoch = int(m.group("epoch"))
            records.append(record)
        records.sort(key=lambda r: r.step)
        return records

    @staticmethod
    def _parse_jsonl(lines: list[str]) -> list[StepRecord]:
        records: list[StepRecord] = []
        seen_steps: set[int] = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = data.get("step") or data.get("global_step")
            if step is None:
                continue
            step = int(step)
            if step in seen_steps:
                continue
            seen_steps.add(step)
            record = StepRecord(
                step=step,
                loss=_float_or_none(data.get("loss") or data.get("train_loss")),
                lr=_float_or_none(data.get("lr") or data.get("learning_rate")),
                grad_norm=_float_or_none(data.get("grad_norm")),
                tokens_per_sec=_float_or_none(
                    data.get("tokens_per_sec") or data.get("tps")
                ),
                epoch=_int_or_none(data.get("epoch")),
                elapsed_sec=_float_or_none(data.get("elapsed") or data.get("elapsed_sec")),
            )
            records.append(record)
        records.sort(key=lambda r: r.step)
        return records

    @staticmethod
    def _parse_csv(lines: list[str]) -> list[StepRecord]:
        records: list[StepRecord] = []
        reader = csv.DictReader(lines)
        seen_steps: set[int] = set()
        for row in reader:
            step = _int_or_none(row.get("step"))
            if step is None or step in seen_steps:
                continue
            seen_steps.add(step)
            records.append(
                StepRecord(
                    step=step,
                    loss=_float_or_none(row.get("loss")),
                    lr=_float_or_none(row.get("lr")),
                    grad_norm=_float_or_none(row.get("grad_norm")),
                    tokens_per_sec=_float_or_none(row.get("tokens_per_sec")),
                    epoch=_int_or_none(row.get("epoch")),
                    elapsed_sec=_float_or_none(row.get("elapsed_sec")),
                )
            )
        records.sort(key=lambda r: r.step)
        return records

    @staticmethod
    def _compute_aggregates(log: TrainingLog) -> None:
        losses = [r.loss for r in log.records if r.loss is not None]
        lrs = [r.lr for r in log.records if r.lr is not None]
        epochs = [r.epoch for r in log.records if r.epoch is not None]

        log.total_steps = max((r.step for r in log.records), default=0)
        log.total_epochs = max(epochs, default=0)
        if losses:
            log.min_loss = min(losses)
            log.max_loss = max(losses)
            log.final_loss = losses[-1]
        if lrs:
            log.min_lr = min(lrs)
            log.max_lr = max(lrs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _float_or_none(val: object) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _int_or_none(val: object) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
