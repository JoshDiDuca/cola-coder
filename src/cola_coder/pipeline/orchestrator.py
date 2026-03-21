"""Pipeline orchestrator for end-to-end train → eval → export workflows.

Runs each pipeline stage as a subprocess, capturing output, tracking artifacts,
and saving state for resume. Stages:
  tokenizer → data_prep → training → smoke_test → evaluation → export

Usage:
    from cola_coder.pipeline.orchestrator import PipelineOrchestrator, PipelineStage

    orch = PipelineOrchestrator("configs/tiny.yaml")
    results = orch.run()
    print(orch.format_report())
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


class PipelineStage(Enum):
    TOKENIZER = "tokenizer"
    DATA_PREP = "data_prep"
    TRAINING = "training"
    SMOKE_TEST = "smoke_test"
    EVALUATION = "evaluation"
    EXPORT = "export"


@dataclass
class StageResult:
    stage: PipelineStage
    success: bool
    message: str
    duration_seconds: float
    artifacts: dict = field(default_factory=dict)
    skipped: bool = False
    returncode: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class PipelineOrchestrator:
    """Run the complete training pipeline end-to-end.

    Each stage is executed as a subprocess using the scripts/ entry points.
    State is persisted to a JSON sidecar so interrupted runs can resume.
    """

    # Map each stage to its handler method name
    _STAGE_HANDLERS = {
        PipelineStage.TOKENIZER: "_run_tokenizer",
        PipelineStage.DATA_PREP: "_run_data_prep",
        PipelineStage.TRAINING: "_run_training",
        PipelineStage.SMOKE_TEST: "_run_smoke_test",
        PipelineStage.EVALUATION: "_run_evaluation",
        PipelineStage.EXPORT: "_run_export",
    }

    def __init__(
        self,
        config_path: str,
        stages: Optional[list[PipelineStage]] = None,
        skip_existing: bool = True,
        auto_resume: bool = True,
        continue_on_failure: bool = False,
        export_format: str = "gguf-q8",
        dry_run: bool = False,
        log_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            config_path: Path to model config YAML (e.g. "configs/tiny.yaml").
            stages: Which stages to run (default: all in order).
            skip_existing: Skip stages whose outputs already exist.
            auto_resume: Pass --auto-resume to the training script.
            continue_on_failure: Don't stop the pipeline when a stage fails.
            export_format: Action passed to export_model.py --action (default: gguf-q8).
            dry_run: Show what would run without actually running anything.
            log_dir: Directory for per-stage log files (default: pipeline_logs/).
        """
        self.config_path = str(config_path)
        self.stages = stages if stages is not None else list(PipelineStage)
        self.skip_existing = skip_existing
        self.auto_resume = auto_resume
        self.continue_on_failure = continue_on_failure
        self.export_format = export_format
        self.dry_run = dry_run
        self.results: list[StageResult] = []

        # Resolve project root: two levels up from this file (src/cola_coder/pipeline/)
        self._project_root = Path(__file__).resolve().parent.parent.parent.parent
        self._scripts_dir = self._project_root / "scripts"

        # Python executable inside the project venv
        if sys.platform == "win32":
            self._python = str(self._project_root / ".venv" / "Scripts" / "python")
        else:
            self._python = str(self._project_root / ".venv" / "bin" / "python")

        # Log directory
        if log_dir is not None:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = self._project_root / "pipeline_logs"

        # State file for resume support
        config_stem = Path(self.config_path).stem
        self._state_file = self._project_root / f".pipeline_state_{config_stem}.json"

        # Accumulated artifacts shared across stages
        self._artifacts: dict = {}
        self._load_state()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> list[StageResult]:
        """Run all configured stages in order.

        Returns the list of StageResult objects (one per stage that was
        attempted, skipped, or executed).
        """
        self._log_dir.mkdir(parents=True, exist_ok=True)

        for stage in self.stages:
            handler_name = self._STAGE_HANDLERS[stage]
            handler = getattr(self, handler_name)

            if self.dry_run:
                complete = self._check_stage_complete(stage)
                skip_reason = " (would skip — outputs exist)" if (self.skip_existing and complete) else ""
                result = StageResult(
                    stage=stage,
                    success=True,
                    message=f"[dry-run] would execute{skip_reason}",
                    duration_seconds=0.0,
                    skipped=True,
                )
                self.results.append(result)
                continue

            if self.skip_existing and self._check_stage_complete(stage):
                result = StageResult(
                    stage=stage,
                    success=True,
                    message="Skipped — outputs already exist",
                    duration_seconds=0.0,
                    skipped=True,
                )
                self.results.append(result)
                logger.info("Stage %s: skipped (outputs exist)", stage.value)
                continue

            logger.info("Stage %s: starting", stage.value)
            result = handler()
            self.results.append(result)
            self._save_state()

            if not result.success and not self.continue_on_failure:
                logger.error(
                    "Stage %s failed: %s — stopping pipeline", stage.value, result.message
                )
                break

        return self.results

    def format_report(self) -> str:
        """Format pipeline results as a Rich table (falls back to plain text)."""
        try:
            return self._format_report_rich()
        except ImportError:
            return self._format_report_plain()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_tokenizer(self) -> StageResult:
        """Train tokenizer if tokenizer.json doesn't exist."""
        cmd = [
            self._python,
            str(self._scripts_dir / "train_tokenizer.py"),
        ]
        return self._run_stage(PipelineStage.TOKENIZER, cmd, timeout=3600)

    def _run_data_prep(self) -> StageResult:
        """Prepare data if train_data.npy doesn't exist."""
        tokenizer_path = self._artifacts.get("tokenizer_path", "tokenizer.json")
        cmd = [
            self._python,
            str(self._scripts_dir / "prepare_data.py"),
            "--config", self.config_path,
            "--tokenizer", tokenizer_path,
            "--output-name", "train_data",
        ]
        result = self._run_stage(PipelineStage.DATA_PREP, cmd, timeout=14400)

        # Track the data file as an artifact
        if result.success:
            try:
                from cola_coder.model.config import Config
                cfg = Config.from_yaml(self.config_path)
                data_dir = Path(cfg.data.data_dir) / "processed"
                data_file = str(data_dir / "train_data.npy")
            except Exception:
                data_file = "data/processed/train_data.npy"
            result.artifacts["data_path"] = data_file
            self._artifacts["data_path"] = data_file

        return result

    def _run_training(self) -> StageResult:
        """Run training with auto-resume."""
        cmd = [
            self._python,
            str(self._scripts_dir / "train.py"),
            "--config", self.config_path,
        ]
        # Pass explicit data path if we know it
        data_path = self._artifacts.get("data_path")
        if data_path and Path(data_path).exists():
            cmd += ["--data", data_path]

        if self.auto_resume:
            cmd.append("--auto-resume")

        result = self._run_stage(PipelineStage.TRAINING, cmd, timeout=None)

        # Discover latest checkpoint as an artifact
        if result.success:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                result.artifacts["checkpoint"] = checkpoint_path
                self._artifacts["checkpoint"] = checkpoint_path

        return result

    def _run_smoke_test(self) -> StageResult:
        """Run smoke test on latest checkpoint."""
        checkpoint = self._artifacts.get("checkpoint") or self._find_latest_checkpoint()
        cmd = [
            self._python,
            str(self._scripts_dir / "smoke_test.py"),
            "--config", self.config_path,
            "--json",  # machine-readable output, no interactive prompts
            "--quick",
        ]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]

        result = self._run_stage(PipelineStage.SMOKE_TEST, cmd, timeout=300)
        if checkpoint:
            result.artifacts["checkpoint"] = checkpoint
        return result

    def _run_evaluation(self) -> StageResult:
        """Run HumanEval evaluation."""
        checkpoint = self._artifacts.get("checkpoint") or self._find_latest_checkpoint()
        cmd = [
            self._python,
            str(self._scripts_dir / "evaluate.py"),
            "--config", self.config_path,
        ]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]

        result = self._run_stage(PipelineStage.EVALUATION, cmd, timeout=7200)
        if checkpoint:
            result.artifacts["checkpoint"] = checkpoint
        return result

    def _run_export(self) -> StageResult:
        """Export to GGUF format."""
        checkpoint = self._artifacts.get("checkpoint") or self._find_latest_checkpoint()
        if not checkpoint:
            return StageResult(
                stage=PipelineStage.EXPORT,
                success=False,
                message="No checkpoint found to export",
                duration_seconds=0.0,
                returncode=1,
            )

        cmd = [
            self._python,
            str(self._scripts_dir / "export_model.py"),
            "--checkpoint", checkpoint,
            "--config", self.config_path,
            "--action", self.export_format,
        ]

        result = self._run_stage(PipelineStage.EXPORT, cmd, timeout=3600)

        if result.success:
            # Derive the expected export path
            export_dir = str(Path(checkpoint).parent / "exports")
            result.artifacts["export_dir"] = export_dir
            self._artifacts["export_dir"] = export_dir

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Stage completion checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_stage_complete(self, stage: PipelineStage) -> bool:
        """Return True if the stage's expected outputs already exist."""
        if stage == PipelineStage.TOKENIZER:
            return self._tokenizer_exists()

        if stage == PipelineStage.DATA_PREP:
            return self._data_exists()

        if stage == PipelineStage.TRAINING:
            return bool(self._find_latest_checkpoint())

        if stage == PipelineStage.SMOKE_TEST:
            # Always re-run smoke test — it's fast and validates the checkpoint
            return False

        if stage == PipelineStage.EVALUATION:
            # Always re-run evaluation (results may change between checkpoints)
            return False

        if stage == PipelineStage.EXPORT:
            return self._export_exists()

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_stage(
        self,
        stage: PipelineStage,
        cmd: list[str],
        timeout: Optional[float],
    ) -> StageResult:
        """Execute a subprocess command and return a StageResult."""
        log_file = self._log_dir / f"{stage.value}.log"
        t0 = time.monotonic()

        with open(log_file, "w", encoding="utf-8", errors="replace") as lf:
            lf.write(f"[{datetime.now().isoformat()}] Running: {' '.join(cmd)}\n\n")
            lf.flush()

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    env={**os.environ},
                    cwd=str(self._project_root),
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                duration = time.monotonic() - t0
                return StageResult(
                    stage=stage,
                    success=False,
                    message=f"Timed out after {timeout}s",
                    duration_seconds=duration,
                    returncode=-1,
                )
            except FileNotFoundError as exc:
                duration = time.monotonic() - t0
                return StageResult(
                    stage=stage,
                    success=False,
                    message=f"Executable not found: {exc}",
                    duration_seconds=duration,
                    returncode=-1,
                )
            except Exception as exc:  # noqa: BLE001
                duration = time.monotonic() - t0
                return StageResult(
                    stage=stage,
                    success=False,
                    message=f"Unexpected error: {exc}",
                    duration_seconds=duration,
                    returncode=-1,
                )

        duration = time.monotonic() - t0
        success = returncode == 0
        message = "OK" if success else f"Failed with exit code {returncode}"
        logger.info(
            "Stage %s finished in %.1fs — %s", stage.value, duration, message
        )

        return StageResult(
            stage=stage,
            success=success,
            message=message,
            duration_seconds=duration,
            returncode=returncode,
        )

    def _tokenizer_exists(self) -> bool:
        """Check whether tokenizer.json exists."""
        try:
            from cola_coder.model.config import get_storage_config
            storage = get_storage_config()
            return Path(storage.tokenizer_path).exists()
        except Exception:
            return Path("tokenizer.json").exists()

    def _data_exists(self) -> bool:
        """Check whether at least one training .npy file exists."""
        # Check for previously tracked artifact first
        data_path = self._artifacts.get("data_path")
        if data_path and Path(data_path).exists():
            return True

        # Fall back to scanning the processed data directory
        try:
            from cola_coder.model.config import Config
            cfg = Config.from_yaml(self.config_path)
            processed_dir = Path(cfg.data.data_dir) / "processed"
        except Exception:
            processed_dir = Path("data") / "processed"

        if not processed_dir.exists():
            return False

        npy_files = [
            f for f in processed_dir.glob("*.npy")
            if not f.name.endswith("_tmp.npy") and not f.name.endswith(".weights.npy")
        ]
        return bool(npy_files)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Discover the most-recent checkpoint for the configured model size."""
        try:
            from cola_coder.training.checkpoint import detect_latest_checkpoint
            from cola_coder.model.config import get_storage_config
            storage = get_storage_config()
            result = detect_latest_checkpoint(storage.checkpoints_dir)
            if result is not None:
                raw_path, _ = result
                resolved = Path(raw_path)
                if not resolved.is_absolute():
                    resolved = Path(storage.checkpoints_dir).parent / resolved
                return str(resolved)
        except Exception:
            pass

        # Heuristic fallback: find the most-recent step_NNNNN dir
        try:
            config_stem = Path(self.config_path).stem  # e.g. "tiny"
            ckpt_dir = self._project_root / "checkpoints" / config_stem
            if ckpt_dir.exists():
                step_dirs = sorted(ckpt_dir.glob("step_*"))
                if step_dirs:
                    return str(step_dirs[-1])
        except Exception:
            pass

        return None

    def _export_exists(self) -> bool:
        """Check whether at least one .gguf export file exists."""
        export_dir = self._artifacts.get("export_dir")
        if export_dir and Path(export_dir).exists():
            gguf_files = list(Path(export_dir).glob("*.gguf"))
            if gguf_files:
                return True

        # Fall back to heuristic path
        checkpoint = self._artifacts.get("checkpoint") or self._find_latest_checkpoint()
        if checkpoint:
            export_path = Path(checkpoint).parent / "exports"
            if export_path.exists():
                return bool(list(export_path.glob("*.gguf")))

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load persisted artifacts from the state JSON file (if it exists)."""
        if self._state_file.exists():
            try:
                with open(self._state_file, encoding="utf-8") as f:
                    state = json.load(f)
                self._artifacts.update(state.get("artifacts", {}))
            except Exception:
                pass  # Corrupt state file — start fresh

    def _save_state(self) -> None:
        """Persist artifacts to the state JSON file."""
        state = {
            "config_path": self.config_path,
            "updated_at": datetime.now().isoformat(),
            "artifacts": self._artifacts,
            "stages_completed": [
                r.stage.value for r in self.results if r.success and not r.skipped
            ],
        }
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def _format_report_rich(self) -> str:
        """Format results using Rich tables."""
        from io import StringIO

        from rich import box
        from rich.console import Console
        from rich.table import Table

        buf = StringIO()
        console = Console(file=buf, width=100)

        table = Table(
            title="Pipeline Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Stage", style="bold white", width=14)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Duration", width=10, justify="right")
        table.add_column("Message")

        total_seconds = 0.0
        all_passed = True

        for r in self.results:
            if r.skipped:
                status_str = "[yellow]-[/yellow]"
                dur_str = "—"
            elif r.success:
                status_str = "[bold green]✓[/bold green]"
                dur_str = _fmt_duration(r.duration_seconds)
                total_seconds += r.duration_seconds
            else:
                status_str = "[bold red]✗[/bold red]"
                dur_str = _fmt_duration(r.duration_seconds)
                total_seconds += r.duration_seconds
                all_passed = False

            table.add_row(r.stage.value, status_str, dur_str, r.message)

        console.print(table)

        summary_color = "green" if all_passed else "red"
        console.print(
            f"\n[{summary_color}]Pipeline {'PASSED' if all_passed else 'FAILED'}[/{summary_color}]"
            f"  Total time: {_fmt_duration(total_seconds)}"
        )

        # Print artifact paths
        if self._artifacts:
            console.print("\n[bold]Artifacts:[/bold]")
            for key, value in self._artifacts.items():
                console.print(f"  {key}: {value}")

        return buf.getvalue()

    def _format_report_plain(self) -> str:
        """Plain-text fallback for format_report()."""
        lines = ["", "=" * 60, "  Pipeline Results", "=" * 60]

        total_seconds = 0.0
        all_passed = True

        for r in self.results:
            if r.skipped:
                mark = "-"
                dur_str = "    —   "
            elif r.success:
                mark = "✓"
                dur_str = _fmt_duration(r.duration_seconds)
                total_seconds += r.duration_seconds
            else:
                mark = "✗"
                dur_str = _fmt_duration(r.duration_seconds)
                total_seconds += r.duration_seconds
                all_passed = False

            lines.append(f"  [{mark}] {r.stage.value:<14} {dur_str:>9}  {r.message}")

        lines.append("=" * 60)
        verdict = "PASSED" if all_passed else "FAILED"
        lines.append(f"  Pipeline {verdict}  |  Total: {_fmt_duration(total_seconds)}")
        lines.append("=" * 60)

        if self._artifacts:
            lines.append("\nArtifacts:")
            for key, value in self._artifacts.items():
                lines.append(f"  {key}: {value}")

        lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m"
