"""One-Click Pipeline: run the full training pipeline with a single command.

Orchestrates all stages in order:
1. Train tokenizer (if not already done)
2. Prepare data (if not already done)
3. Train model
4. Evaluate model
5. Generate samples

Each stage checks if it's already complete and skips if so.
Provides resume capability if interrupted.

For a TS dev: like `npm run build` that handles all the build steps
in order, skipping ones that are already up to date.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import time

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class StageStatus:
    """Status of a single pipeline stage."""
    name: str
    description: str
    completed: bool = False
    skipped: bool = False
    error: str = ""
    duration_ms: float = 0.0
    output_path: str = ""

    @property
    def status_str(self) -> str:
        if self.error:
            return "FAILED"
        if self.skipped:
            return "SKIPPED"
        if self.completed:
            return "DONE"
        return "PENDING"


@dataclass
class PipelineConfig:
    """Configuration for the one-click pipeline."""
    config_path: str = "configs/tiny.yaml"
    tokenizer_path: str = "tokenizer.json"
    data_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    skip_tokenizer: bool = False
    skip_data_prep: bool = False
    skip_training: bool = False
    skip_eval: bool = False
    skip_generate: bool = False
    max_tokens: int | None = None  # Limit data prep size
    use_wandb: bool = False
    dry_run: bool = False  # Just show what would be done


class Pipeline:
    """Orchestrate the full training pipeline."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.stages: list[StageStatus] = []
        self._init_stages()
        self._state_path = Path(".pipeline_state.json")

    def _init_stages(self):
        """Initialize the pipeline stages."""
        self.stages = [
            StageStatus("tokenizer", "Train BPE tokenizer"),
            StageStatus("data_prep", "Prepare training data"),
            StageStatus("train", "Train the model"),
            StageStatus("evaluate", "Run evaluation benchmarks"),
            StageStatus("generate", "Generate code samples"),
        ]

    def get_stage(self, name: str) -> StageStatus | None:
        """Get a stage by name."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def detect_completed_stages(self) -> dict[str, bool]:
        """Check which stages are already complete based on file existence."""
        cfg = self.config
        status = {}

        # Tokenizer: check if tokenizer.json exists
        status["tokenizer"] = Path(cfg.tokenizer_path).exists()

        # Data prep: check if any .npy files exist in data dir
        data_dir = Path(cfg.data_dir)
        status["data_prep"] = (
            data_dir.exists() and
            len(list(data_dir.glob("*.npy"))) > 0
        )

        # Training: check if any checkpoints exist
        ckpt_dir = Path(cfg.checkpoint_dir)
        status["train"] = (
            ckpt_dir.exists() and
            len(list(ckpt_dir.rglob("model.safetensors"))) > 0
        )

        # Evaluate and generate: always re-run
        status["evaluate"] = False
        status["generate"] = False

        return status

    def run(self, on_stage_start=None, on_stage_complete=None) -> list[StageStatus]:
        """Run the full pipeline.

        Args:
            on_stage_start: Callback(stage_name, stage_description)
            on_stage_complete: Callback(stage_name, StageStatus)

        Returns:
            List of StageStatus for all stages
        """
        cfg = self.config
        completed = self.detect_completed_stages()

        for stage in self.stages:
            # Check skip flags
            skip_map = {
                "tokenizer": cfg.skip_tokenizer,
                "data_prep": cfg.skip_data_prep,
                "train": cfg.skip_training,
                "evaluate": cfg.skip_eval,
                "generate": cfg.skip_generate,
            }
            if skip_map.get(stage.name, False):
                stage.skipped = True
                stage.completed = True
                if on_stage_complete:
                    on_stage_complete(stage.name, stage)
                continue

            # Check if already complete
            if completed.get(stage.name, False) and stage.name not in ("evaluate", "generate"):
                stage.skipped = True
                stage.completed = True
                if on_stage_complete:
                    on_stage_complete(stage.name, stage)
                continue

            if on_stage_start:
                on_stage_start(stage.name, stage.description)

            if cfg.dry_run:
                stage.skipped = True
                stage.completed = True
                if on_stage_complete:
                    on_stage_complete(stage.name, stage)
                continue

            start = time.perf_counter()
            try:
                self._run_stage(stage)
                stage.completed = True
            except Exception as e:
                stage.error = str(e)
            stage.duration_ms = (time.perf_counter() - start) * 1000

            if on_stage_complete:
                on_stage_complete(stage.name, stage)

            # Stop if a stage failed
            if stage.error:
                break

        self._save_state()
        return self.stages

    def _run_stage(self, stage: StageStatus):
        """Run a single stage. Override in subclasses for real execution."""
        # This is the stub version — real implementation would call
        # the actual scripts/functions for each stage.
        # Kept as stub since running real training requires GPU + data.
        raise NotImplementedError(
            f"Stage '{stage.name}' execution not configured. "
            f"Use PipelineRunner for actual execution."
        )

    def _save_state(self):
        """Save pipeline state for resume capability."""
        state = {
            "config": {
                "config_path": self.config.config_path,
                "tokenizer_path": self.config.tokenizer_path,
            },
            "stages": [
                {
                    "name": s.name,
                    "completed": s.completed,
                    "skipped": s.skipped,
                    "error": s.error,
                    "duration_ms": s.duration_ms,
                }
                for s in self.stages
            ],
        }
        self._state_path.write_text(json.dumps(state, indent=2))

    def load_state(self) -> bool:
        """Load previous pipeline state.

        Returns:
            True if state was loaded, False if no state file exists
        """
        if not self._state_path.exists():
            return False
        try:
            state = json.loads(self._state_path.read_text())
            for saved in state.get("stages", []):
                stage = self.get_stage(saved["name"])
                if stage:
                    stage.completed = saved.get("completed", False)
                    stage.skipped = saved.get("skipped", False)
                    stage.error = saved.get("error", "")
                    stage.duration_ms = saved.get("duration_ms", 0.0)
            return True
        except Exception:
            return False

    def print_status(self) -> None:
        """Print current pipeline status."""
        from cola_coder.cli import cli

        cli.header("Pipeline Status")
        completed = self.detect_completed_stages()

        for i, stage in enumerate(self.stages, 1):
            is_done = completed.get(stage.name, False) or stage.completed
            status = "DONE" if is_done else ("FAILED" if stage.error else "PENDING")

            if status == "DONE":
                cli.success(f"  {i}. {stage.description}")
            elif status == "FAILED":
                cli.error(f"  {i}. {stage.description}", hint=stage.error)
            else:
                cli.dim(f"  {i}. {stage.description}")

    def summary(self) -> str:
        """Get a text summary of the pipeline run."""
        lines = ["Pipeline Summary:"]
        total_time = 0.0
        for s in self.stages:
            total_time += s.duration_ms
            lines.append(f"  {s.name}: {s.status_str} ({s.duration_ms:.0f}ms)")
        lines.append(f"Total time: {total_time:.0f}ms ({total_time/1000:.1f}s)")

        failed = [s for s in self.stages if s.error]
        if failed:
            lines.append(f"FAILED stages: {', '.join(s.name for s in failed)}")

        return "\n".join(lines)
