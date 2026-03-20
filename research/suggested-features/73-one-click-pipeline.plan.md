# 73 - One-Click Pipeline

## Overview

A single command that chains all Cola-Coder stages: data preparation → training → evaluation → sample generation → report. Configured by `pipeline.yaml`. Handles errors gracefully (saves progress and reports), runs independent stages in parallel where possible, and produces a final HTML or Markdown report with all results.

**Feature flag:** `--pipeline` (standalone command; always optional)

---

## Motivation

Running a full experiment currently requires manually invoking `prepare.py`, then `train.py`, then `eval.py`, then `generate.py`, checking each one manually. This is:

- Tedious for overnight runs (you have to babysit each stage)
- Error-prone (forgetting to pass the right checkpoint path to eval)
- Non-reproducible (hard to know exactly which command sequence was run)

A pipeline orchestrator solves all three:
- Run it before you go to sleep, get results in the morning
- Stage outputs automatically become the next stage's inputs
- The `pipeline.yaml` config is a complete specification of what was run

---

## Architecture / Design

### Stage Graph

```
prepare_data
     │
     ▼
   train
     │
     ├──────────────────┐
     ▼                  ▼
evaluate            generate_samples
     │                  │
     └────────┬──────────┘
              ▼
        final_report
```

`evaluate` and `generate_samples` both depend on `train` but not on each other, so they can run in parallel.

### `pipeline.yaml`

```yaml
name: experiment-2026-03-20
experiment_tracker:
  experiment_name: cola-coder-baseline

stages:
  prepare_data:
    enabled: true
    config: config/data.yaml
    output_dir: data/

  train:
    enabled: true
    config: config/training.yaml
    resume_from: null      # set to checkpoint path to skip training
    max_steps: 10000

  evaluate:
    enabled: true
    config: config/eval.yaml
    checkpoint: auto       # uses best checkpoint from training

  generate_samples:
    enabled: true
    n_samples: 20
    temperature: 0.7
    prompts_file: eval/benchmarks/sample_prompts.yaml
    output_dir: samples/

  final_report:
    enabled: true
    format: html           # html or markdown
    output: reports/pipeline_report.html

parallel:
  evaluate_and_generate: true   # run these two stages in parallel
```

---

## Implementation Steps

### Step 1: Pipeline Orchestrator (`pipeline/orchestrator.py`)

```python
import time
import yaml
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StageResult:
    name: str
    status: StageStatus
    started_at: float = 0.0
    ended_at: float = 0.0
    output: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_sec(self) -> float:
        return self.ended_at - self.started_at

class PipelineOrchestrator:
    def __init__(self, pipeline_config: dict):
        self.config = pipeline_config
        self.name = pipeline_config.get("name", "unnamed-pipeline")
        self.results: dict[str, StageResult] = {}
        self.context: dict = {}   # shared state between stages

    def run(self) -> bool:
        """Run all enabled stages. Returns True if all passed."""
        from rich.console import Console
        from rich.live import Live
        console = Console()

        console.rule(f"[bold cyan]Pipeline: {self.name}[/]")
        console.print(f"  Started: {datetime.utcnow().isoformat()[:16]} UTC\n")

        stages_cfg = self.config.get("stages", {})

        # Stage 1: prepare_data
        if stages_cfg.get("prepare_data", {}).get("enabled", True):
            self._run_stage("prepare_data", self._run_prepare_data)
        else:
            self._skip("prepare_data")

        if not self._all_ok():
            self._abort("prepare_data failed")
            return False

        # Stage 2: train
        if stages_cfg.get("train", {}).get("enabled", True):
            self._run_stage("train", self._run_train)
        else:
            # Load checkpoint path from config
            resume = stages_cfg.get("train", {}).get("resume_from")
            if resume:
                self.context["best_checkpoint"] = resume
            self._skip("train")

        if not self._all_ok():
            self._abort("training failed")
            return False

        # Stages 3 & 4: evaluate + generate_samples (potentially parallel)
        parallel = self.config.get("parallel", {}).get("evaluate_and_generate", False)
        eval_enabled = stages_cfg.get("evaluate", {}).get("enabled", True)
        gen_enabled = stages_cfg.get("generate_samples", {}).get("enabled", True)

        if parallel and eval_enabled and gen_enabled:
            self._run_parallel([
                ("evaluate", self._run_evaluate),
                ("generate_samples", self._run_generate_samples),
            ])
        else:
            if eval_enabled:
                self._run_stage("evaluate", self._run_evaluate)
            else:
                self._skip("evaluate")
            if gen_enabled:
                self._run_stage("generate_samples", self._run_generate_samples)
            else:
                self._skip("generate_samples")

        # Stage 5: final_report (always runs even if earlier stages had issues)
        if stages_cfg.get("final_report", {}).get("enabled", True):
            self._run_stage("final_report", self._run_final_report)

        self._print_summary(console)
        return self._all_ok()

    def _run_stage(self, name: str, fn) -> StageResult:
        from rich.console import Console
        console = Console()
        console.print(f"  [cyan]▶[/] Starting stage: [bold]{name}[/]")
        result = StageResult(name=name, status=StageStatus.RUNNING, started_at=time.time())
        self.results[name] = result

        try:
            output = fn(self.config["stages"].get(name, {}))
            result.output = output or {}
            result.status = StageStatus.COMPLETED
            console.print(f"  [green]✓[/] Completed: {name} ({result.duration_sec:.1f}s)")
        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = traceback.format_exc()
            console.print(f"  [red]✗[/] Failed: {name}\n    {e}")
        finally:
            result.ended_at = time.time()

        return result

    def _run_parallel(self, stages: list[tuple[str, callable]]):
        with ThreadPoolExecutor(max_workers=len(stages)) as executor:
            futures = {
                executor.submit(self._run_stage, name, fn): name
                for name, fn in stages
            }
            for future in as_completed(futures):
                future.result()  # propagate exceptions

    def _skip(self, name: str):
        self.results[name] = StageResult(name=name, status=StageStatus.SKIPPED)

    def _all_ok(self) -> bool:
        return all(
            r.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
            for r in self.results.values()
        )

    def _abort(self, reason: str):
        from rich.console import Console
        Console().print(f"\n  [red bold]Pipeline aborted: {reason}[/]")
        self._save_progress()

    def _save_progress(self):
        """Save pipeline state to allow inspection after failure."""
        progress = {
            "name": self.name,
            "aborted_at": datetime.utcnow().isoformat(),
            "stages": {
                name: {
                    "status": r.status.value,
                    "duration_sec": r.duration_sec,
                    "error": r.error,
                }
                for name, r in self.results.items()
            },
            "context": {k: str(v) for k, v in self.context.items()},
        }
        Path(f"pipeline_progress_{self.name}.json").write_text(
            json.dumps(progress, indent=2)
        )

    # --- Stage implementations ---

    def _run_prepare_data(self, stage_cfg: dict) -> dict:
        from data.prepare import run_prepare
        stats = run_prepare(config_path=stage_cfg.get("config", "config/data.yaml"))
        self.context["data_dir"] = stage_cfg.get("output_dir", "data/")
        self.context["dataset_stats"] = stats
        return {"total_tokens": stats.get("total_tokens", 0)}

    def _run_train(self, stage_cfg: dict) -> dict:
        from training.trainer import run_training
        result = run_training(
            config_path=stage_cfg.get("config", "config/training.yaml"),
            max_steps=stage_cfg.get("max_steps"),
        )
        self.context["best_checkpoint"] = result["best_checkpoint"]
        self.context["final_loss"] = result["final_loss"]
        return result

    def _run_evaluate(self, stage_cfg: dict) -> dict:
        from eval.run_eval import run_full_eval
        checkpoint = stage_cfg.get("checkpoint", "auto")
        if checkpoint == "auto":
            checkpoint = self.context.get("best_checkpoint")
        result = run_full_eval(checkpoint_path=checkpoint)
        self.context["eval_results"] = result
        return result

    def _run_generate_samples(self, stage_cfg: dict) -> dict:
        from generate.sampler import run_batch_generation
        checkpoint = self.context.get("best_checkpoint")
        result = run_batch_generation(
            checkpoint_path=checkpoint,
            n_samples=stage_cfg.get("n_samples", 20),
            temperature=stage_cfg.get("temperature", 0.7),
            output_dir=stage_cfg.get("output_dir", "samples/"),
        )
        self.context["sample_dir"] = result["output_dir"]
        return result

    def _run_final_report(self, stage_cfg: dict) -> dict:
        report_gen = PipelineReportGenerator(
            pipeline_name=self.name,
            stage_results=self.results,
            context=self.context,
        )
        fmt = stage_cfg.get("format", "markdown")
        output = stage_cfg.get("output", f"reports/{self.name}.{fmt[:2]}")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        report_gen.write(output, fmt=fmt)
        return {"report_path": output}

    def _print_summary(self, console):
        from rich.table import Table
        console.rule("[bold]Pipeline Summary[/]")
        table = Table(show_header=True)
        table.add_column("Stage")
        table.add_column("Status")
        table.add_column("Duration", justify="right")

        for name, r in self.results.items():
            color = {
                StageStatus.COMPLETED: "green",
                StageStatus.FAILED: "red",
                StageStatus.SKIPPED: "dim",
                StageStatus.RUNNING: "yellow",
                StageStatus.PENDING: "dim",
            }[r.status]
            dur = f"{r.duration_sec:.1f}s" if r.duration_sec > 0 else "—"
            table.add_row(name, f"[{color}]{r.status.value}[/]", dur)
        console.print(table)
```

### Step 2: Report Generator (`pipeline/report_generator.py`)

```python
class PipelineReportGenerator:
    def __init__(self, pipeline_name, stage_results, context):
        self.name = pipeline_name
        self.results = stage_results
        self.context = context

    def write(self, output_path: str, fmt: str = "markdown"):
        content = self._build_markdown()
        if fmt == "html":
            content = self._markdown_to_html(content)
        Path(output_path).write_text(content, encoding="utf-8")

    def _build_markdown(self) -> str:
        lines = [
            f"# Pipeline Report: {self.name}",
            f"\nGenerated: {datetime.utcnow().isoformat()[:16]} UTC\n",
            "## Stage Results\n",
            "| Stage | Status | Duration |",
            "|-------|--------|----------|",
        ]
        for name, r in self.results.items():
            dur = f"{r.duration_sec:.1f}s" if r.duration_sec > 0 else "—"
            lines.append(f"| {name} | {r.status.value} | {dur} |")

        # Eval results section
        if "eval_results" in self.context:
            lines.append("\n## Evaluation Results\n")
            for k, v in self.context["eval_results"].items():
                if isinstance(v, float):
                    lines.append(f"- **{k}**: {v:.4f}")

        # Training metrics
        if "final_loss" in self.context:
            lines.append(f"\n## Training\n\n- **Final loss**: {self.context['final_loss']:.4f}")
        if "best_checkpoint" in self.context:
            lines.append(f"- **Best checkpoint**: `{self.context['best_checkpoint']}`")

        return "\n".join(lines) + "\n"

    def _markdown_to_html(self, md: str) -> str:
        try:
            import markdown
            return f"<html><body>{markdown.markdown(md, extensions=['tables'])}</body></html>"
        except ImportError:
            # Fallback: wrap raw markdown in pre tag
            return f"<html><body><pre>{md}</pre></body></html>"
```

### Step 3: CLI Entry Point

```python
# cola-coder pipeline run --config pipeline.yaml
# cola-coder pipeline run --config pipeline.yaml --dry-run

@click.command("run")
@click.option("--config", required=True, type=click.Path())
@click.option("--dry-run", is_flag=True, help="Print stages that would run without executing")
def pipeline_run(config, dry_run):
    cfg = yaml.safe_load(Path(config).read_text())
    if dry_run:
        for stage, stage_cfg in cfg.get("stages", {}).items():
            enabled = stage_cfg.get("enabled", True)
            print(f"  {'[ENABLED]' if enabled else '[SKIP]'} {stage}")
        return

    orchestrator = PipelineOrchestrator(cfg)
    success = orchestrator.run()
    sys.exit(0 if success else 1)
```

---

## Key Files to Modify

- `pipeline/orchestrator.py` - New file: pipeline runner
- `pipeline/report_generator.py` - New file: HTML/MD report
- `cli/pipeline_cmd.py` - New file: CLI
- `cli/main.py` - Register `pipeline` command group
- `pipeline.yaml` - New template config (in project root or `config/`)

---

## Testing Strategy

1. **Dry-run test**: pass `--dry-run`, assert all stage names printed, no actual execution.
2. **Stage skip test**: set `train.enabled: false`, assert `train` shows as SKIPPED in results.
3. **Error handling test**: mock `_run_prepare_data` to raise an exception, assert pipeline aborts after prepare_data, `pipeline_progress_*.json` is written.
4. **Parallel stage test**: mock evaluate and generate_samples with sleep(1), assert total wall time is <2s (parallel) vs >2s (sequential).
5. **Report generation test**: run full pipeline on mock stages, assert report file exists and contains stage names.
6. **Context propagation test**: assert that `best_checkpoint` set in `_run_train` is available in `_run_evaluate`.

---

## Performance Considerations

- The `evaluate` and `generate_samples` stages can run in parallel threads since both are read-only with respect to the model. Use `ThreadPoolExecutor(max_workers=2)`.
- Note: if both stages call `model.generate()` simultaneously and the model is on GPU, they will contend for VRAM. Either load the model once and serialize generation calls (thread lock), or load two separate model instances.
- Report generation via the `markdown` library is fast (<100ms). The `markdown` package is optional; fallback to raw markdown if not available.

---

## Dependencies

```
markdown>=3.0   # optional, for HTML report generation
```

All other dependencies are already required by other Cola-Coder components.

---

## Estimated Complexity

**Medium.** The orchestrator pattern is clean and well-understood. The complexity is in the stage implementations calling into existing modules correctly, and in the parallel execution + context sharing. Estimated implementation time: 3-4 days.

---

## 2026 Best Practices

- **Idempotent stages**: each stage should check if its output already exists and skip if so, unless `force: true` is set. This allows resuming a failed pipeline without re-running expensive stages.
- **Context over global state**: pass outputs between stages via the `context` dict, not via global variables or environment variables. This makes the data flow explicit and testable.
- **Atomic stage outputs**: each stage should write to a temp location and rename to the final location only on success. This prevents partial writes from being mistaken for completed stages.
- **Dry-run as first check**: always implement `--dry-run` before implementing the actual pipeline. It forces you to define the stage graph clearly and catches config errors early.
- **Exit code reflects success**: the pipeline CLI must return exit code 0 on success and non-zero on any stage failure. This makes it composable with CI/CD systems.
