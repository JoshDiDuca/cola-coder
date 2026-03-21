"""Automated quality report generation for cola-coder model checkpoints.

Generates a comprehensive report covering model info, training status,
smoke test results, and sample outputs. Optionally runs HumanEval.

For a TS dev: think of this like a CI quality gate — it runs a battery of
checks on a model checkpoint and produces a structured report you can track
over training runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Standard prompts ──────────────────────────────────────────────────────────

STANDARD_PROMPTS = [
    "def fibonacci(n: int) -> int:",
    "function fetchUserData(userId: string): Promise<User> {",
    "class LinkedList:",
    "// Sort an array of numbers in ascending order\nfunction sort(",
    "import React from 'react';\n\nconst Button = (",
]


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class QualityReport:
    """Comprehensive quality report for a model checkpoint."""

    checkpoint_path: str
    config: dict
    timestamp: str

    # Model info
    model_params: int
    model_config: dict

    # Training info (from metadata)
    training_step: int
    training_loss: float

    # Smoke test results
    smoke_test_passed: bool
    smoke_test_details: list[dict]

    # Generation samples
    samples: list[dict] = field(default_factory=list)  # [{prompt, output, tokens, time_ms}]

    # Optional eval scores
    humaneval_pass_at_1: float | None = None

    # ── Formatting ────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines: list[str] = []

        lines.append("# Cola-Coder Quality Report")
        lines.append(f"**Checkpoint:** {self.checkpoint_path}")
        lines.append(f"**Date:** {self.timestamp}")
        lines.append("")

        # Model info
        lines.append("## Model Info")
        param_str = _human_params(self.model_params)
        lines.append(f"- Parameters: {param_str}")
        n_layers = self.model_config.get("n_layers", self.model_config.get("num_layers", "?"))
        dim = self.model_config.get("dim", self.model_config.get("hidden_size", "?"))
        n_heads = self.model_config.get("n_heads", self.model_config.get("num_heads", "?"))
        lines.append(f"- Architecture: {n_layers} layers, dim={dim}, {n_heads} heads")
        lines.append("")

        # Training status
        lines.append("## Training Status")
        total_steps = self.config.get("training", {}).get("max_steps", "?")
        step_str = f"{self.training_step:,}" if isinstance(self.training_step, int) else str(self.training_step)
        total_str = f"{total_steps:,}" if isinstance(total_steps, int) else str(total_steps)
        lines.append(f"- Step: {step_str} / {total_str}")
        loss_str = f"{self.training_loss:.4f}" if isinstance(self.training_loss, float) and self.training_loss == self.training_loss else "N/A"
        lines.append(f"- Loss: {loss_str}")
        lines.append("")

        # Smoke test summary
        num_passed = sum(1 for d in self.smoke_test_details if d.get("passed"))
        num_total = len(self.smoke_test_details)
        status_str = "PASSED" if self.smoke_test_passed else "FAILED"
        lines.append(f"## Smoke Test: {status_str} ({num_passed}/{num_total})")
        if self.smoke_test_details:
            lines.append("| Test | Result | Time |")
            lines.append("|------|--------|------|")
            for detail in self.smoke_test_details:
                name = detail.get("name", "unknown")
                result = "PASS" if detail.get("passed") else "FAIL"
                duration = f"{detail.get('duration_ms', 0):.0f}ms"
                lines.append(f"| {name} | {result} | {duration} |")
        lines.append("")

        # Sample outputs
        if self.samples:
            lines.append("## Sample Outputs")
            for sample in self.samples:
                prompt = sample.get("prompt", "")
                output = sample.get("output", "")
                tokens = sample.get("tokens", 0)
                time_ms = sample.get("time_ms", 0)
                # Truncate prompt for header
                short_prompt = prompt.split("\n")[0][:60]
                lines.append(f"### Prompt: `{short_prompt}`")
                lines.append(f"*{tokens} tokens in {time_ms:.0f}ms*")
                lines.append("```")
                lines.append(output)
                lines.append("```")
                lines.append("")

        # Evaluation
        if self.humaneval_pass_at_1 is not None:
            lines.append("## Evaluation")
            lines.append(f"- HumanEval pass@1: {self.humaneval_pass_at_1 * 100:.1f}%")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        d = asdict(self)
        # Ensure float NaN is handled (JSON doesn't support NaN)
        if d.get("training_loss") != d.get("training_loss"):  # NaN check
            d["training_loss"] = None
        return d


# ── Helpers ───────────────────────────────────────────────────────────────────


def _human_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _count_params(model: Any) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def _read_metadata(checkpoint_path: str) -> dict:
    """Read metadata.json from a checkpoint directory."""
    meta_path = Path(checkpoint_path) / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── Main class ────────────────────────────────────────────────────────────────


class QualityReportGenerator:
    """Generate comprehensive quality reports for model checkpoints."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "auto",
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self._resolved_device: str | None = None

    def _get_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self.device != "auto":
            self._resolved_device = self.device
            return self._resolved_device
        try:
            import torch
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._resolved_device = "cpu"
        return self._resolved_device

    def generate(
        self,
        run_eval: bool = False,
        num_samples: int = 5,
    ) -> QualityReport:
        """Generate full quality report.

        Steps:
        1. Load checkpoint metadata (step, loss)
        2. Load model + tokenizer
        3. Run smoke test
        4. Generate sample outputs for standard prompts
        5. Optionally run HumanEval subset
        6. Compile report
        """
        from cola_coder.model.config import Config, ModelConfig
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        from cola_coder.evaluation.smoke_test import SmokeTest

        device = self._get_device()

        # 1. Load metadata
        metadata = _read_metadata(self.checkpoint_path)
        training_step = int(metadata.get("step", 0))
        training_loss = float(metadata.get("loss", float("nan")))

        # 2. Load config + model
        config_obj = Config.from_yaml(self.config_path)
        config_dict = config_obj.__dict__.copy() if hasattr(config_obj, "__dict__") else {}

        # Build model config dict from metadata or config
        meta_config = metadata.get("config", {})
        model_cfg_raw = meta_config.get("model", {}) or {}

        # Try to load model config from the metadata if available
        model_config_dict: dict = {}
        if model_cfg_raw:
            valid_fields = ModelConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in model_cfg_raw.items() if k in valid_fields}
            model_cfg = ModelConfig(**filtered)
            model_config_dict = {k: getattr(model_cfg, k) for k in valid_fields}
        else:
            model_cfg = config_obj.model
            valid_fields = ModelConfig.__dataclass_fields__.keys()
            model_config_dict = {k: getattr(model_cfg, k) for k in valid_fields}

        # Tokenizer path: look beside checkpoint or fall back to default
        tokenizer_path = self._find_tokenizer()
        tokenizer = CodeTokenizer(tokenizer_path)

        model = Transformer(model_cfg).to(device)
        load_model_only(self.checkpoint_path, model, device=device)
        model.eval()

        model_params = _count_params(model)

        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)

        # 3. Run smoke tests
        smoke = SmokeTest(generator=generator, tokenizer=tokenizer)
        smoke_report = smoke.run_all()

        smoke_details = [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "duration_ms": r.duration_ms,
            }
            for r in smoke_report.results
        ]

        # 4. Generate samples
        prompts_to_use = STANDARD_PROMPTS[:num_samples]
        samples: list[dict] = []
        for prompt in prompts_to_use:
            t0 = time.perf_counter()
            try:
                output = generator.generate(
                    prompt,
                    max_new_tokens=128,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                # Count new tokens (approximate by tokenizing output)
                try:
                    tokens_out = len(tokenizer.encode(output, add_bos=False))
                    tokens_in = len(tokenizer.encode(prompt, add_bos=False))
                    new_tokens = max(0, tokens_out - tokens_in)
                except Exception:
                    new_tokens = 0
                samples.append(
                    {
                        "prompt": prompt,
                        "output": output,
                        "tokens": new_tokens,
                        "time_ms": elapsed_ms,
                    }
                )
            except Exception as exc:
                samples.append(
                    {
                        "prompt": prompt,
                        "output": f"[ERROR: {exc}]",
                        "tokens": 0,
                        "time_ms": 0.0,
                    }
                )

        # 5. Optional HumanEval
        humaneval_pass_at_1: float | None = None
        if run_eval:
            humaneval_pass_at_1 = self._run_humaneval_subset(generator, tokenizer)

        # 6. Compile report
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = QualityReport(
            checkpoint_path=self.checkpoint_path,
            config=config_dict,
            timestamp=timestamp,
            model_params=model_params,
            model_config=model_config_dict,
            training_step=training_step,
            training_loss=training_loss,
            smoke_test_passed=smoke_report.passed,
            smoke_test_details=smoke_details,
            samples=samples,
            humaneval_pass_at_1=humaneval_pass_at_1,
        )
        return report

    def _find_tokenizer(self) -> str:
        """Try to find tokenizer.json in standard locations."""
        candidates = [
            Path(self.checkpoint_path) / "tokenizer.json",
            Path(self.checkpoint_path).parent / "tokenizer.json",
            Path(self.checkpoint_path).parent.parent / "tokenizer.json",
            Path("tokenizer.json"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # Try storage config
        try:
            from cola_coder.model.config import get_storage_config
            storage = get_storage_config()
            p = Path(storage.tokenizer_path)
            if p.exists():
                return str(p)
        except Exception:
            pass
        return "tokenizer.json"

    def _run_humaneval_subset(self, generator: Any, tokenizer: Any) -> float:
        """Run a small HumanEval subset and return pass@1."""
        try:
            from cola_coder.evaluation.humaneval import get_all_problems
            from cola_coder.evaluation.runner import evaluate_solution, extract_function
            from cola_coder.evaluation.metrics import pass_at_k, ProblemResult

            problems = get_all_problems()[:5]  # Small subset for speed
            results = []
            for problem in problems:
                try:
                    output = generator.generate(
                        problem.prompt,
                        max_new_tokens=256,
                        temperature=0.2,
                        top_k=50,
                        top_p=0.9,
                    )
                    extracted = extract_function(output, problem.entry_point)
                    passed, _ = evaluate_solution(problem, extracted)
                    results.append(ProblemResult(
                        task_id=problem.task_id,
                        num_samples=1,
                        num_correct=1 if passed else 0,
                    ))
                except Exception:
                    results.append(ProblemResult(
                        task_id=problem.task_id,
                        num_samples=1,
                        num_correct=0,
                    ))

            if not results:
                return 0.0
            scores = [pass_at_k(r.num_samples, r.num_correct, 1) for r in results]
            return sum(scores) / len(scores)
        except Exception:
            return 0.0

    def save_report(self, report: QualityReport, output_dir: str = "reports/") -> None:
        """Save report as markdown and JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Derive a filename from checkpoint path + step
        ckpt_name = Path(report.checkpoint_path).name
        step_str = f"step_{report.training_step:07d}" if isinstance(report.training_step, int) else "unknown"
        base_name = f"quality_report_{ckpt_name}_{step_str}"

        md_path = out_path / f"{base_name}.md"
        json_path = out_path / f"{base_name}.json"

        md_path.write_text(report.to_markdown(), encoding="utf-8")
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
