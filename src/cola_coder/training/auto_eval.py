"""Automatic evaluation of the model during training.

Periodically pauses training, runs a subset of HumanEval problems, records
pass@1 and pass@5, and tracks whether quality is improving or degrading.

Key design decisions:
- Only run a SUBSET of problems (default 20 out of ~20 available) for speed.
  A full HumanEval run takes minutes; 20 problems takes seconds.
- Fixed random seed for subset sampling so the same problems are tested at
  every evaluation step — apples-to-apples comparisons across steps.
- Model is set to eval mode before generation and back to train mode after.
- Regression detection: score dropped >20% from best is flagged.
- State is serialisable so eval history survives checkpoint resume.
- wandb logging is optional (graceful no-op if not installed / not enabled).

For a TS dev: think of this as a CI check that runs every N commits.  Instead
of "does the build pass?" the question is "can the model still write correct
code?" — tracked over the life of a training run.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalSnapshot:
    """Results from a single evaluation run during training."""

    step: int
    timestamp: str  # ISO-8601, e.g. "2026-03-21T14:05:00"
    pass_at_1: float  # 0.0 – 1.0
    pass_at_5: float  # 0.0 – 1.0 (0.0 if fewer than 5 samples generated)
    num_problems: int  # How many problems were evaluated
    avg_generation_time: float  # Seconds per problem
    is_best: bool = False  # True if this snapshot has the best pass@1 so far

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON / checkpoint embedding)."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "num_problems": self.num_problems,
            "avg_generation_time": self.avg_generation_time,
            "is_best": self.is_best,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalSnapshot":
        """Deserialise from a plain dict."""
        return cls(
            step=d["step"],
            timestamp=d["timestamp"],
            pass_at_1=d["pass_at_1"],
            pass_at_5=d["pass_at_5"],
            num_problems=d["num_problems"],
            avg_generation_time=d["avg_generation_time"],
            is_best=d.get("is_best", False),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AutoEvaluator:
    """Periodically evaluate the model during training on a HumanEval subset.

    Usage in a training loop::

        evaluator = AutoEvaluator(eval_every_steps=5000, eval_subset=20)

        for step in range(max_steps):
            # ... training step ...
            if evaluator.should_eval(step):
                snapshot = evaluator.evaluate(model, tokenizer, step, device)
                print(evaluator.format_report())
    """

    def __init__(
        self,
        eval_every_steps: int = 5000,
        eval_subset: int = 20,
        save_best: bool = True,
        best_metric: str = "pass@1",
        log_to_wandb: bool = False,
        num_samples: int = 5,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        timeout: float = 10.0,
        subset_seed: int = 42,
        regression_threshold: float = 0.20,
        checkpoint_dir: str | None = None,
    ):
        """
        Args:
            eval_every_steps: Run evaluation every N training steps.
            eval_subset: Number of HumanEval problems to sample for each run.
                         Smaller = faster, but noisier signal.
            save_best: If True, save the model when pass@1 improves.
                       Requires checkpoint_dir to be set.
            best_metric: Which metric to track for "best" detection ("pass@1").
            log_to_wandb: Log eval metrics to wandb if available and initialised.
            num_samples: Solutions to generate per problem (needed for pass@5).
                         Set to 1 for the fastest possible eval.
            temperature: Sampling temperature for generation (lower = more deterministic).
            max_new_tokens: Max tokens to generate per solution.
            timeout: Per-solution execution timeout (seconds).
            subset_seed: Fixed seed so the same problems are sampled every time.
            regression_threshold: Flag a regression when pass@1 drops this fraction
                                  below the best recorded score (e.g. 0.20 = 20%).
            checkpoint_dir: Where to save "best" checkpoints.  Only used when
                            save_best=True.
        """
        self.eval_every_steps = eval_every_steps
        self.eval_subset = eval_subset
        self.save_best = save_best
        self.best_metric = best_metric
        self.log_to_wandb = log_to_wandb
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.subset_seed = subset_seed
        self.regression_threshold = regression_threshold
        self.checkpoint_dir = checkpoint_dir

        # State that is persisted across checkpoints
        self.history: list[EvalSnapshot] = []
        self.best_score: float = 0.0
        self.best_step: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def should_eval(self, step: int) -> bool:
        """Return True if we should run evaluation at this step.

        Evaluation is skipped at step 0 so the very first batch (where the
        model is random / freshly loaded) does not skew the history.
        """
        if step <= 0:
            return False
        return step % self.eval_every_steps == 0

    def evaluate(
        self,
        model: "torch.nn.Module",
        tokenizer,
        step: int,
        device: str = "cuda",
    ) -> EvalSnapshot:
        """Run evaluation and record results.

        Steps:
        1. Switch model to eval mode.
        2. Sample a fixed-seed subset of HumanEval problems.
        3. Generate ``num_samples`` solutions per problem.
        4. Execute solutions against test cases.
        5. Compute pass@1 and pass@5 via the unbiased estimator.
        6. Check for regression; save best checkpoint if improved.
        7. Switch model back to train mode.
        8. Log to wandb if enabled.

        Args:
            model: The model being trained (may be torch.compile()-wrapped).
            tokenizer: CodeTokenizer instance.
            step: Current training step (for labelling the snapshot).
            device: "cuda" or "cpu".

        Returns:
            An EvalSnapshot with the evaluation results.
        """
        import torch

        from ..evaluation.humaneval import get_all_problems
        from ..evaluation.metrics import ProblemResult, compute_pass_at_k
        from ..evaluation.runner import evaluate_solution, extract_function
        from ..inference.generator import CodeGenerator

        # 1. Switch to eval mode (disables dropout, enables KV-cache)
        was_training = model.training
        model.eval()

        # Resolve the underlying model when torch.compile() is used
        # (compile wraps the model in a _orig_mod attribute)
        raw_model = getattr(model, "_orig_mod", model)

        # 2. Sample a consistent subset using the fixed seed
        problems = get_all_problems()
        rng = random.Random(self.subset_seed)
        subset_size = min(self.eval_subset, len(problems))
        subset = rng.sample(problems, subset_size)

        # 3–5. Generate solutions and evaluate
        generator = CodeGenerator(model=raw_model, tokenizer=tokenizer, device=device)

        problem_results: list[ProblemResult] = []
        gen_times: list[float] = []

        with torch.no_grad():
            for problem in subset:
                num_correct = 0
                t0 = time.perf_counter()

                for _ in range(self.num_samples):
                    try:
                        generated_text = generator.generate(
                            prompt=problem.prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_k=50,
                            top_p=0.9,
                        )
                        function_code = extract_function(generated_text, problem.entry_point)
                        passed, _ = evaluate_solution(
                            problem, function_code, timeout=self.timeout
                        )
                        if passed:
                            num_correct += 1
                    except Exception:
                        # Count as failed; never let an eval error crash training
                        pass

                elapsed = time.perf_counter() - t0
                gen_times.append(elapsed / max(self.num_samples, 1))

                problem_results.append(
                    ProblemResult(
                        task_id=problem.task_id,
                        num_samples=self.num_samples,
                        num_correct=num_correct,
                    )
                )

        # 5. Compute pass@k
        k_values = [k for k in [1, 5] if k <= self.num_samples]
        metrics = compute_pass_at_k(problem_results, k_values=k_values)
        pass_1 = metrics.get("pass@1", 0.0)
        pass_5 = metrics.get("pass@5", 0.0)

        avg_gen_time = sum(gen_times) / max(len(gen_times), 1)

        # 6. Detect improvement / regression and build snapshot
        is_best = pass_1 > self.best_score
        snapshot = EvalSnapshot(
            step=step,
            timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            pass_at_1=pass_1,
            pass_at_5=pass_5,
            num_problems=subset_size,
            avg_generation_time=avg_gen_time,
            is_best=is_best,
        )

        if is_best:
            self.best_score = pass_1
            self.best_step = step
            # Save best checkpoint if requested and a dir is configured
            if self.save_best and self.checkpoint_dir:
                self._save_best_checkpoint(model, step)

        self.history.append(snapshot)

        # 7. Restore training mode
        if was_training:
            model.train()

        # 8. Optional wandb logging
        if self.log_to_wandb:
            self._log_wandb(snapshot)

        # Print results to the console
        self._print_snapshot(snapshot)

        return snapshot

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def check_regression(self, current: EvalSnapshot) -> bool:
        """Return True if the model's quality dropped significantly.

        A regression is defined as pass@1 dropping more than
        ``regression_threshold`` (default 20%) below the best score.
        Only meaningful after at least two evaluations.
        """
        if self.best_score <= 0.0:
            return False
        drop_fraction = (self.best_score - current.pass_at_1) / self.best_score
        return drop_fraction > self.regression_threshold

    def get_trend(self) -> str:
        """Return a human-readable trend label based on recent pass@1 history.

        Uses the last 3 snapshots to classify as:
        - "improving"  — pass@1 is going up
        - "degrading"  — pass@1 is going down
        - "stable"     — no clear direction
        - "not enough data" — fewer than 2 snapshots
        """
        if len(self.history) < 2:
            return "not enough data"

        tail = self.history[-3:]  # last 3 snapshots
        scores = [s.pass_at_1 for s in tail]
        n = len(scores)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(scores) / n
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, scores))
        denominator = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = numerator / denominator

        # 0.01 means 1 percentage-point per snapshot — meaningful threshold
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_report(self) -> str:
        """Return a multi-line text report of evaluation history.

        Includes a table of snapshots and the current trend.
        """
        if not self.history:
            return "No evaluations recorded yet."

        lines: list[str] = [
            "",
            "=" * 70,
            "  AUTO-EVAL HISTORY",
            "=" * 70,
            f"  {'Step':>8}  {'pass@1':>8}  {'pass@5':>8}  {'N':>5}  "
            f"{'Gen/prob':>9}  {'Best':>5}  Timestamp",
            "  " + "-" * 66,
        ]

        for snap in self.history:
            best_marker = "*" if snap.is_best else " "
            regression_marker = " [!]" if self.check_regression(snap) else ""
            lines.append(
                f"  {snap.step:>8,d}  {snap.pass_at_1:>7.1%}  {snap.pass_at_5:>7.1%}"
                f"  {snap.num_problems:>5d}  {snap.avg_generation_time:>8.2f}s"
                f"  {best_marker:>5}  {snap.timestamp}{regression_marker}"
            )

        lines += [
            "  " + "-" * 66,
            f"  Best:  step {self.best_step:,}  pass@1 {self.best_score:.1%}",
            f"  Trend: {self.get_trend()}",
            "=" * 70,
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Checkpoint state (survives resume)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise state for inclusion in a checkpoint.

        Include this in the training_state.pt alongside optimizer state.
        """
        return {
            "history": [s.to_dict() for s in self.history],
            "best_score": self.best_score,
            "best_step": self.best_step,
            # Config fields — handy for inspection
            "eval_every_steps": self.eval_every_steps,
            "eval_subset": self.eval_subset,
            "num_samples": self.num_samples,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint dict."""
        self.history = [EvalSnapshot.from_dict(d) for d in state.get("history", [])]
        self.best_score = state.get("best_score", 0.0)
        self.best_step = state.get("best_step", 0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_best_checkpoint(self, model: "torch.nn.Module", step: int) -> None:
        """Save the current model weights as the 'best' checkpoint."""
        if not self.checkpoint_dir:
            return
        try:
            from safetensors.torch import save_file

            best_dir = Path(self.checkpoint_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)

            raw_model = getattr(model, "_orig_mod", model)
            raw_state = raw_model.state_dict()
            state_dict = {}
            for k, v in raw_state.items():
                clean_key = k.removeprefix("_orig_mod.")
                if clean_key == "output.weight":
                    continue  # tied weight — skip as in checkpoint.py
                state_dict[clean_key] = v.contiguous()

            save_file(state_dict, str(best_dir / "model.safetensors"))

            import json
            (best_dir / "metadata.json").write_text(
                json.dumps({"step": step, "pass_at_1": self.best_score}, indent=2)
            )
        except Exception as exc:
            # Never let checkpoint saving crash training
            print(f"[auto_eval] WARNING: could not save best checkpoint: {exc}")

    def _log_wandb(self, snapshot: EvalSnapshot) -> None:
        """Log snapshot metrics to wandb (no-op if wandb is not available)."""
        try:
            import wandb

            if wandb.run is None:
                return
            wandb.log(
                {
                    "eval/pass_at_1": snapshot.pass_at_1,
                    "eval/pass_at_5": snapshot.pass_at_5,
                    "eval/num_problems": snapshot.num_problems,
                    "eval/avg_gen_time": snapshot.avg_generation_time,
                    "eval/is_best": int(snapshot.is_best),
                },
                step=snapshot.step,
            )
        except ImportError:
            pass
        except Exception as exc:
            print(f"[auto_eval] WARNING: wandb logging failed: {exc}")

    def _print_snapshot(self, snapshot: EvalSnapshot) -> None:
        """Print a single snapshot to the console using Rich if available."""
        try:
            from rich.console import Console

            console = Console()

            regression = self.check_regression(snapshot)
            best_tag = " [bold yellow]*BEST*[/bold yellow]" if snapshot.is_best else ""
            reg_tag = " [bold red][REGRESSION][/bold red]" if regression else ""

            console.print(
                f"\n[bold cyan]Auto-Eval[/bold cyan] step [bold]{snapshot.step:,}[/bold]"
                f"  pass@1 [bold green]{snapshot.pass_at_1:.1%}[/bold green]"
                f"  pass@5 [green]{snapshot.pass_at_5:.1%}[/green]"
                f"  ({snapshot.num_problems} problems"
                f", {snapshot.avg_generation_time:.2f}s/prob)"
                f"{best_tag}{reg_tag}"
            )
        except ImportError:
            best_tag = " *BEST*" if snapshot.is_best else ""
            reg_tag = " [REGRESSION]" if self.check_regression(snapshot) else ""
            print(
                f"\nAuto-Eval step {snapshot.step:,}"
                f"  pass@1 {snapshot.pass_at_1:.1%}"
                f"  pass@5 {snapshot.pass_at_5:.1%}"
                f"  ({snapshot.num_problems} problems"
                f", {snapshot.avg_generation_time:.2f}s/prob)"
                f"{best_tag}{reg_tag}"
            )


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def create_auto_evaluator(config: dict) -> "AutoEvaluator | None":
    """Create an AutoEvaluator from a config dict, or return None if disabled.

    The config dict is expected to have an ``auto_eval`` sub-dict.  If the key
    is absent, or if ``enabled`` is False, returns None.

    Example YAML / dict structure::

        auto_eval:
          enabled: true
          eval_every_steps: 5000
          eval_subset: 20
          num_samples: 5
          temperature: 0.2
          save_best: true
          log_to_wandb: false
          checkpoint_dir: checkpoints/tiny

    Args:
        config: Training config dict (may contain an ``auto_eval`` sub-dict).

    Returns:
        Configured AutoEvaluator, or None if auto-eval is disabled / not
        configured.
    """
    ae_cfg = config.get("auto_eval")
    if not ae_cfg:
        return None
    if not ae_cfg.get("enabled", True):
        return None

    return AutoEvaluator(
        eval_every_steps=ae_cfg.get("eval_every_steps", 5000),
        eval_subset=ae_cfg.get("eval_subset", 20),
        save_best=ae_cfg.get("save_best", True),
        best_metric=ae_cfg.get("best_metric", "pass@1"),
        log_to_wandb=ae_cfg.get("log_to_wandb", False),
        num_samples=ae_cfg.get("num_samples", 5),
        temperature=ae_cfg.get("temperature", 0.2),
        max_new_tokens=ae_cfg.get("max_new_tokens", 256),
        timeout=ae_cfg.get("timeout", 10.0),
        subset_seed=ae_cfg.get("subset_seed", 42),
        regression_threshold=ae_cfg.get("regression_threshold", 0.20),
        checkpoint_dir=ae_cfg.get("checkpoint_dir"),
    )
