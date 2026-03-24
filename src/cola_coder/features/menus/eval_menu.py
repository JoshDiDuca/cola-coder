"""Evaluation and benchmarking sub-menu for Cola-Coder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cola_coder.cli import cli
from cola_coder.features.master_menu import _print_section_header

if TYPE_CHECKING:
    from cola_coder.features.master_menu import MasterMenu


class EvalMenu:
    """Evaluate & benchmark menu."""

    def __init__(self, master: MasterMenu) -> None:
        self._master = master

    def menu(self) -> None:
        """Evaluation and benchmarking sub-menu."""
        while True:
            _print_section_header("Evaluate & Benchmark", "Measure model quality")

            options = [
                {"label": "HumanEval Evaluation",
                 "detail": "scripts/evaluate.py — pass@k on 164 coding problems"},
                {"label": "Quick Benchmark",
                 "detail": "scripts/benchmark.py — speed + quality benchmark"},
                {"label": "Compare Checkpoints",
                 "detail": "scripts/compare_checkpoints.py — side-by-side comparison"},
                {"label": "Nano Benchmark",
                 "detail": "scripts/nano_benchmark.py — fast generation speed test"},
                {"label": "Generate Model Card",
                 "detail": "scripts/model_card.py — create HuggingFace-style model card"},
                {"label": "Training Status",
                 "detail": "scripts/training_status.py — inspect logs, no GPU needed"},
                {"label": "Smoke Test",
                 "detail": "scripts/smoke_test.py — 8 quick validation checks (<30s)"},
                {"label": "TypeScript Benchmark",
                 "detail": "scripts/ts_benchmark.py — 50 TS-specific coding problems"},
                {"label": "Regression Tests",
                 "detail": "scripts/regression_test.py — track quality across checkpoints"},
                {"label": "Quality Report",
                 "detail": "scripts/quality_report.py — auto-generate markdown report"},
                {"label": "Compare Models",
                 "detail": "scripts/compare_models.py — side-by-side model comparison"},
                {"label": "Safety Evaluation",
                 "detail": "Run safety metrics: harmful output rate, refusal accuracy"},
                {"label": "Routing Accuracy",
                 "detail": "Test semantic router classification accuracy across domains"},
            ]

            choice = cli.choose("Select evaluation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._humaneval_menu()
            elif choice == 1:
                self._benchmark_menu()
            elif choice == 2:
                self._compare_checkpoints_menu()
            elif choice == 3:
                self._master._nano_benchmark()
            elif choice == 4:
                self._model_card_menu()
            elif choice == 5:
                self._master._tools.training_status_menu()
            elif choice == 6:
                self._smoke_test_menu()
            elif choice == 7:
                self._ts_benchmark_menu()
            elif choice == 8:
                self._regression_test_menu()
            elif choice == 9:
                self._quality_report_menu()
            elif choice == 10:
                self._compare_models_menu()
            elif choice == 11:
                self._safety_eval_menu()
            elif choice == 12:
                self._routing_accuracy_menu()

    def _humaneval_menu(self) -> None:
        """HumanEval evaluation with checkpoint selection."""
        _print_section_header("HumanEval Evaluation", "164 Python coding problems — pass@k metric")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to evaluate:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script("evaluate.py", ["--checkpoint", ckpt_path, "--config", config])
        self._master._pause()

    def _benchmark_menu(self) -> None:
        """Benchmark with checkpoint selection."""
        _print_section_header("Quick Benchmark", "Speed and quality benchmark")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to benchmark:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script("benchmark.py", ["--checkpoint", ckpt_path, "--config", config])
        self._master._pause()

    def _model_card_menu(self) -> None:
        """Generate model card."""
        _print_section_header("Generate Model Card", "HuggingFace-style model card")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint for model card:")
        if ckpt_path is None:
            return

        self._master._run_script("model_card.py", ["--checkpoint", ckpt_path])
        self._master._pause()

    def _smoke_test_menu(self) -> None:
        """Quick smoke test for a checkpoint."""
        _print_section_header("Smoke Test", "8 quick validation checks in <30 seconds")

        cli.print("  Checks: token generation, syntax, perplexity, repetition,")
        cli.print("  diversity, special tokens, temperature sensitivity, code keywords")
        cli.print("")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to smoke-test:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        args = ["--checkpoint", ckpt_path, "--config", config]
        if cli.confirm("Quick mode (fewer samples)?", default=True):
            args.append("--quick")

        self._master._run_script("smoke_test.py", args)
        self._master._pause()

    def _ts_benchmark_menu(self) -> None:
        """TypeScript-specific benchmark."""
        _print_section_header("TypeScript Benchmark", "50 TS-specific coding problems")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to benchmark:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script("ts_benchmark.py", ["--checkpoint", ckpt_path, "--config", config])
        self._master._pause()

    def _regression_test_menu(self) -> None:
        """Regression test suite."""
        _print_section_header("Regression Tests", "Track quality across checkpoint versions")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to test:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script(
            "regression_test.py", ["--checkpoint", ckpt_path, "--config", config]
        )
        self._master._pause()

    def _quality_report_menu(self) -> None:
        """Generate quality report."""
        _print_section_header("Quality Report", "Auto-generate markdown quality report")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint for report:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script(
            "quality_report.py", ["--checkpoint", ckpt_path, "--config", config]
        )
        self._master._pause()

    def _compare_checkpoints_menu(self) -> None:
        """Compare two checkpoints from the same model."""
        _print_section_header(
            "Compare Checkpoints",
            "Side-by-side comparison of two checkpoints",
        )

        model = self._master._pick_model("Select model to compare checkpoints:")
        if model is None:
            return
        ckpt_a = self._master._pick_checkpoint(
            "Select checkpoint A:", model=model,
        )
        if ckpt_a is None:
            return
        ckpt_b = self._master._pick_checkpoint(
            "Select checkpoint B:", model=model,
        )
        if ckpt_b is None:
            return

        self._master._run_script(
            "compare_checkpoints.py",
            ["--a", ckpt_a, "--b", ckpt_b],
        )
        self._master._pause()

    def _compare_models_menu(self) -> None:
        """Side-by-side comparison of checkpoints from different models."""
        _print_section_header(
            "Compare Models",
            "Side-by-side comparison of two model checkpoints",
        )

        ckpt_a = self._master._pick_checkpoint("Select first model checkpoint:")
        if ckpt_a is None:
            return
        ckpt_b = self._master._pick_checkpoint("Select second model checkpoint:")
        if ckpt_b is None:
            return

        self._master._run_script(
            "compare_models.py", ["--checkpoints", ckpt_a, ckpt_b],
        )
        self._master._pause()

    def _safety_eval_menu(self) -> None:
        """Run safety evaluation metrics on a checkpoint."""
        _print_section_header(
            "Safety Evaluation",
            "Measure harmful output rate, refusal accuracy, and boundary adherence",
        )

        cli.print(
            "  Safety evaluation tests the model against:\n"
            "    • Harmful content generation (should refuse)\n"
            "    • Prompt injection attempts\n"
            "    • PII leakage in generated code\n"
            "    • License-incompatible code reproduction\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to evaluate:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        suite_options = [
            {"label": "Basic safety suite",
             "detail": "50 prompts — fast, covers main categories (~5 min)"},
            {"label": "Extended safety suite",
             "detail": "200 prompts — comprehensive coverage (~20 min)"},
            {"label": "PII-focused",
             "detail": "Test for personal information leakage in code"},
            {"label": "License compliance",
             "detail": "Check for verbatim reproduction of copyleft code"},
        ]
        suite_choice = cli.choose("Safety suite:", suite_options, allow_cancel=True)
        if suite_choice is None:
            return

        suite_names = ["basic", "extended", "pii", "license"]
        suite = suite_names[suite_choice]

        cli.kv_table({
            "Checkpoint": ckpt_path,
            "Suite": suite,
        }, title="Safety Evaluation Config")

        if not cli.confirm("Run safety evaluation?"):
            return

        # safety_eval.py is the Phase evaluation script
        args = [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--suite", suite,
        ]
        self._master._run_script("safety_eval.py", args)
        self._master._pause()

    def _routing_accuracy_menu(self) -> None:
        """Test semantic router classification accuracy."""
        _print_section_header(
            "Routing Accuracy",
            "Test semantic router classification across domains",
        )

        cli.print(
            "  Evaluates the semantic router on a held-out domain classification\n"
            "  test set. Measures per-domain accuracy, confusion matrix, and\n"
            "  confidence calibration.\n"
        )

        # Router checkpoint
        router_options = [
            {"label": "Latest router checkpoint",
             "detail": "checkpoints/router/latest"},
            {"label": "Select specific checkpoint",
             "detail": "Browse available router checkpoints"},
        ]
        router_choice = cli.choose("Router checkpoint:", router_options, allow_cancel=True)
        if router_choice is None:
            return

        if router_choice == 0:
            router_path = "checkpoints/router/latest"
        else:
            router_path = self._master._pick_checkpoint("Select router checkpoint:") or ""
            if not router_path:
                return

        # Domain filter
        domain_options = [
            {"label": "All domains",
             "detail": "React, Next.js, GraphQL, Prisma, Zod, Testing, General TS"},
            {"label": "TypeScript-focused",
             "detail": "React, Next.js, TypeScript general"},
            {"label": "Backend-focused",
             "detail": "GraphQL, Prisma, API design"},
        ]
        domain_choice = cli.choose("Domain scope:", domain_options, allow_cancel=True)
        if domain_choice is None:
            return

        domain_sets = ["all", "typescript", "backend"]
        domain_set = domain_sets[domain_choice]

        cli.kv_table({
            "Router": router_path,
            "Domains": domain_set,
        }, title="Routing Accuracy Config")

        if not cli.confirm("Run routing accuracy test?"):
            return

        args = [
            "--router-checkpoint", router_path,
            "--domains", domain_set,
        ]
        self._master._run_script("evaluate_router.py", args)
        self._master._pause()
