"""Evaluation and benchmarking sub-menu for Cola-Coder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cola_coder.cli import cli
from cola_coder.features.master_menu import _print_section_header

if TYPE_CHECKING:
    from cola_coder.features.master_menu import MasterMenu


class EvalMenu:
    """Evaluate & benchmark menu — grouped into sub-categories."""

    def __init__(self, master: MasterMenu) -> None:
        self._master = master

    # ── Top-level eval menu (5 groups) ────────────────────────────────────

    def menu(self) -> None:
        """Grouped evaluation and benchmarking menu."""
        while True:
            _print_section_header("Evaluate & Benchmark", "Measure model quality")

            options = [
                {"label": "Benchmarks",
                 "detail": "TypeScript, Python, React, inference profiling"},
                {"label": "Router Evaluation",
                 "detail": "Domain detection, routing accuracy, specialist testing"},
                {"label": "Quality & Regression",
                 "detail": "Smoke test, regression, quality report, model card"},
                {"label": "Compare",
                 "detail": "Checkpoint diffs, side-by-side, model comparison"},
                {"label": "Training Status",
                 "detail": "Inspect training logs — no GPU needed"},
                {"label": "Safety Evaluation",
                 "detail": "Harmful output rate, refusal accuracy, PII, license"},
                {"label": "Robustness Evaluation",
                 "detail": "Functional drift under semantically-preserving docstring rewordings"},
                {"label": "Depth / Early-Exit Profile",
                 "detail": "Per-token logit-lens convergence depth — how many layers each token needs"},
                {"label": "Spectral Health / Divergence Risk",
                 "detail": "Per-layer sign-collapse of weight-activation alignment — early divergence signal"},
                {"label": "Process / Function-Step Credit",
                 "detail": "Verifier-graded per-function process_score + fragile-function flags"},
                {"label": "Routing Accuracy",
                 "detail": "Test semantic router classification accuracy across domains"},
                {"label": "Data Contamination",
                 "detail": "Check eval problems for leakage into the training corpus"},
                {"label": "PLD Acceptance Analysis",
                 "detail": "Offline prompt-lookup speculative-decoding acceptance + speedup (no model)"},
            ]

            choice = cli.choose("Select category:", options, allow_cancel=True)
            if choice is None:
                return

            handlers = [
                self._benchmarks_menu,
                self._router_eval_menu,
                self._quality_menu,
                self._compare_menu,
                self._master._tools.training_status_menu,
                self._safety_eval_menu,
                self._robustness_eval_menu,
                self._depth_profile_menu,
                self._spectral_health_menu,
                self._process_credit_menu,
                self._routing_accuracy_menu,
                self._contamination_menu,
                self._pld_analysis_menu,
            ]
            handlers[choice]()

    # ── Benchmarks sub-menu ───────────────────────────────────────────────

    def _benchmarks_menu(self) -> None:
        """Language-specific and general benchmarks."""
        while True:
            _print_section_header("Benchmarks", "Language-specific and general benchmarks")

            options = [
                {"label": "TypeScript Benchmark",
                 "detail": "50 problems — basics, types, react, nextjs, prisma, zod, testing"},
                {"label": "TypeScript Quick (Nano)",
                 "detail": "10 simple TypeScript problems — fast validation"},
                {"label": "TypeScript/React Benchmark",
                 "detail": "14 problems — React components + Next.js patterns"},
                {"label": "Python Benchmark (HumanEval)",
                 "detail": "62 problems — full HumanEval suite with pass@k"},
                {"label": "Python Completion",
                 "detail": "30 prefix-completion problems — easy/medium/hard"},
                {"label": "Mixed Quick Benchmark",
                 "detail": "5 prompts (Python + TS) — speed + quality"},
                {"label": "Run ALL Benchmarks",
                 "detail": "Full eval suite — HumanEval + TS + smoke + regression"},
                {"label": "Inference Profiler",
                 "detail": "tok/s across temperatures, seq lengths, precisions"},
            ]

            choice = cli.choose("Select benchmark:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._ts_benchmark_menu()
            elif choice == 1:
                self._ts_quick_benchmark()
            elif choice == 2:
                self._ts_react_benchmark()
            elif choice == 3:
                self._python_humaneval_menu()
            elif choice == 4:
                self._python_completion_benchmark()
            elif choice == 5:
                self._benchmark_menu()
            elif choice == 6:
                self._run_eval_suite_menu()
            elif choice == 7:
                self._inference_profiler_menu()

    # ── Router Evaluation sub-menu ────────────────────────────────────────

    def _router_eval_menu(self) -> None:
        """Router and domain detection evaluation."""
        while True:
            _print_section_header(
                "Router Evaluation", "Domain routing accuracy and specialist testing"
            )

            options = [
                {"label": "Domain Detection Test",
                 "detail": "Test heuristic domain classifier on sample code"},
                {"label": "Router Accuracy",
                 "detail": "Precision/recall/F1 per domain (8 domains)"},
                {"label": "Router + Specialist Benchmark",
                 "detail": "Route then generate — combined accuracy report"},
            ]

            choice = cli.choose("Select operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                # Domain detection test — reuse from router_menu
                self._domain_detection_test()
            elif choice == 1:
                self._router_accuracy()
            elif choice == 2:
                self._router_specialist_benchmark()

    # ── Quality & Regression sub-menu ─────────────────────────────────────

    def _quality_menu(self) -> None:
        """Quality checks, regression tests, reports, model card."""
        while True:
            _print_section_header(
                "Quality & Regression", "Model health checks and reporting"
            )

            options = [
                {"label": "Smoke Test",
                 "detail": "8 quick validation checks (<30s)"},
                {"label": "Regression Tests",
                 "detail": "20 baselines — track quality across checkpoints"},
                {"label": "Quality Report",
                 "detail": "Auto-generate markdown + JSON report"},
                {"label": "Generate Model Card",
                 "detail": "HuggingFace-style model card with real examples"},
            ]

            choice = cli.choose("Select operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._smoke_test_menu()
            elif choice == 1:
                self._regression_test_menu()
            elif choice == 2:
                self._quality_report_menu()
            elif choice == 3:
                self._model_card_menu()

    # ── Compare sub-menu ──────────────────────────────────────────────────

    def _compare_menu(self) -> None:
        """Checkpoint and model comparison tools."""
        while True:
            _print_section_header(
                "Compare", "Side-by-side checkpoint and model comparisons"
            )

            options = [
                {"label": "Compare Checkpoints",
                 "detail": "Side-by-side comparison of two checkpoints (same model)"},
                {"label": "Compare Models",
                 "detail": "Cross-model comparison (different architectures)"},
                {"label": "Checkpoint Diff",
                 "detail": "Parameter-level diffs between two checkpoints"},
                {"label": "Checkpoint Info",
                 "detail": "Display metadata, config, file sizes"},
            ]

            choice = cli.choose("Select operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._compare_checkpoints_menu()
            elif choice == 1:
                self._compare_models_menu()
            elif choice == 2:
                self._checkpoint_diff_menu()
            elif choice == 3:
                self._checkpoint_info_menu()

    # ── Benchmark implementations ─────────────────────────────────────────

    def _ts_benchmark_menu(self) -> None:
        """TypeScript benchmark with category filter."""
        _print_section_header(
            "TypeScript Benchmark", "50 problems across 7 categories"
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to benchmark:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        # Category filter
        cat_options = [
            {"label": "All categories (50 problems)",
             "detail": "Full TypeScript benchmark"},
            {"label": "React + Next.js (14 problems)",
             "detail": "Frontend framework focus"},
            {"label": "Types + Basics (18 problems)",
             "detail": "Core TypeScript skills"},
            {"label": "Prisma + Zod (12 problems)",
             "detail": "Schema & validation"},
            {"label": "Testing (6 problems)",
             "detail": "Jest/Vitest patterns"},
            {"label": "Custom selection...",
             "detail": "Pick individual categories"},
        ]

        cat_choice = cli.choose("Category filter:", cat_options, allow_cancel=True)
        if cat_choice is None:
            return

        args = ["--checkpoint", ckpt_path, "--config", config]

        cat_map = {
            1: "react,nextjs",
            2: "types,basics",
            3: "prisma,zod",
            4: "testing",
        }
        if cat_choice in cat_map:
            args += ["--category", cat_map[cat_choice]]
        elif cat_choice == 5:
            # Custom selection
            all_cats = ["basics", "types", "react", "nextjs", "prisma", "zod", "testing"]
            cat_opts = [
                {"label": c, "detail": f"TypeScript {c} problems"}
                for c in all_cats
            ]
            selected = cli.multi_select("Select categories:", cat_opts, preselected=[0, 1])
            if selected:
                cats = ",".join(all_cats[i] for i in selected)
                args += ["--category", cats]

        cli.info("Language", "TypeScript")
        self._master._run_script("ts_benchmark.py", args)
        self._master._pause()

    def _ts_quick_benchmark(self) -> None:
        """Quick 10-problem TypeScript nano benchmark."""
        _print_section_header(
            "TypeScript Quick (Nano)", "10 simple TypeScript problems"
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to benchmark:"
        )
        if ckpt_path is None:
            return

        cli.info("Language", "TypeScript")
        cli.info("Problems", "10 (difficulty 1-2)")
        self._master._run_script("nano_benchmark.py", ["--checkpoint", ckpt_path])
        self._master._pause()

    def _ts_react_benchmark(self) -> None:
        """TypeScript/React benchmark — React + Next.js problems only."""
        _print_section_header(
            "TypeScript/React Benchmark",
            "14 problems — React components + Next.js patterns",
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to benchmark:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        cli.info("Language", "TypeScript React")
        cli.info("Problems", "8 React + 6 Next.js = 14 total")
        self._master._run_script("ts_benchmark.py", [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--category", "react,nextjs",
        ])
        self._master._pause()

    def _python_humaneval_menu(self) -> None:
        """Python HumanEval benchmark with extended problem set."""
        _print_section_header(
            "Python Benchmark (HumanEval)",
            "62 problems — original 20 + 42 extended",
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to evaluate:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        # Problem set selection
        set_options = [
            {"label": "Full suite (62 problems)",
             "detail": "All HumanEval — original 20 + 42 extended"},
            {"label": "Original only (20 problems)",
             "detail": "Standard HumanEval — comparable to published benchmarks"},
            {"label": "Extended only (42 problems)",
             "detail": "Additional problems — harder, more diverse"},
        ]

        set_choice = cli.choose("Problem set:", set_options, allow_cancel=True)
        if set_choice is None:
            return

        args = ["--checkpoint", ckpt_path, "--config", config]
        if set_choice == 0:
            args.append("--extended")
        elif set_choice == 2:
            args += ["--extended", "--extended-only"]

        cli.info("Language", "Python")
        self._master._run_script("evaluate.py", args)
        self._master._pause()

    def _python_completion_benchmark(self) -> None:
        """Python completion benchmark — 30 prefix-completion problems."""
        _print_section_header(
            "Python Completion Benchmark",
            "30 problems — 10 easy, 10 medium, 10 hard",
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to benchmark:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        cli.info("Language", "Python")
        cli.info("Problems", "30 (10 easy, 10 medium, 10 hard)")
        self._master._run_script("completion_benchmark.py", [
            "--checkpoint", ckpt_path, "--config", config,
        ])
        self._master._pause()

    def _benchmark_menu(self) -> None:
        """Quick mixed benchmark (5 prompts)."""
        _print_section_header(
            "Mixed Quick Benchmark", "5 prompts — Python + TypeScript"
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to benchmark:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        cli.info("Languages", "Python, TypeScript, JavaScript")
        cli.info("Prompts", "5 (fibonacci, sort, Calculator, React App, async fetchData)")
        self._master._run_script(
            "benchmark.py", ["--checkpoint", ckpt_path, "--config", config]
        )
        self._master._pause()

    # ── Router evaluation implementations ─────────────────────────────────

    def _domain_detection_test(self) -> None:
        """Test heuristic domain detection on built-in samples."""
        _print_section_header(
            "Domain Detection Test", "Test the heuristic classifier"
        )

        cli.print("  Testing domain detection on built-in samples...\n")

        try:
            from cola_coder.features.router_evaluation import create_test_dataset
            from cola_coder.features.domain_detector import detect_domain

            test_data = create_test_dataset()
            correct = 0

            for sample in test_data:
                scores = detect_domain(sample.prompt)
                predicted = scores[0].domain if scores else "unknown"
                is_correct = predicted == sample.expected_domain
                if is_correct:
                    correct += 1

                icon = "[green]✓[/green]" if is_correct else "[red]✗[/red]"
                conf = f"{scores[0].confidence:.2f}" if scores else "0.00"
                cli.print(
                    f"  {icon}  expected=[cyan]{sample.expected_domain:12s}[/cyan]"
                    f"  predicted={predicted:12s}  conf={conf}"
                )

            acc = correct / len(test_data) if test_data else 0
            cli.info("Accuracy", f"{correct}/{len(test_data)} ({acc:.0%})")

        except Exception as e:
            cli.error(f"Detection test failed: {e}")

        self._master._pause()

    def _router_accuracy(self) -> None:
        """Evaluate router accuracy with precision/recall/F1."""
        _print_section_header(
            "Router Accuracy", "Precision, recall, F1 per domain"
        )

        try:
            from cola_coder.features.router_evaluation import (
                RouterEvaluator, create_test_dataset,
            )
            from cola_coder.features.domain_detector import classify

            evaluator = RouterEvaluator()
            test_data = create_test_dataset()

            for sample in test_data:
                predicted = classify(sample.prompt)
                evaluator.record(
                    predicted=predicted,
                    actual=sample.expected_domain,
                    confidence=0.8,
                )

            metrics = evaluator.compute_metrics()

            cli.kv_table({
                "Overall accuracy": f"{metrics['accuracy']:.1%}",
                "Macro F1": f"{metrics['macro_f1']:.3f}",
                "Weighted F1": f"{metrics['weighted_f1']:.3f}",
            }, title="Router Accuracy")

            cli.print("")
            cli.print("  [bold]Per-domain results:[/bold]")
            for domain, stats in metrics.get("per_domain", {}).items():
                p = stats.get("precision", 0)
                r = stats.get("recall", 0)
                f1 = stats.get("f1", 0)
                cli.print(
                    f"    {domain:12s}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}"
                )

        except Exception as e:
            cli.error(f"Evaluation failed: {e}")

        self._master._pause()

    def _router_specialist_benchmark(self) -> None:
        """Combined router + specialist benchmark."""
        _print_section_header(
            "Router + Specialist Benchmark",
            "Route then generate — combined accuracy report",
        )

        cli.print("  Routes each test prompt through the domain detector,")
        cli.print("  then generates with the appropriate specialist model.")
        cli.print("")

        try:
            from cola_coder.features.specialist_registry import SpecialistRegistry
            from pathlib import Path
            registry = SpecialistRegistry(
                str(self._master.project_root / "configs" / "specialists.yaml")
            )
            specialists = registry.list_specialists()

            if not specialists:
                cli.warn("No specialists registered.")
                cli.dim("Register specialist models in configs/specialists.yaml first.")
                self._master._pause()
                return

            cli.info("Registered specialists", str(len(specialists)))
            for spec in specialists:
                exists = Path(spec.checkpoint).exists() if spec.checkpoint else False
                status = "[green]ready[/green]" if exists else "[red]missing[/red]"
                cli.print(f"    {spec.domain:12s}  {status}")

            cli.print("")
            cli.warn("Full specialist benchmarking requires loading each model.")
            cli.dim("Use the individual TypeScript/Python benchmarks for now.")

        except Exception as e:
            cli.warn(f"Could not load specialist registry: {e}")

        self._master._pause()

    # ── Quality & regression implementations ──────────────────────────────

    def _smoke_test_menu(self) -> None:
        """Quick smoke test for a checkpoint."""
        _print_section_header("Smoke Test", "8 quick validation checks in <30 seconds")

        cli.print("  Checks: token generation, syntax, perplexity, repetition,")
        cli.print("  diversity, special tokens, temperature sensitivity, code keywords")
        cli.print("")

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to smoke-test:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        args = ["--checkpoint", ckpt_path, "--config", config]
        if cli.confirm("Quick mode (fewer samples)?", default=True):
            args.append("--quick")

        self._master._run_script("smoke_test.py", args)
        self._master._pause()

    def _regression_test_menu(self) -> None:
        """Regression test suite."""
        _print_section_header(
            "Regression Tests", "Track quality across checkpoint versions"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to test:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script(
            "regression_test.py",
            ["--checkpoint", ckpt_path, "--config", config],
        )
        self._master._pause()

    def _quality_report_menu(self) -> None:
        """Generate quality report."""
        _print_section_header(
            "Quality Report", "Auto-generate markdown quality report"
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint for report:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script(
            "quality_report.py",
            ["--checkpoint", ckpt_path, "--config", config],
        )
        self._master._pause()

    def _model_card_menu(self) -> None:
        """Generate model card with language selection."""
        _print_section_header(
            "Generate Model Card",
            "HuggingFace-style model card with real examples",
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint for model card:"
        )
        if ckpt_path is None:
            return

        # Language selection for example generation
        languages = cli.pick_languages("Select languages for examples:")

        args = ["--checkpoint", ckpt_path]
        if languages:
            args += ["--languages", ",".join(languages)]

        self._master._run_script("model_card.py", args)
        self._master._pause()

    # ── Compare implementations ───────────────────────────────────────────

    def _compare_checkpoints_menu(self) -> None:
        """Compare two checkpoints from the same model."""
        _print_section_header(
            "Compare Checkpoints",
            "Side-by-side comparison of two checkpoints",
        )

        model = self._master._pick_model(
            "Select model to compare checkpoints:"
        )
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
            "compare_checkpoints.py", ["--a", ckpt_a, "--b", ckpt_b],
        )
        self._master._pause()

    def _compare_models_menu(self) -> None:
        """Cross-model comparison."""
        _print_section_header(
            "Compare Models",
            "Side-by-side comparison of different model checkpoints",
        )

        ckpt_a = self._master._pick_checkpoint(
            "Select first model checkpoint:"
        )
        if ckpt_a is None:
            return
        ckpt_b = self._master._pick_checkpoint(
            "Select second model checkpoint:"
        )
        if ckpt_b is None:
            return

        self._master._run_script(
            "compare_models.py", ["--checkpoints", ckpt_a, ckpt_b],
        )
        self._master._pause()

    def _checkpoint_diff_menu(self) -> None:
        """Compare parameter diffs between two checkpoints."""
        _print_section_header(
            "Checkpoint Diff",
            "Parameter differences between two checkpoints",
        )

        ckpt_a = self._master._pick_checkpoint("Select first checkpoint:")
        if ckpt_a is None:
            return
        ckpt_b = self._master._pick_checkpoint("Select second checkpoint:")
        if ckpt_b is None:
            return

        self._master._run_script(
            "checkpoint_diff.py", ["--a", ckpt_a, "--b", ckpt_b],
        )
        self._master._pause()

    def _checkpoint_info_menu(self) -> None:
        """Display checkpoint metadata."""
        _print_section_header(
            "Checkpoint Info", "Metadata, config, file sizes"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint:")
        if ckpt_path is None:
            return

        self._master._run_script("checkpoint_info.py", [ckpt_path])
        self._master._pause()

    # ── Utility implementations ───────────────────────────────────────────

    def _run_eval_suite_menu(self) -> None:
        """Run all evaluations in sequence."""
        _print_section_header(
            "Run Full Eval Suite",
            "HumanEval + TS + smoke + regression + quality report",
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to evaluate:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script(
            "run_eval_suite.py",
            ["--checkpoint", ckpt_path, "--config", config],
        )
        self._master._pause()

    def _inference_profiler_menu(self) -> None:
        """Inference throughput profiling."""
        _print_section_header(
            "Inference Profiler",
            "tok/s across temperatures, seq lengths, precisions",
        )

        ckpt_path = self._master._pick_checkpoint(
            "Select checkpoint to profile:"
        )
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script(
            "inference_benchmark.py",
            ["--checkpoint", ckpt_path, "--config", config],
        )
        self._master._pause()

    # ── Safety & routing accuracy ─────────────────────────────────────────

    def _safety_eval_menu(self) -> None:
        """Run safety evaluation metrics on a checkpoint."""
        _print_section_header(
            "Safety Evaluation",
            "Measure harmful output rate, refusal accuracy, and boundary adherence",
        )

        cli.print(
            "  Safety evaluation tests the model against:\n"
            "    - Harmful content generation (should refuse)\n"
            "    - Prompt injection attempts\n"
            "    - PII leakage in generated code\n"
            "    - License-incompatible code reproduction\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to evaluate:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        suite_options = [
            {"label": "Basic safety suite",
             "detail": "25 probe prompts — fast, covers main categories"},
            {"label": "Extended safety suite",
             "detail": "65 probe prompts — injection-prone APIs, misconfig, secrets"},
            {"label": "PII-focused",
             "detail": "24 prompts probing for personal-information fabrication"},
            {"label": "License compliance",
             "detail": "18 prompts probing verbatim copyleft/proprietary reproduction"},
            {"label": "Prompt injection",
             "detail": "16 prompts with embedded malicious instructions in comments"},
            {"label": "CWE vulnerability scan",
             "detail": "17 prompts — static CWE scan of completions "
                       "(command exec, eval, deserialization, SQLi, weak crypto)"},
            {"label": "All suites",
             "detail": "every probe from every suite"},
        ]
        suite_choice = cli.choose("Safety suite:", suite_options, allow_cancel=True)
        if suite_choice is None:
            return

        suite_names = ["basic", "extended", "pii", "license", "injection", "cwe", "all"]
        suite = suite_names[suite_choice]

        cli.kv_table({
            "Checkpoint": ckpt_path,
            "Suite": suite,
        }, title="Safety Evaluation Config")

        if not cli.confirm("Run safety evaluation?"):
            return

        args = [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--suite", suite,
        ]
        self._master._run_script("safety_eval.py", args)
        self._master._pause()

    def _robustness_eval_menu(self) -> None:
        """Run verifier-graded functional robustness evaluation on a checkpoint."""
        _print_section_header(
            "Robustness Evaluation",
            "Functional drift under semantically-preserving docstring rewordings",
        )

        cli.print(
            "  Reword each problem's docstring without changing the spec, then\n"
            "  re-grade with the sandbox verifier. Reports:\n"
            "    - robust_pass@1  — solved under the WORST rewording\n"
            "    - consistency    — pass/fail verdict invariant across rewordings\n"
            "    - fragility list — solved clean but failing a mere rewording\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to evaluate:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        set_options = [
            {"label": "Built-in (20 problems)",
             "detail": "Original core HumanEval-style set — fast"},
            {"label": "Extended (62 problems)",
             "detail": "Original + extended problems"},
            {"label": "All (62 problems)",
             "detail": "Alias for extended — full built-in set"},
        ]
        set_choice = cli.choose("Problem set:", set_options, allow_cancel=True)
        if set_choice is None:
            return
        problems = ["builtin", "extended", "all"][set_choice]

        from cola_coder.evaluation.perturbations import ALL_KINDS

        kind_opts = [{"label": k, "detail": f"{k} perturbation"} for k in ALL_KINDS]
        selected = cli.multi_select(
            "Perturbation kinds:", kind_opts, preselected=list(range(len(ALL_KINDS)))
        )
        kinds = [ALL_KINDS[i] for i in selected] if selected else list(ALL_KINDS)

        want_ci = cli.confirm("Attach bootstrap CI to robust_pass@1?", default=False)

        cli.kv_table({
            "Checkpoint": ckpt_path,
            "Problems": problems,
            "Perturbations": ", ".join(kinds),
            "Bootstrap CI": "yes" if want_ci else "no",
        }, title="Robustness Evaluation Config")

        if not cli.confirm("Run robustness evaluation?"):
            return

        args = [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--problems", problems,
            "--kinds", ",".join(kinds),
        ]
        if want_ci:
            args.append("--ci")
        self._master._run_script("robustness_eval.py", args)
        self._master._pause()

    def _depth_profile_menu(self) -> None:
        """Run the logit-lens per-token depth / early-exit profiler on a checkpoint."""
        _print_section_header(
            "Depth / Early-Exit Profile",
            "Per-token logit-lens convergence depth",
        )

        cli.print(
            "  Decodes EVERY transformer layer through the tied output head and\n"
            "  reports, per token, the earliest layer whose next-token prediction\n"
            "  has converged to the final layer's answer. Reports:\n"
            "    - mean / median exit depth  — how many layers tokens actually need\n"
            "    - cumulative convergence-by-depth curve\n"
            "    - optional breakdown by problem difficulty tier\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to profile:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        set_options = [
            {"label": "Built-in (20 problems)",
             "detail": "Original core HumanEval-style set — fast"},
            {"label": "Extended (62 problems)",
             "detail": "Original + extended problems"},
            {"label": "All (62 problems)",
             "detail": "Alias for extended — full built-in set"},
        ]
        set_choice = cli.choose("Problem set:", set_options, allow_cancel=True)
        if set_choice is None:
            return
        problems = ["builtin", "extended", "all"][set_choice]

        mode_options = [
            {"label": "argmax",
             "detail": "Earliest layer whose top-1 matches the final layer (and stays matched)"},
            {"label": "entropy",
             "detail": "Earliest layer whose softmax entropy <= tau"},
        ]
        mode_choice = cli.choose("Convergence criterion:", mode_options, allow_cancel=True)
        if mode_choice is None:
            return
        mode = ["argmax", "entropy"][mode_choice]

        args = [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--problems", problems,
            "--mode", mode,
        ]
        if mode == "entropy":
            args += ["--tau", "0.5"]
        if cli.confirm("Stratify by problem difficulty tier?", default=False):
            args.append("--by-difficulty")

        cli.kv_table({
            "Checkpoint": ckpt_path,
            "Problems": problems,
            "Mode": mode,
        }, title="Depth Profile Config")

        if not cli.confirm("Run depth profile?"):
            return

        self._master._run_script("depth_profile.py", args)
        self._master._pause()

    def _spectral_health_menu(self) -> None:
        """Run the Spectral-Alignment divergence-risk diagnostic on a checkpoint."""
        _print_section_header(
            "Spectral Health / Divergence Risk",
            "Per-layer sign-collapse of weight-activation alignment",
        )

        cli.print(
            "  For each layer, measures the cosine alignment between the layer's\n"
            "  forward response and u1(W) (the weight's principal singular vector,\n"
            "  via cheap power iteration). A healthy layer's alignments are\n"
            "  SIGN-BALANCED (~half +, half -); SIGN-COLLAPSE (all one sign) is an\n"
            "  EARLY divergence-risk signal preceding a loss explosion. Reports:\n"
            "    - worst layer + its sign-collapse fraction (0.50 healthy -> 1.00)\n"
            "    - per-layer sign-collapse and mean alignment\n"
            "    - optional breakdown by problem difficulty tier\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to diagnose:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        set_options = [
            {"label": "Built-in (20 problems)",
             "detail": "Original core HumanEval-style set — fast"},
            {"label": "Extended (62 problems)",
             "detail": "Original + extended problems"},
            {"label": "All (62 problems)",
             "detail": "Alias for extended — full built-in set"},
        ]
        set_choice = cli.choose("Problem set:", set_options, allow_cancel=True)
        if set_choice is None:
            return
        problems = ["builtin", "extended", "all"][set_choice]

        probe_options = [
            {"label": "q (attention query projection)",
             "detail": "Probe the q_proj weight per block — fastest"},
            {"label": "q,fc2 (query + FFN down projection)",
             "detail": "Probe q_proj and the FFN second linear (down_proj)"},
        ]
        probe_choice = cli.choose("Which weights to probe:", probe_options, allow_cancel=True)
        if probe_choice is None:
            return
        layers = ["q", "q,fc2"][probe_choice]

        args = [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--problems", problems,
            "--layers", layers,
        ]
        if cli.confirm("Stratify by problem difficulty tier?", default=False):
            args.append("--by-difficulty")

        cli.kv_table({
            "Checkpoint": ckpt_path,
            "Problems": problems,
            "Probes": layers,
        }, title="Spectral Health Config")

        if not cli.confirm("Run spectral health diagnostic?"):
            return

        self._master._run_script("spectral_health.py", args)
        self._master._pause()

    def _process_credit_menu(self) -> None:
        """Run the verifier-anchored function-step process-credit profiler on a checkpoint."""
        _print_section_header(
            "Process / Function-Step Credit",
            "Verifier-graded per-function process_score",
        )

        cli.print(
            "  A 'poor-man's PRM': decomposes each candidate into its functions\n"
            "  ('steps') and grades every step with the sandbox verifier. Reports:\n"
            "    - per-candidate process_score — length-normalized mean of step scores\n"
            "    - fragile functions — dead / non-executable code that rides along\n"
            "      on a candidate whose top-level tests still pass\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to profile:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        set_options = [
            {"label": "Built-in (20 problems)",
             "detail": "Original core HumanEval-style set — fast"},
            {"label": "Extended (62 problems)",
             "detail": "Original + extended problems"},
            {"label": "All (62 problems)",
             "detail": "Alias for extended — full built-in set"},
        ]
        set_choice = cli.choose("Problem set:", set_options, allow_cancel=True)
        if set_choice is None:
            return
        problems = ["builtin", "extended", "all"][set_choice]

        best_of_options = [
            {"label": "1 (single sample)", "detail": "One candidate per problem — fastest"},
            {"label": "4 (best-of-N)", "detail": "Sandbox-verified best-of-4 — profile the spread"},
            {"label": "8 (best-of-N)", "detail": "Sandbox-verified best-of-8 — slower, denser"},
        ]
        best_choice = cli.choose("Candidates per problem:", best_of_options, allow_cancel=True)
        if best_choice is None:
            return
        best_of = [1, 4, 8][best_choice]

        args = [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--problems", problems,
            "--best-of", str(best_of),
        ]

        cli.kv_table({
            "Checkpoint": ckpt_path,
            "Problems": problems,
            "Best-of": str(best_of),
        }, title="Process-Credit Config")

        if not cli.confirm("Run process-credit profile?"):
            return

        self._master._run_script("process_credit.py", args)
        self._master._pause()

    def _contamination_menu(self) -> None:
        """Check eval problems for leakage into the training corpus."""
        _print_section_header(
            "Data Contamination",
            "Detect eval problems leaking into training data",
        )

        cli.print(
            "  Contaminated benchmarks inflate pass@k. This checks whether the\n"
            "  eval problems appear in your training corpus (containment match).\n"
        )

        from pathlib import Path

        eval_options = [
            {"label": "All built-in (62)", "detail": "Full HumanEval-style set"},
            {"label": "Built-in (20)", "detail": "Original core set"},
            {"label": "TypeScript", "detail": "TypeScript problem set"},
        ]
        eval_choice = cli.choose("Eval problem set:", eval_options, allow_cancel=True)
        if eval_choice is None:
            return
        eval_arg = ["all", "builtin", "typescript"][eval_choice]

        try:
            corpus = input(
                "  Training corpus path (.jsonl text corpus, or .npy + tokenizer): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not corpus or not Path(corpus).exists():
            cli.error("Corpus path not found.")
            self._master._pause()
            return

        args = ["--eval", eval_arg]
        if corpus.endswith(".npy"):
            try:
                from cola_coder.data.dataset_resolver import DatasetResolver
                tok = str(DatasetResolver.get_tokenizer_path())
            except Exception:
                tok = self._master.storage.tokenizer_path
            args += ["--train-npy", corpus, "--tokenizer", tok]
        else:
            args += ["--train-jsonl", corpus]

        self._master._run_script("check_contamination.py", args)
        self._master._pause()

    def _pld_analysis_menu(self) -> None:
        """Offline prompt-lookup-decoding acceptance analysis over a token corpus."""
        _print_section_header(
            "PLD Acceptance Analysis",
            "Offline prompt-lookup speculative-decoding acceptance + speedup",
        )

        cli.print(
            "  Replays the draft-free PLD drafter over a tokenized .npy corpus and\n"
            "  reports acceptance rate, mean accepted length, draft hit rate, and an\n"
            "  idealised step-count speedup. Pure offline — no model, no GPU.\n"
        )

        from pathlib import Path

        try:
            data = input(
                "  Tokenized corpus path (.npy, shape [N, seq] or [tokens]): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not data or not Path(data).exists():
            cli.error("Data path not found.")
            self._master._pause()
            return

        ngram_options = [
            {"label": "Reference (max=3, min=1, num-pred=10)",
             "detail": "Upstream PLD defaults — high hit rate, lots of short matches"},
            {"label": "Specific (max=4, min=2, num-pred=8)",
             "detail": "Longer, more specific matches — fewer false continuations"},
            {"label": "Aggressive (max=5, min=1, num-pred=16)",
             "detail": "Long continuations — best case for repetitive code"},
        ]
        ngram_choice = cli.choose("Drafter preset:", ngram_options, allow_cancel=True)
        if ngram_choice is None:
            return
        presets = [(3, 1, 10), (4, 2, 8), (5, 1, 16)]
        max_ngram, min_ngram, num_pred = presets[ngram_choice]

        sample_options = [
            {"label": "First 200 sequences", "detail": "Fast sample"},
            {"label": "First 1000 sequences", "detail": "Larger sample"},
            {"label": "All sequences", "detail": "Whole corpus — slowest"},
        ]
        sample_choice = cli.choose("Sample size:", sample_options, allow_cancel=True)
        if sample_choice is None:
            return
        sample = [200, 1000, None][sample_choice]

        args = [
            "--data", data,
            "--max-ngram", str(max_ngram),
            "--min-ngram", str(min_ngram),
            "--num-pred", str(num_pred),
            "--seed-len", "1",
        ]
        if sample is not None:
            args += ["--sample", str(sample)]

        cli.kv_table({
            "Corpus": data,
            "max/min ngram": f"{max_ngram}/{min_ngram}",
            "num-pred": str(num_pred),
            "Sample": "all" if sample is None else str(sample),
        }, title="PLD Analysis Config")

        if not cli.confirm("Run PLD acceptance analysis?"):
            return

        self._master._run_script("pld_analysis.py", args)
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
