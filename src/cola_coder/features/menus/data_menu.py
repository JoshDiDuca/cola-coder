"""Data pipeline sub-menu for Cola-Coder."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cola_coder.cli import cli
from cola_coder.features.master_menu import _print_section_header

if TYPE_CHECKING:
    from cola_coder.features.master_menu import MasterMenu


class DataMenu:
    """Data pipeline menu — collect, modify, score, inspect, and prepare data."""

    def __init__(self, master: MasterMenu) -> None:
        self._master = master

    # ── Top-level data menu (5 groups) ────────────────────────────────────

    def menu(self) -> None:
        """Show the grouped data pipeline menu."""
        while True:
            _print_section_header(
                "Data Pipeline", "Download, filter, score, and prepare data"
            )

            options = [
                {"label": "Collect Data",
                 "detail": "GitHub, HuggingFace, Software Heritage, docs scraping"},
                {"label": "Modify Data",
                 "detail": "Combine datasets, generate instruction pairs"},
                {"label": "Score & Filter",
                 "detail": "Quality scoring, classifiers, filter plugins"},
                {"label": "Inspect & View",
                 "detail": "Browse samples, data statistics, FIM data"},
                {"label": "Prepare for Training",
                 "detail": "Tokenize and prepare data for model training"},
            ]

            choice = cli.choose("Select category:", options, allow_cancel=True)
            if choice is None:
                return

            handlers = [
                self._collect_data_menu,
                self._modify_data_menu,
                self._score_filter_menu,
                self._inspect_view_menu,
                self._prepare_for_training_menu,
            ]
            handlers[choice]()

    # ── Collect Data ──────────────────────────────────────────────────────

    def _collect_data_menu(self) -> None:
        """Data collection sub-menu."""
        while True:
            _print_section_header(
                "Collect Data",
                "Gather code from multiple sources",
            )

            options = [
                {"label": "GitHub API Collection",
                 "detail": "Collect code via official GitHub REST API"},
                {"label": "Browse / Import HuggingFace",
                 "detail": "Search any public HF dataset — preview and download"},
                {"label": "Browse Software Heritage",
                 "detail": "Search the universal source code archive (SWH API)"},
                {"label": "Scrape Framework Docs",
                 "detail": "scripts/scrape_docs.py — React/Next.js/Zod/TypeORM"},
                {"label": "Prepare Docs Training Data",
                 "detail": "scripts/prepare_docs_data.py — tokenize scraped docs"},
                {"label": "Prepare Context Training Data",
                 "detail": "scripts/prepare_repo_context_data.py — repo context pairs"},
            ]

            choice = cli.choose("Select source:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._master._run_script("scrape_github.py")
                self._master._pause()
            elif choice == 1:
                self._huggingface_wizard()
            elif choice == 2:
                self._software_heritage_info()
            elif choice == 3:
                self._scrape_docs_menu()
            elif choice == 4:
                self._master._run_script("prepare_docs_data.py")
                self._master._pause()
            elif choice == 5:
                self._master._run_script("prepare_repo_context_data.py")
                self._master._pause()

    # ── Modify Data ───────────────────────────────────────────────────────

    def _modify_data_menu(self) -> None:
        """Data modification sub-menu."""
        while True:
            _print_section_header(
                "Modify Data", "Combine, mix, and transform datasets"
            )

            options = [
                {"label": "Combine Datasets",
                 "detail": "scripts/combine_datasets.py — merge multiple datasets"},
                {"label": "Combine Datasets (weighted)",
                 "detail": "Interactive weighted dataset mixing with per-dataset ratios"},
                {"label": "Generate Instructions",
                 "detail": "scripts/generate_instructions.py — create instruction pairs"},
            ]

            choice = cli.choose("Select operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._master._run_script("combine_datasets.py")
                self._master._pause()
            elif choice == 1:
                self._combine_datasets_menu()
            elif choice == 2:
                self._master._run_script("generate_instructions.py")
                self._master._pause()

    # ── Score & Filter ────────────────────────────────────────────────────

    def _score_filter_menu(self) -> None:
        """Scoring and filtering sub-menu."""
        while True:
            _print_section_header(
                "Score & Filter", "Quality scoring, classifiers, and filter plugins"
            )

            options = [
                {"label": "Score Code Quality",
                 "detail": "Evaluate and rank collected data"},
                {"label": "Score HuggingFace Samples",
                 "detail": "Stream + score samples from any HF dataset"},
                {"label": "Score Repositories",
                 "detail": "scripts/score_repos.py — rank repos by quality"},
                {"label": "Train Quality Classifier",
                 "detail": "scripts/train_quality_classifier.py"},
                {"label": "Advanced Filters",
                 "detail": "PII, dedup, license, syntax — view available plugins"},
            ]

            choice = cli.choose("Select operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._score_quality_menu()
            elif choice == 1:
                self._score_hf_samples()
            elif choice == 2:
                self._score_repos_menu()
            elif choice == 3:
                self._train_quality_classifier_menu()
            elif choice == 4:
                self._advanced_filters_info()

    # ── Inspect & View ────────────────────────────────────────────────────

    def _inspect_view_menu(self) -> None:
        """Data inspection sub-menu."""
        while True:
            _print_section_header(
                "Inspect & View", "Browse and analyze training data"
            )

            options = [
                {"label": "Inspect Dataset",
                 "detail": "Browse random training data samples"},
                {"label": "Data Statistics",
                 "detail": "scripts/data_stats.py — training data size and composition"},
                {"label": "Prepare FIM Data",
                 "detail": "scripts/prepare_fim_data.py — fill-in-the-middle data"},
            ]

            choice = cli.choose("Select operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._inspect_dataset()
                self._master._pause()
            elif choice == 1:
                self._master._run_script("data_stats.py")
                self._master._pause()
            elif choice == 2:
                self._master._run_script("prepare_fim_data.py")
                self._master._pause()

    # ── Prepare for Training ──────────────────────────────────────────────

    def _prepare_for_training_menu(self) -> None:
        """Training data preparation sub-menu."""
        while True:
            _print_section_header(
                "Prepare for Training",
                "Tokenize and prepare data for model training",
            )

            options = [
                {"label": "Prepare Training Data",
                 "detail": "Quick mode selection — tiny/standard/strict/no-filter"},
                {"label": "Interactive Data Prep",
                 "detail": "scripts/prepare_data_interactive.py — guided setup"},
                {"label": "Enhanced Wizard",
                 "detail": "7-step wizard — full config control with summary"},
            ]

            choice = cli.choose("Select mode:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._prepare_data_menu()
            elif choice == 1:
                self._master._run_script("prepare_data_interactive.py")
            elif choice == 2:
                self._prepare_training_wizard()

    # ── HuggingFace Wizard (NEW) ──────────────────────────────────────────

    def _huggingface_wizard(self) -> None:
        """Browse and import any HuggingFace dataset."""
        _print_section_header(
            "Browse / Import HuggingFace",
            "Search, preview, and download HF datasets",
        )

        # Step 1 — Dataset selection
        presets = [
            {"label": "bigcode/starcoderdata",
             "detail": "Default — large multilingual code corpus"},
            {"label": "bigcode/the-stack-v2-train-smol-ids",
             "detail": "The Stack v2 (SWH-derived, deduplicated)"},
            {"label": "Enter custom dataset ID...",
             "detail": "Any public HF dataset identifier"},
        ]

        choice = cli.choose("Select HuggingFace dataset:", presets, allow_cancel=True)
        if choice is None:
            return

        if choice == 2:
            try:
                dataset_id = input("Dataset ID (e.g. owner/name): ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not dataset_id:
                cli.warn("No dataset ID entered.")
                return
        else:
            dataset_id = presets[choice]["label"]

        cli.info("Dataset", dataset_id)

        # Step 2 — Split selection
        splits = [
            {"label": "train", "detail": "Training split (most data)"},
            {"label": "validation", "detail": "Validation split"},
            {"label": "test", "detail": "Test split"},
        ]
        split_idx = cli.choose("Dataset split:", splits, allow_cancel=True)
        if split_idx is None:
            return
        split = splits[split_idx]["label"]

        # Step 3 — Preview
        if cli.confirm("Preview 3 random samples before downloading?", default=True):
            try:
                from cola_coder.data.sources.huggingface import HuggingFaceSource
                source = HuggingFaceSource(
                    dataset=dataset_id, split=split, max_samples=3,
                )
                cli.print("")
                for i, record in enumerate(source.stream(), 1):
                    cli.rule(f"Sample {i}")
                    preview = record.content[:300]
                    if len(record.content) > 300:
                        preview += "  [dim]...[/dim]"
                    cli.print(preview)
                    cli.print("")
            except Exception as e:
                cli.warn(f"Preview failed: {e}")
                cli.dim("Check that HF_TOKEN is set for gated datasets.")

        # Step 4 — Action
        actions = [
            {"label": "Download raw (no tokenization)",
             "detail": f"Save content to data/raw/{dataset_id.replace('/', '_')}/"},
            {"label": "Run through filter pipeline",
             "detail": "Filter + tokenize — launches enhanced wizard"},
            {"label": "Just inspect (no download)",
             "detail": "Browse more samples interactively"},
        ]

        action = cli.choose("What would you like to do?", actions, allow_cancel=True)
        if action is None:
            return

        if action == 0:
            cli.warn("Not yet implemented — use prepare_data.py --stream for now.")
            cli.dim(
                f"  .venv/Scripts/python scripts/prepare_data.py "
                f"--stream --config configs/tiny.yaml "
                f"--tokenizer {self._master.storage.tokenizer_path}"
            )
            self._master._pause()
        elif action == 1:
            self._prepare_training_wizard()
        elif action == 2:
            # Inspect loop
            try:
                from cola_coder.data.sources.huggingface import HuggingFaceSource
                source = HuggingFaceSource(
                    dataset=dataset_id, split=split, max_samples=20,
                )
                for i, record in enumerate(source.stream(), 1):
                    cli.rule(f"Sample {i}")
                    preview = record.content[:600]
                    if len(record.content) > 600:
                        preview += "  [dim]...[/dim]"
                    cli.print(preview)
                    cli.print("")
                    if not cli.confirm("Show next sample?", default=True):
                        break
            except Exception as e:
                cli.warn(f"Inspection failed: {e}")
            self._master._pause()

    # ── Enhanced Prepare Wizard (NEW) ─────────────────────────────────────

    def _prepare_training_wizard(self) -> None:
        """7-step wizard for preparing training data with full config control."""
        _print_section_header(
            "Enhanced Preparation Wizard",
            "Step-by-step training data configuration",
        )

        # Step 1/7 — Config
        cli.step(1, 7, "Model configuration")
        config_options = [
            {"label": "Tiny   (50M)",
             "detail": "configs/tiny.yaml — seq_len=1024"},
            {"label": "Small  (125M)",
             "detail": "configs/small.yaml — seq_len=2048"},
            {"label": "Medium (299M)",
             "detail": "configs/medium.yaml — seq_len=2048"},
            {"label": "4080 Max (455M)",
             "detail": "configs/4080_max.yaml — seq_len=4096"},
            {"label": "No config (manual)",
             "detail": "Configure languages/seq_len manually"},
        ]

        config_choice = cli.choose(
            "Select model config:", config_options, allow_cancel=True
        )
        if config_choice is None:
            return

        config_map = {
            0: "configs/tiny.yaml",
            1: "configs/small.yaml",
            2: "configs/medium.yaml",
            3: "configs/4080_max.yaml",
        }
        config_path = config_map.get(config_choice, "")

        # Step 2/7 — Languages (only if manual)
        languages: list[str] = []
        if config_choice == 4:
            cli.step(2, 7, "Language selection")
            all_langs = [
                "TypeScript", "JavaScript", "Python", "Go",
                "Java", "Rust", "C++", "C",
            ]
            for lang in all_langs:
                if cli.confirm(f"Include {lang}?", default=lang in ("TypeScript", "JavaScript")):
                    languages.append(lang.lower().replace("++", "pp"))
            if not languages:
                cli.warn("No languages selected — defaulting to TypeScript.")
                languages = ["typescript"]
        else:
            cli.step(2, 7, "Language selection — auto from config")
            cli.dim("Languages are defined in the selected config file.")

        # Step 3/7 — Filter mode
        cli.step(3, 7, "Filter mode")
        filter_options = [
            {"label": "Standard",
             "detail": "Balanced quality — ~40% rejection rate (recommended)"},
            {"label": "Strict",
             "detail": "High quality only — ~65-75% rejection rate"},
            {"label": "None",
             "detail": "No filtering — fastest, lowest quality"},
        ]
        filter_choice = cli.choose("Filter mode:", filter_options, allow_cancel=True)
        if filter_choice is None:
            return
        filter_modes = ["standard", "strict", "none"]
        filter_mode = filter_modes[filter_choice]

        # Step 4/7 — Individual filters (informational)
        active_filters: list[str] = []
        if filter_mode != "none":
            cli.step(4, 7, "Individual filters")
            cli.dim(
                "Note: per-filter granularity will be added when "
                "prepare_data.py gains --filters support."
            )
            filter_names = [
                "PII", "Deduplication", "License", "Syntax",
                "Length", "Quality (heuristic)", "Quality (ML classifier)",
            ]
            for name in filter_names:
                if name == "Quality (ML classifier)":
                    cli.dim("  Requires DistilBERT — adds ~2 minutes setup time.")
                if cli.confirm(f"Enable {name} filter?", default=True):
                    active_filters.append(name)
        else:
            cli.step(4, 7, "Individual filters — skipped (no filtering)")

        # Step 5/7 — Quality weights
        cli.step(5, 7, "Quality weights")
        score = cli.confirm(
            "Generate quality weights sidecar (.weights.npy)?", default=False
        )
        if score:
            cli.dim(
                "Adds ~30% preprocessing time. "
                "Enables quality-weighted loss during training."
            )

        # Step 6/7 — Performance
        cli.step(6, 7, "Performance settings")
        cli.dim("Press Enter to accept defaults.")

        try:
            cli.info("Hint", "More workers = faster filtering, more RAM usage")
            workers_str = input("  Worker count [default: auto]: ").strip()
            workers = int(workers_str) if workers_str else None

            cli.info("Hint", "Larger batches = faster tokenization")
            batch_str = input("  Batch size [default: 1000]: ").strip()
            batch_size = int(batch_str) if batch_str else None

            cli.info("Hint", "Limit total tokens for quick test runs")
            max_tok_str = input("  Max tokens [default: no limit]: ").strip()
            max_tokens = int(max_tok_str) if max_tok_str else None
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        except ValueError:
            cli.warn("Invalid number — using defaults.")
            workers = None
            batch_size = None
            max_tokens = None

        # Step 7/7 — Output name
        cli.step(7, 7, "Output configuration")
        try:
            output_name = input(
                "  Output file name [default: auto-named]: "
            ).strip() or ""
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        # Summary
        summary = {
            "Config": config_path or "manual",
            "Languages": ", ".join(languages) if languages else "from config",
            "Filter mode": filter_mode,
            "Active filters": ", ".join(active_filters) or "none",
            "Quality weights": "yes" if score else "no",
            "Workers": str(workers) if workers else "auto",
            "Batch size": str(batch_size) if batch_size else "1000",
            "Max tokens": str(max_tokens) if max_tokens else "no limit",
            "Output name": output_name or "auto-named",
        }
        cli.kv_table(summary, title="Prepare for Training — Configuration")

        if not cli.confirm("Launch with these settings?"):
            return

        # Build arg list (only flags that exist in prepare_data.py)
        args: list[str] = []
        if config_path:
            args += ["--config", config_path]
        args += ["--tokenizer", self._master.storage.tokenizer_path]
        if languages:
            args += ["--languages"] + languages
        if filter_mode == "none":
            args.append("--no-filter")
        elif filter_mode == "strict":
            args.append("--filter-strict")
        if score:
            args.append("--score")
        if workers:
            args += ["--workers", str(workers)]
        if batch_size:
            args += ["--batch-size", str(batch_size)]
        if max_tokens:
            args += ["--max-tokens", str(max_tokens)]
        if output_name:
            args += ["--output-name", output_name]

        self._master._run_script("prepare_data.py", args)
        self._master._pause()

    # ── Score HuggingFace Samples (NEW) ──────────────────────────────────

    def _score_hf_samples(self) -> None:
        """Stream and score samples from a HuggingFace dataset."""
        _print_section_header(
            "Score HuggingFace Samples",
            "Quality-score code samples from any HF dataset",
        )

        # Step 1 — Dataset
        presets = [
            {"label": "bigcode/starcoderdata",
             "detail": "Default — large multilingual code corpus"},
            {"label": "bigcode/the-stack-v2-train-smol-ids",
             "detail": "The Stack v2 (SWH-derived, deduplicated)"},
            {"label": "Enter custom dataset ID...",
             "detail": "Any public HF dataset identifier"},
        ]
        choice = cli.choose(
            "Select dataset to score:", presets, allow_cancel=True
        )
        if choice is None:
            return

        if choice == 2:
            try:
                dataset_id = input("Dataset ID (e.g. owner/name): ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not dataset_id:
                cli.warn("No dataset ID entered.")
                return
        else:
            dataset_id = presets[choice]["label"]

        # Step 2 — Scorer
        scorer_options = [
            {"label": "Heuristic (fast, no deps)",
             "detail": "Pattern-based quality scoring — structure, naming, docs"},
            {"label": "ML Classifier",
             "detail": "DistilBERT/CodeBERTa — needs trained model"},
            {"label": "Full CodeScorer",
             "detail": "12-signal weighted scorer — most detailed breakdown"},
        ]
        scorer_choice = cli.choose(
            "Scoring method:", scorer_options, allow_cancel=True
        )
        if scorer_choice is None:
            return

        # Step 3 — Sample count
        try:
            n_str = input("Number of samples to score [default: 20]: ").strip()
            n_samples = int(n_str) if n_str else 20
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        except ValueError:
            cli.warn("Invalid number — using 20.")
            n_samples = 20

        n_samples = min(max(n_samples, 1), 500)

        # Step 4 — Score
        cli.info("Dataset", dataset_id)
        cli.info("Samples", str(n_samples))
        cli.info("Scorer", scorer_options[scorer_choice]["label"])
        cli.print("")

        try:
            from cola_coder.data.sources.huggingface import HuggingFaceSource
            source = HuggingFaceSource(
                dataset=dataset_id, split="train", max_samples=n_samples,
            )
        except Exception as e:
            cli.error(f"Failed to connect to HuggingFace: {e}")
            cli.dim("Check that HF_TOKEN is set for gated datasets.")
            self._master._pause()
            return

        # Initialize scorer
        def _make_scorer(mode: int):
            if mode == 0:
                from cola_coder.data.filters.quality_classifier import (
                    HeuristicQualityScorer,
                )
                s = HeuristicQualityScorer()
                return lambda code, lang: s.score(code, lang)
            elif mode == 1:
                from cola_coder.data.filters.quality_classifier import (
                    CodeQualityClassifier,
                )
                s = CodeQualityClassifier()
                return lambda code, lang: s.score(code, lang)
            else:
                from cola_coder.features.code_scorer import CodeScorer
                s = CodeScorer()
                return lambda code, lang: s.score(code, lang).overall

        try:
            score_fn = _make_scorer(scorer_choice)
        except Exception as e:
            cli.error(f"Failed to initialize scorer: {e}")
            self._master._pause()
            return

        # Stream and score
        scores: list[float] = []
        grades = {"excellent": 0, "good": 0, "average": 0, "poor": 0, "reject": 0}

        for i, record in enumerate(source.stream(), 1):
            lang = record.metadata.get("language", "")
            s = score_fn(record.content, lang)
            scores.append(s)

            # Grade
            if s >= 0.8:
                grades["excellent"] += 1
            elif s >= 0.6:
                grades["good"] += 1
            elif s >= 0.4:
                grades["average"] += 1
            elif s >= 0.2:
                grades["poor"] += 1
            else:
                grades["reject"] += 1

            # Progress
            if i % 10 == 0 or i == n_samples:
                avg = sum(scores) / len(scores)
                cli.print(
                    f"  Scored {i}/{n_samples} samples — "
                    f"avg: {avg:.3f}"
                )

        if not scores:
            cli.warn("No samples scored.")
            self._master._pause()
            return

        # Summary
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        cli.print("")
        cli.kv_table({
            "Dataset": dataset_id,
            "Samples scored": str(len(scores)),
            "Average score": f"{avg_score:.3f}",
            "Min / Max": f"{min_score:.3f} / {max_score:.3f}",
            "Excellent (0.8+)": str(grades["excellent"]),
            "Good (0.6-0.8)": str(grades["good"]),
            "Average (0.4-0.6)": str(grades["average"]),
            "Poor (0.2-0.4)": str(grades["poor"]),
            "Reject (<0.2)": str(grades["reject"]),
        }, title="Quality Score Summary")

        # Offer next steps
        cli.print("")
        next_options = [
            {"label": "Score more samples",
             "detail": "Run again with different settings"},
            {"label": "View sample breakdown",
             "detail": "Show top/bottom scored samples"},
            {"label": "Done",
             "detail": "Return to Score & Filter menu"},
        ]
        next_choice = cli.choose(
            "What next?", next_options, allow_cancel=True
        )
        if next_choice == 0:
            self._score_hf_samples()
        elif next_choice == 1:
            # Show top 3 and bottom 3
            if len(scores) >= 3:
                cli.print("\n  [bold]Lowest scored samples:[/bold]")
                sorted_indices = sorted(
                    range(len(scores)), key=lambda x: scores[x]
                )
                for idx in sorted_indices[:3]:
                    cli.print(
                        f"    Score: {scores[idx]:.3f}"
                    )
            self._master._pause()

        self._master._pause()

    # ── Existing helper methods (preserved) ───────────────────────────────

    def _prepare_data_menu(self) -> None:
        """Data preparation sub-menu with mode selection."""
        _print_section_header("Prepare Training Data", "Configure and run data pipeline")

        options = [
            {"label": "Interactive Mode",
             "detail": "Guided, menu-driven data preparation"},
            {"label": "Quick Tiny Dataset",
             "detail": "Small dataset for testing — max 500k tokens"},
            {"label": "Standard Preparation",
             "detail": "Full pipeline with defaults from configs/tiny.yaml"},
            {"label": "Standard (Strict Filter)",
             "detail": "Aggressive quality filtering — ~65% rejection rate"},
            {"label": "Standard (No Filter)",
             "detail": "Skip quality filter — faster but lower quality"},
            {"label": "Test/Validation Split",
             "detail": "Prepare test split only"},
        ]

        choice = cli.choose("Preparation mode:", options, allow_cancel=True)
        if choice is None:
            return

        base_args = [
            "--config", "configs/tiny.yaml",
            "--tokenizer", self._master.storage.tokenizer_path,
        ]

        if choice == 0:
            self._master._run_script("prepare_data_interactive.py")
        elif choice == 1:
            self._master._run_script(
                "prepare_data.py", base_args + ["--max-tokens", "500000"]
            )
        elif choice == 2:
            self._master._run_script("prepare_data.py", base_args)
        elif choice == 3:
            self._master._run_script(
                "prepare_data.py", base_args + ["--filter-strict"]
            )
        elif choice == 4:
            self._master._run_script(
                "prepare_data.py", base_args + ["--no-filter"]
            )
        elif choice == 5:
            self._master._run_script(
                "prepare_data.py", base_args + ["--split", "test"]
            )

        self._master._pause()

    def _score_repos_menu(self) -> None:
        """Prompt for a repo path then run score_repos.py."""
        try:
            repo_path = input("Repository/directory path to score: ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not repo_path:
            cli.warn("No path entered — cancelled.")
            return
        path = Path(repo_path)
        # Scraped data directories store repos under _clones/
        clones_dir = path / "_clones"
        if clones_dir.is_dir():
            cli.info("Detected scraped data directory — scoring repos in _clones/")
            path = clones_dir
        args = [str(path)]
        if path.is_dir():
            args.append("--all")
        self._master._run_script("score_repos.py", args)
        self._master._pause()

    def _train_quality_classifier_menu(self) -> None:
        """Pick a subcommand then run train_quality_classifier.py."""
        options = [
            {"label": "demo",
             "detail": "Quick demo of the heuristic scorer (no deps)"},
            {"label": "annotate",
             "detail": "Score code samples with an LLM (needs ANTHROPIC_API_KEY)"},
            {"label": "train",
             "detail": "Fine-tune a small classifier on scored labels"},
            {"label": "evaluate",
             "detail": "Evaluate a trained classifier"},
        ]
        choice = cli.choose("Select command:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            # demo — no required args
            self._master._run_script("train_quality_classifier.py", ["demo"])
        elif choice == 1:
            # annotate — requires --data
            try:
                data_path = input("Path to .npy token data file: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not data_path:
                cli.warn("No path entered — cancelled.")
                return
            self._master._run_script(
                "train_quality_classifier.py", ["annotate", "--data", data_path]
            )
        elif choice == 2:
            # train — requires --labels
            try:
                labels_path = input("Path to quality_labels.jsonl: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not labels_path:
                cli.warn("No path entered — cancelled.")
                return
            self._master._run_script(
                "train_quality_classifier.py", ["train", "--labels", labels_path]
            )
        elif choice == 3:
            # evaluate — requires --model and --labels
            try:
                model_path = input("Path to trained classifier model: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not model_path:
                cli.warn("No path entered — cancelled.")
                return
            try:
                labels_path = input("Path to quality_labels.jsonl: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not labels_path:
                cli.warn("No path entered — cancelled.")
                return
            self._master._run_script(
                "train_quality_classifier.py",
                ["evaluate", "--model", model_path, "--labels", labels_path],
            )
        self._master._pause()

    def _score_quality_menu(self) -> None:
        """Score code quality sub-menu."""
        _print_section_header(
            "Score Code Quality", "Evaluate and rank collected data"
        )

        options = [
            {"label": "Score Repositories",
             "detail": "scripts/score_repos.py — rank repos by code quality"},
            {"label": "Train Quality Classifier",
             "detail": "scripts/train_quality_classifier.py — train ML quality scorer"},
        ]

        choice = cli.choose("Scoring method:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            self._score_repos_menu()
        elif choice == 1:
            self._train_quality_classifier_menu()

    def _software_heritage_info(self) -> None:
        """Show info about Software Heritage data source."""
        _print_section_header(
            "Software Heritage Archive",
            "Universal source code archive — archive.softwareheritage.org",
        )

        cli.print(
            "  Software Heritage is the universal archive of software source code.\n"
            "  It provides deduplicated, archival-quality code with rich metadata.\n"
        )
        cli.print("  [bold cyan]Access methods:[/bold cyan]")
        cli.print(
            "    [cyan]1.[/cyan] SWH REST API — 1,200 req/hr (12,000 with token)"
        )
        cli.print(
            "    [cyan]2.[/cyan] The Stack v2 on HuggingFace — SWH-derived, bulk access"
        )
        cli.print("")
        cli.print("  [bold cyan]Setup:[/bold cyan]")
        cli.print(
            "    Set [green]SWH_API_TOKEN[/green] env var for higher rate limits."
        )
        cli.print(
            "    Get a token at: "
            "[link]https://archive.softwareheritage.org[/link]"
        )
        cli.print("")
        cli.print("  [bold cyan]Code location:[/bold cyan]")
        cli.print(
            "    [dim]src/cola_coder/data/sources/software_heritage.py[/dim]"
        )
        cli.print(
            "    Implements SWHClient, SoftwareHeritageSource (DataSource plugin)"
        )
        cli.print("")
        cli.print(
            "  [dim]Use via the extensible pipeline "
            "(scripts/prepare_data_interactive.py)[/dim]"
        )

        self._master._pause()

    def _advanced_filters_info(self) -> None:
        """Show available data filter plugins."""
        _print_section_header(
            "Advanced Filters",
            "Composable data quality filter plugins",
        )

        filters_info = [
            ("Content Filter", "content.py",
             "Pattern matching — spam, boilerplate, auto-generated"),
            ("Deduplication", "dedup.py",
             "MinHash LSH — near-duplicate detection"),
            ("License Filter", "license_filter.py",
             "SPDX license checking and compliance"),
            ("PII Filter", "pii.py",
             "Detect emails, API keys, secrets, phone numbers"),
            ("Syntax Filter", "syntax.py",
             "Tree-sitter AST parsing (Python, TS, JS, Go, Rust, Java)"),
            ("Length Filter", "length.py",
             "Min/max line count validation"),
            ("Quality Filter", "quality.py",
             "Existing quality filter as composable plugin"),
            ("Quality Classifier", "quality_classifier.py",
             "ML-based quality scoring (DistilBERT)"),
        ]

        table_data = {
            name: f"{filename}  —  {desc}"
            for name, filename, desc in filters_info
        }
        cli.kv_table(table_data, title="Available Filter Plugins")
        cli.print("")
        cli.print("  [bold cyan]Usage:[/bold cyan]")
        cli.print(
            "    Filters are composable plugins in "
            "[dim]src/cola_coder/data/filters/[/dim]"
        )
        cli.print(
            "    Use via the extensible pipeline "
            "(scripts/prepare_data_interactive.py)"
        )
        cli.print(
            "    Or import directly: "
            "[dim]from cola_coder.data.filters import PIIFilter[/dim]"
        )

        self._master._pause()

    def _scrape_docs_menu(self) -> None:
        """Scrape framework docs with framework/version selection."""
        _print_section_header(
            "Scrape Framework Docs", "Download docs for training data"
        )

        options = [
            {"label": "React",   "detail": "reactjs.org / react.dev"},
            {"label": "Next.js", "detail": "nextjs.org"},
            {"label": "Zod",     "detail": "zod.dev"},
            {"label": "TypeORM", "detail": "typeorm.io"},
            {"label": "All",     "detail": "Scrape all four frameworks"},
        ]

        choice = cli.choose("Select framework:", options, allow_cancel=True)
        if choice is None:
            return

        frameworks = ["react", "nextjs", "zod", "typeorm", "all"]
        framework = frameworks[choice]

        try:
            version = input(
                "Enter version (or 'latest'): "
            ).strip() or "latest"
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        args = ["--framework", framework, "--version", version]
        self._master._run_script("scrape_docs.py", args)
        self._master._pause()

    def _combine_datasets_menu(self) -> None:
        """Interactive weighted dataset mixing."""
        import numpy as np

        _print_section_header(
            "Combine Datasets (weighted)",
            "Mix datasets with custom sampling ratios",
        )

        data_dir = (
            self._master._resolve_path(self._master.storage.data_dir)
            / "processed"
        )
        npy_files = sorted(data_dir.glob("*.npy")) if data_dir.exists() else []

        if not npy_files:
            cli.error(f"No .npy datasets found in {data_dir}")
            cli.dim("Run a data preparation step first.")
            self._master._pause()
            return

        # Build file options
        file_options = []
        for f in npy_files:
            try:
                arr = np.load(str(f), mmap_mode="r")
                size_str = f"{f.stat().st_size / 1e6:.1f} MB"
                shape_str = (
                    f"{arr.shape[0]:,} x {arr.shape[1]}"
                    if arr.ndim == 2 else str(arr.shape)
                )
                file_options.append({
                    "label": f.stem, "detail": f"{shape_str}  •  {size_str}"
                })
            except Exception:
                file_options.append({"label": f.stem, "detail": str(f)})

        cli.info("Datasets found", str(len(npy_files)))

        # Sequential selection with weights
        selected_paths: list[str] = []
        selected_weights: list[float] = []

        for i, opt in enumerate(file_options):
            try:
                include = input(
                    f"Include '{opt['label']}' ({opt['detail']})? [y/N]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if include == "y":
                try:
                    weight_str = input(
                        f"  Weight for '{opt['label']}' (e.g. 0.8): "
                    ).strip()
                    weight = float(weight_str) if weight_str else 1.0
                except (ValueError, EOFError, KeyboardInterrupt):
                    weight = 1.0
                selected_paths.append(str(npy_files[i]))
                selected_weights.append(weight)

        if not selected_paths:
            cli.warn("No datasets selected.")
            self._master._pause()
            return

        # Display selection summary
        total = sum(selected_weights)
        cli.info("Selected", str(len(selected_paths)))
        for path, w in zip(selected_paths, selected_weights):
            cli.info("  weight", f"{w / total:.1%}  —  {Path(path).stem}")

        # Build --datasets args with :weight suffix
        datasets_args: list[str] = []
        for path, w in zip(selected_paths, selected_weights):
            datasets_args.append(f"{path}:{w}")

        try:
            output = input("Output path (leave blank for auto): ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        args = ["--datasets"] + datasets_args
        if output:
            args += ["--output", output]

        self._master._run_script("combine_datasets.py", args)
        self._master._pause()

    def _inspect_dataset(self) -> None:
        """Browse random samples from training data (inline inspection)."""
        import numpy as np

        data_dir = (
            self._master._resolve_path(self._master.storage.data_dir)
            / "processed"
        )
        npy_files = list(data_dir.glob("*.npy")) if data_dir.exists() else []

        if not npy_files:
            cli.error(f"No datasets found in {data_dir}")
            return

        # Pick a dataset
        if len(npy_files) == 1:
            npy_path = npy_files[0]
        else:
            options = [
                {
                    "label": f.stem,
                    "detail": str(round(f.stat().st_size / 1e6, 1)) + " MB",
                }
                for f in npy_files
            ]
            choice = cli.choose(
                "Select dataset to inspect:", options, allow_cancel=True
            )
            if choice is None:
                return
            npy_path = npy_files[choice]

        _print_section_header("Dataset Inspector", npy_path.stem)

        data = np.load(str(npy_path), mmap_mode="r")
        cli.info(
            "Shape",
            f"{data.shape[0]:,} chunks x {data.shape[1]} tokens",
        )
        cli.info("Total tokens", f"{data.shape[0] * data.shape[1]:,}")
        cli.info("File", str(npy_path))
        cli.print("")

        try:
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
            tokenizer_path = self._master._resolve_path(
                self._master.storage.tokenizer_path
            )
            if not tokenizer_path.exists():
                cli.warn(
                    f"{tokenizer_path} not found — can't decode samples."
                )
                return

            tokenizer = CodeTokenizer(str(tokenizer_path))
            n_samples = min(3, data.shape[0])
            indices = np.random.choice(
                data.shape[0], size=n_samples, replace=False
            )

            for idx in indices:
                cli.rule(f"Sample #{idx}")
                tokens = data[idx].tolist()
                text = tokenizer.decode(tokens)
                display = text[:600] + ("  [...]" if len(text) > 600 else "")
                cli.print(display)
                cli.print("")

        except Exception as e:
            cli.warn(f"Could not decode samples: {e}")
