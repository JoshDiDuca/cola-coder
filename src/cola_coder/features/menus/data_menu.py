"""Data pipeline sub-menu for Cola-Coder."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cola_coder.cli import cli, HF_LANG_MAP
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
                {"label": "Collect Text Data (FineWeb-Edu)",
                 "detail": "Download high-quality web text for mixed pretraining"},
                {"label": "Collect Math Data (OpenWebMath)",
                 "detail": "Download math reasoning data for code+math pretraining"},
                {"label": "Collect GitHub Issues & PRs",
                 "detail": "Download GitHub issues, PRs, and diff data for SFT"},
                {"label": "Download Instruction Datasets",
                 "detail": "Download public instruction tuning datasets (Alpaca, Orca, etc.)"},
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
            elif choice == 6:
                self._collect_text_data_menu()
            elif choice == 7:
                self._collect_math_data_menu()
            elif choice == 8:
                self._collect_github_artifacts_menu()
            elif choice == 9:
                self._download_instruction_datasets_menu()

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
                {"label": "Run Data Scoring Pipeline",
                 "detail": "Score data with tsc + ESLint + stars + heuristic scorers"},
                {"label": "LLM-as-Judge Annotation",
                 "detail": "Score samples with Claude/Ollama for classifier training"},
                {"label": "Train Quality Classifier (new)",
                 "detail": "Train fast classifier from LLM annotations"},
                {"label": "Apply Curriculum Ordering",
                 "detail": "Reorder data by quality score for curriculum learning"},
                {"label": "Scan Data for Malware",
                 "detail": "Run YARA + Windows Defender on collected data"},
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
                self._run_scoring_pipeline()
            elif choice == 5:
                self._llm_judge_annotation()
            elif choice == 6:
                self._train_judge_classifier()
            elif choice == 7:
                self._apply_curriculum_ordering()
            elif choice == 8:
                self._scan_malware_menu()
            elif choice == 9:
                self._advanced_filters_info()

    # ── Malware Scanning ─────────────────────────────────────────────────

    def _scan_malware_menu(self) -> None:
        """On-demand malware scan of a dataset directory."""
        from cola_coder.security.scanner import CompositeMalwareScanner

        _print_section_header(
            "Scan Data for Malware",
            "Run YARA + Windows Defender on collected data",
        )

        # Step 1: Pick dataset directory
        data_dir = self._master._resolve_path("data")
        if not data_dir.exists():
            cli.warn("No data/ directory found.")
            self._master._pause()
            return

        # Scan for directories containing files
        candidates: list[tuple[str, Path]] = []
        for d in sorted(data_dir.rglob("*")):
            if d.is_dir() and any(d.iterdir()):
                # Only include leaf-ish dirs or dirs with actual files
                files = [f for f in d.iterdir() if f.is_file()]
                if files:
                    rel = d.relative_to(data_dir)
                    candidates.append((str(rel), d))

        if not candidates:
            cli.warn("No data directories with files found.")
            self._master._pause()
            return

        dir_options = [
            {"label": name, "detail": str(path)}
            for name, path in candidates[:20]  # Limit to 20 options
        ]

        dir_choice = cli.choose(
            "Select dataset directory to scan:", dir_options, allow_cancel=True,
        )
        if dir_choice is None:
            return

        _, scan_path = candidates[dir_choice]

        # Step 2: Choose scanners
        scanner_options = [
            {"label": "Full scan (Defender + YARA)",
             "detail": "Recommended for untrusted data"},
            {"label": "Quick scan (YARA only)",
             "detail": "Fast code-specific pattern check"},
        ]
        scanner_choice = cli.choose(
            "Scanner mode:", scanner_options, allow_cancel=True,
        )
        if scanner_choice is None:
            return

        scan_cfg = {
            "scanners": {"yara": True, "defender": scanner_choice == 0},
        }
        scanner = CompositeMalwareScanner.from_config(scan_cfg)

        cli.info("Directory", str(scan_path))
        cli.info("Scanners", ", ".join(scanner.available_scanners))
        cli.print()

        # Step 3: Run scan
        result = scanner.scan_directory(scan_path)

        if result.is_clean:
            cli.success(
                f"Clean: {result.files_scanned} files scanned "
                f"({result.scan_duration_ms:.0f}ms)"
            )
        else:
            cli.warn(
                f"{len(result.threats)} threat(s) found in "
                f"{result.files_scanned} files ({result.scan_duration_ms:.0f}ms)"
            )
            for t in result.threats:
                cli.error(f"  [{t.severity.upper()}] {t.name}: {t.file_path}")
                if t.details:
                    cli.dim(f"    {t.details}")

        self._master._pause()

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
                {"label": "Prepare Mixed Data (code+text+math)",
                 "detail": "Collect multiple sources with per-source ratios (collect_data.py)"},
                {"label": "Prepare Repo-Level Data",
                 "detail": "Format repos with file separators for whole-repo context"},
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
            elif choice == 3:
                self._prepare_mixed_data_menu()
            elif choice == 4:
                self._prepare_repo_level_data_menu()

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

        # Step — Language filter
        languages = cli.pick_languages("Filter by language:")
        if languages is None:
            return

        # Map to HF directory names for download
        hf_langs = set()
        for slug in languages:
            for hf_dir, framework_slugs in HF_LANG_MAP.items():
                if slug in framework_slugs:
                    hf_langs.add(hf_dir)
        hf_lang_list = sorted(hf_langs) if hf_langs else None

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
                    languages=hf_lang_list,
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
            # Download raw samples
            try:
                max_dl_str = input("Max samples to download [default: 1000]: ").strip()
                max_dl = int(max_dl_str) if max_dl_str else 1000
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                self._master._pause()
                return
            except ValueError:
                max_dl = 1000

            safe_name = dataset_id.replace("/", "_")
            out_dir = self._master._resolve_path(f"data/raw/{safe_name}")
            out_dir.mkdir(parents=True, exist_ok=True)

            cli.info("Output", str(out_dir))
            cli.info("Max samples", str(max_dl))

            try:
                from cola_coder.data.sources.huggingface import HuggingFaceSource
                source = HuggingFaceSource(
                    dataset=dataset_id, split=split, max_samples=max_dl,
                    languages=hf_lang_list,
                )
                count = 0
                for record in source.stream():
                    count += 1
                    file_path = out_dir / f"{count:06d}.txt"
                    file_path.write_text(record.content, encoding="utf-8")
                    if count % 100 == 0:
                        cli.print(f"  Downloaded {count}/{max_dl} samples...")

                cli.success(f"Downloaded {count} samples to {out_dir}")
            except Exception as e:
                cli.error(f"Download failed: {e}")
                cli.dim("Check that HF_TOKEN is set for gated datasets.")
            self._master._pause()
        elif action == 1:
            self._prepare_training_wizard()
        elif action == 2:
            # Inspect loop
            try:
                from cola_coder.data.sources.huggingface import HuggingFaceSource
                source = HuggingFaceSource(
                    dataset=dataset_id, split=split, max_samples=20,
                    languages=hf_lang_list,
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
            selected = cli.pick_languages("Select languages for training:")
            if selected is None:
                return
            languages = selected
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

        # Deduplication (raw code is 25-40% exact + 20-30% near-duplicates)
        dedup_choice = cli.choose(
            "Deduplicate chunks after tokenization?",
            [
                {"label": "Exact (recommended)",
                 "detail": "Remove byte-identical chunks (SHA-256, fast)"},
                {"label": "Near-duplicate (MinHash)",
                 "detail": "Also remove similar chunks; needs 'datasketch' (pip install -e '.[dedup]')"},
                {"label": "None",
                 "detail": "Keep all chunks, including exact duplicates"},
            ],
            allow_cancel=False,
        )
        dedup_mode = ["exact", "minhash", "none"][dedup_choice or 0]

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
        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            args += ["--tokenizer", str(DatasetResolver.get_tokenizer_path())]
        except Exception:
            args += ["--tokenizer", self._master.storage.tokenizer_path]
        if languages:
            args += ["--languages"] + languages
        if filter_mode == "none":
            args.append("--no-filter")
        elif filter_mode == "strict":
            args.append("--filter-strict")
        if score:
            args.append("--score")
        if dedup_mode != "exact":  # exact is the script default
            args += ["--dedup", dedup_mode]
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

        # Step — Language selection
        languages = cli.pick_languages("Filter by language:")
        if languages is None:
            return

        # Map framework-language slugs to HF download directories
        hf_langs = set()
        for slug in languages:
            for hf_dir, framework_slugs in HF_LANG_MAP.items():
                if slug in framework_slugs:
                    hf_langs.add(hf_dir)
        hf_lang_list = sorted(hf_langs) if hf_langs else None
        cli.info("Languages", ", ".join(languages))

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
            n_str = input("Number of samples to score [default: all, or enter a number]: ").strip()
            if n_str == "" or n_str.lower() == "all":
                n_samples = None
            else:
                n_samples = int(n_str)
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        except ValueError:
            cli.warn("Invalid number — scoring all samples.")
            n_samples = None

        if n_samples is not None:
            n_samples = max(n_samples, 1)

        # Step 4 — Score
        cli.info("Dataset", dataset_id)
        cli.info("Samples", "all" if n_samples is None else str(n_samples))
        cli.info("Scorer", scorer_options[scorer_choice]["label"])
        cli.print("")

        try:
            from cola_coder.data.sources.huggingface import HuggingFaceSource
            source = HuggingFaceSource(
                dataset=dataset_id, split="train", max_samples=n_samples,
                languages=hf_lang_list,
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

        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            _tok = str(DatasetResolver.get_tokenizer_path())
        except Exception:
            _tok = self._master.storage.tokenizer_path
        base_args = [
            "--config", "configs/tiny.yaml",
            "--tokenizer", _tok,
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
            cli.info("Source", "Detected scraped data directory — scoring repos in _clones/")
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

    def _run_scoring_pipeline(self) -> None:
        """Run the data scoring pipeline (score_data.py)."""
        _print_section_header(
            "Data Scoring Pipeline",
            "Score .npy or .jsonl data with composite scorers",
        )

        # Choose data format
        fmt_options = [
            {"label": "Score .npy data", "detail": "Tokenized data — requires tokenizer"},
            {"label": "Score .jsonl data", "detail": "Raw JSONL (GitHub scraped)"},
        ]
        fmt_choice = cli.choose("Data format:", fmt_options, allow_cancel=True)
        if fmt_choice is None:
            return

        try:
            data_path = input("Path to data file: ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not data_path:
            cli.warn("No path entered — cancelled.")
            return

        args: list[str] = []
        if fmt_choice == 0:
            args.extend(["--data", data_path])
            try:
                tok_path = input("Path to tokenizer.json: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not tok_path:
                cli.warn("Tokenizer path required for .npy data.")
                return
            args.extend(["--tokenizer", tok_path])
        else:
            args.extend(["--jsonl", data_path])

        # Choose scorers
        scorer_options = [
            {"label": "All enabled scorers", "detail": "Use scoring.yaml config"},
            {"label": "Full (tsc + eslint + heuristic)", "detail": "Comprehensive"},
            {"label": "Quick (heuristic only)", "detail": "Fast heuristic-based scoring"},
        ]
        scorer_choice = cli.choose("Scorers:", scorer_options, allow_cancel=True)
        if scorer_choice is None:
            return
        if scorer_choice == 1:
            args.extend(["--scorers", "tsc,eslint,heuristic"])
        elif scorer_choice == 2:
            args.extend(["--scorers", "heuristic"])

        self._master._run_script("score_data.py", args)
        self._master._pause()

    def _llm_judge_annotation(self) -> None:
        """Run LLM-as-Judge annotation via train_judge_classifier.py annotate."""
        _print_section_header(
            "LLM-as-Judge Annotation",
            "Score code samples with Claude or Ollama",
        )

        provider_options = [
            {"label": "Ollama (local)", "detail": "Free, local — requires ollama running"},
            {"label": "Claude (API)", "detail": "Higher quality — requires ANTHROPIC_API_KEY"},
        ]
        prov_choice = cli.choose("LLM provider:", provider_options, allow_cancel=True)
        if prov_choice is None:
            return

        provider = "ollama" if prov_choice == 0 else "claude"
        try:
            model = input(f"Model name (default: {'codellama' if provider == 'ollama' else 'claude-sonnet-4-6'}): ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not model:
            model = "codellama" if provider == "ollama" else "claude-sonnet-4-6"

        try:
            data_path = input("Path to data file (.npy or .jsonl): ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not data_path:
            cli.warn("No path entered — cancelled.")
            return

        args = [
            "annotate",
            "--provider", provider,
            "--model", model,
            "--data", data_path,
        ]

        # Tokenizer for .npy
        if data_path.endswith(".npy"):
            try:
                tok_path = input("Path to tokenizer.json: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if tok_path:
                args.extend(["--tokenizer", tok_path])

        self._master._run_script("train_judge_classifier.py", args)
        self._master._pause()

    def _train_judge_classifier(self) -> None:
        """Train quality classifier from LLM annotations."""
        _print_section_header(
            "Train Quality Classifier",
            "Train fast classifier from LLM annotations",
        )

        cmd_options = [
            {"label": "Train", "detail": "Train classifier from annotations.jsonl"},
            {"label": "Evaluate", "detail": "Evaluate a trained classifier"},
        ]
        cmd_choice = cli.choose("Command:", cmd_options, allow_cancel=True)
        if cmd_choice is None:
            return

        if cmd_choice == 0:
            try:
                ann_path = input("Path to annotations.jsonl: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not ann_path:
                cli.warn("No path entered — cancelled.")
                return
            self._master._run_script(
                "train_judge_classifier.py",
                ["train", "--annotations", ann_path],
            )
        elif cmd_choice == 1:
            try:
                model_dir = input("Path to trained classifier directory: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not model_dir:
                cli.warn("No path entered — cancelled.")
                return
            try:
                ann_path = input("Path to test annotations.jsonl: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not ann_path:
                cli.warn("No path entered — cancelled.")
                return
            self._master._run_script(
                "train_judge_classifier.py",
                ["evaluate", "--model-dir", model_dir, "--annotations", ann_path],
            )
        self._master._pause()

    def _apply_curriculum_ordering(self) -> None:
        """Apply curriculum ordering to scored data."""
        _print_section_header(
            "Curriculum Ordering",
            "Reorder data by quality score for curriculum learning",
        )

        try:
            data_path = input("Path to .npy data file: ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not data_path:
            cli.warn("No path entered — cancelled.")
            return

        try:
            tok_path = input("Path to tokenizer.json: ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not tok_path:
            cli.warn("Tokenizer path required.")
            return

        strategy_options = [
            {"label": "easy_to_hard", "detail": "Start with easy examples, increase difficulty"},
            {"label": "hard_to_easy", "detail": "Start with hard examples first"},
            {"label": "staged", "detail": "Discrete difficulty stages with transitions"},
            {"label": "random", "detail": "Random ordering (baseline)"},
        ]
        strat_choice = cli.choose("Curriculum strategy:", strategy_options, allow_cancel=True)
        if strat_choice is None:
            return

        strategies = ["easy_to_hard", "hard_to_easy", "staged", "random"]
        self._master._run_script("score_data.py", [
            "--data", data_path,
            "--tokenizer", tok_path,
            "--curriculum", strategies[strat_choice],
        ])
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
            try:
                from cola_coder.data.dataset_resolver import DatasetResolver
                tokenizer_path = DatasetResolver.get_tokenizer_path()
            except Exception:
                tokenizer_path = self._master._resolve_path(self._master.storage.tokenizer_path)
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

    # ── New Phase 1/2 data collection methods ─────────────────────────────

    def _collect_text_data_menu(self) -> None:
        """Collect high-quality web text data (FineWeb-Edu / C4 / OpenWebText)."""
        _print_section_header(
            "Collect Text Data",
            "Download web text for mixed code+text pretraining",
        )

        datasets = [
            {"label": "FineWeb-Edu (HuggingFace/FineWeb-Edu)",
             "detail": "High-quality educational web text — best for code+text mix"},
            {"label": "C4 (allenai/c4)",
             "detail": "Cleaned Common Crawl — large general corpus"},
            {"label": "OpenWebText2 (EleutherAI/openwebtext2)",
             "detail": "Reddit-curated web text — high signal-to-noise"},
            {"label": "Custom HuggingFace dataset...",
             "detail": "Enter any HF dataset ID"},
        ]

        choice = cli.choose("Select text dataset:", datasets, allow_cancel=True)
        if choice is None:
            return

        dataset_ids = [
            "HuggingFaceFW/fineweb-edu",
            "allenai/c4",
            "EleutherAI/openwebtext2",
        ]

        if choice == 3:
            try:
                dataset_id = input("HuggingFace dataset ID: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not dataset_id:
                cli.warn("No dataset ID entered.")
                return
        else:
            dataset_id = dataset_ids[choice]

        try:
            max_gb_str = input("Max download size in GB [default: 10]: ").strip()
            max_gb = float(max_gb_str) if max_gb_str else 10.0
        except (ValueError, EOFError, KeyboardInterrupt):
            max_gb = 10.0

        cli.info("Dataset", dataset_id)
        cli.info("Max size", f"{max_gb:.1f} GB")

        if not cli.confirm("Download text data?"):
            return

        cli.info(
            "Command",
            f".venv/Scripts/python scripts/prepare_data.py "
            f"--config configs/tiny.yaml --stream --hf-dataset {dataset_id}",
        )
        cli.dim(
            "Note: Text data download is handled via the HuggingFace pipeline.\n"
            "  After download, use 'Prepare Mixed Data' to combine with code."
        )
        self._master._pause()

    def _collect_math_data_menu(self) -> None:
        """Collect math reasoning data (OpenWebMath / MATH / GSM8K)."""
        _print_section_header(
            "Collect Math Data",
            "Download math + reasoning data for code+math pretraining",
        )

        datasets = [
            {"label": "OpenWebMath (open-web-math/open-web-math)",
             "detail": "Web-scraped mathematical content — proofs, textbooks, forums"},
            {"label": "DeepMind Math (deepmind/math_dataset)",
             "detail": "Procedurally generated math problems — good for reasoning"},
            {"label": "GSM8K (gsm8k)",
             "detail": "Grade school math word problems — chain-of-thought compatible"},
            {"label": "MATH (hendrycks/competition_math)",
             "detail": "Competition mathematics — advanced reasoning"},
        ]

        choice = cli.choose("Select math dataset:", datasets, allow_cancel=True)
        if choice is None:
            return

        dataset_ids = [
            "open-web-math/open-web-math",
            "deepmind/math_dataset",
            "gsm8k",
            "hendrycks/competition_math",
        ]
        dataset_id = dataset_ids[choice]

        cli.info("Dataset", dataset_id)
        cli.info(
            "Hint",
            "Math data is typically small — full download is feasible.",
        )

        if not cli.confirm("Proceed with math data collection?"):
            return

        cli.info(
            "Command",
            f".venv/Scripts/python scripts/prepare_data.py "
            f"--config configs/tiny.yaml --stream --hf-dataset {dataset_id}",
        )
        cli.dim(
            "After download, use 'Prepare Mixed Data' to mix with code+text."
        )
        self._master._pause()

    def _collect_github_artifacts_menu(self) -> None:
        """Collect GitHub issues, PRs, and diff data for SFT."""
        _print_section_header(
            "Collect GitHub Issues & PRs",
            "Download GitHub artifacts for instruction tuning",
        )

        cli.print(
            "  Collects GitHub issues, pull requests, and code diffs —\n"
            "  ideal for instruction-following and code editing SFT.\n"
        )

        source_options = [
            {"label": "GitHub Issues (via API)",
             "detail": "Download issues + comments as instruction pairs"},
            {"label": "GitHub PRs with diffs",
             "detail": "Download PR descriptions + before/after diffs"},
            {"label": "Both issues and PRs",
             "detail": "Full GitHub artifact collection"},
        ]

        choice = cli.choose("Select artifact type:", source_options, allow_cancel=True)
        if choice is None:
            return

        try:
            repos_raw = input(
                "Repos to target (comma-separated, e.g. vercel/next.js,facebook/react): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        if not repos_raw:
            cli.warn("No repos specified — will use default curated list.")

        try:
            max_items_str = input("Max items per repo [default: 500]: ").strip()
            max_items = int(max_items_str) if max_items_str else 500
        except (ValueError, EOFError, KeyboardInterrupt):
            max_items = 500

        artifact_type = ["issues", "prs", "both"][choice]
        cli.kv_table({
            "Artifact type": artifact_type,
            "Repos": repos_raw or "default curated list",
            "Max items/repo": str(max_items),
        }, title="GitHub Collection Config")

        if not cli.confirm("Start GitHub artifact collection?"):
            return

        args = ["--type", artifact_type, "--max-items", str(max_items)]
        if repos_raw:
            for repo in [r.strip() for r in repos_raw.split(",") if r.strip()]:
                args.extend(["--repo", repo])

        self._master._run_script("scrape_github.py", args)
        self._master._pause()

    def _download_instruction_datasets_menu(self) -> None:
        """Download public instruction tuning datasets."""
        _print_section_header(
            "Download Instruction Datasets",
            "Public instruction pairs for SFT",
        )

        datasets = [
            {"label": "Alpaca (tatsu-lab/alpaca)",
             "detail": "52K instruction pairs from GPT-3.5 — general purpose"},
            {"label": "Orca (Open-Orca/OpenOrca)",
             "detail": "1M ChatGPT/GPT-4 completions — diverse reasoning"},
            {"label": "CodeAlpaca (sahil2801/CodeAlpaca-20k)",
             "detail": "20K code instruction pairs — code generation focus"},
            {"label": "ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered)",
             "detail": "Multi-turn ChatGPT conversations"},
            {"label": "WizardCoder (WizardLM/WizardCoder_evol_instruct_110k)",
             "detail": "110K evolved code instructions — complex tasks"},
        ]

        choice = cli.choose("Select instruction dataset:", datasets, allow_cancel=True)
        if choice is None:
            return

        hf_ids = [
            "tatsu-lab/alpaca",
            "Open-Orca/OpenOrca",
            "sahil2801/CodeAlpaca-20k",
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "WizardLM/WizardCoder_evol_instruct_110k",
        ]
        dataset_id = hf_ids[choice]

        cli.info("Dataset", dataset_id)
        cli.dim(
            "Instruction datasets are downloaded via HuggingFace and formatted\n"
            "  into ChatML pairs for SFT. Requires HF_TOKEN for some datasets."
        )

        if not cli.confirm(f"Download {datasets[choice]['label']}?"):
            return

        cli.info(
            "Command",
            f"Use prepare_data.py or the HF wizard to download {dataset_id}",
        )
        # Launch via HF wizard pre-configured
        self._huggingface_wizard()

    # ── New prepare methods ────────────────────────────────────────────────

    def _prepare_mixed_data_menu(self) -> None:
        """Configure and launch mixed code+text+math data collection."""
        _print_section_header(
            "Prepare Mixed Data",
            "Collect code + text + math with per-source ratios",
        )

        cli.print(
            "  Collects multiple sources (code, text, math) and combines them at\n"
            "  the chosen ratios into one training set via collect_data.py.\n"
        )

        # Config selection
        config_options = [
            {"label": "Tiny   (50M)",   "detail": "configs/tiny.yaml"},
            {"label": "Small  (125M)",  "detail": "configs/small.yaml"},
            {"label": "Medium (299M)",  "detail": "configs/medium.yaml"},
            {"label": "4080 Max (455M)", "detail": "configs/4080_max.yaml"},
        ]
        config_choice = cli.choose("Model config:", config_options, allow_cancel=True)
        if config_choice is None:
            return

        config_map = ["tiny", "small", "medium", "4080_max"]
        config_name = config_map[config_choice]
        config_path = f"configs/{config_name}.yaml"

        # Ratio presets
        ratio_options = [
            {"label": "Code-heavy (70% code / 15% text / 15% math)",
             "detail": "Best for code generation — recommended starting point"},
            {"label": "Balanced (50% code / 30% text / 20% math)",
             "detail": "Good generalisation + code quality"},
            {"label": "Text-heavy (30% code / 50% text / 20% math)",
             "detail": "Better language understanding, weaker code"},
            {"label": "Code only (100% code)",
             "detail": "Pure code pretraining — no text/math mixing"},
            {"label": "Custom ratios",
             "detail": "Enter per-source weights manually"},
        ]
        ratio_choice = cli.choose("Mixing ratio preset:", ratio_options, allow_cancel=True)
        if ratio_choice is None:
            return

        ratio_presets = [
            {"code": 0.70, "text": 0.15, "math": 0.15},
            {"code": 0.50, "text": 0.30, "math": 0.20},
            {"code": 0.30, "text": 0.50, "math": 0.20},
            {"code": 1.00, "text": 0.00, "math": 0.00},
        ]

        if ratio_choice == 4:
            try:
                code_w = float(input("  Code weight (0.0-1.0) [default: 0.6]: ").strip() or "0.6")
                text_w = float(input("  Text weight (0.0-1.0) [default: 0.3]: ").strip() or "0.3")
                math_w = float(input("  Math weight (0.0-1.0) [default: 0.1]: ").strip() or "0.1")
            except (ValueError, EOFError, KeyboardInterrupt):
                cli.warn("Invalid input — using balanced preset.")
                code_w, text_w, math_w = 0.5, 0.3, 0.2
            ratios = {"code": code_w, "text": text_w, "math": math_w}
        else:
            ratios = ratio_presets[ratio_choice]

        total = sum(ratios.values())
        cli.kv_table({
            "Config": config_path,
            "Code ratio": f"{ratios['code'] / total:.1%}",
            "Text ratio": f"{ratios['text'] / total:.1%}",
            "Math ratio": f"{ratios['math'] / total:.1%}",
        }, title="Mixed Data Configuration")

        if not cli.confirm("Launch mixed data collection?"):
            return

        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            _tok = str(DatasetResolver.get_tokenizer_path())
        except Exception:
            _tok = self._master.storage.tokenizer_path

        # Write the chosen ratios into a derived data_sources config and run
        # the real multi-source collector. (The old code passed --mix-code/
        # --mix-text/--mix-math to prepare_data.py, which has no such args, so
        # this menu item always errored out with "unrecognized arguments".)
        from cola_coder.data.mixing import write_weighted_data_sources

        derived = write_weighted_data_sources(
            ratios, "configs/auto/data_sources_mixed.yaml"
        )
        cli.dim(f"  Derived data sources: {derived}")
        args = [
            "--config", config_path,
            "--tokenizer", _tok,
            "--data-sources", str(derived),
        ]
        if cli.confirm(
            "Score code quality for weighted training? (--score; text/math stay neutral)"
        ):
            args.append("--score")
        self._master._run_script("collect_data.py", args)
        self._master._pause()

    def _prepare_repo_level_data_menu(self) -> None:
        """Format repos with file separators for whole-repo context training."""
        _print_section_header(
            "Prepare Repo-Level Data",
            "Format repos with file separators for whole-repo context",
        )

        cli.print(
            "  Repo-level formatting packs multiple files from the same repo\n"
            "  into a single context window, separated by file path headers.\n"
            "  This teaches the model cross-file reasoning and import patterns.\n"
        )

        # Config selection
        config_options = [
            {"label": "Tiny   (50M)",    "detail": "configs/tiny.yaml — seq_len=1024"},
            {"label": "Small  (125M)",   "detail": "configs/small.yaml — seq_len=2048"},
            {"label": "Medium (299M)",   "detail": "configs/medium.yaml — seq_len=2048"},
            {"label": "4080 Max (455M)", "detail": "configs/4080_max.yaml — seq_len=4096"},
        ]
        config_choice = cli.choose("Model config:", config_options, allow_cancel=True)
        if config_choice is None:
            return

        config_map = ["tiny", "small", "medium", "4080_max"]
        config_path = f"configs/{config_map[config_choice]}.yaml"

        # Format options
        format_options = [
            {"label": "Path header + content",
             "detail": "# === path/to/file.ts ===\\n<content>"},
            {"label": "XML-style tags",
             "detail": "<file path='...'>\\n<content>\\n</file>"},
            {"label": "Simple separator",
             "detail": "---\\n# file.ts\\n---\\n<content>"},
        ]
        fmt_choice = cli.choose("File separator format:", format_options, allow_cancel=True)
        if fmt_choice is None:
            return

        fmt_names = ["path_header", "xml_tags", "simple_sep"]
        fmt = fmt_names[fmt_choice]

        cli.kv_table({
            "Config": config_path,
            "Format": fmt,
            "Use case": "Cross-file reasoning, imports, project structure",
        }, title="Repo-Level Data Config")

        if not cli.confirm("Prepare repo-level training data?"):
            return

        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            _tok = str(DatasetResolver.get_tokenizer_path())
        except Exception:
            _tok = self._master.storage.tokenizer_path
        args = [
            "--config", config_path,
            "--tokenizer", _tok,
            "--repo-level",
            "--repo-format", fmt,
        ]
        self._master._run_script("prepare_data.py", args)
        self._master._pause()
