"""Unified master menu for Cola-Coder.

Single entry point for all Cola-Coder operations. Replaces 12 separate
PowerShell scripts with one interactive, keyboard-driven menu.

Navigation: Arrow keys to move, Enter to select, ESC/Ctrl-C to go back.
"""

import importlib
import subprocess
from pathlib import Path
from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config

# Feature toggle - this feature is OPTIONAL
FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Check if this feature is enabled."""
    return FEATURE_ENABLED


# ── Feature category definitions ──────────────────────────────────────────────
# Each entry maps a category name to the module stem names that belong to it.
# Any module not listed here falls into "Other".

_FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Training": [
        "training_monitor", "loss_curve_visualizer", "gradient_norm_monitor",
        "overfitting_detector", "perplexity_tracker", "streaming_training",
        "crash_recovery", "resume_detector", "continuous_eval",
        "realtime_data_stats", "training_speed_dashboard", "dead_neuron_detection",
        "validation_split",
    ],
    "Generation": [
        "streaming_generation", "beam_search", "batch_inference",
        "generation_constraints", "multi_turn_chat", "prompt_templates",
        "speculative_decoding", "multi_token_prediction", "thinking_budget",
        "multi_step_reasoning",
    ],
    "Evaluation": [
        "nano_benchmark", "smoke_test", "fim_benchmark", "typescript_benchmark",
        "real_world_eval", "complexity_scorer", "syntax_validity_rate",
        "type_correctness_rate", "token_efficiency_metric",
        "thinking_quality_scorer", "hallucination_detector", "self_verification",
    ],
    "Infrastructure": [
        "config_validator", "vram_estimator", "gpu_status_panel",
        "checkpoint_comparison", "checkpoint_leaderboard", "experiment_tracker",
        "data_versioning", "dataset_inspector", "model_card_generator",
        "onnx_export", "quantization", "knowledge_distillation", "lora_qlora",
    ],
    "Routing & Specialists": [
        "router_model", "router_evaluation", "router_data_generator",
        "specialist_registry", "cascade_routing", "confidence_routing",
        "hot_swap_specialists", "domain_detector", "moe_layer",
        "ensemble_generation",
    ],
    "Code Analysis": [
        "ast_chunking", "import_graph", "docstring_extraction", "code_diff_mode",
        "multi_file_context", "byte_level_fallback", "test_code_pair_extractor",
        "synthetic_bug_injection", "contrastive_code_learning",
        "constitutional_coding",
    ],
    "UI & Dashboard": [
        "master_menu", "quick_actions", "pipeline_status_dashboard",
        "recent_runs_history", "side_by_side_comparison",
        "reasoning_trace_viewer", "one_click_pipeline",
    ],
}


def _get_features_dir() -> Path:
    """Return the features/ directory path."""
    return Path(__file__).parent


def _get_yaml_path(project_root: Path) -> Path:
    """Return the configs/features.yaml path."""
    return project_root / "configs" / "features.yaml"


def _load_feature_states(project_root: Path) -> dict[str, bool]:
    """Load feature enabled states from features.yaml.

    Returns a dict mapping module stem -> bool (True = enabled).
    Falls back to FEATURE_ENABLED in each module if not in yaml.
    """
    yaml_path = _get_yaml_path(project_root)
    yaml_states: dict[str, bool] = {}
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            yaml_states = {str(k): bool(v) for k, v in (data.get("features") or {}).items()}
        except Exception:
            pass
    return yaml_states


def _save_feature_state(project_root: Path, module_name: str, enabled: bool) -> None:
    """Write a single feature's enabled state to features.yaml."""
    try:
        import yaml
        yaml_path = _get_yaml_path(project_root)
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        else:
            data = {}

        if "features" not in data or data["features"] is None:
            data["features"] = {}
        data["features"][module_name] = enabled

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=True)
    except Exception as exc:
        cli.warn(f"Could not save to features.yaml: {exc}")


def _scan_feature_modules(project_root: Path) -> list[dict]:
    """Scan all feature modules and return their metadata.

    Each item:  {name, label, enabled, module_obj_or_none, category}
    """
    features_dir = _get_features_dir()
    yaml_states = _load_feature_states(project_root)

    # Build reverse lookup: module_name -> category
    stem_to_category: dict[str, str] = {}
    for cat, stems in _FEATURE_CATEGORIES.items():
        for s in stems:
            stem_to_category[s] = cat

    results = []
    py_files = sorted(
        f for f in features_dir.glob("*.py")
        if f.stem not in ("__init__",) and not f.stem.startswith("_")
    )

    for py_file in py_files:
        stem = py_file.stem
        label = stem.replace("_", " ").title()
        category = stem_to_category.get(stem, "Other")

        # Determine enabled state: yaml overrides module default
        if stem in yaml_states:
            enabled = yaml_states[stem]
        else:
            # Try to read FEATURE_ENABLED from the module
            try:
                mod = importlib.import_module(f"cola_coder.features.{stem}")
                enabled = bool(getattr(mod, "FEATURE_ENABLED", True))
            except Exception:
                enabled = True  # assume enabled if module can't be loaded

        results.append({
            "name": stem,
            "label": label,
            "enabled": enabled,
            "category": category,
        })

    return results


def _count_enabled(features: list[dict]) -> tuple[int, int]:
    """Return (enabled_count, total_count)."""
    return sum(1 for f in features if f["enabled"]), len(features)


# ── Rich helpers (graceful fallback if rich not installed) ────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich import box as rich_box
    _rich_console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    _rich_console = None  # type: ignore


def _print_status_panel(status: dict[str, str]) -> None:
    """Print a rich status bar panel with colored indicators."""
    if not _HAS_RICH:
        parts = [f"{k}: {v}" for k, v in status.items()]
        print("  Pipeline: " + " | ".join(parts))
        return

    tokens_val = status.get("tokenizer", "missing")
    data_val = status.get("data", "missing")
    ckpt_val = status.get("checkpoints", "none")

    def _indicator(val: str, ok_check) -> str:
        if ok_check(val):
            return f"[bold green]●[/bold green] [green]{val}[/green]"
        return f"[bold red]●[/bold red] [red]{val}[/red]"

    tokenizer_ind = _indicator(tokens_val, lambda v: v == "ready")
    data_ind = _indicator(data_val, lambda v: "dataset" in v)
    ckpt_ind = _indicator(ckpt_val, lambda v: "checkpoint" in v)

    table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
    table.add_column("", justify="center")
    table.add_column("", justify="center")
    table.add_column("", justify="center")
    table.add_row(
        f"Tokenizer  {tokenizer_ind}",
        f"Data  {data_ind}",
        f"Checkpoints  {ckpt_ind}",
    )

    _rich_console.print(Panel(
        table,
        title="[bold dim]Pipeline Status[/bold dim]",
        border_style="dim",
        padding=(0, 1),
    ))


def _print_section_header(title: str, subtitle: str = "", hint: str = "") -> None:
    """Print a polished section header panel."""
    if not _HAS_RICH:
        print(f"\n=== {title}" + (f" — {subtitle}" if subtitle else "") + " ===")
        return

    text = Text()
    text.append(f" {title}", style="bold cyan")
    if subtitle:
        text.append(f"  {subtitle}", style="bold white")
    if hint:
        text.append(f"\n  {hint}", style="dim")

    _rich_console.print(Panel(
        text,
        box=rich_box.HEAVY,
        style="cyan",
        padding=(0, 1),
    ))
    _rich_console.print(
        "  [dim]Arrow keys to navigate  ·  Enter to select  ·  ESC/Ctrl-C to go back[/dim]"
    )
    _rich_console.print("")


class MasterMenu:
    """Unified CLI menu for all Cola-Coder operations."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.storage = get_storage_config()
        # Windows: .venv/Scripts/python.exe — Linux/Mac: .venv/bin/python
        win_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        if win_python.exists():
            self.venv_python = win_python
        else:
            unix_python = self.project_root / ".venv" / "bin" / "python"
            if unix_python.exists():
                self.venv_python = unix_python
            else:
                import sys
                self.venv_python = Path(sys.executable)

    # ── Script runner ─────────────────────────────────────────────────────

    def _run_script(self, script: str, args: list[str] | None = None) -> None:
        """Run a Python script from the scripts/ directory."""
        cmd = [str(self.venv_python), f"scripts/{script}"]
        if args:
            cmd.extend(args)
        cli.dim(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root))
            if result.returncode != 0:
                cli.error(f"Script exited with code {result.returncode}")
        except KeyboardInterrupt:
            cli.warn("Interrupted.")
        except Exception as e:
            cli.error(str(e))

    def _run_shell(self, cmd: list[str]) -> None:
        """Run an arbitrary shell command in the project root."""
        cli.dim(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root))
            if result.returncode != 0:
                cli.error(f"Command exited with code {result.returncode}")
        except KeyboardInterrupt:
            cli.warn("Interrupted.")
        except Exception as e:
            cli.error(str(e))

    def _pause(self) -> None:
        """Wait for Enter before returning to the menu."""
        try:
            input("\nPress Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            pass

    # ── Path resolution ────────────────────────────────────────────────────

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a storage path: absolute paths used as-is, relative to project_root."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.project_root / p

    # ── Pipeline status ───────────────────────────────────────────────────

    def _detect_pipeline_status(self) -> dict[str, str]:
        """Detect current pipeline state: what's been completed."""
        status = {}

        tokenizer_path = self._resolve_path(self.storage.tokenizer_path)
        status["tokenizer"] = "ready" if tokenizer_path.exists() else "missing"

        data_dir = self._resolve_path(self.storage.data_dir) / "processed"
        npy_files = list(data_dir.glob("*.npy")) if data_dir.exists() else []
        status["data"] = f"{len(npy_files)} dataset(s)" if npy_files else "missing"

        ckpt_dir = self._resolve_path(self.storage.checkpoints_dir)
        if ckpt_dir.exists():
            ckpt_dirs = list(ckpt_dir.rglob("model.safetensors"))
            status["checkpoints"] = (
                f"{len(ckpt_dirs)} checkpoint(s)" if ckpt_dirs else "none"
            )
        else:
            status["checkpoints"] = "none"

        return status

    def _show_status(self) -> None:
        """Render the pipeline status panel."""
        try:
            status = self._detect_pipeline_status()
            _print_status_panel(status)
        except Exception:
            pass

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def _list_checkpoints(self) -> list[dict]:
        """Return a list of available checkpoints as dicts with label/path/detail."""
        ckpt_dir = self._resolve_path(self.storage.checkpoints_dir)
        checkpoints = []
        if not ckpt_dir.exists():
            return checkpoints

        for size_dir in sorted(ckpt_dir.iterdir()):
            if not size_dir.is_dir():
                continue
            latest = size_dir / "latest"
            if latest.exists():
                try:
                    detail = latest.read_text().strip()
                except Exception:
                    detail = str(latest)
                checkpoints.append({
                    "label": f"{size_dir.name}/latest",
                    "detail": detail,
                    "path": str(latest),
                })
            for step_dir in sorted(size_dir.glob("step_*")):
                checkpoints.append({
                    "label": f"{size_dir.name}/{step_dir.name}",
                    "detail": str(step_dir),
                    "path": str(step_dir),
                })

        return checkpoints

    def _pick_checkpoint(self, prompt: str = "Select checkpoint:") -> str | None:
        """Show checkpoint picker and return the path, or None if cancelled."""
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            cli.error("No checkpoints found. Train a model first.")
            return None

        choice = cli.choose(
            prompt,
            [{"label": c["label"], "detail": c["detail"]} for c in checkpoints],
            allow_cancel=True,
        )
        if choice is None:
            return None
        return checkpoints[choice]["path"]

    # ── Main menu ─────────────────────────────────────────────────────────

    def main_menu(self) -> None:
        """Show the top-level menu."""
        while True:
            _print_section_header("Cola-Coder", "Master Menu")
            self._show_status()

            options = [
                {"label": "Quick Start Pipeline",
                 "detail": "One-click: tokenizer -> data -> train (auto-detect what's needed)"},
                {"label": "Data Pipeline",
                 "detail": "Download, filter, score, prepare training data"},
                {"label": "Training",
                 "detail": "Train models (tiny -> large), resume, tokenizer, reasoning"},
                {"label": "Generate & Interact",
                 "detail": "Code generation, interactive chat, serve API"},
                {"label": "Evaluate & Benchmark",
                 "detail": "HumanEval, benchmarks, checkpoint comparisons"},
                {"label": "Tools & Utilities",
                 "detail": "Lint, test, GPU status, dataset inspection"},
                {"label": "Settings",
                 "detail": "Feature toggles, storage paths"},
                {"label": "Training Status",
                 "detail": "Check training progress — no GPU needed"},
            ]

            choice = cli.choose("What would you like to do?", options, allow_cancel=True)

            if choice is None:
                cli.dim("Goodbye!")
                break

            handlers = [
                self.quick_start_menu,
                self.data_pipeline_menu,
                self.training_menu,
                self.generate_menu,
                self.evaluate_menu,
                self.tools_menu,
                self.settings_menu,
                self.training_status_menu,
            ]

            handlers[choice]()

    # ── 1. Quick Start Pipeline ───────────────────────────────────────────

    def quick_start_menu(self) -> None:
        """One-click pipeline: detects what's needed and runs it."""
        _print_section_header(
            "Quick Start Pipeline",
            "Runs each stage that hasn't been completed yet",
        )
        self._show_status()

        status = self._detect_pipeline_status()
        stages: list[tuple[str, str, bool]] = [
            ("Tokenizer", "train_tokenizer.py", status["tokenizer"] == "ready"),
            ("Training Data", "prepare_data.py", "dataset" in status["data"]),
            ("Train Model (tiny)", "train.py", "checkpoint" in status["checkpoints"]),
        ]

        if _HAS_RICH:
            _rich_console.print("  [bold]Pipeline stages:[/bold]")
            for name, _, done in stages:
                icon = "[green]✓[/green]" if done else "[yellow]○[/yellow]"
                _rich_console.print(f"    {icon}  {name}")
            _rich_console.print("")

        options = [
            {"label": "Run Full Pipeline",
             "detail": "Runs only the missing stages automatically"},
            {"label": "Run Tokenizer Stage",
             "detail": "scripts/train_tokenizer.py"},
            {"label": "Run Data Stage",
             "detail": "scripts/prepare_data.py with tiny config"},
            {"label": "Run Training Stage",
             "detail": "scripts/train.py --config configs/tiny.yaml"},
        ]

        choice = cli.choose("Quick start action:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            # Auto-run only missing stages
            if not stages[0][2]:
                cli.info("Stage 1/3", "Training tokenizer...")
                self._run_script("train_tokenizer.py")
            else:
                cli.success("Tokenizer already trained — skipping.")

            # Re-check status after potential tokenizer run
            status = self._detect_pipeline_status()
            if status["tokenizer"] != "ready":
                cli.warn("Tokenizer still missing. Stopping pipeline.")
                self._pause()
                return

            if not stages[1][2]:
                cli.info("Stage 2/3", "Preparing training data...")
                self._run_script("prepare_data.py", [
                    "--config", "configs/tiny.yaml",
                    "--tokenizer", self.storage.tokenizer_path,
                ])
            else:
                cli.success("Training data already prepared — skipping.")

            # Re-check
            status = self._detect_pipeline_status()
            if "dataset" not in status["data"]:
                cli.warn("Training data still missing. Stopping pipeline.")
                self._pause()
                return

            if not stages[2][2]:
                cli.info("Stage 3/3", "Training model (tiny)...")
                self._run_script("train.py", ["--config", "configs/tiny.yaml"])
            else:
                cli.success("Checkpoint already exists — skipping training.")

            cli.success("Pipeline complete!")

        elif choice == 1:
            self._run_script("train_tokenizer.py")
        elif choice == 2:
            self._run_script("prepare_data.py", [
                "--config", "configs/tiny.yaml",
                "--tokenizer", self.storage.tokenizer_path,
            ])
        elif choice == 3:
            self._run_script("train.py", ["--config", "configs/tiny.yaml"])

        self._pause()

    # ── 2. Data Pipeline ─────────────────────────────────────────────────

    def data_pipeline_menu(self) -> None:
        """Data pipeline sub-menu."""
        while True:
            _print_section_header("Data Pipeline", "Download, filter, score, and prepare data")

            options = [
                {"label": "Prepare Training Data",
                 "detail": "Download from HuggingFace, filter, tokenize"},
                {"label": "Scrape GitHub Repos",
                 "detail": "scripts/scrape_github.py — crawl repos for training data"},
                {"label": "Score Code Quality",
                 "detail": "Run quality scorer on collected data"},
                {"label": "Combine Datasets",
                 "detail": "scripts/combine_datasets.py — merge multiple datasets"},
                {"label": "Inspect Dataset",
                 "detail": "Browse random training data samples"},
                {"label": "Generate Instructions",
                 "detail": "scripts/generate_instructions.py — create instruction pairs"},
                {"label": "Train Quality Classifier",
                 "detail": "scripts/train_quality_classifier.py"},
                {"label": "Score Repositories",
                 "detail": "scripts/score_repos.py — rank repos by quality"},
                {"label": "Interactive Data Prep",
                 "detail": "scripts/prepare_data_interactive.py — guided data setup"},
            ]

            choice = cli.choose("Select data operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._prepare_data_menu()
            elif choice == 1:
                self._run_script("scrape_github.py")
                self._pause()
            elif choice == 2:
                self._score_quality_menu()
            elif choice == 3:
                self._run_script("combine_datasets.py")
                self._pause()
            elif choice == 4:
                self._inspect_dataset()
                self._pause()
            elif choice == 5:
                self._run_script("generate_instructions.py")
                self._pause()
            elif choice == 6:
                self._run_script("train_quality_classifier.py")
                self._pause()
            elif choice == 7:
                self._run_script("score_repos.py")
                self._pause()
            elif choice == 8:
                self._run_script("prepare_data_interactive.py")

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

        base_args = ["--config", "configs/tiny.yaml", "--tokenizer", self.storage.tokenizer_path]

        if choice == 0:
            self._run_script("prepare_data_interactive.py")
        elif choice == 1:
            self._run_script("prepare_data.py", base_args + ["--max-tokens", "500000"])
        elif choice == 2:
            self._run_script("prepare_data.py", base_args)
        elif choice == 3:
            self._run_script("prepare_data.py", base_args + ["--filter-strict"])
        elif choice == 4:
            self._run_script("prepare_data.py", base_args + ["--no-filter"])
        elif choice == 5:
            self._run_script("prepare_data.py", base_args + ["--split", "test"])

        self._pause()

    def _score_quality_menu(self) -> None:
        """Score code quality sub-menu."""
        _print_section_header("Score Code Quality", "Evaluate and rank collected data")

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
            self._run_script("score_repos.py")
        elif choice == 1:
            self._run_script("train_quality_classifier.py")

        self._pause()

    # ── 3. Training ───────────────────────────────────────────────────────

    def training_menu(self) -> None:
        """Training sub-menu."""
        while True:
            _print_section_header("Training", "Train models, tokenizer, and reasoning")

            options = [
                {"label": "Train Model (select size)",
                 "detail": "tiny (50M) / small (125M) / medium (350M) / large (1B+)"},
                {"label": "Resume Training",
                 "detail": "Auto-detect latest checkpoint and continue"},
                {"label": "Train Tokenizer",
                 "detail": "scripts/train_tokenizer.py — BPE tokenizer from scratch"},
                {"label": "Train Reasoning (GRPO)",
                 "detail": "scripts/train_reasoning.py — GRPO with thinking tokens"},
                {"label": "VRAM Estimation",
                 "detail": "scripts/vram_estimate.py — estimate VRAM before training"},
            ]

            choice = cli.choose("Training operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._train_size_menu(resume=False)
            elif choice == 1:
                self._resume_training_menu()
            elif choice == 2:
                self._train_tokenizer()
            elif choice == 3:
                self._train_reasoning()
            elif choice == 4:
                self._vram_estimate_menu()

    def _train_size_menu(self, resume: bool = False) -> None:
        """Select model size and start training."""
        _print_section_header("Train Model", "Select a model size")

        options = [
            {"label": "Tiny   (50M params)",
             "detail": "~3.6 GB VRAM  |  ~4 hours  |  RTX 3080/4080"},
            {"label": "Small  (125M params)",
             "detail": "~6.5 GB VRAM  |  ~2 days   |  RTX 3080/4080"},
            {"label": "Medium (350M params)",
             "detail": "~8.2 GB VRAM  |  ~7 days   |  RTX 4080 (bf16)"},
            {"label": "Large  (1B+ params)",
             "detail": "~24 GB VRAM   |  cloud only"},
        ]

        choice = cli.choose("Select model size:", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium", "large"]
        size = sizes[choice]
        config = f"configs/{size}.yaml"

        if resume:
            # Look for existing checkpoint
            ckpt_dir = self._resolve_path(self.storage.checkpoints_dir) / size
            resume_path = None
            if ckpt_dir.exists():
                latest = ckpt_dir / "latest"
                if latest.exists():
                    resume_path = str(latest)
                else:
                    step_dirs = sorted(ckpt_dir.glob("step_*"))
                    if step_dirs:
                        resume_path = str(step_dirs[-1])

            if resume_path is None:
                cli.error(f"No checkpoint found for {size}. Start a fresh training run instead.")
                self._pause()
                return

            cli.info("Resuming from", resume_path)
            self._run_script("train.py", ["--config", config, "--resume", resume_path])
        else:
            # Check for existing and offer to resume
            ckpt_dir = self._resolve_path(self.storage.checkpoints_dir) / size
            if ckpt_dir.exists():
                has_ckpt = (ckpt_dir / "latest").exists() or bool(list(ckpt_dir.glob("step_*")))
                if has_ckpt:
                    if cli.confirm(f"Found existing {size} checkpoint. Resume training?"):
                        latest = ckpt_dir / "latest"
                        resume_arg = str(latest) if latest.exists() else str(
                            sorted(ckpt_dir.glob("step_*"))[-1]
                        )
                        self._run_script("train.py", ["--config", config, "--resume", resume_arg])
                        self._pause()
                        return

            use_wandb = cli.confirm("Enable Weights & Biases logging?", default=False)
            args = ["--config", config]
            if use_wandb:
                args.append("--wandb")

            self._run_script("train.py", args)

        self._pause()

    def _resume_training_menu(self) -> None:
        """Auto-detect latest checkpoint and resume."""
        _print_section_header("Resume Training", "Continue from latest checkpoint")

        sizes = ["tiny", "small", "medium", "large"]
        found = []
        for size in sizes:
            ckpt_dir = self._resolve_path(self.storage.checkpoints_dir) / size
            if ckpt_dir.exists():
                latest = ckpt_dir / "latest"
                if latest.exists():
                    try:
                        detail = latest.read_text().strip()
                    except Exception:
                        detail = str(latest)
                    found.append({"label": f"{size}/latest", "detail": detail, "size": size})
                else:
                    step_dirs = sorted(ckpt_dir.glob("step_*"))
                    if step_dirs:
                        found.append({
                            "label": f"{size}/{step_dirs[-1].name}",
                            "detail": str(step_dirs[-1]),
                            "size": size,
                        })

        if not found:
            cli.error("No checkpoints found. Start a fresh training run first.")
            self._pause()
            return

        choice = cli.choose("Select checkpoint to resume:", [
            {"label": c["label"], "detail": c["detail"]} for c in found
        ], allow_cancel=True)
        if choice is None:
            return

        selected = found[choice]
        config = f"configs/{selected['size']}.yaml"
        ckpt_path = (
            str(self._resolve_path(self.storage.checkpoints_dir) / selected["size"] / "latest")
            if "latest" in selected["label"]
            else selected["detail"]
        )

        self._run_script("train.py", ["--config", config, "--resume", ckpt_path])
        self._pause()

    def _train_tokenizer(self) -> None:
        """Train BPE tokenizer."""
        _print_section_header("Train Tokenizer", "BPE tokenizer from scratch")

        tokenizer_path = self._resolve_path(self.storage.tokenizer_path)
        if tokenizer_path.exists():
            if not cli.confirm(f"{tokenizer_path.name} already exists. Retrain?", default=False):
                return

        self._run_script("train_tokenizer.py")
        self._pause()

    def _train_reasoning(self) -> None:
        """GRPO reasoning training."""
        _print_section_header("Train Reasoning (GRPO)", "Fine-tuning with thinking tokens")

        if _HAS_RICH:
            _rich_console.print("  [bold]GRPO[/bold] (Group Relative Policy Optimization)")
            _rich_console.print("  Adds [cyan]<think>[/cyan] / [cyan]</think>[/cyan] chain-of-thought tokens.")
            _rich_console.print("  Generates multiple solutions, tests them, reinforces correct ones.")
            _rich_console.print("")

        if cli.confirm("Start reasoning training?"):
            self._run_script("train_reasoning.py")
            self._pause()

    def _vram_estimate_menu(self) -> None:
        """VRAM estimation."""
        _print_section_header("VRAM Estimation", "Estimate GPU memory before training")

        options = [
            {"label": "Estimate for Tiny  (50M)",   "detail": "configs/tiny.yaml"},
            {"label": "Estimate for Small (125M)",  "detail": "configs/small.yaml"},
            {"label": "Estimate for Medium (350M)", "detail": "configs/medium.yaml"},
            {"label": "Estimate for Large  (1B+)",  "detail": "configs/large.yaml"},
            {"label": "Estimate All Sizes",          "detail": "Compare all four configs"},
        ]

        choice = cli.choose("Estimate for which size?", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium", "large"]
        if choice < 4:
            self._run_script("vram_estimate.py", ["--config", f"configs/{sizes[choice]}.yaml"])
        else:
            for size in sizes:
                cli.rule(size)
                self._run_script("vram_estimate.py", ["--config", f"configs/{size}.yaml"])

        self._pause()

    # ── 4. Generate & Interact ────────────────────────────────────────────

    def generate_menu(self) -> None:
        """Code generation and serving sub-menu."""
        while True:
            _print_section_header("Generate & Interact", "Code generation, chat, API server")

            options = [
                {"label": "Quick Generate (auto-detect)",
                 "detail": "scripts/run.py — auto-detects latest checkpoint + config"},
                {"label": "Interactive Generation",
                 "detail": "scripts/generate.py — select checkpoint manually"},
                {"label": "Serve API",
                 "detail": "scripts/serve.py — FastAPI inference server"},
                {"label": "Nano Benchmark",
                 "detail": "scripts/nano_benchmark.py — quick generation speed test"},
            ]

            choice = cli.choose("Select generation mode:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._run_script("run.py")
                self._pause()
            elif choice == 1:
                self._interactive_generate()
            elif choice == 2:
                self._serve_api()
            elif choice == 3:
                self._run_script("nano_benchmark.py")
                self._pause()

    def _interactive_generate(self) -> None:
        """Interactive generation with checkpoint selection."""
        _print_section_header("Interactive Generation", "Select checkpoint and generate")

        ckpt_path = self._pick_checkpoint("Select checkpoint for generation:")
        if ckpt_path is None:
            return

        self._run_script("generate.py", ["--checkpoint", ckpt_path])
        self._pause()

    def _serve_api(self) -> None:
        """Start the FastAPI inference server."""
        _print_section_header("Serve API", "FastAPI inference server")

        if _HAS_RICH:
            _rich_console.print("  Starts a FastAPI server on [cyan]http://localhost:8000[/cyan]")
            _rich_console.print("  Press [bold]Ctrl-C[/bold] in the terminal to stop the server.")
            _rich_console.print("")

        ckpt_path = self._pick_checkpoint("Select checkpoint to serve:")
        if ckpt_path is None:
            return

        self._run_script("serve.py", ["--checkpoint", ckpt_path])
        self._pause()

    # ── 5. Evaluate & Benchmark ───────────────────────────────────────────

    def evaluate_menu(self) -> None:
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
            ]

            choice = cli.choose("Select evaluation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._humaneval_menu()
            elif choice == 1:
                self._benchmark_menu()
            elif choice == 2:
                self._run_script("compare_checkpoints.py")
                self._pause()
            elif choice == 3:
                self._run_script("nano_benchmark.py")
                self._pause()
            elif choice == 4:
                self._model_card_menu()
            elif choice == 5:
                self._run_script("training_status.py")
                self._pause()

    def _humaneval_menu(self) -> None:
        """HumanEval evaluation with checkpoint selection."""
        _print_section_header("HumanEval Evaluation", "164 Python coding problems — pass@k metric")

        ckpt_path = self._pick_checkpoint("Select checkpoint to evaluate:")
        if ckpt_path is None:
            return

        self._run_script("evaluate.py", ["--checkpoint", ckpt_path])
        self._pause()

    def _benchmark_menu(self) -> None:
        """Benchmark with checkpoint selection."""
        _print_section_header("Quick Benchmark", "Speed and quality benchmark")

        ckpt_path = self._pick_checkpoint("Select checkpoint to benchmark:")
        if ckpt_path is None:
            return

        self._run_script("benchmark.py", ["--checkpoint", ckpt_path])
        self._pause()

    def _model_card_menu(self) -> None:
        """Generate model card."""
        _print_section_header("Generate Model Card", "HuggingFace-style model card")

        ckpt_path = self._pick_checkpoint("Select checkpoint for model card:")
        if ckpt_path is None:
            return

        self._run_script("model_card.py", ["--checkpoint", ckpt_path])
        self._pause()

    # ── 6. Tools & Utilities ─────────────────────────────────────────────

    def tools_menu(self) -> None:
        """Developer tools sub-menu."""
        while True:
            _print_section_header("Tools & Utilities", "Tests, linting, GPU, data inspection")

            # Build feature toggles label
            try:
                features = _scan_feature_modules(self.project_root)
                n_enabled, n_total = _count_enabled(features)
                toggles_detail = f"{n_enabled}/{n_total} features enabled"
            except Exception:
                toggles_detail = "Enable/disable optional features"

            options = [
                {"label": "Run Tests",
                 "detail": "pytest tests/ -v"},
                {"label": "Run Linter",
                 "detail": "ruff check src/ scripts/ tests/"},
                {"label": "GPU Status",
                 "detail": "torch.cuda info + nvidia-smi output"},
                {"label": "Dataset Inspector",
                 "detail": "Browse random samples from training data"},
                {"label": "Test Type Reward",
                 "detail": "scripts/test_type_reward.py — test GRPO reward functions"},
                {"label": "Feature Toggles",
                 "detail": toggles_detail},
            ]

            choice = cli.choose("Select tool:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._run_shell([
                    str(self.venv_python), "-m", "pytest", "tests/", "-v",
                ])
                self._pause()
            elif choice == 1:
                self._run_shell([
                    str(self.venv_python), "-m", "ruff", "check",
                    "src/", "scripts/", "tests/",
                ])
                self._pause()
            elif choice == 2:
                self._gpu_status()
            elif choice == 3:
                self._inspect_dataset()
                self._pause()
            elif choice == 4:
                self._run_script("test_type_reward.py")
                self._pause()
            elif choice == 5:
                self._feature_toggles()

    def _gpu_status(self) -> None:
        """Show GPU info from torch and nvidia-smi."""
        _print_section_header("GPU Status", "CUDA and VRAM information")

        cli.gpu_info()

        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                cli.print(result.stdout)
            else:
                cli.warn("nvidia-smi returned non-zero exit code.")
        except FileNotFoundError:
            cli.warn("nvidia-smi not found — is the NVIDIA driver installed?")

        self._pause()

    # ── 7. Settings ───────────────────────────────────────────────────────

    def settings_menu(self) -> None:
        """Settings and configuration sub-menu."""
        while True:
            _print_section_header("Settings", "Feature toggles and storage configuration")

            try:
                features = _scan_feature_modules(self.project_root)
                n_enabled, n_total = _count_enabled(features)
                toggles_detail = f"{n_enabled}/{n_total} features currently enabled"
            except Exception:
                toggles_detail = "Enable/disable optional features"

            options = [
                {"label": "Feature Toggles",
                 "detail": toggles_detail},
                {"label": "Storage Paths",
                 "detail": "Show data, checkpoint, tokenizer paths"},
                {"label": "Project Info",
                 "detail": "Python, torch, CUDA, project root"},
            ]

            choice = cli.choose("Settings:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._feature_toggles()
            elif choice == 1:
                self._storage_paths()
            elif choice == 2:
                self._project_info()

    def _storage_paths(self) -> None:
        """Show configured storage paths from StorageConfig."""
        _print_section_header("Storage Paths", "Current data and checkpoint locations")

        tokenizer_path = self._resolve_path(self.storage.tokenizer_path)
        data_dir = self._resolve_path(self.storage.data_dir)
        checkpoints_dir = self._resolve_path(self.storage.checkpoints_dir)

        paths = {
            "Project root":     str(self.project_root),
            "Tokenizer":        str(tokenizer_path),
            "Data dir":         str(data_dir),
            "Data processed":   str(data_dir / "processed"),
            "Checkpoints":      str(checkpoints_dir),
            "Configs":          str(self.project_root / "configs"),
            "Scripts":          str(self.project_root / "scripts"),
            "Python (venv)":    str(self.venv_python),
        }

        cli.kv_table(paths, title="Paths")

        # Existence indicators
        if _HAS_RICH:
            _rich_console.print("")
            for label, path_str in paths.items():
                p = Path(path_str)
                if p.exists():
                    _rich_console.print(f"  [green]✓[/green] {label}")
                else:
                    _rich_console.print(f"  [red]✗[/red] [dim]{label} — not found[/dim]")

        self._pause()

    def _project_info(self) -> None:
        """Show project and environment information."""
        _print_section_header("Project Info", "Environment and version details")

        import sys
        info: dict[str, str] = {
            "Python":       sys.version.split()[0],
            "Project root": str(self.project_root),
            "Platform":     sys.platform,
        }

        try:
            import torch
            info["PyTorch"] = torch.__version__
            info["CUDA available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["CUDA version"] = torch.version.cuda or "unknown"
                info["GPU"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram = (
                    getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
                ) / 1e9
                info["VRAM"] = f"{vram:.1f} GB"
        except ImportError:
            info["PyTorch"] = "not installed"

        cli.kv_table(info, title="Environment")
        self._pause()

    # ── 8. Training Status ────────────────────────────────────────────────

    def training_status_menu(self) -> None:
        """Training status — reads logs, no GPU needed."""
        _print_section_header("Training Status", "Inspect training progress (no GPU needed)")

        self._run_script("training_status.py")
        self._pause()

    # ── Dataset inspector ─────────────────────────────────────────────────

    def _inspect_dataset(self) -> None:
        """Browse random samples from training data (inline inspection)."""
        import numpy as np

        data_dir = self._resolve_path(self.storage.data_dir) / "processed"
        npy_files = list(data_dir.glob("*.npy")) if data_dir.exists() else []

        if not npy_files:
            cli.error(f"No datasets found in {data_dir}")
            return

        # Pick a dataset
        if len(npy_files) == 1:
            npy_path = npy_files[0]
        else:
            options = [{"label": f.stem, "detail": str(round(f.stat().st_size / 1e6, 1)) + " MB"}
                       for f in npy_files]
            choice = cli.choose("Select dataset to inspect:", options, allow_cancel=True)
            if choice is None:
                return
            npy_path = npy_files[choice]

        _print_section_header("Dataset Inspector", npy_path.stem)

        data = np.load(str(npy_path), mmap_mode="r")
        cli.info("Shape",        f"{data.shape[0]:,} chunks x {data.shape[1]} tokens")
        cli.info("Total tokens", f"{data.shape[0] * data.shape[1]:,}")
        cli.info("File",         str(npy_path))
        cli.print("")

        try:
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
            tokenizer_path = self._resolve_path(self.storage.tokenizer_path)
            if not tokenizer_path.exists():
                cli.warn(f"{tokenizer_path} not found — can't decode samples.")
                return

            tokenizer = CodeTokenizer(str(tokenizer_path))
            n_samples = min(3, data.shape[0])
            indices = np.random.choice(data.shape[0], size=n_samples, replace=False)

            for idx in indices:
                cli.rule(f"Sample #{idx}")
                tokens = data[idx].tolist()
                text = tokenizer.decode(tokens)
                display = text[:600] + ("  [...]" if len(text) > 600 else "")
                cli.print(display)
                cli.print("")

        except Exception as e:
            cli.warn(f"Could not decode samples: {e}")

    # ── Feature toggles (preserved exactly from original) ─────────────────

    def _feature_toggles(self) -> None:
        """Interactive feature toggle menu — grouped by category.

        All features are OPTIONAL. Disabling one never breaks core functionality;
        it simply prevents that module from being used. Changes are persisted to
        configs/features.yaml immediately.
        """
        while True:
            cli.header("Cola-Coder", "Feature Toggles")
            cli.dim("All features are OPTIONAL. Disabling a feature will not break anything.")
            cli.dim("Persisted to: configs/features.yaml")
            cli.print("")

            try:
                features = _scan_feature_modules(self.project_root)
            except Exception as exc:
                cli.warn(f"Could not scan feature modules: {exc}")
                self._pause()
                return

            n_enabled, n_total = _count_enabled(features)
            cli.info("Status", f"{n_enabled}/{n_total} features currently enabled")

            # Build ordered category list (only categories that have modules)
            categories_in_use: list[str] = []
            for cat in list(_FEATURE_CATEGORIES.keys()) + ["Other"]:
                if any(f["category"] == cat for f in features):
                    categories_in_use.append(cat)

            cat_options = []
            for cat in categories_in_use:
                cat_features = [f for f in features if f["category"] == cat]
                cat_enabled = sum(1 for f in cat_features if f["enabled"])
                cat_total = len(cat_features)
                cat_options.append({
                    "label": cat,
                    "detail": f"{cat_enabled}/{cat_total} enabled",
                })

            cat_options.append({
                "label": "Enable ALL Features",
                "detail": f"Turn on all {n_total} optional features",
            })
            cat_options.append({
                "label": "Disable ALL Features",
                "detail": f"Turn off all {n_total} optional features",
            })

            choice = cli.choose("Select a category to manage:", cat_options, allow_cancel=True)
            if choice is None:
                return  # Back to caller

            if choice == len(cat_options) - 2:
                if cli.confirm(f"Enable all {n_total} features?", default=False):
                    for feat in features:
                        _save_feature_state(self.project_root, feat["name"], True)
                    cli.success(f"Enabled all {n_total} features.")
                    self._pause()
                continue

            if choice == len(cat_options) - 1:
                if cli.confirm(f"Disable all {n_total} features?", default=False):
                    for feat in features:
                        _save_feature_state(self.project_root, feat["name"], False)
                    cli.warn(f"Disabled all {n_total} features. Core functionality is unaffected.")
                    self._pause()
                continue

            selected_cat = categories_in_use[choice]
            self._feature_category_menu(selected_cat, features)

    def _feature_category_menu(self, category: str, features: list[dict]) -> None:
        """Show features in a single category and allow toggling.

        Loops until the user chooses Back / Cancel.
        """
        while True:
            cli.header("Cola-Coder", f"Feature Toggles — {category}")
            cli.dim("All features listed here are OPTIONAL.")
            cli.print("")

            yaml_states = _load_feature_states(self.project_root)
            cat_features = [f for f in features if f["category"] == category]

            for feat in cat_features:
                if feat["name"] in yaml_states:
                    feat["enabled"] = yaml_states[feat["name"]]

            feat_options = []
            for feat in cat_features:
                state_icon = "[green]on [/green]" if feat["enabled"] else "[red]off[/red]"
                feat_options.append({
                    "label": feat["label"],
                    "detail": f"{state_icon}  —  {feat['name']}",
                })

            cat_enabled = sum(1 for f in cat_features if f["enabled"])
            cat_total = len(cat_features)

            feat_options.append({
                "label": "Enable All in Category",
                "detail": f"Turn on all {cat_total} features in {category}",
            })
            feat_options.append({
                "label": "Disable All in Category",
                "detail": f"Turn off all {cat_total} features in {category}",
            })

            cli.info("Category", f"{category}  ({cat_enabled}/{cat_total} enabled)")
            choice = cli.choose("Select a feature to toggle:", feat_options, allow_cancel=True)

            if choice is None:
                return  # Back to category list

            if choice == len(feat_options) - 2:
                if cli.confirm(f"Enable all {cat_total} {category} features?", default=True):
                    for feat in cat_features:
                        feat["enabled"] = True
                        _save_feature_state(self.project_root, feat["name"], True)
                    cli.success(f"Enabled all {cat_total} features in {category}.")
                    self._pause()
                continue

            if choice == len(feat_options) - 1:
                if cli.confirm(f"Disable all {cat_total} {category} features?", default=False):
                    for feat in cat_features:
                        feat["enabled"] = False
                        _save_feature_state(self.project_root, feat["name"], False)
                    cli.warn(
                        f"Disabled all {cat_total} features in {category}. "
                        "Core functionality is unaffected."
                    )
                    self._pause()
                continue

            feat = cat_features[choice]
            new_state = not feat["enabled"]
            feat["enabled"] = new_state
            _save_feature_state(self.project_root, feat["name"], new_state)

            if new_state:
                cli.success(f"{feat['label']} enabled.")
            else:
                cli.warn(
                    f"{feat['label']} disabled. "
                    "(This is optional — core functionality is unaffected.)"
                )
            # Loop back immediately so user sees updated state


# ── Entry point ───────────────────────────────────────────────────────────────

def run_master_menu() -> None:
    """Entry point for the master menu."""
    if not is_enabled():
        cli.error("Master menu feature is disabled.")
        return

    # Find project root (look for pyproject.toml or configs/)
    cwd = Path.cwd()
    if (cwd / "configs").exists():
        root = cwd
    elif (cwd / "cola-coder" / "configs").exists():
        root = cwd / "cola-coder"
    else:
        root = cwd

    menu = MasterMenu(project_root=root)
    menu.main_menu()
