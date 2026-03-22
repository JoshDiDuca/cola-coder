"""Unified master menu for Cola-Coder.

Single entry point for all Cola-Coder operations. Replaces 12 separate
PowerShell scripts with one interactive, keyboard-driven menu.

Navigation: Arrow keys to move, Enter to select, ESC/Ctrl-C to go back.
"""

import importlib
import json
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

    # Known model sizes and their parameter counts for display.
    _MODEL_PARAMS: dict[str, str] = {
        "tiny": "50M", "small": "125M", "medium": "299M",
        "4080_max": "455M", "large": "1B+",
    }
    # Canonical ordering for the model picker.
    _MODEL_ORDER: list[str] = ["tiny", "small", "medium", "4080_max", "large"]

    @staticmethod
    def _read_checkpoint_meta(step_dir: Path) -> dict:
        """Read metadata.json from a checkpoint dir and return a display dict."""
        meta_path = step_dir / "metadata.json"
        info: dict = {
            "path": str(step_dir),
            "step": 0,
            "loss": None,
            "label": step_dir.name,
        }
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                info["step"] = data.get("step", 0)
                info["loss"] = data.get("loss")
            except Exception:
                pass
        return info

    def _scan_all_checkpoints(self) -> dict[str, list[dict]]:
        """Scan all checkpoint locations and return ``{model: [info_dicts]}``."""
        seen: set[Path] = set()
        by_model: dict[str, list[dict]] = {}

        # Scan both the storage.yaml path AND the default ./checkpoints/
        dirs_to_scan: list[Path] = []
        storage_dir = self._resolve_path(self.storage.checkpoints_dir)
        default_dir = self._resolve_path("checkpoints")
        dirs_to_scan.append(storage_dir)
        if default_dir.resolve() != storage_dir.resolve():
            dirs_to_scan.append(default_dir)

        for ckpt_dir in dirs_to_scan:
            if not ckpt_dir.exists():
                continue
            for size_dir in sorted(ckpt_dir.iterdir()):
                if not size_dir.is_dir():
                    continue
                model_name = size_dir.name
                for step_dir in sorted(size_dir.glob("step_*")):
                    resolved = step_dir.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    info = self._read_checkpoint_meta(step_dir)
                    by_model.setdefault(model_name, []).append(info)

        # Sort each model's checkpoints by step descending (newest first).
        for model in by_model:
            by_model[model].sort(key=lambda x: x["step"], reverse=True)

        return by_model

    def _resolve_latest_path(self, model: str) -> Path | None:
        """Return the resolved path that a ``latest`` pointer points to."""
        for base in (
            self._resolve_path(self.storage.checkpoints_dir),
            self._resolve_path("checkpoints"),
        ):
            latest = base / model / "latest"
            if latest.is_file():
                try:
                    return Path(latest.read_text().strip()).resolve()
                except Exception:
                    pass
        return None

    def _pick_model(
        self, prompt: str = "Select model:",
    ) -> str | None:
        """Show model picker with checkpoint counts and latest metrics."""
        by_model = self._scan_all_checkpoints()
        if not by_model:
            cli.error("No checkpoints found. Train a model first.")
            return None

        options: list[dict[str, str]] = []
        model_names: list[str] = []
        for name in self._MODEL_ORDER:
            if name not in by_model:
                continue
            ckpts = by_model[name]
            latest = ckpts[0]  # sorted descending by step
            params = self._MODEL_PARAMS.get(name, "?")
            loss_str = f", loss {latest['loss']:.4f}" if latest.get("loss") else ""
            detail = (
                f"{params} — {len(ckpts)} checkpoint(s)"
                f", latest: step {latest['step']:,}{loss_str}"
            )
            options.append({"label": name, "detail": detail})
            model_names.append(name)

        # Include any unknown model names (in case of custom dirs).
        for name in sorted(by_model):
            if name in model_names:
                continue
            ckpts = by_model[name]
            latest = ckpts[0]
            loss_str = f", loss {latest['loss']:.4f}" if latest.get("loss") else ""
            detail = (
                f"? — {len(ckpts)} checkpoint(s)"
                f", latest: step {latest['step']:,}{loss_str}"
            )
            options.append({"label": name, "detail": detail})
            model_names.append(name)

        choice = cli.choose(prompt, options, allow_cancel=True)
        if choice is None:
            return None
        return model_names[choice]

    def _pick_checkpoint(
        self,
        prompt: str = "Select checkpoint:",
        model: str | None = None,
    ) -> str | None:
        """Model-first checkpoint picker.

        If *model* is ``None``, prompts the user to select a model first.
        Then shows the checkpoints for that model with metadata.
        """
        if model is None:
            model = self._pick_model()
            if model is None:
                return None

        by_model = self._scan_all_checkpoints()
        ckpts = by_model.get(model, [])
        if not ckpts:
            cli.error(f"No checkpoints found for {model}.")
            return None

        latest_path = self._resolve_latest_path(model)

        options: list[dict[str, str]] = []
        for c in ckpts:
            is_latest = (
                latest_path is not None
                and Path(c["path"]).resolve() == latest_path
            )
            tag = "  (latest)" if is_latest else ""
            loss_str = f"loss {c['loss']:.4f}" if c.get("loss") else ""
            options.append({
                "label": f"{c['label']}{tag}",
                "detail": loss_str,
            })

        params = self._MODEL_PARAMS.get(model, "")
        header = f"{prompt}  ({model} — {params})" if params else prompt
        choice = cli.choose(header, options, allow_cancel=True)
        if choice is None:
            return None
        return ckpts[choice]["path"]

    def _config_for_checkpoint(self, ckpt_path: str) -> str:
        """Infer the config file from a checkpoint path (e.g. .../tiny/latest → configs/tiny.yaml)."""
        parts = Path(ckpt_path).parts
        for size in self._MODEL_ORDER:
            if size in parts:
                return f"configs/{size}.yaml"
        return "configs/tiny.yaml"

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
                {"label": "Router & Specialists",
                 "detail": "Domain router, MoE, specialist training & management"},
                {"label": "Tools & Utilities",
                 "detail": "Lint, test, GPU status, dataset inspection, export"},
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
                self.router_menu,
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
                {"label": "GitHub API Data Collection",
                 "detail": "Collect code via official GitHub REST API"},
                {"label": "Browse Software Heritage",
                 "detail": "Search the universal source code archive (SWH API)"},
                {"label": "Score Code Quality",
                 "detail": "Run quality scorer on collected data"},
                {"label": "Combine Datasets",
                 "detail": "scripts/combine_datasets.py — merge multiple datasets"},
                {"label": "Combine Datasets (weighted)",
                 "detail": "Interactive weighted dataset mixing with per-dataset ratios"},
                {"label": "Inspect Dataset",
                 "detail": "Browse random training data samples"},
                {"label": "Generate Instructions",
                 "detail": "scripts/generate_instructions.py — create instruction pairs"},
                {"label": "Train Quality Classifier",
                 "detail": "scripts/train_quality_classifier.py"},
                {"label": "Score Repositories",
                 "detail": "scripts/score_repos.py — rank repos by quality"},
                {"label": "Advanced Filters",
                 "detail": "PII, dedup, license, syntax — view available filter plugins"},
                {"label": "Interactive Data Prep",
                 "detail": "scripts/prepare_data_interactive.py — guided data setup"},
                {"label": "Prepare FIM Data",
                 "detail": "scripts/prepare_fim_data.py — fill-in-the-middle training data"},
                {"label": "Scrape Framework Docs",
                 "detail": "scripts/scrape_docs.py — scrape React/Next.js/Zod/TypeORM docs"},
                {"label": "Prepare Docs Training Data",
                 "detail": "scripts/prepare_docs_data.py — tokenize scraped docs"},
                {"label": "Prepare Context Training Data",
                 "detail": "scripts/prepare_repo_context_data.py — repo context pairs"},
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
                self._software_heritage_info()
            elif choice == 3:
                self._score_quality_menu()
            elif choice == 4:
                self._run_script("combine_datasets.py")
                self._pause()
            elif choice == 5:
                self._combine_datasets_menu()
            elif choice == 6:
                self._inspect_dataset()
                self._pause()
            elif choice == 7:
                self._run_script("generate_instructions.py")
                self._pause()
            elif choice == 8:
                self._run_script("train_quality_classifier.py")
                self._pause()
            elif choice == 9:
                self._run_script("score_repos.py")
                self._pause()
            elif choice == 10:
                self._advanced_filters_info()
            elif choice == 11:
                self._run_script("prepare_data_interactive.py")
            elif choice == 12:
                self._run_script("prepare_fim_data.py")
                self._pause()
            elif choice == 13:
                self._scrape_docs_menu()
            elif choice == 14:
                self._run_script("prepare_docs_data.py")
                self._pause()
            elif choice == 15:
                self._run_script("prepare_repo_context_data.py")
                self._pause()

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

    def _software_heritage_info(self) -> None:
        """Show info about Software Heritage data source."""
        _print_section_header(
            "Software Heritage Archive",
            "Universal source code archive — archive.softwareheritage.org",
        )

        if _HAS_RICH:
            _rich_console.print(
                "  Software Heritage is the universal archive of software source code.\n"
                "  It provides deduplicated, archival-quality code with rich metadata.\n"
            )
            _rich_console.print("  [bold cyan]Access methods:[/bold cyan]")
            _rich_console.print("    [cyan]1.[/cyan] SWH REST API — 1,200 req/hr (12,000 with token)")
            _rich_console.print("    [cyan]2.[/cyan] The Stack v2 on HuggingFace — SWH-derived, bulk access")
            _rich_console.print("")
            _rich_console.print("  [bold cyan]Setup:[/bold cyan]")
            _rich_console.print("    Set [green]SWH_API_TOKEN[/green] env var for higher rate limits.")
            _rich_console.print(
                "    Get a token at: [link]https://archive.softwareheritage.org[/link]"
            )
            _rich_console.print("")
            _rich_console.print("  [bold cyan]Code location:[/bold cyan]")
            _rich_console.print("    [dim]src/cola_coder/data/sources/software_heritage.py[/dim]")
            _rich_console.print(
                "    Implements SWHClient, SoftwareHeritageSource (DataSource plugin)"
            )
            _rich_console.print("")
            _rich_console.print(
                "  [dim]Use via the extensible pipeline "
                "(scripts/prepare_data_interactive.py)[/dim]"
            )
        else:
            print("\n  Software Heritage — universal source code archive")
            print("  SWH API: 1,200 req/hr (12,000 with token)")
            print("  Set SWH_API_TOKEN env var for higher rate limits.")
            print("  Code: src/cola_coder/data/sources/software_heritage.py")

        self._pause()

    def _advanced_filters_info(self) -> None:
        """Show available data filter plugins."""
        _print_section_header(
            "Advanced Filters",
            "Composable data quality filter plugins",
        )

        filters_info = [
            ("Content Filter", "content.py", "Pattern matching — spam, boilerplate, auto-generated"),
            ("Deduplication", "dedup.py", "MinHash LSH — near-duplicate detection"),
            ("License Filter", "license_filter.py", "SPDX license checking and compliance"),
            ("PII Filter", "pii.py", "Detect emails, API keys, secrets, phone numbers"),
            ("Syntax Filter", "syntax.py", "Tree-sitter AST parsing (Python, TS, JS, Go, Rust, Java)"),
            ("Length Filter", "length.py", "Min/max line count validation"),
            ("Quality Filter", "quality.py", "Existing quality filter as composable plugin"),
            ("Quality Classifier", "quality_classifier.py", "ML-based quality scoring (DistilBERT)"),
        ]

        if _HAS_RICH:
            table = Table(
                box=rich_box.ROUNDED, show_header=True, header_style="bold cyan",
                padding=(0, 1), title="[bold]Available Filter Plugins[/bold]",
                title_style="bold white",
            )
            table.add_column("Filter", style="bold white", width=20)
            table.add_column("File", style="dim", width=25)
            table.add_column("Description", style="white")

            for name, filename, desc in filters_info:
                table.add_row(name, filename, desc)

            _rich_console.print(table)
            _rich_console.print("")
            _rich_console.print("  [bold cyan]Usage:[/bold cyan]")
            _rich_console.print(
                "    Filters are composable plugins in "
                "[dim]src/cola_coder/data/filters/[/dim]"
            )
            _rich_console.print(
                "    Use via the extensible pipeline "
                "(scripts/prepare_data_interactive.py)"
            )
            _rich_console.print(
                "    Or import directly: "
                "[dim]from cola_coder.data.filters import PIIFilter[/dim]"
            )
        else:
            print("\n  Available Filter Plugins:")
            print("  " + "-" * 60)
            for name, filename, desc in filters_info:
                print(f"    {name:20s} {filename:25s} {desc}")
            print("\n  Use via scripts/prepare_data_interactive.py")

        self._pause()

    def _scrape_docs_menu(self) -> None:
        """Scrape framework documentation with interactive framework/version selection."""
        _print_section_header("Scrape Framework Docs", "Download docs for training data")

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
            version = input("Enter version (or 'latest'): ").strip() or "latest"
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        args = ["--framework", framework, "--version", version]
        self._run_script("scrape_docs.py", args)
        self._pause()

    def _combine_datasets_menu(self) -> None:
        """Interactive weighted dataset mixing."""
        import numpy as np

        _print_section_header("Combine Datasets (weighted)", "Mix datasets with custom sampling ratios")

        data_dir = self._resolve_path(self.storage.data_dir) / "processed"
        npy_files = sorted(data_dir.glob("*.npy")) if data_dir.exists() else []

        if not npy_files:
            cli.error(f"No .npy datasets found in {data_dir}")
            cli.dim("Run a data preparation step first.")
            self._pause()
            return

        # Multi-select datasets
        file_options = []
        for f in npy_files:
            try:
                arr = np.load(str(f), mmap_mode="r")
                size_str = f"{f.stat().st_size / 1e6:.1f} MB"
                shape_str = f"{arr.shape[0]:,} x {arr.shape[1]}" if arr.ndim == 2 else str(arr.shape)
                file_options.append({"label": f.stem, "detail": f"{shape_str}  •  {size_str}"})
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
                    weight_str = input(f"  Weight for '{opt['label']}' (e.g. 0.8): ").strip()
                    weight = float(weight_str) if weight_str else 1.0
                except (ValueError, EOFError, KeyboardInterrupt):
                    weight = 1.0
                selected_paths.append(str(npy_files[i]))
                selected_weights.append(weight)

        if not selected_paths:
            cli.warn("No datasets selected.")
            self._pause()
            return

        # Normalise weights so they display nicely, but pass raw (script handles it)
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

        self._run_script("combine_datasets.py", args)
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
                {"label": "Background Training",
                 "detail": "Automated overnight/idle training with GPU throttling"},
                {"label": "Train Tokenizer",
                 "detail": "scripts/train_tokenizer.py — BPE tokenizer from scratch"},
                {"label": "Train Reasoning (GRPO)",
                 "detail": "scripts/train_reasoning.py — GRPO with thinking tokens"},
                {"label": "VRAM Estimation",
                 "detail": "scripts/vram_estimate.py — estimate VRAM before training"},
                {"label": "Learning Rate Finder",
                 "detail": "scripts/find_lr.py — find optimal LR via range test"},
                {"label": "Training Dashboard (TUI)",
                 "detail": "scripts/training_dashboard.py — real-time Rich dashboard"},
                {"label": "Auto-Eval History",
                 "detail": "scripts/training_eval_history.py — view eval snapshots"},
            ]

            choice = cli.choose("Training operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._train_size_menu(resume=False)
            elif choice == 1:
                self._resume_training_menu()
            elif choice == 2:
                self._background_training_menu()
            elif choice == 3:
                self._train_tokenizer()
            elif choice == 4:
                self._train_reasoning()
            elif choice == 5:
                self._vram_estimate_menu()
            elif choice == 6:
                self._lr_finder_menu()
            elif choice == 7:
                self._run_script("training_dashboard.py")
                self._pause()
            elif choice == 8:
                self._run_script("training_eval_history.py")
                self._pause()

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

    # ── Background Training ────────────────────────────────────────────

    def _background_training_menu(self) -> None:
        """Background training sub-menu with GPU throttling."""
        while True:
            _print_section_header(
                "Background Training",
                "Automated training with GPU clock/power throttling",
            )

            # Check if background training is running for any model
            is_running, status = self._check_any_background_running()

            if is_running and status:
                cli.info("Status", "[green]RUNNING[/green]")
                cli.info("Config", status.get("config_path", "?"))
                cli.info("Step", f"{status.get('step', '?'):,}")
                cli.info("Loss", f"{status.get('loss', 0):.4f}")
                cli.info(
                    "GPU",
                    f"{status.get('gpu_clock_mhz', '?')} MHz / "
                    f"{status.get('gpu_power_watts', '?')}W",
                )
                cli.info("GPU Util (other)", f"{status.get('gpu_util', 0):.0f}%")
                cli.info("Sleep", f"{status.get('sleep_ms', 0)} ms/step")
                elapsed = status.get("elapsed_seconds", 0)
                h, m = int(elapsed) // 3600, (int(elapsed) % 3600) // 60
                cli.info("Elapsed", f"{h}h {m}m")

            options = []
            if not is_running:
                options.append({
                    "label": "Start Background Training",
                    "detail": "Select model, configure GPU throttle, launch in background",
                })
            else:
                options.append({
                    "label": "Stop Background Training",
                    "detail": "Graceful shutdown — saves checkpoint before stopping",
                })
            options.append({
                "label": "Background Training Status",
                "detail": "View current or last training session details",
            })
            options.append({
                "label": "Schedule Overnight Training",
                "detail": "Set up Windows Task Scheduler for automatic training",
            })
            options.append({
                "label": "Remove Overnight Schedule",
                "detail": "Unregister the Windows scheduled task",
            })

            choice = cli.choose("Background training:", options, allow_cancel=True)
            if choice is None:
                return

            if not is_running:
                # Start / Status / Schedule / Remove
                if choice == 0:
                    self._start_background_training()
                elif choice == 1:
                    self._show_background_status()
                elif choice == 2:
                    self._schedule_overnight_training()
                elif choice == 3:
                    self._remove_overnight_schedule()
            else:
                # Stop / Status / Schedule / Remove
                if choice == 0:
                    self._stop_background_training()
                elif choice == 1:
                    self._show_background_status()
                elif choice == 2:
                    self._schedule_overnight_training()
                elif choice == 3:
                    self._remove_overnight_schedule()

    def _start_background_training(self) -> None:
        """Configure and launch background training."""
        _print_section_header("Start Background Training", "Configure and launch")

        # 1. Pick model size
        options = [
            {"label": "Tiny   (50M params)",
             "detail": "~3.6 GB VRAM  |  configs/tiny.yaml"},
            {"label": "Small  (125M params)",
             "detail": "~6.5 GB VRAM  |  configs/small.yaml"},
            {"label": "Medium (299M params)",
             "detail": "~14 GB VRAM   |  configs/medium.yaml"},
            {"label": "4080 Max (455M params)",
             "detail": "~14.1 GB VRAM |  configs/4080_max.yaml"},
            {"label": "Large  (1B+ params)",
             "detail": "~24 GB VRAM   |  configs/large.yaml"},
        ]
        choice = cli.choose("Select model to train:", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium", "4080_max", "large"]
        size = sizes[choice]
        config_path = f"configs/{size}.yaml"

        # 2. Duration mode
        dur_options = [
            {"label": "8 hours (overnight)",
             "detail": "Good for sleeping — about 8 hours of training"},
            {"label": "Until morning (stop at 7:00 AM)",
             "detail": "Trains until 7:00 AM then saves and stops"},
            {"label": "Run indefinitely",
             "detail": "Runs until max_steps or manually stopped"},
            {"label": "Custom duration",
             "detail": "Specify hours (e.g. 4h, 2h30m)"},
        ]
        dur_choice = cli.choose("Training duration:", dur_options, allow_cancel=True)
        if dur_choice is None:
            return

        duration_arg = None
        stop_at_arg = None
        if dur_choice == 0:
            duration_arg = "8h"
        elif dur_choice == 1:
            stop_at_arg = "07:00"
        elif dur_choice == 2:
            pass  # No limit
        elif dur_choice == 3:
            duration_arg = input("  Enter duration (e.g. 4h, 2h30m): ").strip()
            if not duration_arg:
                return

        # 3. GPU throttle level
        gpu_options = [
            {"label": "Light throttle (1800 MHz / 250W)",
             "detail": "~75% speed, desktop should be smooth"},
            {"label": "Medium throttle (1500 MHz / 200W) (Recommended)",
             "detail": "~55% speed, YouTube/browsing very smooth"},
            {"label": "Heavy throttle (1200 MHz / 175W)",
             "detail": "~40% speed, can game while training"},
            {"label": "No throttle (full speed)",
             "detail": "100% speed, desktop may lag during steps"},
        ]
        gpu_choice = cli.choose("GPU throttle level:", gpu_options, allow_cancel=True)
        if gpu_choice is None:
            return

        clock_power = [
            (1800, 250), (1500, 200), (1200, 175), (0, 0),
        ]
        gpu_clock, gpu_power = clock_power[gpu_choice]

        # 4. Show summary and confirm
        cli.rule("Summary")
        cli.info("Model", f"{size} ({config_path})")
        if duration_arg:
            cli.info("Duration", duration_arg)
        elif stop_at_arg:
            cli.info("Stop at", stop_at_arg)
        else:
            cli.info("Duration", "Until max_steps or manual stop")
        if gpu_clock > 0:
            cli.info("GPU Throttle", f"{gpu_clock} MHz / {gpu_power}W")
        else:
            cli.info("GPU Throttle", "None (full speed)")
        cli.info("Save every", "1000 steps")
        cli.print("")

        if not cli.confirm("Launch background training?"):
            return

        # 5. Build command and launch as detached process
        cmd = [
            str(self.venv_python), "scripts/background_train.py",
            "--config", config_path,
            "--save-every", "1000",
        ]
        if duration_arg:
            cmd.extend(["--duration", duration_arg])
        if stop_at_arg:
            cmd.extend(["--stop-at", stop_at_arg])
        if gpu_clock > 0:
            cmd.extend(["--gpu-clock", str(gpu_clock)])
            cmd.extend(["--gpu-power", str(gpu_power)])
        else:
            cmd.append("--no-throttle")

        try:
            # Windows: CREATE_NO_WINDOW + DETACHED_PROCESS
            # This lets the background process survive after the menu exits
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008

            log_path = self.project_root / "logs" / "background_train.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=open(str(log_path), "a"),
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
            )

            cli.success(f"Background training launched! (PID: {proc.pid})")
            cli.info("Log file", str(log_path))
            cli.print("\n  You can close this menu — training continues in background.")
            cli.print("  Use 'Background Training → Stop' to stop gracefully.")
            cli.print("  Use 'Background Training → Status' to check progress.\n")
        except Exception as e:
            cli.error(f"Failed to launch background training: {e}")

        self._pause()

    def _stop_background_training(self) -> None:
        """Send stop signal to running background training."""
        from cola_coder.features.background_trainer import send_stop_signal

        # Find the checkpoint dir that has an active lock
        for size in self._MODEL_ORDER:
            ckpt_dir = self._resolve_path(self.storage.checkpoints_dir) / size
            lock_file = ckpt_dir / ".background_train.lock"
            if lock_file.exists():
                success = send_stop_signal(str(ckpt_dir))
                if success:
                    cli.success(
                        f"Stop signal sent to {size} training. "
                        "It will save a checkpoint and exit within one step."
                    )
                else:
                    cli.error("Failed to send stop signal.")
                self._pause()
                return

        # Also check default ./checkpoints/
        default_dir = self._resolve_path("checkpoints")
        for size_dir in default_dir.iterdir() if default_dir.exists() else []:
            lock_file = size_dir / ".background_train.lock"
            if lock_file.exists():
                success = send_stop_signal(str(size_dir))
                if success:
                    cli.success("Stop signal sent. Training will save and exit.")
                else:
                    cli.error("Failed to send stop signal.")
                self._pause()
                return

        cli.warn("No running background training found.")
        self._pause()

    def _show_background_status(self) -> None:
        """Show background training status."""
        from cola_coder.features.background_trainer import (
            is_background_running, read_background_status,
        )

        _print_section_header("Background Training Status", "Current session details")

        found_any = False
        for size in self._MODEL_ORDER:
            ckpt_dir = self._resolve_path(self.storage.checkpoints_dir) / size
            running, status = is_background_running(str(ckpt_dir))
            if running and status:
                found_any = True
                cli.info("Model", size)
                cli.info("Status", "[green]RUNNING[/green]")
                cli.info("Step", f"{status.get('step', '?'):,}")
                cli.info("Loss", f"{status.get('loss', 0):.4f}")
                cli.info("Tokens/sec", f"{status.get('tokens_per_sec', 0):,.0f}")
                cli.info(
                    "GPU",
                    f"{status.get('gpu_clock_mhz', '?')} MHz / "
                    f"{status.get('gpu_power_watts', '?')}W",
                )
                cli.info("GPU Util (other)", f"{status.get('gpu_util', 0):.0f}%")
                cli.info("Sleep", f"{status.get('sleep_ms', 0)} ms/step")
                elapsed = status.get("elapsed_seconds", 0)
                h, m = int(elapsed) // 3600, (int(elapsed) % 3600) // 60
                cli.info("Elapsed", f"{h}h {m}m")
                cli.info("Updated", status.get("updated", "?"))
            elif not running:
                # Check for stale status file (last session info)
                last_status = read_background_status(str(ckpt_dir))
                if last_status:
                    found_any = True
                    cli.info("Model", size)
                    cli.info("Status", "[dim]STOPPED[/dim]")
                    cli.info("Last step", f"{last_status.get('step', '?'):,}")
                    cli.info("Last loss", f"{last_status.get('loss', 0):.4f}")

        if not found_any:
            cli.dim("No background training sessions found.")

        # Also show log file info
        log_path = self.project_root / "logs" / "background_train.log"
        if log_path.exists():
            size_mb = log_path.stat().st_size / (1024 * 1024)
            cli.info("Log file", f"{log_path} ({size_mb:.1f} MB)")

        self._pause()

    def _schedule_overnight_training(self) -> None:
        """Register a Windows Task Scheduler task for overnight training."""
        from cola_coder.features.background_scheduler import WindowsTaskScheduler

        _print_section_header(
            "Schedule Overnight Training",
            "Register a daily task in Windows Task Scheduler",
        )

        scheduler = WindowsTaskScheduler(self.project_root)

        if scheduler.is_task_registered():
            cli.warn(
                f"Task '{scheduler.task_name}' already exists. "
                "It will be overwritten."
            )

        # Pick model
        options = [
            {"label": size, "detail": f"configs/{size}.yaml"}
            for size in self._MODEL_ORDER
        ]
        choice = cli.choose("Select model for overnight training:", options,
                            allow_cancel=True)
        if choice is None:
            return

        size = self._MODEL_ORDER[choice]
        config_path = f"configs/{size}.yaml"

        # Show defaults and confirm
        cli.rule("Overnight Schedule")
        cli.info("Model", size)
        cli.info("Start time", "10:00 PM (22:00)")
        cli.info("Stop time", "7:00 AM (07:00)")
        cli.info("GPU", "1500 MHz / 200W")
        cli.info("Save every", "1000 steps")
        cli.print("")

        if not cli.confirm("Register this schedule?"):
            return

        success, msg = scheduler.create_overnight_task(
            config_path=config_path,
            start_time="22:00",
            stop_time="07:00",
            gpu_clock=1500,
            gpu_power=200,
        )

        if success:
            cli.success(msg)
        else:
            cli.error(msg)

        self._pause()

    def _remove_overnight_schedule(self) -> None:
        """Remove the Windows Task Scheduler overnight training task."""
        from cola_coder.features.background_scheduler import WindowsTaskScheduler

        scheduler = WindowsTaskScheduler(self.project_root)

        if not scheduler.is_task_registered():
            cli.dim("No overnight training task is registered.")
            self._pause()
            return

        if cli.confirm("Remove the overnight training schedule?"):
            success, msg = scheduler.remove_task()
            if success:
                cli.success(msg)
            else:
                cli.error(msg)

        self._pause()

    def _check_any_background_running(self) -> tuple[bool, dict | None]:
        """Check if background training is running for any model."""
        from cola_coder.features.background_trainer import is_background_running

        for size in self._MODEL_ORDER:
            ckpt_dir = self._resolve_path(self.storage.checkpoints_dir) / size
            running, status = is_background_running(str(ckpt_dir))
            if running:
                return True, status

        # Also check default ./checkpoints/
        default_dir = self._resolve_path("checkpoints")
        if default_dir.exists():
            for size_dir in default_dir.iterdir():
                if size_dir.is_dir():
                    running, status = is_background_running(str(size_dir))
                    if running:
                        return True, status

        return False, None

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
        """GRPO reasoning training with optional enhancements."""
        _print_section_header("Train Reasoning (GRPO)", "Fine-tuning with thinking tokens")

        if _HAS_RICH:
            _rich_console.print("  [bold]GRPO[/bold] (Group Relative Policy Optimization)")
            _rich_console.print("  Adds [cyan]<think>[/cyan] / [cyan]</think>[/cyan] chain-of-thought tokens.")
            _rich_console.print("  Generates multiple solutions, tests them, reinforces correct ones.")
            _rich_console.print("")

        args: list[str] = []

        # SFT Warmup
        if cli.confirm("Enable SFT warmup phase? (DeepSeek-R1 approach)", default=True):
            args.append("--sft-warmup")
            if _HAS_RICH:
                _rich_console.print("  [green]✓[/green] SFT warmup enabled")

        # Reward function
        reward_options = [
            {"label": "Python Execution (default)",
             "detail": "Run code and check output correctness"},
            {"label": "TypeScript Type Checking",
             "detail": "tsc --noEmit --strict validation"},
            {"label": "Combined (multi-signal)",
             "detail": "Type-check + syntax + style + completeness"},
        ]
        reward_choice = cli.choose("Reward function:", reward_options, allow_cancel=True)
        if reward_choice is None:
            return
        reward_names = ["python_exec", "typescript", "combined"]
        args.extend(["--reward", reward_names[reward_choice]])

        # Problem set
        if cli.confirm("Use expanded problem set? (60+ problems)", default=True):
            args.append("--problems")
            args.append("builtin")
            if cli.confirm("Enable curriculum learning? (easy→hard)", default=False):
                args.append("--curriculum")

        if cli.confirm("Start reasoning training?"):
            self._run_script("train_reasoning.py", args)
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

    def _lr_finder_menu(self) -> None:
        """Learning rate finder."""
        _print_section_header("Learning Rate Finder", "Smith's LR Range Test")

        if _HAS_RICH:
            _rich_console.print("  Sweeps learning rate from low to high, plotting loss vs LR.")
            _rich_console.print("  Pick the LR where loss drops fastest (steepest descent).")
            _rich_console.print("")

        options = [
            {"label": "Tiny   (50M)",   "detail": "configs/tiny.yaml"},
            {"label": "Small  (125M)",  "detail": "configs/small.yaml"},
            {"label": "Medium (350M)",  "detail": "configs/medium.yaml"},
        ]

        choice = cli.choose("Select model config:", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium"]
        self._run_script("find_lr.py", ["--config", f"configs/{sizes[choice]}.yaml"])
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
                {"label": "Context-Aware Generation",
                 "detail": "scripts/generate.py --repo <dir> — uses repo context for generation"},
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
                self._context_aware_generate()
            elif choice == 3:
                self._serve_api()
            elif choice == 4:
                self._nano_benchmark()

    def _interactive_generate(self) -> None:
        """Interactive generation with checkpoint selection."""
        _print_section_header("Interactive Generation", "Select checkpoint and generate")

        ckpt_path = self._pick_checkpoint("Select checkpoint for generation:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("generate.py", ["--checkpoint", ckpt_path, "--config", config])
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

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("serve.py", ["--checkpoint", ckpt_path, "--config", config])
        self._pause()

    def _nano_benchmark(self) -> None:
        """Nano benchmark with checkpoint selection."""
        _print_section_header("Nano Benchmark", "Quick generation speed test")

        ckpt_path = self._pick_checkpoint("Select checkpoint to benchmark:")
        if ckpt_path is None:
            return

        self._run_script("nano_benchmark.py", ["--checkpoint", ckpt_path])
        self._pause()

    def _context_aware_generate(self) -> None:
        """Context-aware generation using a repository directory."""
        _print_section_header("Context-Aware Generation", "Generate with repo context")

        ckpt_path = self._pick_checkpoint("Select checkpoint for generation:")
        if ckpt_path is None:
            return

        try:
            repo_dir = input("Repository directory path: ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        if not repo_dir:
            cli.error("No repository path provided.")
            self._pause()
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("generate.py", [
            "--checkpoint", ckpt_path,
            "--config", config,
            "--repo", repo_dir,
        ])
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
                self._nano_benchmark()
            elif choice == 4:
                self._model_card_menu()
            elif choice == 5:
                self.training_status_menu()
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

    def _humaneval_menu(self) -> None:
        """HumanEval evaluation with checkpoint selection."""
        _print_section_header("HumanEval Evaluation", "164 Python coding problems — pass@k metric")

        ckpt_path = self._pick_checkpoint("Select checkpoint to evaluate:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("evaluate.py", ["--checkpoint", ckpt_path, "--config", config])
        self._pause()

    def _benchmark_menu(self) -> None:
        """Benchmark with checkpoint selection."""
        _print_section_header("Quick Benchmark", "Speed and quality benchmark")

        ckpt_path = self._pick_checkpoint("Select checkpoint to benchmark:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("benchmark.py", ["--checkpoint", ckpt_path, "--config", config])
        self._pause()

    def _model_card_menu(self) -> None:
        """Generate model card."""
        _print_section_header("Generate Model Card", "HuggingFace-style model card")

        ckpt_path = self._pick_checkpoint("Select checkpoint for model card:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("model_card.py", ["--checkpoint", ckpt_path, "--config", config])
        self._pause()

    def _smoke_test_menu(self) -> None:
        """Quick smoke test for a checkpoint."""
        _print_section_header("Smoke Test", "8 quick validation checks in <30 seconds")

        if _HAS_RICH:
            _rich_console.print("  Checks: token generation, syntax, perplexity, repetition,")
            _rich_console.print("  diversity, special tokens, temperature sensitivity, code keywords")
            _rich_console.print("")

        ckpt_path = self._pick_checkpoint("Select checkpoint to smoke-test:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        args = ["--checkpoint", ckpt_path, "--config", config]
        if cli.confirm("Quick mode (fewer samples)?", default=True):
            args.append("--quick")

        self._run_script("smoke_test.py", args)
        self._pause()

    def _ts_benchmark_menu(self) -> None:
        """TypeScript-specific benchmark."""
        _print_section_header("TypeScript Benchmark", "50 TS-specific coding problems")

        ckpt_path = self._pick_checkpoint("Select checkpoint to benchmark:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("ts_benchmark.py", ["--checkpoint", ckpt_path, "--config", config])
        self._pause()

    def _regression_test_menu(self) -> None:
        """Regression test suite."""
        _print_section_header("Regression Tests", "Track quality across checkpoint versions")

        ckpt_path = self._pick_checkpoint("Select checkpoint to test:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("regression_test.py", ["--checkpoint", ckpt_path, "--config", config])
        self._pause()

    def _quality_report_menu(self) -> None:
        """Generate quality report."""
        _print_section_header("Quality Report", "Auto-generate markdown quality report")

        ckpt_path = self._pick_checkpoint("Select checkpoint for report:")
        if ckpt_path is None:
            return

        config = self._config_for_checkpoint(ckpt_path)
        self._run_script("quality_report.py", ["--checkpoint", ckpt_path, "--config", config])
        self._pause()

    def _compare_checkpoints_menu(self) -> None:
        """Compare two checkpoints from the same model."""
        _print_section_header(
            "Compare Checkpoints",
            "Side-by-side comparison of two checkpoints",
        )

        model = self._pick_model("Select model to compare checkpoints:")
        if model is None:
            return
        ckpt_a = self._pick_checkpoint(
            "Select checkpoint A:", model=model,
        )
        if ckpt_a is None:
            return
        ckpt_b = self._pick_checkpoint(
            "Select checkpoint B:", model=model,
        )
        if ckpt_b is None:
            return

        self._run_script(
            "compare_checkpoints.py",
            ["--a", ckpt_a, "--b", ckpt_b],
        )
        self._pause()

    def _compare_models_menu(self) -> None:
        """Side-by-side comparison of checkpoints from different models."""
        _print_section_header(
            "Compare Models",
            "Side-by-side comparison of two model checkpoints",
        )

        ckpt_a = self._pick_checkpoint("Select first model checkpoint:")
        if ckpt_a is None:
            return
        ckpt_b = self._pick_checkpoint("Select second model checkpoint:")
        if ckpt_b is None:
            return

        self._run_script(
            "compare_models.py", ["--checkpoints", ckpt_a, ckpt_b],
        )
        self._pause()

    # ── 5b. Router & Specialists ─────────────────────────────────────────

    def router_menu(self) -> None:
        """Router model and specialist management."""
        while True:
            _print_section_header(
                "Router & Specialists",
                "Domain routing, MoE, specialist training & management",
            )

            options = [
                {"label": "Generate Router Training Data",
                 "detail": "Auto-label code samples for router training"},
                {"label": "Train Router Model",
                 "detail": "scripts/train_router.py — train MLP or Transformer router"},
                {"label": "Evaluate Router",
                 "detail": "Test router accuracy on labeled examples"},
                {"label": "Manage Specialist Registry",
                 "detail": "View/add/remove specialist checkpoints"},
                {"label": "MoE Configuration",
                 "detail": "Configure Mixture of Experts layer settings"},
                {"label": "Domain Detection Test",
                 "detail": "Test heuristic domain detector on sample code"},
            ]

            choice = cli.choose("Select operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._generate_router_data_menu()
            elif choice == 1:
                self._train_router_menu()
            elif choice == 2:
                self._evaluate_router()
            elif choice == 3:
                self._specialist_registry_menu()
            elif choice == 4:
                self._moe_config_menu()
            elif choice == 5:
                self._domain_detection_test()

    def _generate_router_data_menu(self) -> None:
        """Generate router training data sub-menu."""
        _print_section_header("Generate Router Data", "Create labeled data for router training")

        options = [
            {"label": "From Training Data (.npy)",
             "detail": "Decode existing tokenized data and auto-label domains"},
            {"label": "From Source Directory",
             "detail": "Scan a directory of source code files"},
            {"label": "Synthetic (Bootstrap)",
             "detail": "Generate template-based synthetic examples"},
        ]

        choice = cli.choose("Data source:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            self._run_script("generate_router_data.py", [
                "--source", "data/processed/train_data.npy",
                "--tokenizer", self.storage.tokenizer_path,
            ])
        elif choice == 1:
            cli.info("Tip", "Enter the path to a directory containing .ts/.tsx/.js files")
            self._run_script("generate_router_data.py", ["--source-dir", "."])
        elif choice == 2:
            self._run_script("generate_router_data.py", ["--synthetic"])

        self._pause()

    def _train_router_menu(self) -> None:
        """Train router model sub-menu."""
        _print_section_header("Train Router Model", "Lightweight domain classifier (<5M params)")

        options = [
            {"label": "MLP Router (fast, ~100us inference)",
             "detail": "Bag-of-embeddings → MLP → softmax"},
            {"label": "Transformer Router (better quality, ~1ms)",
             "detail": "Embedding → 2 transformer layers → classification"},
        ]

        choice = cli.choose("Router architecture:", options, allow_cancel=True)
        if choice is None:
            return

        arch = "mlp" if choice == 0 else "transformer"
        args = ["--arch", arch, "--generate-data"]

        # Check if training data exists
        data_path = Path("data/router_training_data.jsonl")
        if data_path.exists():
            if cli.confirm("Router training data exists. Regenerate?", default=False):
                args.append("--generate-data")
            else:
                args = ["--arch", arch, "--data", str(data_path)]

        self._run_script("train_router.py", args)
        self._pause()

    def _evaluate_router(self) -> None:
        """Evaluate router accuracy."""
        _print_section_header("Evaluate Router", "Test routing accuracy")

        if _HAS_RICH:
            _rich_console.print("  Running router evaluation on built-in test dataset...")
            _rich_console.print("  Checks: accuracy, per-domain precision/recall/F1,")
            _rich_console.print("  confusion matrix, confidence calibration")
            _rich_console.print("")

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

            cli.info("Overall accuracy", f"{metrics['accuracy']:.1%}")
            cli.info("Macro F1", f"{metrics['macro_f1']:.3f}")
            cli.info("Weighted F1", f"{metrics['weighted_f1']:.3f}")

            if _HAS_RICH:
                _rich_console.print("")
                _rich_console.print("  [bold]Per-domain results:[/bold]")
                for domain, stats in metrics.get("per_domain", {}).items():
                    p = stats.get("precision", 0)
                    r = stats.get("recall", 0)
                    f1 = stats.get("f1", 0)
                    _rich_console.print(
                        f"    {domain:12s}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}"
                    )

        except Exception as e:
            cli.error(f"Evaluation failed: {e}")

        self._pause()

    def _specialist_registry_menu(self) -> None:
        """View and manage specialist registry."""
        _print_section_header("Specialist Registry", "Manage domain specialist checkpoints")

        try:
            from cola_coder.features.specialist_registry import SpecialistRegistry
            registry = SpecialistRegistry(str(self.project_root / "configs" / "specialists.yaml"))
            specialists = registry.list_specialists()

            if not specialists:
                cli.warn("No specialists registered yet.")
                cli.dim("Train domain-specific models and register them here.")
                cli.dim("Registry file: configs/specialists.yaml")
            else:
                for spec in specialists:
                    exists = Path(spec.checkpoint).exists() if spec.checkpoint else False
                    status = "[green]ready[/green]" if exists else "[red]missing[/red]"
                    if _HAS_RICH:
                        _rich_console.print(
                            f"  {spec.domain:12s}  {status}  {spec.checkpoint}"
                        )
                    else:
                        cli.print(f"  {spec.domain:12s}  {spec.checkpoint}")

        except Exception as e:
            cli.warn(f"Could not load registry: {e}")
            cli.dim("Registry file: configs/specialists.yaml")

        self._pause()

    def _moe_config_menu(self) -> None:
        """Configure MoE layer settings."""
        _print_section_header("Mixture of Experts (MoE)", "Configure sparse expert layers")

        if _HAS_RICH:
            _rich_console.print("  [bold]What is MoE?[/bold]")
            _rich_console.print(
                "  Replaces standard FFN layers with multiple expert FFNs.\n"
                "  A router sends each token to the top-k experts.\n"
                "  Result: more parameters without proportionally more compute.\n"
            )
            _rich_console.print("  [bold]Current settings:[/bold]")
            _rich_console.print("    Experts: 8  |  Top-K: 2  |  Capacity: 1.25")
            _rich_console.print("    Aux loss weight: 0.01 (prevents expert collapse)")
            _rich_console.print("")
            _rich_console.print("  [bold]Status:[/bold]", end=" ")

            try:
                from cola_coder.features import moe_layer
                if moe_layer.is_enabled():
                    _rich_console.print("[green]Enabled[/green]")
                else:
                    _rich_console.print("[red]Disabled[/red] (toggle in Feature Toggles)")
            except Exception:
                _rich_console.print("[red]Disabled[/red]")

            _rich_console.print("")
            _rich_console.print("  [dim]MoE is experimental. Enable via Settings → Feature Toggles → Training → moe_layer[/dim]")
            _rich_console.print("  [dim]When enabled, add to model config: moe_layers: [4, 8, 12][/dim]")

        self._pause()

    def _domain_detection_test(self) -> None:
        """Test heuristic domain detection on sample code."""
        _print_section_header("Domain Detection Test", "Test the heuristic classifier")

        if _HAS_RICH:
            _rich_console.print("  Testing domain detection on built-in samples...\n")

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

                if _HAS_RICH:
                    _rich_console.print(
                        f"  {icon}  expected=[cyan]{sample.expected_domain:12s}[/cyan]"
                        f"  predicted={predicted:12s}  conf={conf}"
                    )
                else:
                    mark = "✓" if is_correct else "✗"
                    cli.print(
                        f"  {mark}  expected={sample.expected_domain:12s}"
                        f"  predicted={predicted:12s}  conf={conf}"
                    )

            acc = correct / len(test_data) if test_data else 0
            cli.info("Accuracy", f"{correct}/{len(test_data)} ({acc:.0%})")

        except Exception as e:
            cli.error(f"Detection test failed: {e}")

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
                {"label": "Export Model (GGUF/Ollama/Quantize)",
                 "detail": "scripts/export_model.py — export, quantize, create Modelfile"},
                {"label": "Average Checkpoints",
                 "detail": "scripts/average_checkpoints.py — uniform/EMA checkpoint merging"},
                {"label": "Run Full Pipeline",
                 "detail": "scripts/run_pipeline.py — tokenize→train→eval→export"},
                {"label": "Scan Repository",
                 "detail": "Scan a source repo and display structure/stats"},
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
            elif choice == 6:
                self._run_script("export_model.py")
                self._pause()
            elif choice == 7:
                self._run_script("average_checkpoints.py")
                self._pause()
            elif choice == 8:
                self._pipeline_menu()
            elif choice == 9:
                self._scan_repository()

    def _pipeline_menu(self) -> None:
        """Full pipeline orchestrator."""
        _print_section_header("Pipeline Orchestrator", "tokenize → train → eval → export")

        if _HAS_RICH:
            _rich_console.print("  Runs up to 6 stages: tokenizer, data_prep, training,")
            _rich_console.print("  smoke_test, evaluation, export. Smart caching skips done stages.")
            _rich_console.print("")

        options = [
            {"label": "Run All Stages",
             "detail": "Full pipeline with smart caching"},
            {"label": "Dry Run",
             "detail": "Show what would run without executing"},
            {"label": "Continue from Failure",
             "detail": "Resume pipeline, skip failed stages"},
        ]

        choice = cli.choose("Pipeline mode:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            self._run_script("run_pipeline.py", ["--config", "configs/tiny.yaml"])
        elif choice == 1:
            self._run_script("run_pipeline.py", ["--config", "configs/tiny.yaml", "--dry-run"])
        elif choice == 2:
            self._run_script("run_pipeline.py", [
                "--config", "configs/tiny.yaml", "--continue-on-failure",
            ])
        self._pause()

    def _scan_repository(self) -> None:
        """Scan a repository directory and display file structure and stats."""
        _print_section_header("Scan Repository", "Analyse a source code repository")

        try:
            repo_path = input("Repository path: ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        if not repo_path:
            cli.error("No path provided.")
            self._pause()
            return

        repo = Path(repo_path)
        if not repo.exists():
            cli.error(f"Path does not exist: {repo_path}")
            self._pause()
            return

        cli.info("Scanning", str(repo))

        ext_counts: dict[str, int] = {}
        total_files = 0
        total_lines = 0

        for p in repo.rglob("*"):
            if not p.is_file():
                continue
            # Skip hidden dirs and common noise
            parts = p.parts
            if any(part.startswith(".") or part in ("node_modules", "__pycache__", ".venv")
                   for part in parts):
                continue
            total_files += 1
            suffix = p.suffix.lower() or "(no ext)"
            ext_counts[suffix] = ext_counts.get(suffix, 0) + 1
            try:
                total_lines += sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
            except OSError:
                pass

        cli.info("Total files", str(total_files))
        cli.info("Total lines", f"{total_lines:,}")

        if ext_counts:
            top = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            if _HAS_RICH:
                _rich_console.print("\n  [bold]Top file types:[/bold]")
                for ext, count in top:
                    bar = "█" * min(count, 40)
                    _rich_console.print(f"    {ext:12s}  {count:5d}  [dim]{bar}[/dim]")
            else:
                cli.print("\nTop file types:")
                for ext, count in top:
                    cli.print(f"  {ext:12s}  {count:5d}")

        self._pause()

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
                {"label": "Migrate Storage",
                 "detail": "Copy/move data to configured storage location"},
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
                self._run_script("migrate_storage.py")
                self._pause()
            elif choice == 3:
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

        options = [
            {"label": "All Models",    "detail": "Show status for every model size"},
            {"label": "Tiny   (50M)",  "detail": "configs/tiny.yaml checkpoints only"},
            {"label": "Small  (125M)", "detail": "configs/small.yaml checkpoints only"},
            {"label": "Medium (350M)", "detail": "configs/medium.yaml checkpoints only"},
            {"label": "Large  (1B+)",  "detail": "configs/large.yaml checkpoints only"},
        ]

        choice = cli.choose("Which model?", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            self._run_script("training_status.py")
        else:
            sizes = ["tiny", "small", "medium", "large"]
            self._run_script("training_status.py", ["--size", sizes[choice - 1]])
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
