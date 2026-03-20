"""Unified master menu for Cola-Coder.

Single entry point for all Cola-Coder operations. Replaces 12 separate
PowerShell scripts with one interactive, keyboard-driven menu.
"""

import importlib
import subprocess
from pathlib import Path
from cola_coder.cli import cli

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


class MasterMenu:
    """Unified CLI menu for all Cola-Coder operations."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        # Windows: .venv/Scripts/python.exe — Linux/Mac: .venv/bin/python
        win_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        if win_python.exists():
            self.venv_python = win_python
        else:
            unix_python = self.project_root / ".venv" / "bin" / "python"
            if unix_python.exists():
                self.venv_python = unix_python
            else:
                # Last resort: try sys.executable (the current Python)
                import sys
                self.venv_python = Path(sys.executable)

    def _run_script(self, script: str, args: list[str] | None = None):
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

    def _detect_pipeline_status(self) -> dict[str, str]:
        """Detect current pipeline state: what's been completed."""
        status = {}

        # Tokenizer
        tokenizer_path = self.project_root / "tokenizer.json"
        status["tokenizer"] = "ready" if tokenizer_path.exists() else "missing"

        # Data
        data_dir = self.project_root / "data" / "processed"
        npy_files = list(data_dir.glob("*.npy")) if data_dir.exists() else []
        status["data"] = f"{len(npy_files)} dataset(s)" if npy_files else "missing"

        # Checkpoints
        ckpt_dir = self.project_root / "checkpoints"
        if ckpt_dir.exists():
            ckpt_dirs = [d for d in ckpt_dir.rglob("model.safetensors")]
            status["checkpoints"] = f"{len(ckpt_dirs)} checkpoint(s)" if ckpt_dirs else "none"
        else:
            status["checkpoints"] = "none"

        return status

    def show_status_bar(self):
        """Show pipeline status at the top of the menu."""
        status = self._detect_pipeline_status()
        parts = []
        for key, val in status.items():
            if "missing" in val or val == "none":
                parts.append(f"[red]{key}: {val}[/red]")
            else:
                parts.append(f"[green]{key}: {val}[/green]")
        cli.print(f"  Pipeline: {' | '.join(parts)}")

    def main_menu(self):
        """Show the main menu and handle selection."""
        while True:
            cli.header("Cola-Coder", "Master Menu")
            self.show_status_bar()

            options = [
                {"label": "Train Model", "detail": "Train tiny/small/medium/large models"},
                {"label": "Prepare Data", "detail": "Download, filter, tokenize training data"},
                {"label": "Generate Code", "detail": "Interactive code generation"},
                {"label": "Evaluate Model", "detail": "Run HumanEval benchmark"},
                {"label": "Serve API", "detail": "Start FastAPI inference server"},
                {"label": "Train Tokenizer", "detail": "Train BPE tokenizer from scratch"},
                {"label": "Train Reasoning", "detail": "GRPO fine-tuning with thinking tokens"},
                {"label": "Tools", "detail": "Tests, linting, data inspection"},
            ]

            choice = cli.choose("What would you like to do?", options, allow_cancel=True)

            if choice is None:
                cli.dim("Goodbye!")
                break

            handlers = [
                self.train_menu,
                self.prepare_menu,
                self.generate_menu,
                self.evaluate_menu,
                self.serve_menu,
                self.tokenizer_menu,
                self.reasoning_menu,
                self.tools_menu,
            ]

            handlers[choice]()

    def train_menu(self):
        """Training sub-menu."""
        cli.header("Cola-Coder", "Train Model")

        options = [
            {"label": "Tiny (50M)", "detail": "~3.6GB VRAM, ~4 hours"},
            {"label": "Small (125M)", "detail": "~6.5GB VRAM, ~2 days"},
            {"label": "Medium (350M)", "detail": "~8.2GB VRAM, ~7 days"},
            {"label": "Large (1B+)", "detail": "Cloud only, ~24GB VRAM"},
        ]

        choice = cli.choose("Select model size:", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium", "large"]
        size = sizes[choice]

        # Check for existing checkpoints to resume
        ckpt_dir = self.project_root / "checkpoints" / size
        if ckpt_dir.exists():
            latest = ckpt_dir / "latest"
            if latest.exists() or any(ckpt_dir.glob("step_*")):
                if cli.confirm(f"Found existing {size} checkpoints. Resume training?"):
                    self._run_script("train.py", [
                        "--config", f"configs/{size}.yaml",
                        "--resume", str(ckpt_dir / "latest"),
                    ])
                    return

        # Check for wandb
        use_wandb = cli.confirm("Enable Weights & Biases logging?", default=False)
        args = ["--config", f"configs/{size}.yaml"]
        if use_wandb:
            args.append("--wandb")

        self._run_script("train.py", args)

    def prepare_menu(self):
        """Data preparation sub-menu."""
        cli.header("Cola-Coder", "Prepare Data")

        options = [
            {"label": "Interactive Mode", "detail": "Full menu-driven data preparation"},
            {"label": "Quick Tiny Dataset", "detail": "Small dataset for testing (~5 min)"},
            {"label": "Standard Preparation", "detail": "Full data pipeline with defaults"},
            {"label": "Test/Validation Data", "detail": "Prepare test split"},
        ]

        choice = cli.choose("Data preparation mode:", options, allow_cancel=True)
        if choice is None:
            return

        scripts = [
            ("prepare_data_interactive.py", []),
            ("prepare_data.py", ["--config", "configs/tiny.yaml", "--tokenizer", "tokenizer.json",
                                 "--max-tokens", "500000"]),
            ("prepare_data.py", ["--config", "configs/tiny.yaml", "--tokenizer", "tokenizer.json"]),
            ("prepare_data.py", ["--config", "configs/tiny.yaml", "--tokenizer", "tokenizer.json",
                                 "--split", "test"]),
        ]

        script, args = scripts[choice]
        self._run_script(script, args)

    def generate_menu(self):
        """Code generation."""
        cli.header("Cola-Coder", "Generate Code")

        # Find available checkpoints
        ckpt_dir = self.project_root / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            for size_dir in sorted(ckpt_dir.iterdir()):
                if size_dir.is_dir():
                    latest = size_dir / "latest"
                    if latest.exists():
                        # Resolve "latest" pointer to show the actual checkpoint path as detail
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

        if not checkpoints:
            cli.error("No checkpoints found. Train a model first.")
            return

        # "Quick Generate" is always the first option — run.py auto-detects
        # the latest checkpoint, its config from metadata.json, and uses
        # default sampling params. No manual selection needed.
        quick_option = {
            "label": "Quick Generate (auto-detect everything)",
            "detail": "Uses latest checkpoint + auto-detected config + default sampling",
        }
        menu_options = [quick_option] + [{"label": c["label"], "detail": c["detail"]} for c in checkpoints]

        choice = cli.choose("Select checkpoint:", menu_options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            # Quick Generate: run.py handles checkpoint + config detection automatically
            self._run_script("run.py", [])
        else:
            # Offset by 1 because index 0 is the quick option
            selected = checkpoints[choice - 1]
            self._run_script("generate.py", ["--checkpoint", selected["path"]])

    def evaluate_menu(self):
        """Evaluation menu."""
        cli.header("Cola-Coder", "Evaluate Model")

        # Find checkpoints (same pattern as generate)
        ckpt_dir = self.project_root / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            for size_dir in sorted(ckpt_dir.iterdir()):
                if size_dir.is_dir():
                    latest = size_dir / "latest"
                    if latest.exists():
                        checkpoints.append({"label": f"{size_dir.name}/latest", "path": str(latest)})

        if not checkpoints:
            cli.error("No checkpoints found. Train a model first.")
            return

        choice = cli.choose("Select checkpoint to evaluate:",
                            [{"label": c["label"], "detail": ""} for c in checkpoints],
                            allow_cancel=True)
        if choice is None:
            return

        self._run_script("evaluate.py", ["--checkpoint", checkpoints[choice]["path"]])

    def serve_menu(self):
        """Start API server."""
        cli.header("Cola-Coder", "Serve API")
        ckpt_dir = self.project_root / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            for size_dir in sorted(ckpt_dir.iterdir()):
                if size_dir.is_dir():
                    latest = size_dir / "latest"
                    if latest.exists():
                        checkpoints.append({"label": f"{size_dir.name}/latest", "path": str(latest)})

        if not checkpoints:
            cli.error("No checkpoints found.")
            return

        choice = cli.choose("Select checkpoint to serve:",
                            [{"label": c["label"], "detail": ""} for c in checkpoints],
                            allow_cancel=True)
        if choice is None:
            return

        self._run_script("serve.py", ["--checkpoint", checkpoints[choice]["path"]])

    def tokenizer_menu(self):
        """Train tokenizer."""
        cli.header("Cola-Coder", "Train Tokenizer")

        tokenizer_path = self.project_root / "tokenizer.json"
        if tokenizer_path.exists():
            if not cli.confirm("tokenizer.json already exists. Retrain?", default=False):
                return

        self._run_script("train_tokenizer.py", [])

    def reasoning_menu(self):
        """Reasoning/GRPO training."""
        cli.header("Cola-Coder", "Reasoning Training")
        cli.info("Mode", "GRPO with thinking tokens")

        if cli.confirm("Start reasoning training?"):
            self._run_script("train_reasoning.py", [])

    def tools_menu(self):
        """Developer tools sub-menu."""
        cli.header("Cola-Coder", "Tools")

        # Build the N/M indicator for feature toggles
        try:
            features = _scan_feature_modules(self.project_root)
            n_enabled, n_total = _count_enabled(features)
            toggles_detail = f"{n_enabled}/{n_total} enabled — enable/disable optional features"
        except Exception:
            toggles_detail = "Enable/disable optional features"

        options = [
            {"label": "Run Tests", "detail": "pytest tests/ -v"},
            {"label": "Run Linter", "detail": "ruff check src/ scripts/ tests/"},
            {"label": "GPU Status", "detail": "Show GPU info and VRAM usage"},
            {"label": "Dataset Inspector", "detail": "Browse training data samples"},
            {"label": "Feature Toggles", "detail": toggles_detail},
        ]

        choice = cli.choose("Select tool:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            subprocess.run([str(self.venv_python), "-m", "pytest", "tests/", "-v"],
                           cwd=str(self.project_root))
        elif choice == 1:
            subprocess.run([str(self.venv_python), "-m", "ruff", "check", "src/", "scripts/",
                            "tests/"],
                           cwd=str(self.project_root))
        elif choice == 2:
            cli.gpu_info()
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode == 0:
                    cli.print(result.stdout)
            except FileNotFoundError:
                cli.warn("nvidia-smi not found")
        elif choice == 3:
            self._inspect_dataset()
            input("\nPress Enter to continue...")
            return
        elif choice == 4:
            self._feature_toggles()
            return  # _feature_toggles manages its own loop; skip the shared prompt

        input("\nPress Enter to continue...")

    def _inspect_dataset(self):
        """Browse random samples from training data."""
        import numpy as np
        data_dir = self.project_root / "data" / "processed"
        npy_files = list(data_dir.glob("*.npy")) if data_dir.exists() else []

        if not npy_files:
            cli.error("No datasets found.")
            return

        # Pick a dataset
        if len(npy_files) == 1:
            npy_path = npy_files[0]
        else:
            options = [{"label": f.stem, "detail": ""} for f in npy_files]
            choice = cli.choose("Select dataset:", options, allow_cancel=True)
            if choice is None:
                return
            npy_path = npy_files[choice]

        data = np.load(str(npy_path), mmap_mode="r")
        cli.info("Shape", f"{data.shape[0]:,} chunks x {data.shape[1]} tokens")
        cli.info("Total tokens", f"{data.shape[0] * data.shape[1]:,}")

        # Show random samples
        try:
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
            tokenizer_path = self.project_root / "tokenizer.json"
            if tokenizer_path.exists():
                tokenizer = CodeTokenizer(str(tokenizer_path))
                indices = np.random.choice(data.shape[0], size=min(3, data.shape[0]), replace=False)
                for idx in indices:
                    cli.rule(f"Sample #{idx}")
                    tokens = data[idx].tolist()
                    text = tokenizer.decode(tokens)
                    # Truncate for display
                    display_text = text[:500] + ("..." if len(text) > 500 else "")
                    cli.print(display_text)
        except Exception as e:
            cli.warn(f"Could not decode samples: {e}")

    def _feature_toggles(self):
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

            # Reload states fresh each time we return to the category list
            try:
                features = _scan_feature_modules(self.project_root)
            except Exception as exc:
                cli.warn(f"Could not scan feature modules: {exc}")
                input("\nPress Enter to continue...")
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

            # Bulk actions at the bottom
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
                return  # Back to Tools menu

            # "Enable ALL Features"
            if choice == len(cat_options) - 2:
                if cli.confirm(f"Enable all {n_total} features?", default=False):
                    for feat in features:
                        _save_feature_state(self.project_root, feat["name"], True)
                    cli.success(f"Enabled all {n_total} features.")
                    input("\nPress Enter to continue...")
                continue

            # "Disable ALL Features"
            if choice == len(cat_options) - 1:
                if cli.confirm(f"Disable all {n_total} features?", default=False):
                    for feat in features:
                        _save_feature_state(self.project_root, feat["name"], False)
                    cli.warn(f"Disabled all {n_total} features.  Core functionality is unaffected.")
                    input("\nPress Enter to continue...")
                continue

            # Category drill-down
            selected_cat = categories_in_use[choice]
            self._feature_category_menu(selected_cat, features)

    def _feature_category_menu(self, category: str, features: list[dict]):
        """Show features in a single category and allow toggling.

        Loops until the user chooses Back / Cancel.
        """
        while True:
            cli.header("Cola-Coder", f"Feature Toggles — {category}")
            cli.dim("All features listed here are OPTIONAL.")
            cli.print("")

            # Reload yaml states so the list is always current
            yaml_states = _load_feature_states(self.project_root)
            cat_features = [f for f in features if f["category"] == category]

            # Apply latest persisted states before displaying
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

            # Bulk actions for this category
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

            # "Enable All in Category"
            if choice == len(feat_options) - 2:
                if cli.confirm(f"Enable all {cat_total} {category} features?", default=True):
                    for feat in cat_features:
                        feat["enabled"] = True
                        _save_feature_state(self.project_root, feat["name"], True)
                    cli.success(f"Enabled all {cat_total} features in {category}.")
                    input("\nPress Enter to continue...")
                continue

            # "Disable All in Category"
            if choice == len(feat_options) - 1:
                if cli.confirm(f"Disable all {cat_total} {category} features?", default=False):
                    for feat in cat_features:
                        feat["enabled"] = False
                        _save_feature_state(self.project_root, feat["name"], False)
                    cli.warn(
                        f"Disabled all {cat_total} features in {category}.  "
                        "Core functionality is unaffected."
                    )
                    input("\nPress Enter to continue...")
                continue

            # Toggle a single feature
            feat = cat_features[choice]
            new_state = not feat["enabled"]
            feat["enabled"] = new_state
            _save_feature_state(self.project_root, feat["name"], new_state)

            if new_state:
                cli.success(f"{feat['label']} enabled.")
            else:
                cli.warn(
                    f"{feat['label']} disabled.  "
                    "(This is optional — core functionality is unaffected.)"
                )
            # Loop back immediately so the user sees the updated state


def run_master_menu():
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
