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


def _print_status_panel(status: dict[str, str]) -> None:
    """Print pipeline status bar."""
    cli.kv_table(status, title="Pipeline Status")


def _print_section_header(title: str, subtitle: str = "", hint: str = "") -> None:
    """Print a section header."""
    cli.header(title, subtitle)
    if hint:
        cli.dim(f"  {hint}")


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

        # Sub-menu modules
        from cola_coder.features.menus.data_menu import DataMenu
        from cola_coder.features.menus.training_menu import TrainingMenu
        from cola_coder.features.menus.eval_menu import EvalMenu
        from cola_coder.features.menus.tools_menu import ToolsMenu
        self._data = DataMenu(self)
        self._training = TrainingMenu(self)
        self._eval = EvalMenu(self)
        self._tools = ToolsMenu(self)

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
                self._data.menu,
                self._training.menu,
                self.generate_menu,
                self._eval.menu,
                self.router_menu,
                self._tools.menu,
                self._tools.settings_menu,
                self._tools.training_status_menu,
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

        cli.print("  [bold]Pipeline stages:[/bold]")
        for name, _, done in stages:
            icon = "[green]✓[/green]" if done else "[yellow]○[/yellow]"
            cli.print(f"    {icon}  {name}")
        cli.print("")

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

        cli.print("  Starts a FastAPI server on [cyan]http://localhost:8000[/cyan]")
        cli.print("  Press [bold]Ctrl-C[/bold] in the terminal to stop the server.")
        cli.print("")

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

        cli.print("  Running router evaluation on built-in test dataset...")
        cli.print("  Checks: accuracy, per-domain precision/recall/F1,")
        cli.print("  confusion matrix, confidence calibration")
        cli.print("")

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
                    cli.print(f"  {spec.domain:12s}  {status}  {spec.checkpoint}")

        except Exception as e:
            cli.warn(f"Could not load registry: {e}")
            cli.dim("Registry file: configs/specialists.yaml")

        self._pause()

    def _moe_config_menu(self) -> None:
        """Configure MoE layer settings."""
        _print_section_header("Mixture of Experts (MoE)", "Configure sparse expert layers")

        cli.print("  [bold]What is MoE?[/bold]")
        cli.print(
            "  Replaces standard FFN layers with multiple expert FFNs.\n"
            "  A router sends each token to the top-k experts.\n"
            "  Result: more parameters without proportionally more compute.\n"
        )
        cli.print("  [bold]Current settings:[/bold]")
        cli.print("    Experts: 8  |  Top-K: 2  |  Capacity: 1.25")
        cli.print("    Aux loss weight: 0.01 (prevents expert collapse)")
        cli.print("")

        try:
            from cola_coder.features import moe_layer
            if moe_layer.is_enabled():
                cli.print("  [bold]Status:[/bold] [green]Enabled[/green]")
            else:
                cli.print("  [bold]Status:[/bold] [red]Disabled[/red] (toggle in Feature Toggles)")
        except Exception:
            cli.print("  [bold]Status:[/bold] [red]Disabled[/red]")

        cli.print("")
        cli.print("  [dim]MoE is experimental. Enable via Settings → Feature Toggles → Training → moe_layer[/dim]")
        cli.print("  [dim]When enabled, add to model config: moe_layers: [4, 8, 12][/dim]")

        self._pause()

    def _domain_detection_test(self) -> None:
        """Test heuristic domain detection on sample code."""
        _print_section_header("Domain Detection Test", "Test the heuristic classifier")

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

        self._pause()


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
