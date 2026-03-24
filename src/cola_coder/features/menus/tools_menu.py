"""Tools, utilities, and settings sub-menu for Cola-Coder."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from cola_coder.cli import cli
from cola_coder.features.master_menu import (
    _FEATURE_CATEGORIES,
    _count_enabled,
    _load_feature_states,
    _print_section_header,
    _save_feature_state,
    _scan_feature_modules,
)

if TYPE_CHECKING:
    from cola_coder.features.master_menu import MasterMenu


class ToolsMenu:
    """Tools, utilities, and settings menu."""

    def __init__(self, master: MasterMenu) -> None:
        self._master = master

    def menu(self) -> None:
        """Show the tools menu."""
        while True:
            _print_section_header("Tools & Utilities", "Tests, linting, GPU, data inspection")

            # Build feature toggles label
            try:
                features = _scan_feature_modules(self._master.project_root)
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
                {"label": "Environment Check",
                 "detail": "Verify Python, PyTorch, CUDA, GPU, disk, HF_TOKEN"},
                {"label": "Tokenizer Health",
                 "detail": "Vocab size, special tokens, roundtrip, avg token length"},
                {"label": "Project Health",
                 "detail": "Overall project health score"},
            ]

            choice = cli.choose("Select tool:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._master._run_shell([
                    str(self._master.venv_python), "-m", "pytest", "tests/", "-v",
                ])
                self._master._pause()
            elif choice == 1:
                self._master._run_shell([
                    str(self._master.venv_python), "-m", "ruff", "check",
                    "src/", "scripts/", "tests/",
                ])
                self._master._pause()
            elif choice == 2:
                self._gpu_status()
            elif choice == 3:
                self._master._data._inspect_dataset()
                self._master._pause()
            elif choice == 4:
                self._master._run_script("test_type_reward.py")
                self._master._pause()
            elif choice == 5:
                self._feature_toggles()
            elif choice == 6:
                self._export_model_menu()
            elif choice == 7:
                self._average_checkpoints_menu()
            elif choice == 8:
                self._pipeline_menu()
            elif choice == 9:
                self._scan_repository()
            elif choice == 10:
                self._master._run_script("env_check.py")
                self._master._pause()
            elif choice == 11:
                self._master._run_script("tokenizer_health.py")
                self._master._pause()
            elif choice == 12:
                self._master._run_script("project_health.py")
                self._master._pause()

    def _export_model_menu(self) -> None:
        """Export model to GGUF/Ollama/quantized format."""
        _print_section_header("Export Model", "GGUF / Ollama / quantize")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to export:")
        if ckpt_path is None:
            return
        config_path = self._master._config_for_checkpoint(ckpt_path)
        self._master._run_script(
            "export_model.py", ["--checkpoint", ckpt_path, "--config", config_path]
        )
        self._master._pause()

    def _average_checkpoints_menu(self) -> None:
        """Average multiple checkpoints for improved quality."""
        _print_section_header("Average Checkpoints", "Uniform / EMA checkpoint merging")

        mode_options = [
            {"label": "By directory",
             "detail": "Auto-pick the N most-recent checkpoints in a folder"},
            {"label": "By explicit list",
             "detail": "Manually specify checkpoint paths to average"},
        ]
        mode = cli.choose("Selection method:", mode_options, allow_cancel=True)
        if mode is None:
            return

        if mode == 0:
            model = self._master._pick_model("Select model:")
            if model is None:
                return
            ckpt_dir = str(Path(self._master.storage.checkpoints_dir) / model)
            self._master._run_script("average_checkpoints.py", ["--checkpoint-dir", ckpt_dir])
        else:
            ckpt_a = self._master._pick_checkpoint("Select checkpoint 1 (oldest first):")
            if ckpt_a is None:
                return
            ckpt_b = self._master._pick_checkpoint("Select checkpoint 2:")
            if ckpt_b is None:
                return
            self._master._run_script(
                "average_checkpoints.py", ["--checkpoints", ckpt_a, ckpt_b]
            )
        self._master._pause()

    def _pipeline_menu(self) -> None:
        """Full pipeline orchestrator."""
        _print_section_header("Pipeline Orchestrator", "tokenize → train → eval → export")

        cli.print("  Runs up to 6 stages: tokenizer, data_prep, training,")
        cli.print("  smoke_test, evaluation, export. Smart caching skips done stages.")
        cli.print("")

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
            self._master._run_script("run_pipeline.py", ["--config", "configs/tiny.yaml"])
        elif choice == 1:
            self._master._run_script(
                "run_pipeline.py", ["--config", "configs/tiny.yaml", "--dry-run"]
            )
        elif choice == 2:
            self._master._run_script("run_pipeline.py", [
                "--config", "configs/tiny.yaml", "--continue-on-failure",
            ])
        self._master._pause()

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
            self._master._pause()
            return

        repo = Path(repo_path)
        if not repo.exists():
            cli.error(f"Path does not exist: {repo_path}")
            self._master._pause()
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
            cli.print("\n  [bold]Top file types:[/bold]")
            for ext, count in top:
                bar = "█" * min(count, 40)
                cli.print(f"    {ext:12s}  {count:5d}  [dim]{bar}[/dim]")

        self._master._pause()

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

        self._master._pause()

    def settings_menu(self) -> None:
        """Settings and configuration sub-menu."""
        while True:
            _print_section_header("Settings", "Feature toggles and storage configuration")

            try:
                features = _scan_feature_modules(self._master.project_root)
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
                self._master._run_script("migrate_storage.py")
                self._master._pause()
            elif choice == 3:
                self._project_info()

    def _storage_paths(self) -> None:
        """Show configured storage paths from StorageConfig."""
        _print_section_header("Storage Paths", "Current data and checkpoint locations")

        tokenizer_path = self._master._resolve_path(self._master.storage.tokenizer_path)
        data_dir = self._master._resolve_path(self._master.storage.data_dir)
        checkpoints_dir = self._master._resolve_path(self._master.storage.checkpoints_dir)

        paths = {
            "Project root":     str(self._master.project_root),
            "Tokenizer":        str(tokenizer_path),
            "Data dir":         str(data_dir),
            "Data processed":   str(data_dir / "processed"),
            "Checkpoints":      str(checkpoints_dir),
            "Configs":          str(self._master.project_root / "configs"),
            "Scripts":          str(self._master.project_root / "scripts"),
            "Python (venv)":    str(self._master.venv_python),
        }

        cli.kv_table(paths, title="Paths")

        # Existence indicators
        cli.print("")
        for label, path_str in paths.items():
            p = Path(path_str)
            if p.exists():
                cli.print(f"  [green]✓[/green] {label}")
            else:
                cli.print(f"  [red]✗[/red] [dim]{label} — not found[/dim]")

        self._master._pause()

    def _project_info(self) -> None:
        """Show project and environment information."""
        _print_section_header("Project Info", "Environment and version details")

        import sys
        info: dict[str, str] = {
            "Python":       sys.version.split()[0],
            "Project root": str(self._master.project_root),
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
        self._master._pause()

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
            self._master._run_script("training_status.py")
        else:
            sizes = ["tiny", "small", "medium", "large"]
            self._master._run_script("training_status.py", ["--size", sizes[choice - 1]])
        self._master._pause()

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
                features = _scan_feature_modules(self._master.project_root)
            except Exception as exc:
                cli.warn(f"Could not scan feature modules: {exc}")
                self._master._pause()
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
                        _save_feature_state(self._master.project_root, feat["name"], True)
                    cli.success(f"Enabled all {n_total} features.")
                    self._master._pause()
                continue

            if choice == len(cat_options) - 1:
                if cli.confirm(f"Disable all {n_total} features?", default=False):
                    for feat in features:
                        _save_feature_state(self._master.project_root, feat["name"], False)
                    cli.warn(f"Disabled all {n_total} features. Core functionality is unaffected.")
                    self._master._pause()
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

            yaml_states = _load_feature_states(self._master.project_root)
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
                        _save_feature_state(self._master.project_root, feat["name"], True)
                    cli.success(f"Enabled all {cat_total} features in {category}.")
                    self._master._pause()
                continue

            if choice == len(feat_options) - 1:
                if cli.confirm(f"Disable all {cat_total} {category} features?", default=False):
                    for feat in cat_features:
                        feat["enabled"] = False
                        _save_feature_state(self._master.project_root, feat["name"], False)
                    cli.warn(
                        f"Disabled all {cat_total} features in {category}. "
                        "Core functionality is unaffected."
                    )
                    self._master._pause()
                continue

            feat = cat_features[choice]
            new_state = not feat["enabled"]
            feat["enabled"] = new_state
            _save_feature_state(self._master.project_root, feat["name"], new_state)

            if new_state:
                cli.success(f"{feat['label']} enabled.")
            else:
                cli.warn(
                    f"{feat['label']} disabled. "
                    "(This is optional — core functionality is unaffected.)"
                )
            # Loop back immediately so user sees updated state
