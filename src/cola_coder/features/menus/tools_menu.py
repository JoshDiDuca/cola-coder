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
                {"label": "Project Memory",
                 "detail": "Init, view, edit, compact, and inspect project memory store"},
                {"label": "Index Repository",
                 "detail": "Build vector index for RAG-based code retrieval"},
                {"label": "Configure Agent Tools",
                 "detail": "Enable/disable code execution, web search, and other agent tools"},
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
            elif choice == 13:
                self._project_memory_menu()
            elif choice == 14:
                self._index_repository_menu()
            elif choice == 15:
                self._configure_agent_tools_menu()

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

    # ── New tool methods ───────────────────────────────────────────────────

    def _project_memory_menu(self) -> None:
        """Project memory store — init, view, edit, compact, stats."""
        while True:
            _print_section_header(
                "Project Memory",
                "Long-term memory store for codebase context",
            )

            options = [
                {"label": "Initialize Memory Store",
                 "detail": "Create or reset the project memory database"},
                {"label": "View Memory",
                 "detail": "Browse stored memories by type and recency"},
                {"label": "Edit Memory Entry",
                 "detail": "Update or delete a specific memory entry"},
                {"label": "Compact Memory",
                 "detail": "Summarize and deduplicate old memories to save space"},
                {"label": "Memory Stats",
                 "detail": "Show entry count, size, oldest/newest entries"},
            ]

            choice = cli.choose("Memory operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._memory_init()
            elif choice == 1:
                self._memory_view()
            elif choice == 2:
                self._memory_edit()
            elif choice == 3:
                self._memory_compact()
            elif choice == 4:
                self._memory_stats()

    def _memory_init(self) -> None:
        """Initialize or reset the project memory store."""
        _print_section_header("Initialize Memory Store", "Create project memory database")

        memory_path = self._master.project_root / "data" / "memory" / "project.db"
        if memory_path.exists():
            if not cli.confirm(
                f"Memory store already exists at {memory_path}. Reset it?", default=False
            ):
                return

        try:
            from cola_coder.memory.manager import MemoryManager
            manager = MemoryManager(str(memory_path.parent))
            manager.initialize()
            cli.success(f"Memory store initialized at {memory_path}")
        except ImportError:
            cli.warn("cola_coder.memory not available — creating directory structure.")
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            cli.info("Memory dir", str(memory_path.parent))
            cli.dim("Install memory dependencies or implement cola_coder.memory module.")
        except Exception as e:
            cli.error(f"Failed to initialize memory: {e}")

        self._master._pause()

    def _memory_view(self) -> None:
        """Browse stored memory entries."""
        _print_section_header("View Memory", "Browse stored project memories")

        try:
            from cola_coder.memory.manager import MemoryManager
            memory_path = self._master.project_root / "data" / "memory"
            manager = MemoryManager(str(memory_path))
            entries = manager.list_recent(limit=20)

            if not entries:
                cli.dim("No memory entries found. Run 'Initialize Memory Store' first.")
                self._master._pause()
                return

            cli.info("Entries", str(len(entries)))
            for i, entry in enumerate(entries[:10], 1):
                cli.rule(f"Entry {i}")
                cli.info("Type",    entry.get("type", "unknown"))
                cli.info("Created", entry.get("created_at", "?"))
                content = str(entry.get("content", ""))
                cli.print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")
                cli.print("")
        except ImportError:
            cli.warn("cola_coder.memory not available.")
            cli.dim("Implement src/cola_coder/memory/manager.py to enable this feature.")
        except Exception as e:
            cli.error(f"Failed to read memory: {e}")

        self._master._pause()

    def _memory_edit(self) -> None:
        """Edit or delete a memory entry."""
        _print_section_header("Edit Memory Entry", "Update or delete a stored memory")

        try:
            entry_id_str = input("Memory entry ID to edit (or 'list' to show IDs): ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        if entry_id_str.lower() == "list":
            self._memory_view()
            return

        if not entry_id_str:
            cli.warn("No entry ID provided.")
            self._master._pause()
            return

        edit_options = [
            {"label": "Delete entry",   "detail": "Permanently remove this memory"},
            {"label": "Edit content",   "detail": "Update the stored text"},
            {"label": "Change type",    "detail": "Re-categorise the memory"},
        ]
        edit_choice = cli.choose("Edit action:", edit_options, allow_cancel=True)
        if edit_choice is None:
            return

        try:
            from cola_coder.memory.updater import MemoryUpdater
            memory_path = self._master.project_root / "data" / "memory"
            updater = MemoryUpdater(str(memory_path))

            if edit_choice == 0:
                if cli.confirm(f"Delete entry '{entry_id_str}'?", default=False):
                    updater.delete(entry_id_str)
                    cli.success("Entry deleted.")
            elif edit_choice == 1:
                try:
                    new_content = input("New content: ").strip()
                except (EOFError, KeyboardInterrupt):
                    cli.warn("Cancelled.")
                    return
                updater.update_content(entry_id_str, new_content)
                cli.success("Content updated.")
            elif edit_choice == 2:
                try:
                    new_type = input("New type (fact/code/decision/note): ").strip()
                except (EOFError, KeyboardInterrupt):
                    cli.warn("Cancelled.")
                    return
                updater.update_type(entry_id_str, new_type)
                cli.success("Type updated.")
        except ImportError:
            cli.warn("cola_coder.memory not available.")
        except Exception as e:
            cli.error(f"Failed to edit memory: {e}")

        self._master._pause()

    def _memory_compact(self) -> None:
        """Compact and deduplicate memory entries."""
        _print_section_header("Compact Memory", "Summarise and deduplicate old memories")

        cli.print(
            "  Compaction:\n"
            "    1. Groups similar memories by embedding similarity\n"
            "    2. Summarises clusters into single entries\n"
            "    3. Archives raw entries older than threshold\n"
        )

        threshold_options = [
            {"label": "7 days",  "detail": "Compact entries older than one week"},
            {"label": "30 days", "detail": "Compact entries older than one month"},
            {"label": "All",     "detail": "Compact all non-pinned entries"},
        ]
        threshold_choice = cli.choose("Compaction threshold:", threshold_options, allow_cancel=True)
        if threshold_choice is None:
            return

        thresholds = [7, 30, None]
        threshold_days = thresholds[threshold_choice]

        if not cli.confirm("Run memory compaction?"):
            return

        try:
            from cola_coder.memory.manager import MemoryManager
            memory_path = self._master.project_root / "data" / "memory"
            manager = MemoryManager(str(memory_path))
            result = manager.compact(older_than_days=threshold_days)
            cli.success(f"Compacted: {result.get('merged', 0)} entries merged, "
                        f"{result.get('archived', 0)} archived.")
        except ImportError:
            cli.warn("cola_coder.memory not available.")
        except Exception as e:
            cli.error(f"Compaction failed: {e}")

        self._master._pause()

    def _memory_stats(self) -> None:
        """Show memory store statistics."""
        _print_section_header("Memory Stats", "Project memory database statistics")

        try:
            from cola_coder.memory.manager import MemoryManager
            memory_path = self._master.project_root / "data" / "memory"
            manager = MemoryManager(str(memory_path))
            stats = manager.stats()
            cli.kv_table({
                "Total entries":   str(stats.get("total", 0)),
                "Pinned":          str(stats.get("pinned", 0)),
                "Oldest entry":    stats.get("oldest", "N/A"),
                "Newest entry":    stats.get("newest", "N/A"),
                "Database size":   stats.get("size_mb", "N/A"),
                "Unique types":    ", ".join(stats.get("types", [])) or "none",
            }, title="Memory Statistics")
        except ImportError:
            cli.warn("cola_coder.memory not available.")
            memory_path = self._master.project_root / "data" / "memory"
            if memory_path.exists():
                size_mb = sum(
                    f.stat().st_size for f in memory_path.rglob("*") if f.is_file()
                ) / 1e6
                cli.info("Memory dir", str(memory_path))
                cli.info("Dir size", f"{size_mb:.2f} MB")
            else:
                cli.dim("Memory store not initialised.")
        except Exception as e:
            cli.error(f"Failed to read stats: {e}")

        self._master._pause()

    def _index_repository_menu(self) -> None:
        """Build a vector index for RAG-based code retrieval."""
        _print_section_header(
            "Index Repository",
            "Build vector index for RAG-based code retrieval",
        )

        cli.print(
            "  Indexes a codebase into a vector store for semantic search.\n"
            "  Enables retrieval-augmented generation (RAG) during inference.\n"
        )

        try:
            repo_path = input(
                "Repository path to index [default: current project]: "
            ).strip() or str(self._master.project_root)
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        index_options = [
            {"label": "Full repository",
             "detail": "Index all source files"},
            {"label": "Source only (src/)",
             "detail": "Index only src/ directory — faster"},
            {"label": "Custom glob pattern",
             "detail": "Enter a glob pattern to match files"},
        ]
        index_choice = cli.choose("Index scope:", index_options, allow_cancel=True)
        if index_choice is None:
            return

        chunk_options = [
            {"label": "Function-level",
             "detail": "Split at function/class boundaries (AST-aware)"},
            {"label": "Fixed 512 tokens",
             "detail": "Sliding window — simpler, works for all languages"},
            {"label": "File-level",
             "detail": "One vector per file — coarse but fast"},
        ]
        chunk_choice = cli.choose("Chunking strategy:", chunk_options, allow_cancel=True)
        if chunk_choice is None:
            return

        chunk_names = ["function", "fixed_512", "file"]
        chunk_strategy = chunk_names[chunk_choice]

        cli.kv_table({
            "Repository": repo_path,
            "Chunking": chunk_strategy,
        }, title="Indexing Config")

        if not cli.confirm("Build vector index?"):
            return

        try:
            from cola_coder.retrieval.indexer import RepositoryIndexer
            indexer = RepositoryIndexer(
                repo_path=repo_path,
                chunk_strategy=chunk_strategy,
                index_dir=str(self._master.project_root / "data" / "vector_index"),
            )
            indexer.build()
            cli.success("Vector index built successfully.")
            cli.info("Index location", str(self._master.project_root / "data" / "vector_index"))
        except ImportError:
            cli.warn("cola_coder.retrieval not available.")
            cli.dim("Implement src/cola_coder/retrieval/indexer.py to enable indexing.")
            cli.dim(
                f"Would index: {repo_path}\n"
                f"  Strategy: {chunk_strategy}\n"
                f"  Output: data/vector_index/"
            )
        except Exception as e:
            cli.error(f"Indexing failed: {e}")

        self._master._pause()

    def _configure_agent_tools_menu(self) -> None:
        """Enable/disable agent tools (code execution, web search, etc.)."""
        _print_section_header(
            "Configure Agent Tools",
            "Enable or disable tools available to the agent",
        )

        cli.print(
            "  Agent tools extend the model's capabilities during inference.\n"
            "  Each tool can be enabled/disabled independently.\n"
        )

        tool_definitions = [
            {
                "name": "python_exec",
                "label": "Python Execution",
                "detail": "Execute Python code snippets and return output",
                "risk": "MEDIUM — sandboxed, but can access filesystem",
            },
            {
                "name": "typescript_exec",
                "label": "TypeScript Execution",
                "detail": "Run TypeScript via ts-node — requires Node.js",
                "risk": "MEDIUM — requires ts-node/Node.js",
            },
            {
                "name": "web_search",
                "label": "Web Search",
                "detail": "Search the web for documentation and examples",
                "risk": "LOW — read-only, external requests",
            },
            {
                "name": "file_read",
                "label": "File Read",
                "detail": "Read files from the project directory",
                "risk": "LOW — read-only filesystem access",
            },
            {
                "name": "file_write",
                "label": "File Write",
                "detail": "Write/modify files in the project directory",
                "risk": "HIGH — can modify source files",
            },
            {
                "name": "shell_exec",
                "label": "Shell Execution",
                "detail": "Run shell commands (git, npm, pip, etc.)",
                "risk": "HIGH — full shell access",
            },
        ]

        # Show current states
        try:
            from cola_coder.tools.registry import ToolRegistry
            registry = ToolRegistry()
            current_states = {t["name"]: registry.is_enabled(t["name"]) for t in tool_definitions}
        except ImportError:
            current_states = {t["name"]: False for t in tool_definitions}

        cli.print("")
        for tool in tool_definitions:
            state = "[green]ON [/green]" if current_states.get(tool["name"]) else "[red]OFF[/red]"
            cli.print(
                f"  {state}  {tool['label']:<22s}  "
                f"[dim]{tool['risk']}[/dim]"
            )
        cli.print("")

        tool_options = [
            {"label": t["label"], "detail": f"{t['detail']}  |  Risk: {t['risk']}"}
            for t in tool_definitions
        ]
        tool_options.append({"label": "Enable All",  "detail": "Turn on all agent tools"})
        tool_options.append({"label": "Disable All", "detail": "Turn off all agent tools"})

        tool_choice = cli.choose("Select tool to toggle:", tool_options, allow_cancel=True)
        if tool_choice is None:
            return

        if tool_choice == len(tool_definitions):
            # Enable all
            if cli.confirm("Enable ALL agent tools? (includes high-risk tools)", default=False):
                try:
                    from cola_coder.tools.registry import ToolRegistry
                    reg = ToolRegistry()
                    for t in tool_definitions:
                        reg.set_enabled(t["name"], True)
                    cli.success("All agent tools enabled.")
                except ImportError:
                    cli.warn("cola_coder.tools not available — update configs/features.yaml manually.")
        elif tool_choice == len(tool_definitions) + 1:
            # Disable all
            if cli.confirm("Disable ALL agent tools?", default=True):
                try:
                    from cola_coder.tools.registry import ToolRegistry
                    reg = ToolRegistry()
                    for t in tool_definitions:
                        reg.set_enabled(t["name"], False)
                    cli.success("All agent tools disabled.")
                except ImportError:
                    cli.warn("cola_coder.tools not available — update configs/features.yaml manually.")
        else:
            tool = tool_definitions[tool_choice]
            current = current_states.get(tool["name"], False)
            new_state = not current
            action = "Enable" if new_state else "Disable"

            if tool.get("risk", "").startswith("HIGH") and new_state:
                if not cli.confirm(
                    f"[bold red]WARNING[/bold red]: {tool['label']} has HIGH risk. Enable?",
                    default=False,
                ):
                    self._master._pause()
                    return

            try:
                from cola_coder.tools.registry import ToolRegistry
                reg = ToolRegistry()
                reg.set_enabled(tool["name"], new_state)
                if new_state:
                    cli.success(f"{tool['label']} enabled.")
                else:
                    cli.warn(f"{tool['label']} disabled.")
            except ImportError:
                cli.warn(f"cola_coder.tools not available — {action} '{tool['name']}' manually.")

        self._master._pause()
