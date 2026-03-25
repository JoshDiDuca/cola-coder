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
    "Training Diagnostics": [
        "activation_monitor", "batch_size_finder", "grad_accum_calculator",
        "grad_accum_monitor", "gradient_flow", "gradient_noise",
        "hyperparam_logger", "loss_component_analyzer", "loss_landscape",
        "lr_range_test", "plateau_detector", "progress_estimator",
        "stability_monitor", "training_anomaly_detector", "training_efficiency",
        "training_log_parser", "training_summary",
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
        "experiment_tracker", "data_versioning", "dataset_inspector",
        "model_card_generator", "onnx_export", "quantization",
        "knowledge_distillation", "lora_qlora",
    ],
    "Checkpoint Tools": [
        "checkpoint_comparison", "checkpoint_leaderboard",
        "checkpoint_converter", "checkpoint_health",
        "checkpoint_merger", "checkpoint_scheduler",
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
    "Code Quality": [
        "code_dedup_checker", "code_entropy", "code_normalizer",
        "code_pattern_miner", "code_scorer", "code_similarity",
        "code_smell_detector", "code_style_analyzer", "comment_quality",
        "complexity_heatmap", "docstring_scorer", "formatting_standardizer",
        "readability_scorer", "syntax_error_classifier",
        "type_annotation_scorer", "variable_name_quality",
    ],
    "Model Analysis": [
        "architecture_visualizer", "attention_analyzer", "attention_patterns",
        "confidence_calibrator", "embedding_analyzer", "inference_profiler",
        "memory_profiler", "model_comparison_dashboard", "model_fingerprint",
        "model_size_estimator", "param_counter", "pruning_analyzer",
        "weight_init_analyzer", "adaptive_computation",
        "position_interpolation", "sparse_attention", "perplexity_analyzer",
    ],
    "Tokenizer": [
        "tokenizer_coverage", "tokenizer_debugger", "multilang_tokenizer_eval",
        "token_frequency", "token_stats_tracker", "token_merging",
        "vocab_efficiency",
    ],
    "Data Quality": [
        "data_augmentation", "data_balancer", "data_leakage_detector",
        "data_quality_report", "cv_splitter", "repetition_detector",
    ],
    "Experiment Tracking": [
        "benchmark_store", "cost_estimator", "experiment_comparator",
        "distillation_helper", "background_scheduler", "background_trainer",
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

        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            tokenizer_path = DatasetResolver.get_tokenizer_path()
        except Exception:
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
                {"label": "Instruction Tuning",
                 "detail": "SFT data generation, instruction fine-tuning, GRPO"},
                {"label": "Generate & Interact",
                 "detail": "Code generation, interactive chat, serve API"},
                {"label": "Evaluate & Benchmark",
                 "detail": "HumanEval, benchmarks, checkpoint comparisons"},
                {"label": "Router & Specialists",
                 "detail": "Domain router, MoE, specialist training & management"},
                {"label": "Tools & Utilities",
                 "detail": "Lint, test, GPU status, dataset inspection, export"},
                {"label": "Project Memory",
                 "detail": "Long-term memory store — init, view, compact, stats"},
                {"label": "Retrieval & Search",
                 "detail": "RAG configuration, vector index, semantic search"},
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
                self.instruction_tuning_menu,
                self.generate_menu,
                self._eval.menu,
                self.router_menu,
                self._tools.menu,
                self._tools._project_memory_menu,
                self._retrieval_search_menu,
                self._tools.settings_menu,
                self._tools.training_status_menu,
            ]

            handlers[choice]()

    # ── Retrieval & Search ────────────────────────────────────────────────

    def _retrieval_search_menu(self) -> None:
        """Retrieval & Search sub-menu — RAG config, vector index, semantic search."""
        while True:
            _print_section_header(
                "Retrieval & Search",
                "RAG configuration, vector index, and semantic code search",
            )

            options = [
                {"label": "Index Repository",
                 "detail": "Build vector index for semantic code retrieval"},
                {"label": "Semantic Search",
                 "detail": "Search indexed codebase by natural language query"},
                {"label": "RAG Configuration",
                 "detail": "Configure retrieval-augmented generation settings"},
                {"label": "Vector Store Stats",
                 "detail": "Show index size, document count, embedding model"},
            ]

            choice = cli.choose("Retrieval operation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._tools._index_repository_menu()
            elif choice == 1:
                self._semantic_search_menu()
            elif choice == 2:
                self._rag_config_menu()
            elif choice == 3:
                self._vector_store_stats()

    def _semantic_search_menu(self) -> None:
        """Search the indexed codebase by natural language query."""
        _print_section_header(
            "Semantic Search",
            "Natural language search over indexed codebase",
        )

        try:
            query = input("Search query: ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        if not query:
            cli.warn("No query entered.")
            return

        try:
            top_k_str = input("Number of results [default: 5]: ").strip()
            top_k = int(top_k_str) if top_k_str else 5
        except (ValueError, EOFError, KeyboardInterrupt):
            top_k = 5

        cli.info("Query", query)
        cli.info("Top K", str(top_k))

        try:
            from cola_coder.retrieval.rag import RAGRetriever
            index_dir = str(self.project_root / "data" / "vector_index")
            retriever = RAGRetriever(index_dir=index_dir)
            results = retriever.search(query=query, top_k=top_k)

            if not results:
                cli.warn("No results found. Run 'Index Repository' first.")
            else:
                for i, result in enumerate(results, 1):
                    cli.rule(f"Result {i} (score: {result.get('score', 0):.3f})")
                    cli.info("File", result.get("file_path", "?"))
                    content = result.get("content", "")
                    cli.print(f"  {content[:400]}{'...' if len(content) > 400 else ''}")
                    cli.print("")
        except ImportError:
            cli.warn("cola_coder.retrieval not available.")
            cli.dim("Implement src/cola_coder/retrieval/rag.py to enable search.")
        except Exception as e:
            cli.error(f"Search failed: {e}")

        self._pause()

    def _rag_config_menu(self) -> None:
        """Configure retrieval-augmented generation settings."""
        _print_section_header(
            "RAG Configuration",
            "Configure retrieval-augmented generation",
        )

        rag_options = [
            {"label": "Top-K = 3",  "detail": "Retrieve 3 snippets per query — low context overhead"},
            {"label": "Top-K = 5",  "detail": "Retrieve 5 snippets — recommended"},
            {"label": "Top-K = 10", "detail": "Retrieve 10 snippets — high context, slower"},
        ]
        rag_choice = cli.choose("Top-K retrieval count:", rag_options, allow_cancel=True)
        if rag_choice is None:
            return

        top_k = [3, 5, 10][rag_choice]

        embed_options = [
            {"label": "sentence-transformers/all-MiniLM-L6-v2",
             "detail": "Fast, small — 384-dim, good for code"},
            {"label": "microsoft/codebert-base",
             "detail": "Code-specific — better for code retrieval"},
            {"label": "BAAI/bge-small-en-v1.5",
             "detail": "High quality general embeddings"},
        ]
        embed_choice = cli.choose("Embedding model:", embed_options, allow_cancel=True)
        if embed_choice is None:
            return

        embed_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "microsoft/codebert-base",
            "BAAI/bge-small-en-v1.5",
        ]
        embed_model = embed_models[embed_choice]

        cli.kv_table({
            "Top-K": str(top_k),
            "Embedding model": embed_model,
        }, title="RAG Configuration")

        if cli.confirm("Save RAG configuration?"):
            # Write to configs/rag.yaml
            rag_config_path = self.project_root / "configs" / "rag.yaml"
            try:
                import yaml
                config_data = {
                    "retrieval": {
                        "top_k": top_k,
                        "embedding_model": embed_model,
                        "index_dir": "data/vector_index",
                    }
                }
                with open(rag_config_path, "w", encoding="utf-8") as fh:
                    yaml.dump(config_data, fh, default_flow_style=False)
                cli.success(f"RAG config saved to {rag_config_path}")
            except ImportError:
                cli.warn("PyYAML not available — config not saved.")
            except Exception as e:
                cli.error(f"Failed to save config: {e}")

        self._pause()

    def _vector_store_stats(self) -> None:
        """Show vector store statistics."""
        _print_section_header("Vector Store Stats", "Index size, documents, embedding model")

        index_dir = self.project_root / "data" / "vector_index"

        if not index_dir.exists():
            cli.warn("No vector index found.")
            cli.dim("Run 'Index Repository' to build one.")
            self._pause()
            return

        try:
            from cola_coder.retrieval.vector_store import VectorStore
            store = VectorStore(str(index_dir))
            stats = store.stats()
            cli.kv_table({
                "Documents":        str(stats.get("document_count", "?")),
                "Embedding dim":    str(stats.get("embedding_dim", "?")),
                "Embedding model":  stats.get("embedding_model", "?"),
                "Index size":       stats.get("size_mb", "?"),
                "Last updated":     stats.get("last_updated", "?"),
            }, title="Vector Store Statistics")
        except ImportError:
            # Fallback: just show directory info
            size_mb = sum(
                f.stat().st_size for f in index_dir.rglob("*") if f.is_file()
            ) / 1e6
            files = list(index_dir.iterdir())
            cli.kv_table({
                "Index directory": str(index_dir),
                "Files":           str(len(files)),
                "Total size":      f"{size_mb:.2f} MB",
            }, title="Vector Store (filesystem view)")
            cli.dim("Install cola_coder.retrieval for detailed stats.")
        except Exception as e:
            cli.error(f"Failed to read stats: {e}")

        self._pause()

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
                from cola_coder.data.dataset_resolver import DatasetResolver
                self._run_script("prepare_data.py", [
                    "--config", "configs/tiny.yaml",
                    "--tokenizer", str(DatasetResolver.get_tokenizer_path()),
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
            from cola_coder.data.dataset_resolver import DatasetResolver
            self._run_script("prepare_data.py", [
                "--config", "configs/tiny.yaml",
                "--tokenizer", str(DatasetResolver.get_tokenizer_path()),
            ])
        elif choice == 3:
            self._run_script("train.py", ["--config", "configs/tiny.yaml"])

        self._pause()

    # ── 3b. Instruction Tuning ──────────────────────────────────────────

    def instruction_tuning_menu(self) -> None:
        """Instruction tuning pipeline: data generation, SFT, GRPO."""
        while True:
            _print_section_header(
                "Instruction Tuning",
                "Transform base model into instruction-following assistant",
            )

            options = [
                {"label": "Generate SFT Data",
                 "detail": "Extract instruction/response pairs from code"},
                {"label": "Train SFT (Instruction Tune)",
                 "detail": "Supervised fine-tuning with ChatML format"},
                {"label": "Train GRPO (Reinforcement Learning)",
                 "detail": "Group RL with execution-based rewards"},
                {"label": "Full Pipeline (Data → SFT → GRPO)",
                 "detail": "Run all stages in sequence"},
            ]

            choice = cli.choose(
                "Instruction Tuning", options, allow_cancel=True
            )
            if choice is None:
                break

            if choice == 0:
                self._generate_sft_data()
            elif choice == 1:
                self._train_sft()
            elif choice == 2:
                self._run_script("train_reasoning.py", [
                    "--config", "configs/tiny.yaml",
                    "--reward", "combined",
                    "--problems", "all",
                ])
            elif choice == 3:
                self._instruction_pipeline()

            self._pause()

    def _generate_sft_data(self) -> None:
        """Interactive SFT data generation."""
        cli.step(1, 3, "Configure data source")
        source = cli.choose("Source code directory:", [
            {"label": "Training data directory",
             "detail": f"{self.storage.data_dir}"},
            {"label": "Custom path",
             "detail": "Specify a directory of .py/.ts/.js files"},
        ])
        if source is None:
            return

        if source == 0:
            source_path = str(self._resolve_path(self.storage.data_dir))
        else:
            source_path = input("Enter source path: ").strip()
            if not source_path:
                return

        cli.step(2, 3, "Configure output")
        output_path = "data/sft/instructions.jsonl"
        cli.info("Output", output_path)

        cli.step(3, 3, "Generating...")
        self._run_script("generate_sft_data.py", [
            "--source", source_path,
            "--output", output_path,
            "--num-samples", "1000",
        ])

    def _train_sft(self) -> None:
        """Interactive SFT training setup."""
        # Find available configs
        configs_dir = self.project_root / "configs"
        configs = sorted(configs_dir.glob("*.yaml"))
        config_names = [c.stem for c in configs
                        if c.stem not in ("features", "storage", "reasoning", "sft")]

        config_choice = cli.choose("Model config:", [
            {"label": name, "detail": str(c)}
            for name, c in zip(config_names, configs)
            if name in config_names
        ], allow_cancel=True)
        if config_choice is None:
            return

        config_path = f"configs/{config_names[config_choice]}.yaml"

        # Find checkpoint
        checkpoint_dir = self._resolve_path("checkpoints") / config_names[config_choice]
        if (checkpoint_dir / "latest").exists():
            checkpoint = str(checkpoint_dir / "latest")
        else:
            checkpoint = input("Enter checkpoint path: ").strip()
            if not checkpoint:
                return

        # Data path
        data_path = "data/sft/instructions.jsonl"
        if not (self.project_root / data_path).exists():
            cli.warn(f"SFT data not found at {data_path}")
            cli.info("Run", "'Generate SFT Data' first")
            return

        self._run_script("train_sft.py", [
            "--data", data_path,
            "--config", config_path,
            "--checkpoint", checkpoint,
        ])

    def _instruction_pipeline(self) -> None:
        """Run the full instruction tuning pipeline."""
        cli.header("Cola-Coder", "Full Instruction Tuning Pipeline")
        cli.info("Stage 1/3", "Generating SFT data...")
        self._generate_sft_data()

        data_path = self.project_root / "data" / "sft" / "instructions.jsonl"
        if not data_path.exists():
            cli.error("SFT data generation failed. Stopping pipeline.")
            return

        cli.info("Stage 2/3", "Training SFT...")
        self._train_sft()

        cli.info("Stage 3/3", "GRPO training...")
        cli.info("Note", "GRPO uses the SFT checkpoint as starting point")
        self._run_script("train_reasoning.py", [
            "--config", "configs/tiny.yaml",
            "--reward", "combined",
            "--problems", "all",
        ])

        cli.done("Instruction tuning pipeline complete!")

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
            from cola_coder.data.dataset_resolver import DatasetResolver
            self._run_script("generate_router_data.py", [
                "--source", "data/processed/train_data.npy",
                "--tokenizer", str(DatasetResolver.get_tokenizer_path()),
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
