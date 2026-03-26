"""Interactive pipeline manager menu.

Provides named pipeline runs with state persistence, resume from any stage,
input overrides, and re-run capabilities.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from cola_coder.cli import cli
from cola_coder.features.master_menu import _print_section_header
from cola_coder.pipeline.run_manager import (
    ALL_STAGE_NUMS,
    OPTIONAL_STAGES,
    STAGE_DEFS,
    PipelineRun,
    PipelineRunManager,
)

if TYPE_CHECKING:
    from cola_coder.features.master_menu import MasterMenu

# Default data sources config path — single source of truth for all stage handlers
_DATA_SOURCES_PATH = "configs/data_sources.yaml"

# Status icons for display
_ICON = {
    "completed": "[green]✓[/green]",
    "failed": "[red]✗[/red]",
    "running": "[yellow]▶[/yellow]",
    "skipped": "[dim]⊘[/dim]",
    "pending": "[dim]○[/dim]",
}


class PipelineMenu:
    """Interactive pipeline manager — create, resume, and manage named runs."""

    def __init__(self, master: MasterMenu) -> None:
        self._master = master
        runs_dir = Path(self._master.project_root) / "pipeline_runs"
        self._mgr = PipelineRunManager(runs_dir)

    def menu(self) -> None:
        """Top-level pipeline manager menu."""
        while True:
            _print_section_header(
                "Pipeline Manager",
                "Named pipeline runs with resume, stage override, and state tracking",
            )

            runs = self._mgr.list_runs()
            if runs:
                cli.dim(f"  {len(runs)} saved run(s)")

            options = [
                {"label": "New Pipeline Run",
                 "detail": "Create a named run, select config and stages, then start"},
                {"label": "Resume Pipeline Run",
                 "detail": "Continue a previous run from where it left off"},
                {"label": "View Pipeline Runs",
                 "detail": "List all runs with stage-by-stage status"},
                {"label": "Run Single Stage",
                 "detail": "Execute one specific stage from an existing run"},
                {"label": "Reset to Stage",
                 "detail": "Reset a run back to a specific stage and re-run from there"},
                {"label": "Delete Pipeline Run",
                 "detail": "Remove a saved run and its state file"},
                {"label": "Quick Pipeline (legacy)",
                 "detail": "Run full_pipeline.py directly with stage selection"},
            ]

            choice = cli.choose("Pipeline Manager:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._create_run()
            elif choice == 1:
                self._resume_run()
            elif choice == 2:
                self._view_runs()
            elif choice == 3:
                self._run_single_stage()
            elif choice == 4:
                self._reset_to_stage()
            elif choice == 5:
                self._delete_run()
            elif choice == 6:
                self._legacy_pipeline()

    # ── Create ────────────────────────────────────────────────────────

    def _create_run(self) -> None:
        _print_section_header("New Pipeline Run", "Configure and start a named run")

        # 1. Name
        try:
            name = input("  Run name (e.g. small-v1): ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not name:
            cli.warn("Name required.")
            return
        if self._mgr.exists(name):
            cli.error(f"Run '{name}' already exists. Pick a different name.")
            self._master._pause()
            return

        # 2. Config
        config_options = [
            {"label": "Tiny   (50M)",    "detail": "configs/tiny.yaml — fastest for testing"},
            {"label": "Small  (125M)",   "detail": "configs/small.yaml"},
            {"label": "Medium (299M)",   "detail": "configs/medium.yaml"},
            {"label": "4080 Max (455M)", "detail": "configs/4080_max.yaml — recommended"},
            {"label": "Large  (1B+)",    "detail": "configs/large.yaml — cloud only"},
        ]
        config_choice = cli.choose("Model config:", config_options, allow_cancel=True)
        if config_choice is None:
            return
        config_names = ["tiny", "small", "medium", "4080_max", "large"]
        config_path = f"configs/{config_names[config_choice]}.yaml"

        # 3. Stage selection
        stage_options = []
        preselected = []
        for i, num in enumerate(ALL_STAGE_NUMS):
            defn = STAGE_DEFS[num]
            opt_tag = " (optional)" if num in OPTIONAL_STAGES else ""
            stage_options.append({
                "label": f"Stage {num}: {defn['name']}{opt_tag}",
                "detail": str(defn["description"]),
            })
            if num not in OPTIONAL_STAGES:
                preselected.append(i)

        selected = cli.multi_select(
            "Select stages to include:", stage_options, preselected=preselected,
        )
        if not selected:
            cli.warn("No stages selected.")
            return

        selected_nums = {ALL_STAGE_NUMS[i] for i in selected}
        skip = set(ALL_STAGE_NUMS) - selected_nums

        # 4. Summary
        cli.kv_table({
            "Name": name,
            "Config": config_path,
            "Stages": ", ".join(str(n) for n in sorted(selected_nums)),
            "Skipped": ", ".join(str(n) for n in sorted(skip)) or "none",
        }, title="Pipeline Run")

        if not cli.confirm("Create this pipeline run?"):
            return

        run = self._mgr.create(name, config_path, skip_stages=skip)
        cli.success(f"Created pipeline run: {name}")

        if cli.confirm("Start running now?", default=True):
            self._execute_from(run, self._mgr.next_pending(run))

    # ── Resume ────────────────────────────────────────────────────────

    def _resume_run(self) -> None:
        run = self._pick_run("Select run to resume:")
        if run is None:
            return

        self._show_stage_table(run)

        options = [
            {"label": "Continue from next pending stage",
             "detail": f"Stage {self._mgr.next_pending(run) or '—'}: "
                       f"{self._next_stage_name(run)}"},
            {"label": "Re-run a failed stage",
             "detail": "Pick a specific stage to re-execute"},
            {"label": "Override a stage's input",
             "detail": "Set a custom checkpoint/data path for a stage"},
            {"label": "View run details",
             "detail": "Full stage-by-stage breakdown with artifacts"},
        ]

        choice = cli.choose("Action:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            nxt = self._mgr.next_pending(run)
            if nxt is None:
                cli.success("All stages complete!")
                self._master._pause()
                return
            self._execute_from(run, nxt)
        elif choice == 1:
            self._rerun_stage(run)
        elif choice == 2:
            self._set_override(run)
        elif choice == 3:
            self._show_run_details(run)
            self._master._pause()

    # ── View ──────────────────────────────────────────────────────────

    def _view_runs(self) -> None:
        runs = self._mgr.list_runs()
        if not runs:
            cli.dim("  No pipeline runs found. Create one first.")
            self._master._pause()
            return

        _print_section_header("Pipeline Runs", f"{len(runs)} saved run(s)")

        for run in runs:
            summary = self._mgr.summary_line(run)
            cli.print(f"  {summary}")
            cli.dim(f"    Config: {run.config_path}  |  Updated: {run.updated_at[:10]}")

        self._master._pause()

    # ── Run single stage ──────────────────────────────────────────────

    def _run_single_stage(self) -> None:
        run = self._pick_run("Select run:")
        if run is None:
            return

        self._show_stage_table(run)

        stage_options = []
        for num in ALL_STAGE_NUMS:
            st = run.stages.get(num)
            if st and st.status != "skipped":
                defn = STAGE_DEFS[num]
                icon = _ICON.get(st.status, "?")
                stage_options.append({
                    "label": f"{icon} Stage {num}: {defn['name']}",
                    "detail": f"Status: {st.status}",
                })

        choice = cli.choose("Select stage to run:", stage_options, allow_cancel=True)
        if choice is None:
            return

        # Map choice back to actual stage number
        active_nums = [
            n for n in ALL_STAGE_NUMS
            if run.stages.get(n) and run.stages[n].status != "skipped"
        ]
        stage_num = active_nums[choice]

        # Reset to pending so it can be re-run
        run.stages[stage_num].status = "pending"
        self._mgr.save(run)
        self._execute_stage(run, stage_num)

    # ── Reset to stage ─────────────────────────────────────────────────

    def _reset_to_stage(self) -> None:
        """Reset a run back to a specific stage and optionally re-run."""
        run = self._pick_run("Select run to reset:")
        if run is None:
            return

        # Show current stage status
        _print_section_header("Reset to Stage", f"Run: {run.name}")
        for num in sorted(run.stages):
            st = run.stages[num]
            stage_def = STAGE_DEFS.get(num, {})
            name = stage_def.get("name", f"stage-{num}")
            icon = {"completed": "✓", "failed": "✗", "skipped": "⊘",
                    "running": "▶", "pending": "○"}.get(st.status, "?")
            dur = f"({st.duration_secs:.0f}s)" if st.duration_secs else ""
            cli.dim(f"  {icon} {num:>2}. {name:<26} {st.status:<10} {dur}")

        # Pick which stage to reset to
        stage_options = []
        for num in sorted(run.stages):
            stage_def = STAGE_DEFS.get(num, {})
            name = stage_def.get("name", f"stage-{num}")
            desc = stage_def.get("description", "")
            stage_options.append({
                "label": f"Stage {num}: {name}",
                "detail": desc,
            })

        choice = cli.choose("Reset to which stage?", stage_options, allow_cancel=True)
        if choice is None:
            return

        target_stage = sorted(run.stages)[choice]
        stage_name = STAGE_DEFS.get(target_stage, {}).get("name", f"stage-{target_stage}")

        # Count what will be reset
        reset_count = sum(
            1 for num in run.stages
            if num >= target_stage and run.stages[num].status != "skipped"
        )

        if not cli.confirm(
            f"Reset '{run.name}' to stage {target_stage} ({stage_name})? "
            f"This will reset {reset_count} stage(s) to pending."
        ):
            return

        self._mgr.reset_to_stage(run, target_stage)
        cli.success(
            f"Reset to stage {target_stage} ({stage_name}). "
            f"Stages {target_stage}-10 are now pending."
        )

        # Offer to start running immediately
        if cli.confirm("Start running now?"):
            self._execute_from(run, target_stage)

    # ── Delete ────────────────────────────────────────────────────────

    def _delete_run(self) -> None:
        run = self._pick_run("Select run to delete:")
        if run is None:
            return

        if cli.confirm(f"Delete pipeline run '{run.name}'? This cannot be undone."):
            self._mgr.delete(run.name)
            cli.success(f"Deleted: {run.name}")
        self._master._pause()

    # ── Legacy pipeline ───────────────────────────────────────────────

    def _legacy_pipeline(self) -> None:
        """Delegate to the old full_pipeline.py script."""
        # Import from training_menu to access the existing method
        from cola_coder.features.menus.training_menu import TrainingMenu
        # Create a temporary instance to call the method
        tm = TrainingMenu(self._master)
        tm._full_pipeline_menu()

    # ── Stage execution ───────────────────────────────────────────────

    def _execute_from(self, run: PipelineRun, start: int | None) -> None:
        """Execute stages sequentially from *start* onwards."""
        if start is None:
            cli.success("All stages complete!")
            return

        for num in ALL_STAGE_NUMS:
            if num < start:
                continue
            st = run.stages.get(num)
            if not st or st.status in ("completed", "skipped"):
                continue

            success = self._execute_stage(run, num)
            if not success:
                if not cli.confirm("Stage failed. Continue to next stage?", default=False):
                    cli.dim("  Pipeline paused. Use Resume to continue later.")
                    return

        done = self._mgr.completed_count(run)
        total = self._mgr.total_active(run)
        cli.done(f"Pipeline run '{run.name}' finished: {done}/{total} stages completed.")
        self._master._pause()

    def _execute_stage(self, run: PipelineRun, stage_num: int) -> bool:
        """Execute a single pipeline stage.  Returns True on success."""
        defn = STAGE_DEFS[stage_num]
        cli.print(f"\n  [bold cyan]▶ Stage {stage_num}: {defn['name']}[/bold cyan]")
        cli.dim(f"    {defn['description']}")

        input_path = self._mgr.resolve_input(run, stage_num)
        if input_path:
            cli.info("Input", input_path)

        self._mgr.mark_running(run, stage_num)
        start_time = time.perf_counter()

        try:
            artifact = self._dispatch_stage(run, stage_num, input_path)
            elapsed = time.perf_counter() - start_time
            self._mgr.mark_completed(run, stage_num, artifact=artifact, duration=elapsed)
            cli.success(f"Stage {stage_num} completed ({elapsed:.1f}s)")
            if artifact:
                cli.info("Artifact", artifact)
            return True
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self._mgr.mark_failed(run, stage_num, error=str(e), duration=elapsed)
            cli.error(f"Stage {stage_num} failed ({elapsed:.1f}s)", str(e))
            return False

    def _run_stage_script(self, script: str, args: list[str]) -> None:
        """Run a pipeline stage script via the project venv.

        Unlike master_menu._run_script, this raises RuntimeError on non-zero
        exit so that _execute_stage can catch it and mark the stage as failed.
        Output is teed to the session log file if active.
        """
        cmd = [str(self._master.venv_python), f"scripts/{script}", *args]
        cli.dim(f"Running: {' '.join(cmd)}")
        try:
            from cola_coder.session_log import get_session_log
            session = get_session_log()
            if session is not None:
                result = session.run_and_tee(cmd, cwd=self._master.project_root)
            else:
                result = subprocess.run(cmd, cwd=str(self._master.project_root))
        except KeyboardInterrupt:
            raise RuntimeError(f"{script} interrupted by user")
        if result.returncode != 0:
            raise RuntimeError(f"{script} exited with code {result.returncode}")

    def _dispatch_stage(
        self, run: PipelineRun, stage_num: int, input_path: str,
    ) -> str:
        """Run the actual stage logic.  Returns artifact path on success."""
        from cola_coder.model.config import Config

        config = Config.from_yaml(run.config_path)

        if stage_num == 1:
            return self._stage_collect(run, config)
        elif stage_num == 2:
            return self._stage_prepare(run, config)
        elif stage_num == 3:
            return self._stage_pretrain(run, config)
        elif stage_num == 4:
            return self._stage_extend_context(run, config)
        elif stage_num == 5:
            return self._stage_generate_instructions(run, config, input_path)
        elif stage_num == 6:
            return self._stage_instruction_tune(run, config, input_path)
        elif stage_num == 7:
            return self._stage_upcycle_moe(run, config, input_path)
        elif stage_num == 8:
            return self._stage_train_router(run, config)
        elif stage_num == 9:
            return self._stage_train_reasoning(run, config, input_path)
        elif stage_num == 10:
            return self._stage_evaluate(run, config, input_path)
        else:
            raise ValueError(f"Unknown stage: {stage_num}")

    # ── Individual stage handlers ─────────────────────────────────────

    def _stage_collect(self, run: PipelineRun, config) -> str:
        """Stage 1: Data collection — auto or interactive."""
        from cola_coder.data.dataset_resolver import DatasetResolver

        ds_path = _DATA_SOURCES_PATH
        cfg_path = run.config_path
        dataset_name = DatasetResolver.get_dataset_name(ds_path, config_path=cfg_path)
        dataset_dir = DatasetResolver.get_dataset_dir(ds_path, config_path=cfg_path)

        # ── Tokenizer check (BEFORE data collection) ──────────────────────
        cli.info("Dataset", dataset_name)
        tok_path = DatasetResolver.get_tokenizer_path(ds_path, config_path=cfg_path)

        if not DatasetResolver.tokenizer_exists(ds_path, config_path=cfg_path):
            if cli.confirm(f"No tokenizer found for dataset '{dataset_name}'. Train one now?"):
                self._train_tokenizer_for_run(run)
            else:
                raise RuntimeError(
                    f"Tokenizer required. Train it first with: "
                    f"python scripts/train_tokenizer.py --config {run.config_path!r}"
                )
        else:
            meta = DatasetResolver.get_tokenizer_meta(tok_path)
            trained_at = meta.get("trained_at", "unknown")
            if isinstance(trained_at, str) and len(trained_at) >= 10:
                trained_at = trained_at[:10]
            if cli.confirm(f"Tokenizer found (trained {trained_at}). Retrain? (No = use existing)", default=False):
                self._train_tokenizer_for_run(run)

        # ── Dataset selection: use auto-resolved or pick existing ─────────
        selected_dir = self._select_dataset_dir(dataset_dir, dataset_name)
        if selected_dir is not None:
            # Offer scoring on the existing data
            npy_files = list(selected_dir.glob("*.npy"))
            if npy_files:
                score_options = [
                    {"label": "Full scoring (tsc + eslint + heuristic)",
                     "detail": "Comprehensive quality assessment"},
                    {"label": "Quick scoring (heuristic only)",
                     "detail": "Fast heuristic-based scoring"},
                    {"label": "Skip scoring",
                     "detail": "Use data without quality weights"},
                ]
                score_choice = cli.choose(
                    "Score collected data?", score_options, allow_cancel=True,
                )
                if score_choice == 0:
                    self._run_stage_script("score_data.py", [
                        "--data", str(npy_files[0]),
                        "--tokenizer", str(tok_path),
                        "--scorers", "tsc,eslint,heuristic",
                    ])
                elif score_choice == 1:
                    self._run_stage_script("score_data.py", [
                        "--data", str(npy_files[0]),
                        "--tokenizer", str(tok_path),
                        "--scorers", "heuristic",
                    ])
            return str(selected_dir)

        # Fall through: no existing data selected → collect new data

        collect_options = [
            {"label": "Auto-collect (code + text + math)",
             "detail": "Uses collect_data.py with data_sources.yaml ratios (70/20/10)"},
            {"label": "Code only (standard)",
             "detail": "Uses prepare_data.py with config languages"},
            {"label": "Interactive (Data Pipeline menu)",
             "detail": "Full control — pick sources, languages, filters manually"},
        ]
        choice = cli.choose("Collection mode:", collect_options, allow_cancel=True)
        if choice is None:
            return str(dataset_dir)

        tok_args: list[str] = ["--tokenizer", str(tok_path)] if tok_path.exists() else []

        if choice == 0:
            self._run_stage_script("collect_data.py", [
                "--config", run.config_path,
                "--sources", "code,text,math",
                *tok_args,
            ])
        elif choice == 1:
            self._run_stage_script("prepare_data.py", [
                "--config", run.config_path,
                "--score",
                *tok_args,
            ])
        elif choice == 2:
            self._master._data.menu()

        # Verify data files exist after collection
        new_files = list(dataset_dir.glob("*.npy")) if dataset_dir.exists() else []
        if not new_files:
            raise RuntimeError(
                f"No .npy data files found in {dataset_dir} after collection. "
                "Check the collection output above for errors."
            )

        # ── Malware scan option ──────────────────────────────────────────
        scan_options = [
            {"label": "Full scan (Defender + YARA)",
             "detail": "Recommended for untrusted data"},
            {"label": "Quick scan (YARA only)",
             "detail": "Fast code-specific pattern check"},
            {"label": "Skip scanning",
             "detail": "Trusted data sources only"},
        ]
        scan_choice = cli.choose(
            "Scan collected data for malware?", scan_options, allow_cancel=True,
        )
        if scan_choice is not None and scan_choice < 2:
            from cola_coder.security.scanner import CompositeMalwareScanner

            scan_cfg = {
                "scanners": {"yara": True, "defender": scan_choice == 0},
            }
            scanner = CompositeMalwareScanner.from_config(scan_cfg)
            result = scanner.scan_directory(dataset_dir)
            if not result.is_clean:
                cli.warn(f"Found {len(result.threats)} threat(s)")
                for t in result.threats:
                    cli.error(
                        f"  [{t.severity}] {t.name}: {Path(t.file_path).name}"
                    )
            else:
                cli.success(f"Clean: {result.files_scanned} files scanned")

        return str(dataset_dir)

    @staticmethod
    def _select_dataset_dir(
        auto_dir: Path,
        auto_name: str,
    ) -> Path | None:
        """Let the user pick the auto-resolved dataset or an existing one.

        Scans the parent of *auto_dir* for sibling directories that contain
        ``.npy`` files and presents them as options.

        Returns:
            A Path if the user picked an existing dataset (skip collection).
            None if the user wants to collect new data.
        """
        base_dir = auto_dir.parent
        if not base_dir.exists():
            return None

        # Scan for sibling dataset dirs with .npy files
        siblings: list[tuple[str, Path, int]] = []
        for d in sorted(base_dir.iterdir()):
            if not d.is_dir():
                continue
            npy_files = list(d.glob("*.npy"))
            if npy_files:
                siblings.append((d.name, d, len(npy_files)))

        # Auto-resolved dir may also have data
        auto_npy = list(auto_dir.glob("*.npy")) if auto_dir.exists() else []

        if not siblings and not auto_npy:
            # Nothing exists anywhere — go straight to collection
            return None

        # Build options list
        options: list[dict[str, str]] = []

        if auto_npy:
            options.append({
                "label": f"{auto_name} (auto-resolved)",
                "detail": f"{len(auto_npy)} .npy file(s) — matches current config",
            })

        for name, path, count in siblings:
            if path == auto_dir:
                continue  # Already listed as auto-resolved
            options.append({
                "label": name,
                "detail": f"{count} .npy file(s)",
            })

        options.append({
            "label": "Collect new data",
            "detail": "Download and process fresh data for the current config",
        })

        choice = cli.choose("Select dataset:", options, allow_cancel=True)
        if choice is None:
            # Cancelled — use auto-resolved without re-collecting
            return auto_dir if auto_npy else None

        # Last option = collect new
        if choice == len(options) - 1:
            return None

        # Auto-resolved is first when it has data
        if auto_npy and choice == 0:
            return auto_dir

        # Otherwise map to sibling
        offset = 1 if auto_npy else 0
        _, selected_path, _ = siblings[choice - offset]
        # Skip siblings that are the auto_dir (already handled)
        filtered = [(n, p, c) for n, p, c in siblings if p != auto_dir]
        if choice - offset < len(filtered):
            _, selected_path, _ = filtered[choice - offset]
        return selected_path

    def _train_tokenizer_for_run(self, run: PipelineRun) -> None:
        """Train tokenizer for the current dataset configuration."""
        self._run_stage_script("train_tokenizer.py", ["--config", run.config_path])

    def _stage_prepare(self, run: PipelineRun, config) -> str:
        """Stage 2: Prepare and tokenize data.

        If Stage 1 already produced scored .npy data (code_data.npy + .weights.npy),
        we skip re-downloading and offer to use it directly. Re-preparing from scratch
        only makes sense if you want different filters, chunk size, or languages.
        """
        from cola_coder.data.dataset_resolver import DatasetResolver

        data_dir = DatasetResolver.get_dataset_dir(
            _DATA_SOURCES_PATH, config_path=run.config_path,
        )
        tokenizer = DatasetResolver.get_tokenizer_path(
            _DATA_SOURCES_PATH, config_path=run.config_path,
        )

        # Check if Stage 1 already produced scored data
        existing_npys = sorted(data_dir.glob("*.npy")) if data_dir.exists() else []
        # Filter out .weights.npy and .scores.* files — only count actual data files
        data_npys = [f for f in existing_npys if ".weights" not in f.name and ".scores" not in f.name]
        weights_files = [f for f in existing_npys if ".weights" in f.name]

        if data_npys:
            data_file = data_npys[0]
            has_weights = bool(weights_files)

            cli.info("Existing data", str(data_file.name))
            cli.dim(f"  Path: {data_file}")
            cli.dim(f"  Size: {data_file.stat().st_size / (1024**3):.2f} GB")
            if has_weights:
                cli.dim(f"  Weights: {weights_files[0].name} (scored)")

            prepare_options = [
                {"label": "Use existing data" + (" + weights" if has_weights else ""),
                 "detail": "Skip re-downloading — data is already collected, tokenized" +
                           (", and scored" if has_weights else "")},
                {"label": "Re-prepare from scratch",
                 "detail": "Re-download, re-filter, re-tokenize (use if you changed config)"},
            ]

            if not has_weights:
                prepare_options.insert(1, {
                    "label": "Score existing data",
                    "detail": "Run quality scoring on the existing .npy (generates .weights.npy)",
                })

            choice = cli.choose("Data already exists from Stage 1:", prepare_options)

            if choice == 0:
                # Use existing data as-is
                cli.success(f"Using existing data: {data_file.name}")
                return str(data_dir)

            if not has_weights and choice == 1:
                # Score the existing data
                score_args = [
                    "--data", str(data_file),
                    "--tokenizer", str(tokenizer),
                ]
                self._run_stage_script("score_data.py", score_args)
                return str(data_dir)

            # Fall through to re-prepare

        # Re-prepare from scratch
        args = ["--config", run.config_path]
        if tokenizer.exists():
            args.extend(["--tokenizer", str(tokenizer)])
        args.append("--score")
        self._run_stage_script("prepare_data.py", args)

        # Find the output .npy in the per-dataset directory
        npys = sorted(data_dir.glob("*.npy")) if data_dir.exists() else []
        data_npys = [f for f in npys if ".weights" not in f.name and ".scores" not in f.name]
        return str(data_npys[-1]) if data_npys else str(data_dir)

    def _stage_pretrain(self, run: PipelineRun, config) -> str:
        """Stage 3: Base model pretraining.

        Resolves the training data from the per-dataset directory (from Stage 1/2)
        and passes it explicitly via --data to avoid the legacy 'processed/' lookup.
        """
        from cola_coder.data.dataset_resolver import DatasetResolver

        args = ["--config", run.config_path, "--auto-resume"]

        # Find the .npy data file from per-dataset directory
        data_dir = DatasetResolver.get_dataset_dir(
            _DATA_SOURCES_PATH, config_path=run.config_path,
        )
        if data_dir.exists():
            data_npys = sorted([
                f for f in data_dir.glob("*.npy")
                if ".weights" not in f.name and ".scores" not in f.name
            ])
            if data_npys:
                args.extend(["--data", str(data_npys[0])])
                cli.dim(f"  Training data: {data_npys[0].name} ({data_dir.name}/)")

        self._run_stage_script("train.py", args)
        ckpt_dir = Path(config.checkpoint.output_dir)
        latest = ckpt_dir / "latest"
        return str(latest) if latest.exists() else str(ckpt_dir)

    def _stage_extend_context(self, run: PipelineRun, config) -> str:
        """Stage 4: Context window extension via RoPE scaling."""
        ckpt_dir = Path(config.checkpoint.output_dir)
        latest = ckpt_dir / "latest"

        # Check if rope_scaling is configured
        rope_type = getattr(config.model.rope_scaling, "type", "none")
        rope_factor = getattr(config.model.rope_scaling, "factor", 1.0)

        if rope_type == "none" or rope_factor <= 1.0:
            cli.warn("RoPE scaling not configured in this config.")
            cli.dim("  To enable: add rope_scaling: {type: yarn, factor: 4.0} to config.")
            cli.dim("  Skipping context extension.")
            return str(latest) if latest.exists() else ""

        seq_len = getattr(config.model, "max_seq_len", 2048)
        cli.info("RoPE scaling", f"type={rope_type}, factor={rope_factor}")
        cli.info("Context", f"{seq_len} → {int(seq_len * rope_factor)} tokens")
        cli.dim("  Fine-tune with --auto-resume for context adaptation.")
        cli.dim("  Stop manually after ~1000-2000 steps.")

        if not cli.confirm("Run context extension fine-tune?"):
            return str(latest) if latest.exists() else ""

        self._run_stage_script("train.py", [
            "--config", run.config_path, "--auto-resume",
        ])
        return str(latest) if latest.exists() else ""

    def _stage_generate_instructions(
        self, run: PipelineRun, config, input_path: str,
    ) -> str:
        """Stage 5: Generate instruction-tuning data."""
        output = "data/sft/instructions.jsonl"
        args = [
            "--non-interactive",
            "--source", "huggingface",
            "--mode", "template",
            "--count", "5000",
            "--output", output,
        ]
        self._run_stage_script("generate_instructions.py", args)
        return output

    def _stage_instruction_tune(
        self, run: PipelineRun, config, input_path: str,
    ) -> str:
        """Stage 6: SFT instruction tuning via train_sft.py."""
        # Checkpoint = pretrained model from stage 3/4, NOT stage 5 artifact
        # (input_path from resolve_input walks back to stage 5's JSONL file,
        # which is instruction data, not a model checkpoint)
        ckpt_dir = Path(config.checkpoint.output_dir)
        latest = ckpt_dir / "latest"
        if latest.exists():
            # "latest" is a text file pointing to the real checkpoint dir
            checkpoint = latest.read_text(encoding="utf-8").strip()
        else:
            checkpoint = str(latest)

        instruction_data = "data/sft/instructions.jsonl"
        # Use stage 5 artifact if available (this IS the instructions data)
        st5 = run.stages.get(5)
        if st5 and st5.artifact:
            instruction_data = st5.artifact

        if not Path(instruction_data).exists():
            raise FileNotFoundError(
                f"Instruction data not found at {instruction_data}. "
                "Run Stage 5 (generate-instructions) first."
            )

        args = [
            "--data", instruction_data,
            "--config", run.config_path,
            "--checkpoint", checkpoint,
            "--epochs", "2",
            "--lr", "2e-5",
        ]
        self._run_stage_script("train_sft.py", args)

        # train_sft.py saves to checkpoints/{config_stem}_sft/
        cfg_stem = Path(run.config_path).stem
        sft_dir = Path(f"checkpoints/{cfg_stem}_sft")
        sft_latest = sft_dir / "latest"
        return str(sft_latest) if sft_latest.exists() else str(sft_dir)

    @staticmethod
    def _resolve_checkpoint(
        run: PipelineRun, config, input_path: str,
    ) -> str:
        """Resolve a model checkpoint path for stages that need one.

        ``resolve_input()`` walks back through all completed stages and may
        return a JSONL data file (from stage 5) rather than a model
        checkpoint directory.  This helper validates the path and falls back
        through sensible alternatives:

        1. User override (already in ``input_path`` if set).
        2. SFT checkpoint from stage 6 (``checkpoints/<stem>_sft/``).
        3. Base pretrained checkpoint (``config.checkpoint.output_dir/latest``).
        """
        # If input_path points to a real checkpoint directory, use it
        if input_path:
            p = Path(input_path)
            # If it's a "latest" pointer file, read the actual path
            if p.name == "latest" and p.exists():
                resolved = p.read_text(encoding="utf-8").strip()
                if (Path(resolved) / "model.safetensors").exists():
                    return resolved
            # Accept directories that contain model.safetensors (actual LM checkpoints).
            # Reject router checkpoints, JSONL, or plain data files.
            if p.is_dir() and (p / "model.safetensors").exists():
                return input_path

        # Try SFT checkpoint (stage 6 output)
        cfg_stem = Path(run.config_path).stem
        sft_dir = Path(f"checkpoints/{cfg_stem}_sft")
        sft_latest = sft_dir / "latest"
        if sft_latest.exists():
            return sft_latest.read_text(encoding="utf-8").strip()
        if sft_dir.exists():
            return str(sft_dir)

        # Fallback to base pretrained checkpoint
        ckpt_dir = Path(config.checkpoint.output_dir)
        latest = ckpt_dir / "latest"
        if latest.exists():
            return latest.read_text(encoding="utf-8").strip()
        return str(latest)

    def _stage_upcycle_moe(
        self, run: PipelineRun, config, input_path: str,
    ) -> str:
        """Stage 7: MoE upcycling."""
        checkpoint = self._resolve_checkpoint(run, config, input_path)

        args = [
            "--checkpoint", checkpoint,
            "--config", run.config_path,
        ]
        self._run_stage_script("upcycle_to_moe.py", args)
        moe_dir = Path("checkpoints/moe")
        return str(moe_dir) if moe_dir.exists() else checkpoint

    def _stage_train_router(self, run: PipelineRun, config) -> str:
        """Stage 8: Train semantic router."""
        save_dir = f"checkpoints/router/{run.name}"
        data_path = Path("data/router_training_data.jsonl")

        if data_path.exists():
            cli.info("Router data", f"Using existing {data_path}")
            args = ["--data", str(data_path), "--arch", "mlp", "--save-dir", save_dir]
        else:
            cli.dim("  Generating router training data...")
            args = ["--generate-data", "--arch", "mlp", "--save-dir", save_dir]

        self._run_stage_script("train_router.py", args)
        return save_dir

    def _stage_train_reasoning(
        self, run: PipelineRun, config, input_path: str,
    ) -> str:
        """Stage 9: GRPO reasoning training."""
        checkpoint = self._resolve_checkpoint(run, config, input_path)

        args = [
            "--config", run.config_path,
            "--base-checkpoint", checkpoint,
            "--reward", "combined",
            "--problems", "all",
        ]
        self._run_stage_script("train_reasoning.py", args)

        # train_reasoning.py saves to config.checkpoint.output_dir or
        # checkpoints/reasoning/ — return whichever exists
        reasoning_dir = Path("checkpoints/reasoning")
        if reasoning_dir.exists():
            return str(reasoning_dir)
        return checkpoint

    def _stage_evaluate(
        self, run: PipelineRun, config, input_path: str,
    ) -> str:
        """Stage 10: Full evaluation suite (smoke + HumanEval + quality report)."""
        checkpoint = self._resolve_checkpoint(run, config, input_path)

        # 1. Smoke test
        cli.step(1, 3, "Running smoke test")
        self._master._run_script("smoke_test.py", [
            "--checkpoint", checkpoint,
            "--config", run.config_path,
        ])

        # 2. HumanEval pass@k
        cli.step(2, 3, "Running HumanEval evaluation")
        self._master._run_script("evaluate.py", [
            "--checkpoint", checkpoint,
            "--config", run.config_path,
        ])

        # 3. Quality report
        cli.step(3, 3, "Generating quality report")
        self._master._run_script("quality_report.py", [
            "--checkpoint", checkpoint,
            "--config", run.config_path,
            "--eval",
        ])

        return checkpoint

    # ── UI helpers ────────────────────────────────────────────────────

    def _pick_run(self, prompt: str) -> PipelineRun | None:
        """Let the user select a pipeline run."""
        runs = self._mgr.list_runs()
        if not runs:
            cli.dim("  No pipeline runs found.")
            self._master._pause()
            return None

        options = []
        for run in runs:
            options.append({
                "label": run.name,
                "detail": self._mgr.summary_line(run),
            })

        choice = cli.choose(prompt, options, allow_cancel=True)
        if choice is None:
            return None
        return runs[choice]

    def _show_stage_table(self, run: PipelineRun) -> None:
        """Display a compact stage status table."""
        cli.print("")
        for num in ALL_STAGE_NUMS:
            st = run.stages.get(num)
            if st is None:
                continue
            defn = STAGE_DEFS[num]
            icon = _ICON.get(st.status, "?")
            dur = f"({st.duration_secs:.0f}s)" if st.duration_secs else ""
            override = " [cyan](override)[/cyan]" if st.override else ""
            artifact = f" → {st.artifact}" if st.artifact else ""
            cli.print(
                f"  {icon} {num:2d}. {defn['name']:<25s} "
                f"{st.status:<10s} {dur}{override}{artifact}"
            )
        cli.print("")

    def _show_run_details(self, run: PipelineRun) -> None:
        """Show full details for a pipeline run."""
        _print_section_header(f"Run: {run.name}", run.config_path)

        cli.kv_table({
            "Name": run.name,
            "Config": run.config_path,
            "Created": run.created_at[:19] if run.created_at else "?",
            "Updated": run.updated_at[:19] if run.updated_at else "?",
            "Progress": (
                f"{self._mgr.completed_count(run)}/{self._mgr.total_active(run)}"
            ),
        }, title="Run Info")

        self._show_stage_table(run)

        # Show errors if any
        for num in ALL_STAGE_NUMS:
            st = run.stages.get(num)
            if st and st.error:
                cli.error(f"Stage {num} error", st.error)

    def _rerun_stage(self, run: PipelineRun) -> None:
        """Pick a stage and re-run it."""
        stage_options = []
        rerunnable = []
        for num in ALL_STAGE_NUMS:
            st = run.stages.get(num)
            if st and st.status in ("completed", "failed"):
                defn = STAGE_DEFS[num]
                icon = _ICON.get(st.status, "?")
                stage_options.append({
                    "label": f"{icon} Stage {num}: {defn['name']}",
                    "detail": f"Status: {st.status}",
                })
                rerunnable.append(num)

        if not stage_options:
            cli.dim("  No stages available to re-run.")
            self._master._pause()
            return

        choice = cli.choose("Select stage to re-run:", stage_options, allow_cancel=True)
        if choice is None:
            return

        stage_num = rerunnable[choice]
        run.stages[stage_num].status = "pending"
        run.stages[stage_num].error = ""
        self._mgr.save(run)
        self._execute_stage(run, stage_num)
        self._master._pause()

    def _set_override(self, run: PipelineRun) -> None:
        """Set a custom input override for a stage."""
        stage_options = []
        overrideable = []
        for num in ALL_STAGE_NUMS:
            st = run.stages.get(num)
            if st and st.status != "skipped":
                defn = STAGE_DEFS[num]
                current = f" (current: {st.override})" if st.override else ""
                stage_options.append({
                    "label": f"Stage {num}: {defn['name']}",
                    "detail": f"Override input path{current}",
                })
                overrideable.append(num)

        choice = cli.choose(
            "Select stage to override:", stage_options, allow_cancel=True,
        )
        if choice is None:
            return

        stage_num = overrideable[choice]

        try:
            path = input("  Input path (checkpoint or data file): ").strip()
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        if not path:
            # Clear override
            self._mgr.set_override(run, stage_num, "")
            cli.success(f"Cleared override for stage {stage_num}")
        else:
            self._mgr.set_override(run, stage_num, path)
            cli.success(f"Set override for stage {stage_num}: {path}")
        self._master._pause()

    def _next_stage_name(self, run: PipelineRun) -> str:
        nxt = self._mgr.next_pending(run)
        if nxt is None:
            return "All done"
        return str(STAGE_DEFS[nxt]["name"])
