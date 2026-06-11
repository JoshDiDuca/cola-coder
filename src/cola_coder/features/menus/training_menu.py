"""Training sub-menu for Cola-Coder."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from cola_coder.cli import cli
from cola_coder.features.master_menu import _print_section_header

if TYPE_CHECKING:
    from cola_coder.features.master_menu import MasterMenu
    from cola_coder.features.menus.pipeline_menu import PipelineMenu


class TrainingMenu:
    """Training menu — grouped by pipeline stage for end-to-end model development."""

    def __init__(self, master: MasterMenu) -> None:
        self._master = master
        self._pipeline: PipelineMenu | None = None

    def _get_pipeline_menu(self):
        """Lazy-import pipeline menu to avoid circular imports."""
        if self._pipeline is None:
            from cola_coder.features.menus.pipeline_menu import PipelineMenu
            self._pipeline = PipelineMenu(self._master)
        return self._pipeline

    def menu(self) -> None:
        """Show the training menu — grouped by pipeline stage."""
        while True:
            _print_section_header(
                "Training",
                "End-to-end model development: data → pretrain → post-train → align → eval",
            )

            options = [
                {"label": "Pipeline Manager",
                 "detail": "Named runs with resume, stage override, and state tracking"},
                {"label": "Foundation (Tokenizer & Data)",
                 "detail": "Stage 1-2: Train tokenizer, prepare/mix training data"},
                {"label": "Pre-Training",
                 "detail": "Stage 3: Train base model from scratch or resume"},
                {"label": "Post-Training",
                 "detail": "Stage 4-7: Context extension, SFT, MoE upcycling"},
                {"label": "Alignment & Reasoning",
                 "detail": "Stage 8-9: Semantic routing, GRPO, self-play"},
                {"label": "Monitoring & Tools",
                 "detail": "Dashboard, eval history, VRAM, LR finder"},
            ]

            choice = cli.choose("Training area:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._get_pipeline_menu().menu()
            elif choice == 1:
                self._foundation_menu()
            elif choice == 2:
                self._pretraining_menu()
            elif choice == 3:
                self._post_training_menu()
            elif choice == 4:
                self._alignment_menu()
            elif choice == 5:
                self._monitoring_menu()

    # ── Grouped sub-menus ─────────────────────────────────────────────────

    def _foundation_menu(self) -> None:
        """Foundation: tokenizer and data preparation."""
        while True:
            _print_section_header(
                "Foundation (Stage 1-2)",
                "Tokenizer training and data preparation",
            )

            options = [
                {"label": "Train Tokenizer",
                 "detail": "BPE tokenizer from scratch (vocab 32K-64K)"},
                {"label": "Prepare Data",
                 "detail": "Open the Data Pipeline menu (collect, filter, score, mix)"},
            ]

            choice = cli.choose("Foundation:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._train_tokenizer()
            elif choice == 1:
                self._master._data.menu()

    def _pretraining_menu(self) -> None:
        """Pre-training: base model training."""
        while True:
            _print_section_header(
                "Pre-Training (Stage 3)",
                "Train or resume base model pretraining",
            )

            options = [
                {"label": "Train Model (select size)",
                 "detail": "tiny (50M) / small (125M) / medium (350M) / 4080_max (455M)"},
                {"label": "Resume Training",
                 "detail": "Auto-detect latest checkpoint and continue"},
                {"label": "Background Training",
                 "detail": "Automated overnight/idle training with GPU throttling"},
            ]

            choice = cli.choose("Pre-Training:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._train_size_menu(resume=False)
            elif choice == 1:
                self._resume_training_menu()
            elif choice == 2:
                self._background_training_menu()

    def _post_training_menu(self) -> None:
        """Post-training: context extension, instruction tuning, MoE."""
        while True:
            _print_section_header(
                "Post-Training (Stage 4-7)",
                "Enhance the base model with specialized training",
            )

            options = [
                {"label": "Extend Context Window",
                 "detail": "Stage 4: YaRN RoPE scaling for longer context (e.g. 4x-8x)"},
                {"label": "Generate Instruction Data",
                 "detail": "Stage 5: Create SFT pairs from code (SelfCodeAlign)"},
                {"label": "Instruction Tuning (SFT)",
                 "detail": "Stage 6: Fine-tune on ChatML instruction data"},
                {"label": "MoE Upcycling",
                 "detail": "Stage 7: Convert dense checkpoint to Mixture of Experts"},
                {"label": "Fine-tune Upcycled MoE",
                 "detail": "Stage 7.5: Differentiate experts (low LR, short schedule)"},
            ]

            choice = cli.choose("Post-Training:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._extend_context_menu()
            elif choice == 1:
                self._generate_instructions_menu()
            elif choice == 2:
                self._instruction_tuning_menu()
            elif choice == 3:
                self._moe_upcycling_menu()
            elif choice == 4:
                self._moe_finetune_menu()

    def _alignment_menu(self) -> None:
        """Alignment: routing, reasoning, self-play."""
        while True:
            _print_section_header(
                "Alignment & Reasoning (Stage 8-9)",
                "Domain routing, GRPO reasoning, self-play improvement",
            )

            options = [
                {"label": "Train Semantic Router",
                 "detail": "Stage 8: MLP/Transformer domain classifier (<5M params)"},
                {"label": "Train Reasoning (GRPO)",
                 "detail": "Stage 9: Group RL with <think> tokens and test-based rewards"},
                {"label": "Self-Play Training",
                 "detail": "Iterative generate-test-improve reasoning loop"},
            ]

            choice = cli.choose("Alignment:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._train_router_menu()
            elif choice == 1:
                self._train_reasoning()
            elif choice == 2:
                self._self_play_training_menu()

    def _monitoring_menu(self) -> None:
        """Monitoring and utility tools."""
        while True:
            _print_section_header(
                "Monitoring & Tools",
                "Training dashboards, estimation, and analysis",
            )

            options = [
                {"label": "VRAM Estimation",
                 "detail": "Estimate GPU memory before training"},
                {"label": "Learning Rate Finder",
                 "detail": "Smith's LR range test to find optimal learning rate"},
                {"label": "Training Dashboard (TUI)",
                 "detail": "Real-time Rich dashboard with loss/LR curves"},
                {"label": "Auto-Eval History",
                 "detail": "View evaluation snapshots from training runs"},
            ]

            choice = cli.choose("Monitoring:", options, allow_cancel=True)
            if choice is None:
                return

            if choice == 0:
                self._vram_estimate_menu()
            elif choice == 1:
                self._lr_finder_menu()
            elif choice == 2:
                self._training_dashboard()
            elif choice == 3:
                self._eval_history_menu()

    # ── New menu entries ──────────────────────────────────────────────────

    def _generate_instructions_menu(self) -> None:
        """Generate instruction-tuning data from code."""
        _print_section_header(
            "Generate Instruction Data (Stage 5)",
            "Create SFT pairs from raw code using SelfCodeAlign",
        )

        cli.print(
            "  Extracts functions/classes from code, generates instruction-response\n"
            "  pairs in ChatML format. Three modes:\n"
            "    [cyan]template[/cyan]  — regex-based extraction (fast, no LLM needed)\n"
            "    [cyan]llm[/cyan]       — use an external LLM for higher quality\n"
            "    [cyan]self[/cyan]      — bootstrap from the model's own generations\n"
        )

        source_options = [
            {"label": "HuggingFace dataset",
             "detail": "Generate from HF code dataset (recommended)"},
            {"label": "Local code directory",
             "detail": "Generate from local .py/.ts/.js files"},
            {"label": "Demo mode",
             "detail": "Quick test with built-in sample code"},
        ]
        source_choice = cli.choose("Source:", source_options, allow_cancel=True)
        if source_choice is None:
            return

        sources = ["huggingface", "local", "demo"]
        args: list[str] = ["--non-interactive", "--source", sources[source_choice]]

        if source_choice == 1:
            try:
                path = input("  Path to code directory: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if not path:
                return
            args.extend(["--paths", path])

        mode_options = [
            {"label": "Template (fast, no LLM)",
             "detail": "Regex-based function/class extraction"},
            {"label": "LLM-assisted (higher quality)",
             "detail": "Uses an external LLM to generate instructions"},
            {"label": "Self-instruct (bootstrap)",
             "detail": "Model generates its own instruction pairs"},
        ]
        mode_choice = cli.choose("Generation mode:", mode_options, allow_cancel=True)
        if mode_choice is None:
            return
        modes = ["template", "llm", "self"]
        args.extend(["--mode", modes[mode_choice]])

        try:
            count_str = input("  Number of pairs to generate [default: 1000]: ").strip()
            count = int(count_str) if count_str else 1000
        except (ValueError, EOFError, KeyboardInterrupt):
            count = 1000
        args.extend(["--count", str(count)])

        output_path = "data/sft/instructions.jsonl"
        args.extend(["--output", output_path])

        cli.kv_table({
            "Source": sources[source_choice],
            "Mode": modes[mode_choice],
            "Count": str(count),
            "Output": output_path,
        }, title="Instruction Generation Config")

        if cli.confirm("Start generating instruction data?"):
            self._master._run_script("generate_instructions.py", args)
            self._master._pause()

    def _train_router_menu(self) -> None:
        """Train semantic router model."""
        _print_section_header(
            "Train Semantic Router (Stage 8)",
            "Lightweight domain classifier (<5M params)",
        )

        cli.print(
            "  Routes code to domain-specialist models at inference time.\n"
            "  Trained on code snippets labeled by domain (React, Next.js, etc.).\n"
        )

        arch_options = [
            {"label": "MLP Router (fast, ~100us inference)",
             "detail": "Bag-of-embeddings → MLP → softmax"},
            {"label": "Transformer Router (better quality, ~1ms)",
             "detail": "Embedding → 2 transformer layers → classification"},
        ]

        choice = cli.choose("Router architecture:", arch_options, allow_cancel=True)
        if choice is None:
            return

        arch = "mlp" if choice == 0 else "transformer"
        args = ["--arch", arch]

        # Check if training data already exists
        data_path = Path("data/router_training_data.jsonl")
        if data_path.exists():
            if cli.confirm("Router training data exists. Regenerate?", default=False):
                args.append("--generate-data")
            else:
                args.extend(["--data", str(data_path)])
        else:
            args.append("--generate-data")

        if cli.confirm("Start router training?"):
            self._master._run_script("train_router.py", args)
            self._master._pause()

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
            resume_path = self._master._pick_checkpoint(
                f"Select {size} checkpoint to resume:", model=size,
            )
            if resume_path is None:
                self._master._pause()
                return
            cli.info("Resuming from", resume_path)
            self._master._run_script("train.py", ["--config", config, "--resume", resume_path])
        else:
            # Check for existing and offer to resume
            by_model = self._master._scan_all_checkpoints()
            if by_model.get(size):
                if cli.confirm(f"Found existing {size} checkpoint. Resume training?"):
                    resume_path = self._master._pick_checkpoint(
                        f"Select {size} checkpoint:", model=size,
                    )
                    if resume_path:
                        self._master._run_script(
                            "train.py", ["--config", config, "--resume", resume_path],
                        )
                        self._master._pause()
                        return

            use_wandb = cli.confirm("Enable Weights & Biases logging?", default=False)
            args = ["--config", config]
            if use_wandb:
                args.append("--wandb")

            self._master._run_script("train.py", args)

        self._master._pause()

    def _resume_training_menu(self) -> None:
        """Select a checkpoint to resume training from."""
        _print_section_header("Resume Training", "Continue from a checkpoint")

        ckpt_path = self._master._pick_checkpoint("Select checkpoint to resume:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)
        cli.info("Resuming from", ckpt_path)
        self._master._run_script("train.py", ["--config", config, "--resume", ckpt_path])
        self._master._pause()

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
            str(self._master.venv_python), "scripts/background_train.py",
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

            log_path = self._master.project_root / "logs" / "background_train.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            proc = subprocess.Popen(
                cmd,
                cwd=str(self._master.project_root),
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

        self._master._pause()

    def _stop_background_training(self) -> None:
        """Send stop signal to running background training."""
        from cola_coder.features.background_trainer import send_stop_signal

        # Find the checkpoint dir that has an active lock
        for size in self._master._MODEL_ORDER:
            ckpt_dir = self._master._resolve_path(self._master.storage.checkpoints_dir) / size
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
                self._master._pause()
                return

        # Also check default ./checkpoints/
        default_dir = self._master._resolve_path("checkpoints")
        for size_dir in default_dir.iterdir() if default_dir.exists() else []:
            lock_file = size_dir / ".background_train.lock"
            if lock_file.exists():
                success = send_stop_signal(str(size_dir))
                if success:
                    cli.success("Stop signal sent. Training will save and exit.")
                else:
                    cli.error("Failed to send stop signal.")
                self._master._pause()
                return

        cli.warn("No running background training found.")
        self._master._pause()

    def _show_background_status(self) -> None:
        """Show background training status."""
        from cola_coder.features.background_trainer import (
            is_background_running, read_background_status,
        )

        _print_section_header("Background Training Status", "Current session details")

        found_any = False
        for size in self._master._MODEL_ORDER:
            ckpt_dir = self._master._resolve_path(self._master.storage.checkpoints_dir) / size
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
        log_path = self._master.project_root / "logs" / "background_train.log"
        if log_path.exists():
            size_mb = log_path.stat().st_size / (1024 * 1024)
            cli.info("Log file", f"{log_path} ({size_mb:.1f} MB)")

        self._master._pause()

    def _schedule_overnight_training(self) -> None:
        """Register a Windows Task Scheduler task for overnight training."""
        from cola_coder.features.background_scheduler import WindowsTaskScheduler

        _print_section_header(
            "Schedule Overnight Training",
            "Register a daily task in Windows Task Scheduler",
        )

        scheduler = WindowsTaskScheduler(self._master.project_root)

        if scheduler.is_task_registered():
            cli.warn(
                f"Task '{scheduler.task_name}' already exists. "
                "It will be overwritten."
            )

        # Pick model
        options = [
            {"label": size, "detail": f"configs/{size}.yaml"}
            for size in self._master._MODEL_ORDER
        ]
        choice = cli.choose("Select model for overnight training:", options,
                            allow_cancel=True)
        if choice is None:
            return

        size = self._master._MODEL_ORDER[choice]
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

        self._master._pause()

    def _remove_overnight_schedule(self) -> None:
        """Remove the Windows Task Scheduler overnight training task."""
        from cola_coder.features.background_scheduler import WindowsTaskScheduler

        scheduler = WindowsTaskScheduler(self._master.project_root)

        if not scheduler.is_task_registered():
            cli.dim("No overnight training task is registered.")
            self._master._pause()
            return

        if cli.confirm("Remove the overnight training schedule?"):
            success, msg = scheduler.remove_task()
            if success:
                cli.success(msg)
            else:
                cli.error(msg)

        self._master._pause()

    def _check_any_background_running(self) -> tuple[bool, dict | None]:
        """Check if background training is running for any model."""
        from cola_coder.features.background_trainer import is_background_running

        for size in self._master._MODEL_ORDER:
            ckpt_dir = self._master._resolve_path(self._master.storage.checkpoints_dir) / size
            running, status = is_background_running(str(ckpt_dir))
            if running:
                return True, status

        # Also check default ./checkpoints/
        default_dir = self._master._resolve_path("checkpoints")
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

        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            tokenizer_path = DatasetResolver.get_tokenizer_path()
        except Exception:
            tokenizer_path = self._master._resolve_path(self._master.storage.tokenizer_path)
        if tokenizer_path.exists():
            if not cli.confirm(
                f"{tokenizer_path.name} already exists. Retrain?", default=False
            ):
                return

        self._master._run_script("train_tokenizer.py")
        self._master._pause()

    def _train_reasoning(self) -> None:
        """GRPO reasoning training with optional enhancements."""
        _print_section_header("Train Reasoning (GRPO)", "Fine-tuning with thinking tokens")

        cli.print("  [bold]GRPO[/bold] (Group Relative Policy Optimization)")
        cli.print("  Adds [cyan]<think>[/cyan] / [cyan]</think>[/cyan] chain-of-thought tokens.")
        cli.print("  Generates multiple solutions, tests them, reinforces correct ones.")
        cli.print("")

        ckpt_path = self._master._pick_checkpoint("Select base model checkpoint:")
        if ckpt_path is None:
            return
        args: list[str] = ["--base-checkpoint", ckpt_path]

        # SFT Warmup
        if cli.confirm("Enable SFT warmup phase? (DeepSeek-R1 approach)", default=True):
            args.append("--sft-warmup")
            cli.print("  [green]✓[/green] SFT warmup enabled")

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
            if cli.confirm("Enable curriculum learning? (easy→hard)", default=False):
                args.extend(["--problems", "curriculum"])
            else:
                args.extend(["--problems", "all"])

        if cli.confirm("Start reasoning training?"):
            self._master._run_script("train_reasoning.py", args)
            self._master._pause()

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
            self._master._run_script(
                "vram_estimate.py", ["--config", f"configs/{sizes[choice]}.yaml"]
            )
        else:
            for size in sizes:
                cli.rule(size)
                self._master._run_script(
                    "vram_estimate.py", ["--config", f"configs/{size}.yaml"]
                )

        self._master._pause()

    def _lr_finder_menu(self) -> None:
        """Learning rate finder with batch size selection."""
        _print_section_header("Learning Rate Finder", "Smith's LR Range Test")

        cli.print("  Sweeps learning rate from low to high, plotting loss vs LR.")
        cli.print("  Pick the LR where loss drops fastest (steepest descent).")
        cli.print("")

        options = [
            {"label": "Tiny   (50M)",   "detail": "configs/tiny.yaml"},
            {"label": "Small  (125M)",  "detail": "configs/small.yaml"},
            {"label": "Medium (350M)",  "detail": "configs/medium.yaml"},
            {"label": "4080 Max (455M)", "detail": "configs/4080_max.yaml"},
        ]

        choice = cli.choose("Select model config:", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium", "4080_max"]
        args = ["--config", f"configs/{sizes[choice]}.yaml"]

        # Batch size — smaller to avoid OOM during LR sweep
        cli.dim("  Tip: Use a smaller batch size than training to avoid OOM.")
        try:
            bs_str = input("  Batch size [default: 4]: ").strip()
            bs = int(bs_str) if bs_str else 4
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return
        except ValueError:
            bs = 4
        args += ["--batch-size", str(bs)]

        self._master._run_script("find_lr.py", args)
        self._master._pause()

    def _training_dashboard(self) -> None:
        """Training dashboard with model selection."""
        _print_section_header("Training Dashboard", "Real-time training metrics")

        model = self._master._pick_model("Select model to monitor:")
        if model is None:
            return

        # Build checkpoint dir path
        from pathlib import Path
        ckpt_dir = Path(self._master.storage.checkpoints_dir) / model
        if not ckpt_dir.exists():
            cli.error(f"No checkpoint directory found: {ckpt_dir}")
            cli.dim("Train a model first, then monitor it here.")
            self._master._pause()
            return

        self._master._run_script(
            "training_dashboard.py",
            ["--checkpoint-dir", str(ckpt_dir)],
        )
        self._master._pause()

    def _eval_history_menu(self) -> None:
        """Auto-eval history with model selection."""
        _print_section_header("Auto-Eval History", "View eval snapshots from training runs")

        model = self._master._pick_model("Select model to view history for:")
        if model is None:
            return

        from pathlib import Path
        ckpt_dir = Path(self._master.storage.checkpoints_dir) / model
        if not ckpt_dir.exists():
            cli.error(f"No checkpoint directory found: {ckpt_dir}")
            cli.dim("Train a model first, then view its eval history here.")
            self._master._pause()
            return

        self._master._run_script(
            "training_eval_history.py",
            ["--checkpoint-dir", str(ckpt_dir)],
        )
        self._master._pause()

    # ── New training methods ───────────────────────────────────────────────

    def _instruction_tuning_menu(self) -> None:
        """SFT instruction tuning on ChatML-formatted data."""
        _print_section_header(
            "Instruction Tuning (SFT)",
            "Fine-tune on ChatML instruction data",
        )

        cli.print(
            "  Supervised fine-tuning (SFT) on instruction pairs.\n"
            "  Uses ChatML format: <|im_start|>user / <|im_start|>assistant.\n"
            "  Recommended: 2-3 epochs, lr=2e-5, batch=8.\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select base checkpoint to fine-tune:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        # Instruction data path
        try:
            data_path = input(
                "  Instruction data path (.jsonl) [default: data/instructions.jsonl]: "
            ).strip() or "data/instructions.jsonl"
        except (EOFError, KeyboardInterrupt):
            cli.warn("Cancelled.")
            return

        # Training hyperparams
        epoch_options = [
            {"label": "1 epoch",  "detail": "Fast — risk of under-fitting"},
            {"label": "2 epochs", "detail": "Recommended — balanced"},
            {"label": "3 epochs", "detail": "Thorough — watch for over-fitting"},
        ]
        epoch_choice = cli.choose("Training epochs:", epoch_options, allow_cancel=True)
        if epoch_choice is None:
            return
        epochs = epoch_choice + 1

        lr_options = [
            {"label": "1e-5 (conservative)", "detail": "Safest — minimal forgetting"},
            {"label": "2e-5 (recommended)",  "detail": "Standard SFT LR"},
            {"label": "5e-5 (aggressive)",   "detail": "Faster convergence, more forgetting"},
        ]
        lr_choice = cli.choose("Learning rate:", lr_options, allow_cancel=True)
        if lr_choice is None:
            return
        lr_values = ["1e-5", "2e-5", "5e-5"]
        lr = lr_values[lr_choice]

        cli.kv_table({
            "Base checkpoint": ckpt_path,
            "Config": config,
            "Data": data_path,
            "Epochs": str(epochs),
            "LR": lr,
        }, title="Instruction Tuning Config")

        if not cli.confirm("Start instruction tuning?"):
            return

        args = [
            "--data", data_path,
            "--config", config,
            "--checkpoint", ckpt_path,
            "--epochs", str(epochs),
            "--lr", lr,
        ]
        self._master._run_script("train_sft.py", args)
        self._master._pause()

    def _moe_upcycling_menu(self) -> None:
        """Convert a dense checkpoint to a Mixture-of-Experts model."""
        _print_section_header(
            "MoE Upcycling",
            "Convert dense checkpoint to Mixture-of-Experts",
        )

        cli.print(
            "  MoE upcycling copies dense FFN weights into N experts,\n"
            "  then trains the router from scratch. Increases capacity\n"
            "  with minimal additional compute during inference.\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select dense checkpoint to upcycle:")
        if ckpt_path is None:
            return

        config = self._master._config_for_checkpoint(ckpt_path)

        # Expert count
        expert_options = [
            {"label": "4 experts (2 active)",  "detail": "Lightest — minimal VRAM overhead"},
            {"label": "8 experts (2 active)",  "detail": "Recommended — good capacity/cost"},
            {"label": "16 experts (4 active)", "detail": "Higher capacity — more VRAM"},
        ]
        expert_choice = cli.choose("Number of experts:", expert_options, allow_cancel=True)
        if expert_choice is None:
            return

        expert_configs = [(4, 2), (8, 2), (16, 4)]
        num_experts, num_active = expert_configs[expert_choice]

        # Shared experts
        shared_options = [
            {"label": "0 shared experts", "detail": "Pure MoE — all experts gated"},
            {"label": "1 shared expert",  "detail": "1 always-active dense expert"},
            {"label": "2 shared experts", "detail": "DeepSeek-style 2 shared + N gated"},
        ]
        shared_choice = cli.choose("Shared experts:", shared_options, allow_cancel=True)
        if shared_choice is None:
            return
        num_shared = shared_choice

        cli.kv_table({
            "Source checkpoint": ckpt_path,
            "Total experts": str(num_experts),
            "Active experts": str(num_active),
            "Shared experts": str(num_shared),
        }, title="MoE Upcycling Config")

        if not cli.confirm("Start MoE upcycling?"):
            return

        args = [
            "--config", config,
            "--checkpoint", ckpt_path,
            "--num-experts", str(num_experts),
            "--num-active", str(num_active),
            "--num-shared", str(num_shared),
        ]
        self._master._run_script("upcycle_to_moe.py", args)
        self._master._pause()

    def _moe_finetune_menu(self) -> None:
        """Stage 7.5: fine-tune an upcycled MoE checkpoint to differentiate experts.

        Upcycling (stage 7) copies the dense FFN into every expert, so they
        start identical. A short, low-LR fine-tune differentiates them without
        destroying the inherited dense knowledge (MODEL-003). Derives a
        finetune config (scaled LR + max_steps) and runs `train.py --resume`
        on the MoE checkpoint — the trainer auto-detects MoE from the checkpoint.
        """
        import yaml
        from pathlib import Path

        from cola_coder.model.config import derive_moe_finetune_config

        _print_section_header(
            "Fine-tune Upcycled MoE",
            "Differentiate experts after upcycling (low LR, short schedule)",
        )
        cli.print(
            "  Upcycling copies the dense FFN into every expert (identical at first).\n"
            "  This short, low-LR fine-tune differentiates them. Run AFTER MoE\n"
            "  upcycling, on the upcycled MoE checkpoint (e.g. checkpoints/moe/).\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select upcycled MoE checkpoint:")
        if ckpt_path is None:
            return
        config_path = self._master._config_for_checkpoint(ckpt_path)

        # Recipe presets: (lr_fraction, step_fraction)
        presets = [
            {"label": "Gentle (10% LR, 10% steps)", "detail": "Safest — minimal drift"},
            {"label": "Recommended (10% LR, 15% steps)", "detail": "Good differentiation"},
            {"label": "Aggressive (20% LR, 25% steps)", "detail": "More differentiation, more drift"},
        ]
        pick = cli.choose("Fine-tune recipe:", presets, allow_cancel=True)
        if pick is None:
            return
        lr_frac, step_frac = [(0.1, 0.10), (0.1, 0.15), (0.2, 0.25)][pick]

        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except OSError as exc:
            cli.error(f"Could not read config {config_path}: {exc}")
            return

        derived = derive_moe_finetune_config(raw, lr_fraction=lr_frac, step_fraction=step_frac)

        auto_dir = Path("configs/auto")
        auto_dir.mkdir(parents=True, exist_ok=True)
        derived_path = str(auto_dir / f"{Path(config_path).stem}_moe_ft.yaml")
        Path(derived_path).write_text(
            yaml.safe_dump(derived, sort_keys=False), encoding="utf-8"
        )

        base_tr = raw.get("training", {})
        new_tr = derived.get("training", {})
        cli.kv_table({
            "MoE checkpoint": ckpt_path,
            "Base config": config_path,
            "Derived config": derived_path,
            "Learning rate": f"{new_tr.get('learning_rate')} (was {base_tr.get('learning_rate')})",
            "Max steps": f"{new_tr.get('max_steps')} (was {base_tr.get('max_steps')})",
        }, title="MoE Fine-tune Config")

        if not cli.confirm("Start MoE fine-tuning?"):
            return

        self._master._run_script(
            "train.py", ["--config", derived_path, "--resume", ckpt_path]
        )
        self._master._pause()

    def _self_play_training_menu(self) -> None:
        """Iterative self-play: generate → test → improve reasoning loop."""
        _print_section_header(
            "Self-Play Training",
            "Iterative generate-test-improve loop",
        )

        cli.print(
            "  Self-play training iteratively:\n"
            "    1. Generates candidate solutions\n"
            "    2. Runs test suites to score them\n"
            "    3. Fine-tunes on correct solutions\n"
            "    4. Repeats with harder problems\n"
        )

        ckpt_path = self._master._pick_checkpoint("Select base checkpoint:")
        if ckpt_path is None:
            return

        # Iteration count
        iter_options = [
            {"label": "3 iterations",  "detail": "Quick test — ~1-2 hours"},
            {"label": "5 iterations",  "detail": "Recommended — solid improvement"},
            {"label": "10 iterations", "detail": "Full self-play — ~4-8 hours"},
        ]
        iter_choice = cli.choose("Number of iterations:", iter_options, allow_cancel=True)
        if iter_choice is None:
            return
        iterations = [3, 5, 10][iter_choice]

        # Problem set
        problem_options = [
            {"label": "Built-in (62 problems)", "detail": "HumanEval subset — Python focus"},
            {"label": "Extended (150 problems)", "detail": "HumanEval + TypeScript + math"},
            {"label": "Custom JSONL",            "detail": "Load problems from file"},
        ]
        prob_choice = cli.choose("Problem set:", problem_options, allow_cancel=True)
        if prob_choice is None:
            return

        args = [
            "--base-checkpoint", ckpt_path,
            "--iterations", str(iterations),
            "--reward", "combined",
            "--sft-warmup",
        ]
        if prob_choice == 0:
            args.extend(["--problems", "builtin"])
        elif prob_choice == 1:
            args.extend(["--problems", "extended"])
        elif prob_choice == 2:
            try:
                problems_path = input("  Path to problems.jsonl: ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if problems_path:
                args.extend(["--problems-file", problems_path])

        cli.kv_table({
            "Base checkpoint": ckpt_path,
            "Iterations": str(iterations),
            "Reward": "combined",
        }, title="Self-Play Config")

        if not cli.confirm("Start self-play training?"):
            return

        self._master._run_script("train_reasoning.py", args)
        self._master._pause()

    def _extend_context_menu(self) -> None:
        """Configure YaRN RoPE scaling to extend context window."""
        _print_section_header(
            "Extend Context Window",
            "Apply YaRN RoPE scaling for longer context",
        )

        cli.print(
            "  YaRN (Yet Another RoPE extensioN) scales the RoPE frequencies\n"
            "  to support longer sequences than the model was trained with.\n"
            "  After config update, fine-tune for 1000-2000 steps on long data.\n"
        )

        scale_options = [
            {"label": "2x  (e.g. 2048 → 4096)",   "detail": "factor=2.0 — minimal degradation"},
            {"label": "4x  (e.g. 2048 → 8192)",   "detail": "factor=4.0 — slight perplexity rise"},
            {"label": "8x  (e.g. 4096 → 32768)",  "detail": "factor=8.0 — needs fine-tuning data"},
            {"label": "16x (e.g. 4096 → 65536)",  "detail": "factor=16.0 — significant fine-tuning"},
        ]
        scale_choice = cli.choose("Scale factor:", scale_options, allow_cancel=True)
        if scale_choice is None:
            return

        factors = [2.0, 4.0, 8.0, 16.0]
        factor = factors[scale_choice]

        config_options = [
            {"label": "Tiny   (50M)",    "detail": "configs/tiny.yaml"},
            {"label": "Small  (125M)",   "detail": "configs/small.yaml"},
            {"label": "Medium (299M)",   "detail": "configs/medium.yaml"},
            {"label": "4080 Max (455M)", "detail": "configs/4080_max.yaml"},
        ]
        config_choice = cli.choose("Target config:", config_options, allow_cancel=True)
        if config_choice is None:
            return

        config_names = ["tiny", "small", "medium", "4080_max"]
        config_path = f"configs/{config_names[config_choice]}.yaml"

        cli.rule("Context Extension Instructions")
        cli.print(f"\n  1. Add to [bold]{config_path}[/bold]:\n")
        cli.print("     [cyan]rope_scaling:[/cyan]")
        cli.print("       [cyan]type: yarn[/cyan]")
        cli.print(f"       [cyan]factor: {factor}[/cyan]")
        cli.print("")
        cli.print("  2. Fine-tune for 1000-2000 steps on long sequences:")
        cli.print(
            f"     .venv/Scripts/python scripts/train.py "
            f"--config {config_path} --auto-resume"
        )
        cli.print("")
        cli.print("  3. Collect long-context training data (repos, books, multi-file).")
        cli.print("")

        if cli.confirm("Open config file for editing? (shows path)"):
            cli.info("Config path", str(self._master.project_root / config_path))
            cli.dim("Edit the file manually to add rope_scaling settings above.")

        self._master._pause()

    def _full_pipeline_menu(self) -> None:
        """Launch the full 10-stage training pipeline."""
        _print_section_header(
            "Full Training Pipeline",
            "All 10 stages: collect → prepare → pretrain → SFT → MoE → router → reasoning → eval",
        )

        cli.print(
            "  Runs all training stages in sequence with configurable\n"
            "  stage selection, dry-run mode, and failure recovery.\n"
        )

        config_options = [
            {"label": "Tiny   (50M)",    "detail": "configs/tiny.yaml — fastest for testing"},
            {"label": "Small  (125M)",   "detail": "configs/small.yaml"},
            {"label": "Medium (299M)",   "detail": "configs/medium.yaml"},
            {"label": "4080 Max (455M)", "detail": "configs/4080_max.yaml — recommended"},
        ]
        config_choice = cli.choose("Model config:", config_options, allow_cancel=True)
        if config_choice is None:
            return

        config_names = ["tiny", "small", "medium", "4080_max"]
        config_path = f"configs/{config_names[config_choice]}.yaml"

        mode_options = [
            {"label": "Run all stages",
             "detail": "Full pipeline — stages 1-10"},
            {"label": "Dry run",
             "detail": "Show what would run — no changes"},
            {"label": "Select specific stages",
             "detail": "Enter comma-separated stage numbers (e.g. 3,6,10)"},
            {"label": "Start from stage N",
             "detail": "Resume after a failed stage"},
            {"label": "Skip optional stages (4, 7)",
             "detail": "Skip context extension and MoE upcycling"},
        ]
        mode_choice = cli.choose("Pipeline mode:", mode_options, allow_cancel=True)
        if mode_choice is None:
            return

        args = ["--config", config_path]

        if mode_choice == 0:
            pass  # All stages, no extra args
        elif mode_choice == 1:
            args.append("--dry-run")
        elif mode_choice == 2:
            try:
                stages_raw = input("  Stages (e.g. 1,2,3): ").strip()
            except (EOFError, KeyboardInterrupt):
                cli.warn("Cancelled.")
                return
            if stages_raw:
                args.extend(["--stages", stages_raw])
        elif mode_choice == 3:
            try:
                start_str = input("  Start from stage (1-10): ").strip()
                start_n = int(start_str) if start_str else 1
            except (ValueError, EOFError, KeyboardInterrupt):
                start_n = 1
            args.extend(["--start-from", str(start_n)])
        elif mode_choice == 4:
            args.append("--skip-optional")

        if cli.confirm("Launch full pipeline?"):
            self._master._run_script("full_pipeline.py", args)
            self._master._pause()
