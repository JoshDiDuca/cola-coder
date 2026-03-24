"""Training sub-menu for Cola-Coder."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from cola_coder.cli import cli
from cola_coder.features.master_menu import _print_section_header

if TYPE_CHECKING:
    from cola_coder.features.master_menu import MasterMenu


class TrainingMenu:
    """Training menu — train models, resume, background training, utilities."""

    def __init__(self, master: MasterMenu) -> None:
        self._master = master

    def menu(self) -> None:
        """Show the training menu."""
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
                self._master._run_script("training_dashboard.py")
                self._master._pause()
            elif choice == 8:
                self._master._run_script("training_eval_history.py")
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

        args: list[str] = []

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
            args.append("--problems")
            args.append("builtin")
            if cli.confirm("Enable curriculum learning? (easy→hard)", default=False):
                args.append("--curriculum")

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
        """Learning rate finder."""
        _print_section_header("Learning Rate Finder", "Smith's LR Range Test")

        cli.print("  Sweeps learning rate from low to high, plotting loss vs LR.")
        cli.print("  Pick the LR where loss drops fastest (steepest descent).")
        cli.print("")

        options = [
            {"label": "Tiny   (50M)",   "detail": "configs/tiny.yaml"},
            {"label": "Small  (125M)",  "detail": "configs/small.yaml"},
            {"label": "Medium (350M)",  "detail": "configs/medium.yaml"},
        ]

        choice = cli.choose("Select model config:", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium"]
        self._master._run_script("find_lr.py", ["--config", f"configs/{sizes[choice]}.yaml"])
        self._master._pause()
