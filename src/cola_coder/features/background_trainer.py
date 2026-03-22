"""Background Trainer: automated GPU-throttled training for overnight/idle use.

Wraps the existing Trainer without modifying training math. Controls:
1. GPU clock/power limits (nvidia-smi) — makes kernels slower but identical
2. Sleep between steps — yields GPU time for desktop apps
3. Session management — lock files, status reporting, stop signals

Model output is bit-for-bit identical regardless of throttling. Only pace changes.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

try:
    import pynvml  # type: ignore[import]
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False

logger = logging.getLogger("background_trainer")

# RTX 4080 Super defaults (used for restore)
_DEFAULT_GPU_POWER_WATTS = 320
_NVIDIA_SMI = r"C:\Windows\System32\nvidia-smi.exe"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BackgroundTrainingConfig:
    """Configuration for a background training session."""
    config_path: str                        # YAML config (e.g. "configs/medium.yaml")
    data_path: str | None = None            # Explicit data path, or None for auto-detect
    gpu_clock_mhz: int = 1500              # GPU clock lock (MHz). Default RTX 4080S boost ~2550
    gpu_power_watts: int = 200             # GPU power limit (W). Default RTX 4080S = 320W
    save_every_override: int = 1000        # More aggressive saves for background training
    max_duration_seconds: int | None = None  # Max session duration (None = until max_steps)
    stop_at_time: str | None = None        # Wall-clock stop time (e.g. "07:00")
    max_sleep_ms: int = 200                # Max sleep between steps when throttled
    gpu_busy_threshold: float = 40.0       # Other-process GPU util above this = throttle
    no_throttle: bool = False              # Disable all throttling (full speed)
    lock_file: str = ""                    # Auto-set based on checkpoint dir
    status_file: str = ""                  # Auto-set based on checkpoint dir
    log_file: str = "logs/background_train.log"


# ---------------------------------------------------------------------------
# GPU Throttle Controller
# ---------------------------------------------------------------------------

class GPUThrottleController:
    """Manages GPU clock/power limits and monitors utilization for sleep decisions.

    Two throttling layers (both model-safe):
    1. nvidia-smi clock/power limits — slower kernels, identical math
    2. time.sleep() between steps — yields GPU completely
    """

    def __init__(
        self,
        clock_mhz: int = 1500,
        power_watts: int = 200,
        max_sleep_ms: int = 200,
        busy_threshold: float = 40.0,
    ):
        self.clock_mhz = clock_mhz
        self.power_watts = power_watts
        self.max_sleep_ms = max_sleep_ms
        self.busy_threshold = busy_threshold
        self._is_admin = False
        self._clocks_applied = False
        self._power_applied = False
        self._other_gpu_util: float = 0.0
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitor = threading.Event()

    # -- Clock / Power Management --

    def apply_limits(self) -> dict[str, bool]:
        """Apply GPU clock and power limits. Returns what succeeded."""
        results = {"clock": False, "power": False, "cpu_priority": False}

        # Try clock lock
        ret = self._run_nvidia_smi(["-lgc", str(self.clock_mhz)])
        if ret == 0:
            results["clock"] = True
            self._clocks_applied = True
            logger.info("GPU clocks locked to %d MHz", self.clock_mhz)
        else:
            logger.warning(
                "Failed to lock GPU clocks (need admin). "
                "Training will use sleep-only throttling."
            )

        # Try power limit
        ret = self._run_nvidia_smi(["-pl", str(self.power_watts)])
        if ret == 0:
            results["power"] = True
            self._power_applied = True
            logger.info("GPU power limit set to %d W", self.power_watts)
        else:
            logger.warning("Failed to set GPU power limit (need admin).")

        # CPU priority (always works)
        results["cpu_priority"] = self._set_low_cpu_priority()

        return results

    def restore_defaults(self) -> None:
        """Restore GPU to default clocks and power."""
        if self._clocks_applied:
            ret = self._run_nvidia_smi(["-rgc"])
            if ret == 0:
                logger.info("GPU clocks restored to default.")
            self._clocks_applied = False

        if self._power_applied:
            ret = self._run_nvidia_smi(["-pl", str(_DEFAULT_GPU_POWER_WATTS)])
            if ret == 0:
                logger.info("GPU power limit restored to %d W.", _DEFAULT_GPU_POWER_WATTS)
            self._power_applied = False

    # -- Utilization Monitor --

    def start_monitor(self) -> None:
        """Start background thread that samples GPU utilization."""
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="gpu-monitor"
        )
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        """Stop the monitoring thread."""
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

    def _monitor_loop(self) -> None:
        """Sample GPU utilization every 3 seconds."""
        while not self._stop_monitor.is_set():
            self._other_gpu_util = self._sample_gpu_util()
            self._stop_monitor.wait(3.0)

    def _sample_gpu_util(self) -> float:
        """Read current GPU utilization percentage."""
        # Try pynvml first (fast, no subprocess)
        if _NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.gpu)
            except Exception:
                pass

        # Fallback: nvidia-smi subprocess
        try:
            result = subprocess.run(
                [_NVIDIA_SMI, "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass

        return 0.0

    def compute_sleep_ms(self) -> int:
        """Compute how long to sleep based on other-process GPU utilization.

        Called between training steps (when our GPU kernels aren't running),
        so the utilization reading reflects other processes.
        """
        util = self._other_gpu_util

        # Effectively idle — full speed (within clock limits)
        if util < 15.0:
            return 0

        # Heavy load (game, video editing) — pause entirely
        if util > 90.0:
            return 30_000  # 30 seconds, will be clamped by caller

        # Moderate — linear interpolation
        if util > self.busy_threshold:
            return self.max_sleep_ms

        # Between 15% and busy_threshold — proportional
        ratio = (util - 15.0) / max(1.0, self.busy_threshold - 15.0)
        return int(ratio * self.max_sleep_ms)

    def should_pause(self) -> bool:
        """Return True if GPU is saturated (game running, etc)."""
        return self._other_gpu_util > 90.0

    # -- Helpers --

    @staticmethod
    def _run_nvidia_smi(args: list[str]) -> int:
        """Run nvidia-smi with given args. Returns exit code."""
        try:
            result = subprocess.run(
                [_NVIDIA_SMI] + args,
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                logger.debug("nvidia-smi %s failed: %s", args, result.stderr.strip())
            return result.returncode
        except FileNotFoundError:
            logger.warning("nvidia-smi not found at %s", _NVIDIA_SMI)
            return 1
        except Exception as e:
            logger.warning("nvidia-smi error: %s", e)
            return 1

    @staticmethod
    def _set_low_cpu_priority() -> bool:
        """Set this process to Below Normal CPU priority."""
        try:
            if _PSUTIL_AVAILABLE:
                psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                logger.info("CPU priority set to Below Normal (psutil).")
                return True
        except Exception:
            pass

        # Fallback: ctypes
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetCurrentProcess()
            BELOW_NORMAL = 0x00004000
            kernel32.SetPriorityClass(handle, BELOW_NORMAL)
            logger.info("CPU priority set to Below Normal (ctypes).")
            return True
        except Exception as e:
            logger.warning("Could not set CPU priority: %s", e)
            return False


# ---------------------------------------------------------------------------
# Training Session
# ---------------------------------------------------------------------------

class TrainingSession:
    """Manages a background training run: lock, throttle, status, stop signals."""

    def __init__(self, bg_config: BackgroundTrainingConfig):
        self.config = bg_config
        self._start_time: float = 0.0
        self._throttle: GPUThrottleController | None = None
        self._lock_acquired = False

    # -- Lock File --

    def acquire_lock(self) -> bool:
        """Create lock file atomically. Returns True if acquired."""
        lock_path = Path(self.config.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for stale lock
        if lock_path.exists():
            try:
                data = json.loads(lock_path.read_text())
                old_pid = data.get("pid", -1)
                if self._is_pid_alive(old_pid):
                    logger.error(
                        "Background training already running (PID %d). "
                        "Stop it first or delete %s", old_pid, lock_path
                    )
                    return False
                else:
                    logger.warning("Stale lock file (PID %d dead). Removing.", old_pid)
                    lock_path.unlink()
            except Exception:
                lock_path.unlink(missing_ok=True)

        # Atomic create
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            lock_data = json.dumps({
                "pid": os.getpid(),
                "start_time": datetime.now().isoformat(),
                "config_path": self.config.config_path,
            })
            os.write(fd, lock_data.encode())
            os.close(fd)
            self._lock_acquired = True
            logger.info("Lock acquired: %s", lock_path)
            return True
        except FileExistsError:
            logger.error("Lock file race condition — another process beat us.")
            return False
        except OSError as e:
            logger.error("Failed to create lock file: %s", e)
            return False

    def release_lock(self) -> None:
        """Remove lock file."""
        if self._lock_acquired:
            Path(self.config.lock_file).unlink(missing_ok=True)
            self._lock_acquired = False
            logger.info("Lock released.")

    # -- Status File --

    def write_status(
        self, step: int, loss: float, tokens_per_sec: float = 0.0,
        sleep_ms: int = 0, gpu_util: float = 0.0,
    ) -> None:
        """Write status JSON for monitoring by the menu."""
        status_path = Path(self.config.status_file)
        elapsed = time.time() - self._start_time if self._start_time else 0
        status = {
            "pid": os.getpid(),
            "step": step,
            "loss": round(loss, 4),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "sleep_ms": sleep_ms,
            "gpu_util": round(gpu_util, 1),
            "elapsed_seconds": round(elapsed, 0),
            "config_path": self.config.config_path,
            "gpu_clock_mhz": self.config.gpu_clock_mhz,
            "gpu_power_watts": self.config.gpu_power_watts,
            "updated": datetime.now().isoformat(),
        }
        # Atomic write
        tmp_path = status_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(status, indent=2))
            tmp_path.replace(status_path)
        except Exception as e:
            logger.debug("Failed to write status: %s", e)

    def clear_status(self) -> None:
        """Remove status file on exit."""
        Path(self.config.status_file).unlink(missing_ok=True)

    # -- Stop Conditions --

    def should_stop(self) -> bool:
        """Check all stop conditions."""
        # 1. Stop signal file from menu
        stop_file = Path(self.config.lock_file).parent / ".background_stop"
        if stop_file.exists():
            logger.info("Stop signal received.")
            stop_file.unlink(missing_ok=True)
            return True

        # 2. Duration exceeded
        if self.config.max_duration_seconds is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= self.config.max_duration_seconds:
                logger.info(
                    "Duration limit reached (%d seconds).",
                    self.config.max_duration_seconds,
                )
                return True

        # 3. Wall-clock stop time
        if self.config.stop_at_time:
            now = datetime.now()
            try:
                h, m = map(int, self.config.stop_at_time.split(":"))
                stop_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
                # If stop time is in the past (e.g. "07:00" and it's 23:00),
                # it means tomorrow
                if stop_time <= datetime.now().replace(
                    hour=0, minute=0, second=0
                ) + timedelta(hours=self._start_time_hour()):
                    # We started before midnight, stop is after midnight — skip
                    pass
                if now >= stop_time and (time.time() - self._start_time) > 60:
                    logger.info("Stop time reached (%s).", self.config.stop_at_time)
                    return True
            except (ValueError, AttributeError):
                pass

        return False

    def _start_time_hour(self) -> float:
        """Return the hour we started at."""
        if self._start_time:
            return datetime.fromtimestamp(self._start_time).hour
        return 0.0

    # -- Main Run --

    def run(self) -> None:
        """Execute the full background training session."""
        self._start_time = time.time()

        # 1. Acquire lock
        if not self.acquire_lock():
            return

        try:
            self._run_training()
        finally:
            # Always clean up
            if self._throttle:
                self._throttle.restore_defaults()
                self._throttle.stop_monitor()
            self.release_lock()
            logger.info("Background training session ended.")

    def _run_training(self) -> None:
        """Core training logic — imports here to avoid circular deps."""
        # Late imports to keep module lightweight
        from cola_coder.model.config import Config
        from cola_coder.training.trainer import Trainer
        from cola_coder.training.checkpoint import detect_latest_checkpoint

        # 2. Apply GPU throttle
        if not self.config.no_throttle:
            self._throttle = GPUThrottleController(
                clock_mhz=self.config.gpu_clock_mhz,
                power_watts=self.config.gpu_power_watts,
                max_sleep_ms=self.config.max_sleep_ms,
                busy_threshold=self.config.gpu_busy_threshold,
            )
            results = self._throttle.apply_limits()
            logger.info("Throttle results: %s", results)
            self._throttle.start_monitor()

            # Register cleanup for unexpected exit
            atexit.register(self._throttle.restore_defaults)
        else:
            logger.info("Throttling disabled (--no-throttle).")

        # 3. Load config and override save frequency
        config = Config.from_yaml(self.config.config_path)
        if self.config.save_every_override:
            config.checkpoint.save_every = self.config.save_every_override
            logger.info(
                "Checkpoint save_every overridden to %d",
                self.config.save_every_override,
            )

        # 4. Auto-resume from latest checkpoint
        resume_from = detect_latest_checkpoint(config.checkpoint.output_dir)
        if resume_from:
            logger.info("Resuming from checkpoint: %s", resume_from)
        else:
            logger.info("No existing checkpoint — starting fresh.")

        # 5. Create trainer
        trainer = Trainer(config, resume_from=resume_from)

        # 6. Set step callback for throttling + status + stop
        step_count = [0]  # mutable for closure

        def on_step(step: int, loss: float) -> None:
            step_count[0] = step

            # Throttle: sleep based on GPU utilization
            if self._throttle:
                sleep_ms = self._throttle.compute_sleep_ms()

                # Heavy pause mode (game running, etc)
                while self._throttle and self._throttle.should_pause():
                    logger.info("GPU saturated — pausing training (30s)...")
                    self.write_status(
                        step, loss, sleep_ms=30000,
                        gpu_util=self._throttle._other_gpu_util,
                    )
                    time.sleep(30.0)
                    if self.should_stop():
                        raise KeyboardInterrupt("Stop signal during pause")

                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
            else:
                sleep_ms = 0

            # Write status every 100 steps
            if step % 100 == 0:
                gpu_util = (
                    self._throttle._other_gpu_util if self._throttle else 0.0
                )
                self.write_status(
                    step, loss, sleep_ms=sleep_ms, gpu_util=gpu_util,
                )

            # Check stop conditions
            if self.should_stop():
                raise KeyboardInterrupt("Background training stop condition met")

        trainer._step_callback = on_step

        # 7. Register signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Signal %d received — stopping after checkpoint save.", signum)
            # Create stop file so should_stop() picks it up
            stop_file = Path(self.config.lock_file).parent / ".background_stop"
            stop_file.write_text("signal")

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (OSError, ValueError):
            pass

        # 8. Find data path
        data_path = self.config.data_path
        if not data_path:
            data_path = self._auto_detect_data(config)
        if not data_path:
            logger.error("No training data found. Exiting.")
            return

        # 9. Run training
        logger.info(
            "Starting background training: config=%s, data=%s",
            self.config.config_path, data_path,
        )
        try:
            trainer.train(data_path)
        except KeyboardInterrupt:
            logger.info("Training stopped gracefully.")
        except Exception as e:
            logger.error("Training error: %s", e, exc_info=True)
        finally:
            self.clear_status()

    @staticmethod
    def _auto_detect_data(config) -> str | None:
        """Find the most recent .npy training data file."""
        data_dir = Path("data/processed")
        if not data_dir.exists():
            return None
        npy_files = sorted(data_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime,
                           reverse=True)
        # Prefer train_data.npy
        for f in npy_files:
            if f.name == "train_data.npy":
                return str(f)
        return str(npy_files[0]) if npy_files else None

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Check if a PID is still running."""
        if pid <= 0:
            return False
        if _PSUTIL_AVAILABLE:
            return psutil.pid_exists(pid)
        # Fallback: try to open the process (Windows)
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Helpers for menu integration
# ---------------------------------------------------------------------------

def parse_duration(duration_str: str) -> int | None:
    """Parse duration string like '8h', '30m', '2h30m' to seconds."""
    if not duration_str:
        return None
    total = 0
    current = ""
    for ch in duration_str.strip().lower():
        if ch.isdigit():
            current += ch
        elif ch == "h" and current:
            total += int(current) * 3600
            current = ""
        elif ch == "m" and current:
            total += int(current) * 60
            current = ""
        elif ch == "s" and current:
            total += int(current)
            current = ""
    # If just a number, treat as hours
    if current and total == 0:
        total = int(current) * 3600
    return total if total > 0 else None


def read_background_status(checkpoint_dir: str | Path) -> dict | None:
    """Read the background training status file."""
    status_path = Path(checkpoint_dir) / ".background_status.json"
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text())
    except Exception:
        return None


def is_background_running(checkpoint_dir: str | Path) -> tuple[bool, dict | None]:
    """Check if background training is currently running."""
    lock_path = Path(checkpoint_dir) / ".background_train.lock"
    if not lock_path.exists():
        return False, None
    try:
        data = json.loads(lock_path.read_text())
        pid = data.get("pid", -1)
        if TrainingSession._is_pid_alive(pid):
            status = read_background_status(checkpoint_dir)
            return True, status
        else:
            # Stale lock
            return False, None
    except Exception:
        return False, None


def send_stop_signal(checkpoint_dir: str | Path) -> bool:
    """Send stop signal to background training."""
    stop_path = Path(checkpoint_dir) / ".background_stop"
    try:
        stop_path.write_text(datetime.now().isoformat())
        return True
    except Exception:
        return False
