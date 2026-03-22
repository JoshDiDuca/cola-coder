"""Windows Task Scheduler integration for overnight background training.

Uses schtasks.exe (built into Windows, no admin required for user-level tasks).
Registers a daily task that launches background_train.py at a configurable time.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

FEATURE_ENABLED = True

logger = logging.getLogger(__name__)

# Default task name in Windows Task Scheduler
TASK_NAME = "ColaCoder-BackgroundTraining"


def is_enabled() -> bool:
    return FEATURE_ENABLED


class WindowsTaskScheduler:
    """Manages a Windows Task Scheduler task for overnight training."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.task_name = TASK_NAME

        # Find Python executable
        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            self.python_path = str(venv_python)
        else:
            self.python_path = sys.executable

        self.script_path = str(
            self.project_root / "scripts" / "background_train.py"
        )

    def create_overnight_task(
        self,
        config_path: str,
        start_time: str = "22:00",
        stop_time: str = "07:00",
        gpu_clock: int = 1500,
        gpu_power: int = 200,
        save_every: int = 1000,
    ) -> tuple[bool, str]:
        """Register a daily scheduled task for overnight training.

        Args:
            config_path: YAML config path (e.g. "configs/medium.yaml").
            start_time: Time to start training (HH:MM, 24-hour).
            stop_time: Time to stop training (HH:MM, 24-hour).
            gpu_clock: GPU clock limit in MHz.
            gpu_power: GPU power limit in Watts.
            save_every: Checkpoint save interval in steps.

        Returns:
            (success, message) tuple.
        """
        # Build the command the task will run
        cmd = (
            f'"{self.python_path}" "{self.script_path}" '
            f'--config "{config_path}" '
            f'--stop-at {stop_time} '
            f'--gpu-clock {gpu_clock} '
            f'--gpu-power {gpu_power} '
            f'--save-every {save_every}'
        )

        # schtasks /Create
        schtasks_args = [
            "schtasks", "/Create",
            "/TN", self.task_name,
            "/TR", cmd,
            "/SC", "DAILY",
            "/ST", start_time,
            "/F",  # Force overwrite if exists
        ]

        try:
            result = subprocess.run(
                schtasks_args,
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                msg = (
                    f"Scheduled task '{self.task_name}' created.\n"
                    f"Starts daily at {start_time}, stops at {stop_time}.\n"
                    f"GPU: {gpu_clock} MHz / {gpu_power}W\n"
                    f"Verify in Task Scheduler (taskschd.msc)."
                )
                logger.info(msg)
                return True, msg
            else:
                msg = f"schtasks failed: {result.stderr.strip()}"
                logger.error(msg)
                return False, msg
        except FileNotFoundError:
            msg = "schtasks.exe not found — Windows Task Scheduler unavailable."
            logger.error(msg)
            return False, msg
        except subprocess.TimeoutExpired:
            msg = "schtasks timed out."
            logger.error(msg)
            return False, msg

    def remove_task(self) -> tuple[bool, str]:
        """Unregister the scheduled task."""
        try:
            result = subprocess.run(
                ["schtasks", "/Delete", "/TN", self.task_name, "/F"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                msg = f"Scheduled task '{self.task_name}' removed."
                logger.info(msg)
                return True, msg
            else:
                msg = f"Failed to remove task: {result.stderr.strip()}"
                logger.error(msg)
                return False, msg
        except Exception as e:
            msg = f"Error removing task: {e}"
            logger.error(msg)
            return False, msg

    def is_task_registered(self) -> bool:
        """Check if our task exists in Task Scheduler."""
        try:
            result = subprocess.run(
                ["schtasks", "/Query", "/TN", self.task_name],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_task_info(self) -> dict | None:
        """Get info about the registered task."""
        try:
            result = subprocess.run(
                ["schtasks", "/Query", "/TN", self.task_name,
                 "/FO", "LIST", "/V"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None

            # Parse the LIST format output
            info: dict[str, str] = {}
            for line in result.stdout.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    info[key.strip()] = value.strip()
            return info
        except Exception:
            return None
