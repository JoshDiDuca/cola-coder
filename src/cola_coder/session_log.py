"""Session logging — tee all console output to a timestamped log file.

Captures everything: CLI output, subprocess stdout/stderr, Python logging.
One log file per session (menu.py invocation or standalone script run).

Usage:
    from cola_coder.session_log import start_session_log, get_session_log

    # At script startup:
    start_session_log()  # creates logs/session_2026-03-26_14-30-00.log

    # To write directly:
    get_session_log().write("custom message")

    # Subprocess teeing (used by _run_script / _run_stage_script):
    get_session_log().run_and_tee(cmd, cwd=project_root)
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO


class SessionLogger:
    """Manages a session log file that captures all console output."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._lock = threading.Lock()

        # Write session header
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"\n{'=' * 80}\n")
        self._file.write(f"  Cola-Coder Session Log — {now}\n")
        self._file.write(f"  Python: {sys.executable}\n")
        self._file.write(f"  CWD: {os.getcwd()}\n")
        self._file.write(f"{'=' * 80}\n\n")
        self._file.flush()

        # Also set up Python logging to this file
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def write(self, text: str) -> None:
        """Write text to the log file (thread-safe)."""
        with self._lock:
            self._file.write(text)
            if not text.endswith("\n"):
                self._file.write("\n")
            self._file.flush()

    def write_line(self, line: str) -> None:
        """Write a single line with timestamp prefix."""
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._file.write(f"[{ts}] {line}\n")
            self._file.flush()

    def write_subprocess_header(self, cmd: list[str]) -> None:
        """Log the start of a subprocess execution."""
        ts = datetime.now().strftime("%H:%M:%S")
        cmd_str = " ".join(cmd)
        with self._lock:
            self._file.write(f"\n[{ts}] ── subprocess: {cmd_str}\n")
            self._file.flush()

    def write_subprocess_footer(self, returncode: int, duration_s: float) -> None:
        """Log the end of a subprocess execution."""
        ts = datetime.now().strftime("%H:%M:%S")
        status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
        with self._lock:
            self._file.write(f"[{ts}] ── subprocess {status} ({duration_s:.1f}s)\n\n")
            self._file.flush()

    def run_and_tee(
        self,
        cmd: list[str],
        cwd: str | Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run a subprocess, streaming output to both console and log file.

        Returns CompletedProcess with returncode (stdout/stderr are not captured
        since they stream in real time).
        """
        import time

        self.write_subprocess_header(cmd)
        start = time.monotonic()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError:
            self.write_line(f"ERROR: command not found: {cmd[0]}")
            return subprocess.CompletedProcess(cmd, returncode=-2)

        # Stream output line-by-line to both console and log
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                # Write to console (real-time)
                sys.stdout.write(line)
                sys.stdout.flush()
                # Write to log file (strip trailing newline, we add our own)
                stripped = line.rstrip("\n\r")
                if stripped:
                    with self._lock:
                        self._file.write(f"    {stripped}\n")
                        self._file.flush()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            duration = time.monotonic() - start
            self.write_subprocess_footer(-9, duration)
            raise

        proc.wait()
        duration = time.monotonic() - start
        self.write_subprocess_footer(proc.returncode, duration)

        return subprocess.CompletedProcess(cmd, returncode=proc.returncode)

    def close(self) -> None:
        """Close the log file."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._file.write(f"\n{'=' * 80}\n")
            self._file.write(f"  Session ended — {ts}\n")
            self._file.write(f"{'=' * 80}\n")
            self._file.close()


# ── Module-level singleton ──────────────────────────────────────────────────

_session: SessionLogger | None = None


def start_session_log(log_dir: str = "logs") -> SessionLogger:
    """Start session logging. Creates a timestamped log file.

    Safe to call multiple times — returns existing session if already started.
    """
    global _session
    if _session is not None:
        return _session

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(log_dir) / f"session_{ts}.log"
    _session = SessionLogger(log_path)
    return _session


def get_session_log() -> SessionLogger | None:
    """Get the current session logger, or None if not started."""
    return _session
