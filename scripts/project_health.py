"""Project Health Check: verify the cola-coder project is in a healthy state.

Checks:
    1. All scripts in scripts/ have --help (argparse)
    2. All features in src/cola_coder/features/ have FEATURE_ENABLED
    3. All tests pass (pytest --tb=no -q)
    4. Ruff lint is clean (ruff check src/ scripts/ tests/)
    5. Checkpoint tests pass (pytest tests/test_checkpoint.py)

Prints an overall health score (n checks passed / n total).

Usage:
    python scripts/project_health.py
    python scripts/project_health.py --skip tests ruff
    python scripts/project_health.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result for a single health check."""

    name: str
    passed: bool
    skipped: bool = False
    details: str = ""
    issues: list[str] = field(default_factory=list)
    duration_sec: float = 0.0


@dataclass
class HealthReport:
    """Full health check report."""

    checks: list[CheckResult] = field(default_factory=list)
    total_sec: float = 0.0

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed and not c.skipped)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed and not c.skipped)

    @property
    def n_skipped(self) -> int:
        return sum(1 for c in self.checks if c.skipped)

    @property
    def n_total(self) -> int:
        return sum(1 for c in self.checks if not c.skipped)

    @property
    def score_pct(self) -> float:
        return (100.0 * self.n_passed / self.n_total) if self.n_total else 0.0

    @property
    def healthy(self) -> bool:
        return self.n_failed == 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _find_python() -> str:
    project_root = Path(__file__).resolve().parent.parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _find_ruff() -> str:
    project_root = Path(__file__).resolve().parent.parent
    ruff = project_root / ".venv" / "Scripts" / "ruff.exe"
    if ruff.exists():
        return str(ruff)
    return "ruff"


def _find_pytest() -> str:
    project_root = Path(__file__).resolve().parent.parent
    pytest_bin = project_root / ".venv" / "Scripts" / "pytest.exe"
    if pytest_bin.exists():
        return str(pytest_bin)
    return "pytest"


def _check_scripts_have_help() -> CheckResult:
    """Verify every script in scripts/ responds to --help without crashing."""
    t0 = time.monotonic()
    scripts_dir = Path(__file__).resolve().parent
    python = _find_python()

    issues: list[str] = []
    skipped_scripts = {"menu.py", "__init__.py"}  # menu.py uses questionary
    script_files = [
        p for p in scripts_dir.glob("*.py")
        if p.name not in skipped_scripts
    ]

    for script in sorted(script_files):
        try:
            proc = subprocess.run(
                [python, str(script), "--help"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            # --help should exit 0 or print something
            if proc.returncode not in (0, 1) and not proc.stdout:
                issues.append(f"{script.name}: returned rc={proc.returncode} with no output")
        except subprocess.TimeoutExpired:
            issues.append(f"{script.name}: timed out on --help")
        except Exception as exc:
            issues.append(f"{script.name}: {exc}")

    duration = time.monotonic() - t0
    return CheckResult(
        name="scripts_have_help",
        passed=len(issues) == 0,
        details=f"Checked {len(script_files)} scripts",
        issues=issues,
        duration_sec=duration,
    )


def _check_features_have_enabled() -> CheckResult:
    """Verify every feature module has a FEATURE_ENABLED constant."""
    t0 = time.monotonic()
    features_dir = (
        Path(__file__).resolve().parent.parent
        / "src" / "cola_coder" / "features"
    )

    issues: list[str] = []
    skip_files = {"__init__.py", "__pycache__"}
    feature_files = [
        p for p in features_dir.glob("*.py")
        if p.name not in skip_files
    ]

    for feature in sorted(feature_files):
        content = feature.read_text(encoding="utf-8")
        if "FEATURE_ENABLED" not in content:
            issues.append(feature.name)

    duration = time.monotonic() - t0
    return CheckResult(
        name="features_have_enabled",
        passed=len(issues) == 0,
        details=f"Checked {len(feature_files)} feature modules",
        issues=issues,
        duration_sec=duration,
    )


def _run_subprocess(cmd: list[str], timeout: int = 300) -> tuple[bool, str]:
    """Run a subprocess and return (success, output)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout + proc.stderr)[-3000:]
        return proc.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)


def _check_tests_pass() -> CheckResult:
    """Run the full test suite and check it passes."""
    t0 = time.monotonic()
    pytest = _find_pytest()

    passed, output = _run_subprocess(
        [pytest, "tests/", "--tb=short", "-q", "--no-header"],
        timeout=300,
    )
    duration = time.monotonic() - t0

    issues: list[str] = []
    if not passed:
        # Extract failed test names from output
        for line in output.splitlines():
            if "FAILED" in line or "ERROR" in line:
                issues.append(line.strip()[:100])
        if not issues:
            issues.append(output[-500:])

    return CheckResult(
        name="tests_pass",
        passed=passed,
        details=f"pytest tests/  ({duration:.1f}s)",
        issues=issues[:10],  # cap at 10
        duration_sec=duration,
    )


def _check_ruff_clean() -> CheckResult:
    """Run ruff and check there are no lint errors."""
    t0 = time.monotonic()
    ruff = _find_ruff()

    passed, output = _run_subprocess(
        [ruff, "check", "src/", "scripts/", "tests/"],
        timeout=60,
    )
    duration = time.monotonic() - t0

    issues: list[str] = []
    if not passed:
        for line in output.splitlines():
            line = line.strip()
            if line and not line.startswith("Found"):
                issues.append(line[:100])
        issues = issues[:20]

    return CheckResult(
        name="ruff_clean",
        passed=passed,
        details="ruff check src/ scripts/ tests/",
        issues=issues,
        duration_sec=duration,
    )


def _check_checkpoint_tests() -> CheckResult:
    """Run the checkpoint-specific tests (critical for training safety)."""
    t0 = time.monotonic()
    pytest = _find_pytest()

    passed, output = _run_subprocess(
        [pytest, "tests/test_checkpoint.py", "--tb=short", "-q", "--no-header"],
        timeout=120,
    )
    duration = time.monotonic() - t0

    issues: list[str] = []
    if not passed:
        for line in output.splitlines():
            if "FAILED" in line or "ERROR" in line:
                issues.append(line.strip()[:100])
        if not issues:
            issues.append(output[-500:])

    return CheckResult(
        name="checkpoint_tests",
        passed=passed,
        details=f"pytest tests/test_checkpoint.py  ({duration:.1f}s)",
        issues=issues[:10],
        duration_sec=duration,
    )


# ---------------------------------------------------------------------------
# Health check registry
# ---------------------------------------------------------------------------


ALL_CHECKS = ["help", "features", "tests", "ruff", "checkpoint"]

_CHECK_RUNNERS = {
    "help": _check_scripts_have_help,
    "features": _check_features_have_enabled,
    "tests": _check_tests_pass,
    "ruff": _check_ruff_clean,
    "checkpoint": _check_checkpoint_tests,
}

_CHECK_LABELS = {
    "help": "Scripts have --help",
    "features": "Features have FEATURE_ENABLED",
    "tests": "Tests pass",
    "ruff": "Ruff lint clean",
    "checkpoint": "Checkpoint tests pass",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cola-coder project health checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=ALL_CHECKS,
        help="Checks to skip.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--output", type=str, default=None, help="Save JSON to file.")
    args = parser.parse_args()

    skip_set = set(args.skip)

    if not args.json:
        cli.header("Cola-Coder", "Project Health Check")

    report = HealthReport()
    t0 = time.monotonic()

    for check_key in ALL_CHECKS:
        label = _CHECK_LABELS[check_key]
        if check_key in skip_set:
            result = CheckResult(name=check_key, passed=True, skipped=True)
            report.checks.append(result)
            if not args.json:
                cli.dim(f"  SKIP  {label}")
            continue

        if not args.json:
            cli.info("Checking", label)

        runner = _CHECK_RUNNERS[check_key]
        result = runner()
        result.name = check_key
        report.checks.append(result)

        if not args.json:
            if result.passed:
                cli.success(f"  PASS  {label}  ({result.duration_sec:.1f}s)")
            else:
                cli.error(f"  FAIL  {label}  ({result.duration_sec:.1f}s)")
                for issue in result.issues[:5]:
                    cli.dim(f"    {issue}")

    report.total_sec = time.monotonic() - t0

    # Summary
    if not args.json:
        score_bar_width = 40
        filled = int(score_bar_width * report.score_pct / 100)
        bar = "█" * filled + "░" * (score_bar_width - filled)
        print()
        print(f"  Health Score: [{bar}]  {report.score_pct:.0f}%  "
              f"({report.n_passed}/{report.n_total} checks)")
        print()
        if report.healthy:
            cli.success("Project is healthy!")
        else:
            cli.error(f"{report.n_failed} check(s) failed.")

    if args.json or args.output:
        report_dict = {
            "score_pct": round(report.score_pct, 1),
            "n_passed": report.n_passed,
            "n_failed": report.n_failed,
            "n_skipped": report.n_skipped,
            "n_total": report.n_total,
            "healthy": report.healthy,
            "total_sec": round(report.total_sec, 2),
            "checks": [asdict(c) for c in report.checks],
        }
        json_str = json.dumps(report_dict, indent=2)
        if args.json:
            print(json_str)
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
            if not args.json:
                cli.info("Saved", args.output)

    sys.exit(0 if report.healthy else 1)


if __name__ == "__main__":
    main()
