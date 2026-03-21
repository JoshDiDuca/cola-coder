"""Run all cola-coder evaluations in sequence and produce a summary report.

Runs: HumanEval, TypeScript benchmark, smoke test, regression test, quality report.
Collects all results into a single JSON/text summary.

Usage:
    python scripts/run_eval_suite.py --checkpoint checkpoints/tiny/latest
    python scripts/run_eval_suite.py --checkpoint checkpoints/tiny/latest --skip humaneval ts
    python scripts/run_eval_suite.py --checkpoint checkpoints/tiny/latest --json
    python scripts/run_eval_suite.py --checkpoint checkpoints/tiny/latest --output eval_results.json

Flags:
    --skip   Space-separated list of benchmark names to skip.
             Choices: humaneval, ts, smoke, regression, quality
    --json   Print machine-readable JSON summary to stdout.
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
from cola_coder.model.config import get_storage_config


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result for a single benchmark."""

    name: str
    skipped: bool = False
    passed: bool = False
    return_code: int = -1
    duration_sec: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error: str = ""


@dataclass
class EvalSuiteReport:
    """Summary of all benchmark results."""

    checkpoint: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_sec: float = 0.0

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_script(
    script: str,
    args: list[str],
    timeout: int = 600,
) -> BenchmarkResult:
    """Run a script subprocess and return its result."""
    scripts_dir = Path(__file__).resolve().parent
    python = _find_python()
    cmd = [python, str(scripts_dir / script)] + args

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - t0
        return BenchmarkResult(
            name=script,
            passed=proc.returncode == 0,
            return_code=proc.returncode,
            duration_sec=duration,
            stdout=proc.stdout[-4000:] if proc.stdout else "",
            stderr=proc.stderr[-2000:] if proc.stderr else "",
        )
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t0
        return BenchmarkResult(
            name=script,
            passed=False,
            return_code=-1,
            duration_sec=duration,
            error=f"Timed out after {timeout}s",
        )
    except Exception as exc:
        duration = time.monotonic() - t0
        return BenchmarkResult(
            name=script,
            passed=False,
            return_code=-1,
            duration_sec=duration,
            error=str(exc),
        )


def _find_python() -> str:
    """Find the virtualenv python or fall back to sys.executable."""
    project_root = Path(__file__).resolve().parent.parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


# ---------------------------------------------------------------------------
# Suite definition
# ---------------------------------------------------------------------------


ALL_BENCHMARKS = ["humaneval", "ts", "smoke", "regression", "quality"]


def _build_suite(
    checkpoint: str,
    config: str | None,
    skip: list[str],
) -> list[tuple[str, str, list[str], int]]:
    """Return list of (name, script, extra_args, timeout_sec)."""
    ckpt_args = ["--checkpoint", checkpoint] if checkpoint else []
    cfg_args = ["--config", config] if config else []

    suite = [
        ("humaneval", "evaluate.py", ckpt_args + cfg_args + ["--num-samples", "5"], 300),
        ("ts", "ts_benchmark.py", ckpt_args + cfg_args, 300),
        ("smoke", "smoke_test.py", ckpt_args + cfg_args + ["--quick"], 120),
        ("regression", "regression_test.py", ckpt_args + cfg_args, 120),
        ("quality", "quality_report.py", ckpt_args + cfg_args, 180),
    ]

    return [(name, script, args, t) for name, script, args, t in suite if name not in skip]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all cola-coder evaluations and produce a summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML path.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=ALL_BENCHMARKS,
        help="Benchmarks to skip.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON report to file.",
    )
    args = parser.parse_args()

    # Auto-detect checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        try:
            storage = get_storage_config()
            from cola_coder.training.checkpoint import detect_latest_checkpoint

            result = detect_latest_checkpoint(storage.checkpoints_dir)
            if result:
                checkpoint, _ = result
                if not args.json:
                    cli.info("Auto-detected checkpoint", checkpoint)
        except Exception:
            pass

    if not args.json:
        cli.header("Cola-Coder", "Evaluation Suite")
        if checkpoint:
            cli.info("Checkpoint", checkpoint)
        if args.skip:
            cli.info("Skipping", ", ".join(args.skip))

    suite = _build_suite(checkpoint or "", args.config, args.skip)
    skip_set = set(args.skip)

    report = EvalSuiteReport(checkpoint=checkpoint or "auto")
    # Add skipped entries first
    for name in ALL_BENCHMARKS:
        if name in skip_set:
            report.results.append(BenchmarkResult(name=name, skipped=True))

    suite_t0 = time.monotonic()

    for name, script, extra_args, timeout in suite:
        if not args.json:
            cli.info("Running", f"{name}  ({script})")

        result = _run_script(script, extra_args, timeout=timeout)
        result.name = name
        report.results.append(result)

        if not args.json:
            status = "PASS" if result.passed else ("SKIP" if result.skipped else "FAIL")
            cli.info(
                f"  {status}",
                f"{result.duration_sec:.1f}s  (rc={result.return_code})",
            )
            if not result.passed and result.error:
                cli.warn(f"  Error: {result.error}")

    report.total_sec = time.monotonic() - suite_t0

    # Summary
    if not args.json:
        cli.kv_table(
            {
                "Passed": str(report.n_passed),
                "Failed": str(report.n_failed),
                "Skipped": str(report.n_skipped),
                "Total time": f"{report.total_sec:.1f}s",
            },
            title="Suite Summary",
        )
        if report.all_passed:
            cli.success("All benchmarks passed!")
        else:
            cli.error(f"{report.n_failed} benchmark(s) failed.")

    # JSON output
    if args.json or args.output:
        report_dict = {
            "checkpoint": report.checkpoint,
            "total_sec": round(report.total_sec, 2),
            "n_passed": report.n_passed,
            "n_failed": report.n_failed,
            "n_skipped": report.n_skipped,
            "all_passed": report.all_passed,
            "results": [asdict(r) for r in report.results],
        }
        json_str = json.dumps(report_dict, indent=2)
        if args.json:
            print(json_str)
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
            if not args.json:
                cli.info("Saved", args.output)

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
