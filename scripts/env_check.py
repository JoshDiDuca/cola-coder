"""Environment check script.

Verifies the runtime environment before starting training:
- Python version (3.10+)
- PyTorch version and CUDA availability
- GPU memory (per device)
- Disk space for data and checkpoint directories
- HF_TOKEN environment variable
- Internet connectivity (optional, just pings HuggingFace)

Prints PASS / FAIL / WARN for each check so issues are immediately visible.

Usage:
    python scripts/env_check.py
    python scripts/env_check.py --no-internet   # skip connectivity check
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cola_coder.cli import cli  # noqa: E402

# ── Individual checks ─────────────────────────────────────────────────────────

CheckResult = tuple[str, str, str]  # (status: PASS|FAIL|WARN, label, detail)


def _check_python_version() -> CheckResult:
    vi = sys.version_info
    version_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if vi >= (3, 10):
        return ("PASS", "Python version", version_str)
    return ("FAIL", "Python version", f"{version_str} — requires 3.10+")


def _check_torch() -> CheckResult:
    try:
        import torch  # noqa: PLC0415

        v = torch.__version__
        return ("PASS", "PyTorch version", v)
    except ImportError:
        return ("FAIL", "PyTorch version", "torch not installed")


def _check_cuda() -> CheckResult:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return ("WARN", "CUDA", "Not available — training will use CPU (very slow)")
        v = torch.version.cuda or "unknown"
        device_count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        return ("PASS", "CUDA", f"v{v}, {device_count} GPU(s): {', '.join(names)}")
    except ImportError:
        return ("FAIL", "CUDA", "torch not installed")
    except Exception as exc:
        return ("WARN", "CUDA", f"Could not query CUDA: {exc}")


def _check_gpu_memory() -> list[CheckResult]:
    """Return one result per GPU."""
    results: list[CheckResult] = []
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return [("WARN", "GPU memory", "No CUDA GPUs found")]
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            name = props.name
            status = "PASS" if total_gb >= 8 else "WARN"
            detail = f"{name}: {total_gb:.1f} GB total"
            if total_gb < 8:
                detail += " (may be tight for medium/large configs)"
            results.append((status, f"GPU {i} memory", detail))
    except ImportError:
        results.append(("FAIL", "GPU memory", "torch not installed"))
    except Exception as exc:
        results.append(("WARN", "GPU memory", str(exc)))
    return results


def _check_disk_space(label: str, path: Path, min_gb: float = 2.0) -> CheckResult:
    """Check free disk space for a given path."""
    # Walk up to find an existing parent
    check_path = path
    while not check_path.exists() and check_path != check_path.parent:
        check_path = check_path.parent

    try:
        usage = shutil.disk_usage(str(check_path))
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        detail = f"{free_gb:.1f} GB free / {total_gb:.1f} GB total  ({path})"
        if free_gb < min_gb:
            return ("FAIL", label, f"Only {free_gb:.1f} GB free — need at least {min_gb} GB  ({path})")
        if free_gb < min_gb * 3:
            return ("WARN", label, detail + " — getting low")
        return ("PASS", label, detail)
    except Exception as exc:
        return ("WARN", label, f"Could not check disk space: {exc}")


def _check_hf_token() -> CheckResult:
    token = os.environ.get("HF_TOKEN", "")
    if token and len(token) > 8:
        # Mask for display
        masked = token[:4] + "..." + token[-4:]
        return ("PASS", "HF_TOKEN", f"Set ({masked})")
    if token:
        return ("WARN", "HF_TOKEN", "Set but very short — may be invalid")
    return ("WARN", "HF_TOKEN", "Not set — needed for bigcode/starcoderdata (gated dataset)")


def _check_internet() -> CheckResult:
    """Try to reach HuggingFace."""
    try:
        import urllib.request  # noqa: PLC0415

        with urllib.request.urlopen("https://huggingface.co", timeout=5) as resp:
            code = resp.status
        if code == 200:
            return ("PASS", "Internet (HuggingFace)", "Reachable")
        return ("WARN", "Internet (HuggingFace)", f"HTTP {code}")
    except Exception as exc:
        return ("WARN", "Internet (HuggingFace)", f"Not reachable: {exc}")


def _check_safetensors() -> CheckResult:
    try:
        import safetensors  # noqa: PLC0415, F401

        return ("PASS", "safetensors", "Installed")
    except ImportError:
        return ("FAIL", "safetensors", "Not installed — required for checkpoints")


def _check_tokenizers() -> CheckResult:
    try:
        import tokenizers  # noqa: PLC0415, F401

        return ("PASS", "tokenizers (HF)", "Installed")
    except ImportError:
        return ("FAIL", "tokenizers (HF)", "Not installed — required for tokenizer training")


# ── Rendering ─────────────────────────────────────────────────────────────────

_STATUS_STYLE = {
    "PASS": "[bold green]PASS[/bold green]",
    "FAIL": "[bold red]FAIL[/bold red]",
    "WARN": "[bold yellow]WARN[/bold yellow]",
}


def _print_result(status: str, label: str, detail: str) -> None:
    badge = _STATUS_STYLE.get(status, status)
    cli.print(f"  {badge}  {label:<35} {detail}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Check the cola-coder runtime environment.")
    parser.add_argument(
        "--no-internet",
        action="store_true",
        help="Skip internet connectivity check",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Environment Check")

    all_results: list[CheckResult] = []

    # Python / torch / CUDA
    all_results.append(_check_python_version())
    all_results.append(_check_torch())
    all_results.append(_check_cuda())
    all_results.extend(_check_gpu_memory())

    # Disk space
    data_dir = _PROJECT_ROOT / "data"
    ckpt_dir = _PROJECT_ROOT / "checkpoints"
    all_results.append(_check_disk_space("Disk (data dir)", data_dir, min_gb=10.0))
    all_results.append(_check_disk_space("Disk (checkpoints dir)", ckpt_dir, min_gb=5.0))

    # Dependencies
    all_results.append(_check_safetensors())
    all_results.append(_check_tokenizers())

    # Credentials
    all_results.append(_check_hf_token())

    # Internet (optional)
    if not args.no_internet:
        all_results.append(_check_internet())

    # Print
    cli.print("")
    for status, label, detail in all_results:
        _print_result(status, label, detail)

    # Summary
    n_pass = sum(1 for s, _, _ in all_results if s == "PASS")
    n_warn = sum(1 for s, _, _ in all_results if s == "WARN")
    n_fail = sum(1 for s, _, _ in all_results if s == "FAIL")

    cli.print("")
    if n_fail == 0:
        cli.success(f"All checks passed ({n_pass} PASS, {n_warn} WARN, 0 FAIL)")
        return 0
    else:
        cli.error(f"{n_fail} check(s) failed ({n_pass} PASS, {n_warn} WARN, {n_fail} FAIL)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
