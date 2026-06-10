"""Hardware profiler: detect GPUs and recommend a training configuration.

Detects CUDA GPUs (count, name, VRAM, compute capability, bf16 support) with
a clean CPU-only fallback, then maps the available hardware to the best model
config (tiny / small / medium / 4080_max / large) and derives safe training
overrides (precision, batch size, gradient accumulation, checkpointing).

Recommendations are validated against the VRAM estimator: if the estimate
does not fit, the profiler progressively enables gradient checkpointing,
halves the batch size (doubling accumulation to keep the effective batch),
and finally steps down to a smaller config.

In TypeScript terms: ``profile_hardware()`` is a typed system probe and
``recommend_config()`` is a pure function from that probe to a build config —
all side-effect free until ``generate_auto_config()`` writes the YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cola_coder.cli import cli

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# Config tiers ordered smallest → largest. Thresholds are the minimum GPU VRAM
# (GB) to *start* from that tier; the estimator-driven adjustment loop below
# is the real gatekeeper and may step a recommendation down further.
_CONFIG_TIERS: list[tuple[str, float]] = [
    ("tiny", 3.0),
    ("small", 5.5),
    ("medium", 11.0),
    ("4080_max", 15.0),
    ("large", 22.0),
]

# Fraction of total VRAM the training estimate must stay under.
_VRAM_SAFETY = 0.92


@dataclass
class GPUInfo:
    """One detected CUDA device."""

    index: int
    name: str
    total_vram_gb: float
    compute_capability: tuple[int, int]

    @property
    def supports_bf16(self) -> bool:
        """Ampere (SM 8.0) and newer have native bf16."""
        return self.compute_capability >= (8, 0)


@dataclass
class HardwareProfile:
    """Snapshot of the machine's training-relevant hardware."""

    has_cuda: bool
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_count: int = 0
    total_ram_gb: float | None = None
    torch_version: str = ""
    cuda_version: str = ""

    @property
    def best_gpu(self) -> GPUInfo | None:
        """The GPU with the most VRAM (training runs on a single device)."""
        return max(self.gpus, key=lambda g: g.total_vram_gb) if self.gpus else None


@dataclass
class TrainingRecommendation:
    """Hardware-derived config choice plus safe training overrides."""

    config_name: str
    config_path: str
    precision: str
    batch_size: int
    gradient_accumulation: int
    gradient_checkpointing: bool
    estimated_vram_gb: float | None
    gpu: GPUInfo | None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def profile_hardware() -> HardwareProfile:
    """Detect GPUs, CPU count, and system RAM. Never raises."""
    profile = HardwareProfile(has_cuda=False, cpu_count=os.cpu_count() or 1)
    profile.total_ram_gb = _detect_ram_gb()

    try:
        import torch
    except ImportError:
        return profile

    profile.torch_version = torch.__version__
    try:
        if not torch.cuda.is_available():
            return profile
        profile.has_cuda = True
        profile.cuda_version = torch.version.cuda or ""
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            profile.gpus.append(GPUInfo(
                index=i,
                name=torch.cuda.get_device_name(i),
                total_vram_gb=props.total_memory / 1e9,
                compute_capability=(props.major, props.minor),
            ))
    except Exception:
        # A broken CUDA install must not take the menu down — fall back to CPU.
        profile.has_cuda = bool(profile.gpus)
    return profile


def _detect_ram_gb() -> float | None:
    """Total system RAM in GB, or None if undetectable."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    try:
        # Windows fallback — no extra dependency needed.
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 1e9
    except Exception:
        pass
    return None


def recommend_config(
    profile: HardwareProfile,
    configs_dir: str | Path = "configs",
) -> TrainingRecommendation:
    """Map a hardware profile to the best config + safe training overrides."""
    configs_dir = Path(configs_dir)
    gpu = profile.best_gpu

    if gpu is None:
        rec = TrainingRecommendation(
            config_name="tiny",
            config_path=str(configs_dir / "tiny.yaml"),
            precision="fp32",
            batch_size=2,
            gradient_accumulation=16,
            gradient_checkpointing=False,
            estimated_vram_gb=None,
            gpu=None,
            reasons=["No CUDA GPU detected — CPU-only mode with the smallest config."],
            warnings=[
                "CPU training is 100-1000x slower than GPU. "
                "Expect smoke tests only, not real training runs.",
            ],
        )
        return rec

    # Pick the largest tier whose VRAM floor we clear.
    tier_idx = 0
    for i, (_, min_vram) in enumerate(_CONFIG_TIERS):
        if gpu.total_vram_gb >= min_vram:
            tier_idx = i

    precision = "bf16" if gpu.supports_bf16 else "fp16"
    reasons = [
        f"GPU: {gpu.name} with {gpu.total_vram_gb:.1f} GB VRAM "
        f"(compute capability {gpu.compute_capability[0]}.{gpu.compute_capability[1]})",
        f"Precision: {precision} "
        + ("(native bf16 — no GradScaler needed)" if precision == "bf16"
           else "(pre-Ampere GPU — fp16 with GradScaler)"),
    ]
    warnings: list[str] = []
    if len(profile.gpus) > 1:
        reasons.append(
            f"{len(profile.gpus)} GPUs detected — training uses GPU {gpu.index} "
            "(multi-GPU training is not supported yet)."
        )
    if gpu.total_vram_gb < _CONFIG_TIERS[0][1]:
        warnings.append(
            f"Only {gpu.total_vram_gb:.1f} GB VRAM — below the {_CONFIG_TIERS[0][1]:.1f} GB "
            "floor for the tiny config. Batch size reduced; OOM is still possible."
        )

    # Estimator-driven adjustment: shrink until the estimate fits.
    name, batch, accum, ckpt, estimate = _fit_to_vram(
        tier_idx, gpu.total_vram_gb, precision, configs_dir, reasons,
    )

    if name == "large":
        warnings.append(
            "The large (1B+) config is designed for cloud GPUs — "
            "verify data and storage throughput before a long run."
        )

    return TrainingRecommendation(
        config_name=name,
        config_path=str(configs_dir / f"{name}.yaml"),
        precision=precision,
        batch_size=batch,
        gradient_accumulation=accum,
        gradient_checkpointing=ckpt,
        estimated_vram_gb=estimate,
        gpu=gpu,
        reasons=reasons,
        warnings=warnings,
    )


def _fit_to_vram(
    tier_idx: int,
    vram_gb: float,
    precision: str,
    configs_dir: Path,
    reasons: list[str],
) -> tuple[str, int, int, bool, float | None]:
    """Walk tiers downward until the VRAM estimate fits.

    Within a tier: try the config's own batch settings, then enable gradient
    checkpointing, then halve batch (doubling accumulation). Returns
    (config_name, batch_size, gradient_accumulation, checkpointing, estimate_gb).
    """
    from cola_coder.model.config import Config

    budget = vram_gb * _VRAM_SAFETY

    for idx in range(tier_idx, -1, -1):
        name = _CONFIG_TIERS[idx][0]
        config_path = configs_dir / f"{name}.yaml"
        if not config_path.exists():
            continue
        try:
            cfg = Config.from_yaml(config_path)
        except Exception:
            continue

        cfg.training.precision = precision
        batch = cfg.training.batch_size
        accum = cfg.training.gradient_accumulation
        ckpt = cfg.training.gradient_checkpointing

        while True:
            cfg.training.batch_size = batch
            cfg.training.gradient_accumulation = accum
            cfg.training.gradient_checkpointing = ckpt
            estimate = _estimate_training_gb(cfg)
            if estimate is None or estimate <= budget:
                if name != _CONFIG_TIERS[tier_idx][0]:
                    reasons.append(
                        f"Stepped down to '{name}' so the VRAM estimate fits "
                        f"({estimate:.1f} GB ≤ {budget:.1f} GB budget)."
                        if estimate is not None else
                        f"Stepped down to '{name}' (estimator unavailable)."
                    )
                elif estimate is not None:
                    reasons.append(
                        f"Config '{name}': estimated {estimate:.1f} GB "
                        f"of {vram_gb:.1f} GB VRAM (budget {budget:.1f} GB)."
                    )
                return name, batch, accum, ckpt, estimate

            if not ckpt:
                ckpt = True
                reasons.append(
                    f"Enabled gradient checkpointing for '{name}' "
                    f"(estimate {estimate:.1f} GB exceeded {budget:.1f} GB budget)."
                )
                continue
            if batch > 1:
                batch = max(1, batch // 2)
                accum *= 2
                continue
            break  # batch=1 + checkpointing still too big → smaller tier

    # Even tiny at batch=1 doesn't fit — return tiny minimal and let the
    # caller's warning stand.
    return "tiny", 1, 32, True, None


def _estimate_training_gb(cfg) -> float | None:
    """Training VRAM estimate in GB, or None if the estimator fails."""
    try:
        from cola_coder.features.vram_estimator import estimate_vram
        return estimate_vram(
            model_config=cfg.model, training_config=cfg.training,
        ).total_training_gb
    except Exception:
        return None


# ── Auto-config generation ─────────────────────────────────────────────────

def generate_auto_config(
    rec: TrainingRecommendation,
    output_dir: str | Path = "configs/auto",
    smoke: bool = False,
) -> Path:
    """Write a derived YAML config applying the recommendation's overrides.

    The generated file is based on the recommended base config with only the
    hardware-derived training keys changed, so it stays loadable by
    ``Config.from_yaml`` and usable by every pipeline script via ``--config``.

    Smoke mode shrinks the run to minutes (for validating pipeline wiring,
    not for producing a usable model).
    """
    import yaml

    base_path = Path(rec.config_path)
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    training = raw.setdefault("training", {})
    training["precision"] = rec.precision
    training["batch_size"] = rec.batch_size
    training["gradient_accumulation"] = rec.gradient_accumulation
    training["gradient_checkpointing"] = rec.gradient_checkpointing

    checkpoint = raw.setdefault("checkpoint", {})
    suffix = "smoke" if smoke else "auto"
    if smoke:
        training["max_steps"] = 30
        training["warmup_steps"] = 5
        training["batch_size"] = min(rec.batch_size, 4)
        checkpoint["save_every"] = 15
        checkpoint["max_checkpoints"] = 2
    # Keep generated checkpoints separate from hand-run ones so a smoke run
    # can never pollute or prune a real training directory.
    base_out = checkpoint.get("output_dir", f"./checkpoints/{rec.config_name}")
    checkpoint["output_dir"] = f"{base_out.rstrip('/')}_{suffix}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"auto_{rec.config_name}{'_smoke' if smoke else ''}.yaml"

    gpu_desc = f"{rec.gpu.name} ({rec.gpu.total_vram_gb:.1f} GB)" if rec.gpu else "CPU only"
    header = (
        f"# AUTO-GENERATED by hardware_profiler — do not hand-edit (regenerated each run)\n"
        f"# Base config: {base_path.name}\n"
        f"# Hardware:    {gpu_desc}\n"
        f"# Mode:        {'smoke test (minutes — validates wiring only)' if smoke else 'full run'}\n"
    )
    out_path.write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path


# ── CLI output ──────────────────────────────────────────────────────────────

def print_hardware_profile(profile: HardwareProfile) -> None:
    """Print the detected hardware."""
    rows: dict[str, str] = {
        "CUDA available": "yes" if profile.has_cuda else "no",
        "CPU cores": str(profile.cpu_count),
    }
    if profile.total_ram_gb is not None:
        rows["System RAM"] = f"{profile.total_ram_gb:.0f} GB"
    if profile.torch_version:
        rows["PyTorch"] = profile.torch_version
    if profile.cuda_version:
        rows["CUDA"] = profile.cuda_version
    for gpu in profile.gpus:
        cc = f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
        rows[f"GPU {gpu.index}"] = (
            f"{gpu.name} — {gpu.total_vram_gb:.1f} GB, SM {cc}, "
            f"{'bf16' if gpu.supports_bf16 else 'fp16'}"
        )
    cli.kv_table(rows, title="Detected Hardware")


def print_recommendation(rec: TrainingRecommendation) -> None:
    """Print the recommended config and overrides."""
    cli.kv_table({
        "Config": f"{rec.config_name} ({rec.config_path})",
        "Precision": rec.precision,
        "Batch size": str(rec.batch_size),
        "Grad accumulation": f"{rec.gradient_accumulation} "
                             f"(effective batch {rec.batch_size * rec.gradient_accumulation})",
        "Grad checkpointing": "on" if rec.gradient_checkpointing else "off",
        "Estimated VRAM": f"{rec.estimated_vram_gb:.1f} GB"
                          if rec.estimated_vram_gb is not None else "n/a",
    }, title="Recommended Training Setup")
    for reason in rec.reasons:
        cli.dim(f"  • {reason}")
    for warning in rec.warnings:
        cli.warn(warning)
