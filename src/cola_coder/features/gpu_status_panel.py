"""GPU Status Panel: display real-time GPU memory, utilization, and temperature.

Supports NVIDIA GPUs via torch.cuda and pynvml (if available), with a graceful
CPU-only fallback when no CUDA device is present.
"""

from dataclasses import dataclass

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Optional imports — fail gracefully so the module works on CPU-only machines
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import pynvml  # type: ignore[import]
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Snapshot of a single GPU's status."""
    name: str
    index: int
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float
    temperature: float | None


# ---------------------------------------------------------------------------
# Panel implementation
# ---------------------------------------------------------------------------

class GPUStatusPanel:
    """Query and format GPU status information."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def is_gpu_available(self) -> bool:
        """Return True when at least one CUDA-capable GPU is detected."""
        return _TORCH_AVAILABLE

    def get_gpu_info(self) -> list[GPUInfo]:
        """Return a GPUInfo snapshot for every visible GPU.

        Falls back to an empty list when no GPU is available.
        """
        if not _TORCH_AVAILABLE or torch is None:
            return []

        infos: list[GPUInfo] = []
        device_count = torch.cuda.device_count()

        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            name = props.name

            # Memory (bytes -> MB)
            mem = torch.cuda.mem_get_info(idx)          # (free, total)
            total_mb = mem[1] / (1024 ** 2)
            used_mb = (mem[1] - mem[0]) / (1024 ** 2)

            # Utilization + temperature via pynvml when available
            utilization_pct: float = 0.0
            temperature: float | None = None
            if _NVML_AVAILABLE and pynvml is not None:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization_pct = float(util.gpu)
                    temperature = float(pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ))
                except Exception:
                    pass

            infos.append(GPUInfo(
                name=name,
                index=idx,
                memory_used_mb=round(used_mb, 1),
                memory_total_mb=round(total_mb, 1),
                utilization_pct=utilization_pct,
                temperature=temperature,
            ))

        return infos

    def get_memory_usage(self, device_index: int = 0) -> dict:
        """Return a memory breakdown dict for *device_index*.

        Keys: used_mb, total_mb, free_mb, used_pct
        """
        if not _TORCH_AVAILABLE or torch is None:
            return {"used_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0, "used_pct": 0.0}

        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        used_bytes = total_bytes - free_bytes
        total_mb = total_bytes / (1024 ** 2)
        used_mb = used_bytes / (1024 ** 2)
        free_mb = free_bytes / (1024 ** 2)
        used_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

        return {
            "used_mb": round(used_mb, 1),
            "total_mb": round(total_mb, 1),
            "free_mb": round(free_mb, 1),
            "used_pct": round(used_pct, 2),
        }

    def memory_pressure(self, device_index: int = 0) -> float:
        """Return VRAM fullness as a 0–1 float (0 = empty, 1 = full)."""
        if not _TORCH_AVAILABLE or torch is None:
            return 0.0

        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        if total_bytes == 0:
            return 0.0
        used_bytes = total_bytes - free_bytes
        return max(0.0, min(1.0, used_bytes / total_bytes))

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    def format_panel(self) -> str:
        """Return a human-readable text panel summarising all GPUs."""
        lines: list[str] = ["=== GPU Status ==="]

        if not self.is_gpu_available():
            lines.append("No CUDA GPU detected — running on CPU.")
            return "\n".join(lines)

        gpus = self.get_gpu_info()
        if not gpus:
            lines.append("No GPUs found.")
            return "\n".join(lines)

        for gpu in gpus:
            lines.append(f"\n[GPU {gpu.index}] {gpu.name}")
            lines.append(
                f"  Memory : {gpu.memory_used_mb:.0f} MB / {gpu.memory_total_mb:.0f} MB"
                f"  ({gpu.memory_used_mb / gpu.memory_total_mb * 100:.1f}% used)"
            )
            lines.append(f"  Util   : {gpu.utilization_pct:.1f}%")
            if gpu.temperature is not None:
                lines.append(f"  Temp   : {gpu.temperature:.0f} °C")
            else:
                lines.append("  Temp   : N/A")

        return "\n".join(lines)

    def summary(self) -> dict:
        """Return a compact dict summary suitable for logging or dashboards."""
        available = self.is_gpu_available()
        gpus = self.get_gpu_info()

        result: dict = {
            "gpu_available": available,
            "gpu_count": len(gpus),
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_used_mb": g.memory_used_mb,
                    "memory_total_mb": g.memory_total_mb,
                    "utilization_pct": g.utilization_pct,
                    "temperature": g.temperature,
                }
                for g in gpus
            ],
        }

        if gpus:
            result["memory_pressure"] = self.memory_pressure(gpus[0].index)

        return result


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def quick_gpu_check() -> str:
    """Return a single-line GPU status string."""
    panel = GPUStatusPanel()

    if not panel.is_gpu_available():
        return "GPU: not available (CPU only)"

    gpus = panel.get_gpu_info()
    if not gpus:
        return "GPU: no devices found"

    g = gpus[0]
    temp_str = f", {g.temperature:.0f}°C" if g.temperature is not None else ""
    return (
        f"GPU[{g.index}] {g.name} | "
        f"{g.memory_used_mb:.0f}/{g.memory_total_mb:.0f} MB "
        f"({g.memory_used_mb / g.memory_total_mb * 100:.1f}%) | "
        f"util {g.utilization_pct:.1f}%{temp_str}"
    )
