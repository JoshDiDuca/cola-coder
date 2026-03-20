# Feature 22: Hot-Swap Specialists

**Status:** Optional | **CLI Flag:** `--hot-swap` | **Complexity:** Medium

---

## Overview

Load and unload specialist models on demand to stay within VRAM budget. Only one specialist is resident in GPU memory at a time (plus the always-resident router). Uses memory-mapped safetensors for fast checkpoint loading. An LRU (Least Recently Used) cache with a configurable slot count evicts the least recently used specialist when a new one needs to be loaded. Automatically monitors VRAM usage and evicts if headroom falls below a safety threshold. Target: ~100ms swap time.

---

## Motivation

With an RTX 3080 (10GB VRAM) or RTX 4080 (16GB VRAM), keeping multiple 125M–350M parameter models resident simultaneously is feasible but constrains batch sizes and context lengths. Hot-swapping enables:

- Clean VRAM utilization: load only what is needed for the current prompt
- Support for arbitrarily many specialists without VRAM scaling cost
- Graceful degradation: if VRAM is low, swap aggressively
- Development ergonomics: add specialists without worrying about VRAM budget

The 100ms swap target is achievable with safetensors' memory-mapped loading (weights are loaded lazily from disk on first access, not eagerly).

---

## Architecture / Design

### VRAM Slot Model

```
GPU Memory Layout:
┌─────────────────────────────────────┐
│  Router Model (always loaded, ~20MB) │
├─────────────────────────────────────┤
│  Active Specialist Slot 1           │  ← ~480MB (small, 125M params)
│  (most recently used)               │
├─────────────────────────────────────┤
│  Active Specialist Slot 2 (optional)│  ← only if VRAM allows
├─────────────────────────────────────┤
│  KV Cache + Activations (dynamic)   │
└─────────────────────────────────────┘
```

### LRU Cache Strategy

```
Request domain "react":
  → Cache hit: already in slot → return immediately
  → Cache miss:
      1. Check VRAM headroom
      2. If headroom < min_free_mb: evict LRU specialist
      3. Load "react" from disk via safetensors mmap
      4. Move to GPU
      5. Update LRU order
```

---

## Implementation Steps

### Step 1: VRAM Monitor

```python
# cola_coder/memory/vram_monitor.py
import torch
from dataclasses import dataclass

@dataclass
class VRAMStats:
    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float

    @property
    def utilization(self) -> float:
        return self.allocated_mb / self.total_mb if self.total_mb > 0 else 0.0


def get_vram_stats(device: int = 0) -> VRAMStats:
    if not torch.cuda.is_available():
        return VRAMStats(0, 0, 0, 0)
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1024**2
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    free = total - reserved
    return VRAMStats(total, allocated, reserved, free)


def estimate_model_vram_mb(num_params: int, dtype: torch.dtype = torch.float16) -> float:
    """Estimate VRAM usage of a model in MB."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)
    return (num_params * bytes_per_param) / 1024**2
```

### Step 2: SpecialistCache (LRU)

```python
# cola_coder/memory/specialist_cache.py
import time
import threading
import torch
import gc
from collections import OrderedDict
from typing import Optional
from .vram_monitor import get_vram_stats, estimate_model_vram_mb

class SpecialistCache:
    """
    LRU cache for specialist models. Manages VRAM by evicting
    least recently used models when capacity is exceeded.
    """
    def __init__(
        self,
        max_slots: int = 1,
        min_free_vram_mb: float = 512.0,
        device: str = "cuda",
        preload_domain: Optional[str] = None,
    ):
        self.max_slots = max_slots
        self.min_free_vram_mb = min_free_vram_mb
        self.device = device
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._load_times: dict[str, float] = {}
        self._access_counts: dict[str, int] = {}
        self._lock = threading.RLock()

        if preload_domain:
            self._preload_domain = preload_domain

    def get(self, domain: str) -> Optional[object]:
        """Return cached model or None if not in cache."""
        with self._lock:
            if domain in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(domain)
                self._access_counts[domain] = self._access_counts.get(domain, 0) + 1
                return self._cache[domain]
        return None

    def put(self, domain: str, model: object):
        """Add model to cache, evicting LRU if necessary."""
        with self._lock:
            if domain in self._cache:
                self._cache.move_to_end(domain)
                self._cache[domain] = model
                return

            # Check if we need to evict
            while (len(self._cache) >= self.max_slots
                   or self._vram_pressure()):
                if not self._cache:
                    break
                evicted_domain, evicted_model = self._cache.popitem(last=False)
                self._evict_model(evicted_domain, evicted_model)

            self._cache[domain] = model
            self._load_times[domain] = time.perf_counter()
            self._access_counts[domain] = 1

    def _vram_pressure(self) -> bool:
        """True if VRAM free headroom is below safety threshold."""
        stats = get_vram_stats()
        return stats.free_mb < self.min_free_vram_mb

    def _evict_model(self, domain: str, model: object):
        """Move model off GPU and release memory."""
        print(f"[cache] Evicting specialist '{domain}'")
        try:
            if hasattr(model, "cpu"):
                model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[cache] Warning: eviction error for '{domain}': {e}")

    def evict(self, domain: str):
        """Manually evict a specific domain."""
        with self._lock:
            if domain in self._cache:
                model = self._cache.pop(domain)
                self._evict_model(domain, model)

    def evict_all(self):
        """Evict all cached specialists."""
        with self._lock:
            for domain in list(self._cache.keys()):
                self.evict(domain)

    def stats(self) -> dict:
        with self._lock:
            vram = get_vram_stats()
            return {
                "cached_domains": list(self._cache.keys()),
                "num_cached": len(self._cache),
                "max_slots": self.max_slots,
                "vram_free_mb": round(vram.free_mb, 1),
                "vram_used_mb": round(vram.allocated_mb, 1),
                "vram_utilization": round(vram.utilization, 3),
                "access_counts": dict(self._access_counts),
            }
```

### Step 3: HotSwapManager

```python
# cola_coder/memory/hot_swap.py
import time
from safetensors.torch import load_file
from .specialist_cache import SpecialistCache
from .vram_monitor import get_vram_stats

class HotSwapManager:
    """
    High-level interface for hot-swapping specialist models.
    Combines SpecialistCache with the SpecialistRegistry.
    """
    def __init__(
        self,
        registry,
        cache: SpecialistCache,
        model_loader_fn,
        device: str = "cuda",
    ):
        self.registry = registry
        self.cache = cache
        self.load_fn = model_loader_fn
        self.device = device
        self._swap_times: list[float] = []

    def get_or_load(self, domain: str) -> object:
        """
        Return model for domain. Load from disk if not cached.
        Uses memory-mapped safetensors for fast loading.
        """
        # Cache hit
        model = self.cache.get(domain)
        if model is not None:
            return model

        # Cache miss — need to load
        entry = self.registry.get_specialist(domain)
        if entry is None:
            raise ValueError(f"Domain '{domain}' not in registry")

        t0 = time.perf_counter()
        model = self._load_mmap(entry.checkpoint, entry.model_config)
        swap_ms = (time.perf_counter() - t0) * 1000
        self._swap_times.append(swap_ms)
        print(f"[hot-swap] Loaded '{domain}' in {swap_ms:.1f}ms")

        self.cache.put(domain, model)
        return model

    def _load_mmap(self, checkpoint_path: str, config_path: str) -> object:
        """
        Load model using safetensors memory-mapped IO.
        Weights are mapped into virtual memory and paged in on first access.
        """
        from cola_coder.model.loader import build_model_from_config
        from cola_coder.model.config import ModelConfig
        import yaml

        cfg = ModelConfig(**yaml.safe_load(open(config_path)))
        model = build_model_from_config(cfg)

        # Use safetensors mmap for fast loading
        state_dict = load_file(checkpoint_path, device=self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def swap_stats(self) -> dict:
        if not self._swap_times:
            return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}
        return {
            "count": len(self._swap_times),
            "avg_ms": round(sum(self._swap_times) / len(self._swap_times), 1),
            "min_ms": round(min(self._swap_times), 1),
            "max_ms": round(max(self._swap_times), 1),
            "p95_ms": round(sorted(self._swap_times)[int(len(self._swap_times) * 0.95)], 1),
        }
```

### Step 4: CLI Commands

```python
@app.command()
def vram_status():
    """Show current VRAM usage and cached specialists."""
    from cola_coder.memory.vram_monitor import get_vram_stats
    stats = get_vram_stats()
    bar_len = int(stats.utilization * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    color = "red" if stats.utilization > 0.85 else "yellow" if stats.utilization > 0.65 else "green"
    console.print(f"VRAM: [{color}]{bar}[/{color}] {stats.allocated_mb:.0f}/{stats.total_mb:.0f} MB")

    if hot_swap_manager:
        cache_stats = hot_swap_manager.cache.stats()
        console.print(f"Cached specialists: {cache_stats['cached_domains']}")
        swap_s = hot_swap_manager.swap_stats()
        if swap_s["count"] > 0:
            console.print(f"Swap stats: avg={swap_s['avg_ms']}ms, p95={swap_s['p95_ms']}ms")


@app.command()
def evict_specialist(
    domain: str = typer.Argument(...),
):
    """Manually evict a specialist from VRAM."""
    hot_swap_manager.cache.evict(domain)
    console.print(f"[green]Evicted '{domain}' from VRAM[/green]")
```

---

## Key Files to Modify

- `cola_coder/memory/__init__.py` — new package
- `cola_coder/memory/vram_monitor.py` — VRAM stats
- `cola_coder/memory/specialist_cache.py` — LRU cache
- `cola_coder/memory/hot_swap.py` — HotSwapManager
- `cola_coder/generate.py` — use HotSwapManager instead of direct model loading
- `cola_coder/cli.py` — `vram-status`, `evict-specialist` commands
- `configs/hot_swap.yaml` — max_slots, min_free_vram_mb, preload settings

---

## Testing Strategy

```python
def test_lru_eviction():
    cache = SpecialistCache(max_slots=2)
    cache.put("react", object())
    cache.put("prisma", object())
    cache.put("zod", object())  # Should evict "react" (LRU)
    assert "react" not in cache._cache
    assert "prisma" in cache._cache
    assert "zod" in cache._cache

def test_cache_hit_updates_lru():
    cache = SpecialistCache(max_slots=2)
    cache.put("react", object())
    cache.put("prisma", object())
    cache.get("react")           # Access react → it's now MRU
    cache.put("zod", object())   # Should evict "prisma" (now LRU)
    assert "react" in cache._cache
    assert "prisma" not in cache._cache

def test_vram_stats_available():
    from cola_coder.memory.vram_monitor import get_vram_stats
    stats = get_vram_stats()
    # On CPU-only machine, should return zeros gracefully
    assert stats.total_mb >= 0
```

---

## Performance Considerations

- **100ms swap target:** safetensors mmap loading of a 125M param model (fp16 = ~250MB) from NVMe SSD takes ~50-100ms. From HDD, ~200-500ms. From RAM-cached file, ~10-20ms.
- **Preload most-used:** Track access counts and proactively preload the most-requested specialist at startup to avoid first-request latency.
- **Pinned memory:** Use `torch.cuda.pin_memory()` for the CPU staging buffer to speed up GPU transfers.
- **fp16 loading:** Always load specialists in fp16/bf16, not fp32. Halves load time and VRAM usage.
- **max_slots=1 is usually correct:** For a single-user CLI, one specialist at a time is optimal. Increase to 2 only if running multiple sessions or the most common pattern is alternating between two domains.
- **KV cache VRAM:** Account for KV cache when estimating headroom. At 2048 context, 125M model uses ~200MB for KV cache. Factor into `min_free_vram_mb`.

---

## Dependencies

- `safetensors` (already used in Cola-Coder)
- `torch` (CUDA memory management APIs)
- Feature 18 (SpecialistRegistry) — source of checkpoint paths
- Feature 20 (cascade routing) — triggers rapid model switches

---

## Estimated Complexity

| Task                        | Effort  |
|-----------------------------|---------|
| VRAM monitor                | 1h      |
| SpecialistCache (LRU)       | 2h      |
| HotSwapManager              | 2h      |
| Generator integration       | 2h      |
| CLI commands                | 1h      |
| Tests                       | 1.5h    |
| **Total**                   | **~9.5h** |

Overall complexity: **Medium** (well-understood LRU pattern, main challenge is VRAM accounting)

---

## 2026 Best Practices

- **safetensors over pickle/torch.load:** safetensors is both safer (no arbitrary code execution) and faster for partial loading. Already the Cola-Coder standard — maintain this.
- **Avoid torch.save/load for hot-swap:** torch.save pickles the model; safetensors loads individual tensors without unpickling overhead.
- **Memory pressure monitoring:** Use `torch.cuda.memory_stats()` for detailed allocation tracking beyond simple reserved/allocated. Particularly useful for debugging fragmentation.
- **Quantized specialists:** For even faster swaps and lower VRAM, keep specialist weights in INT8 on disk (via `bitsandbytes` or `torch.ao.quantization`) and dequantize on load.
- **Torch compile compatibility:** If using `torch.compile()`, be aware that compiled models cannot easily be moved between devices. Either compile after loading to GPU, or disable compile for hot-swapped models.
- **Graceful CPU fallback:** If VRAM is completely full and eviction fails, fall back to CPU inference rather than crashing. Log a warning.
