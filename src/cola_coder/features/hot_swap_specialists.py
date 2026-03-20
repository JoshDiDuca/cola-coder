"""Hot-Swap Specialists: load and unload specialist models at runtime without restarting.

Manages a VRAM budget across registered models, using LRU eviction when needed.
Designed to be self-contained and work without GPU hardware (simulated loading for testing).

Usage:
    manager = HotSwapManager(vram_budget_mb=8000.0)
    manager.register('react', '/path/to/react', vram_mb=480.0)
    manager.load('react')
    model = manager.get('react')
    manager.unload('react')
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ModelSlot:
    """A slot in the hot-swap registry for one specialist model."""

    name: str
    path: str
    loaded: bool = False
    model: Optional[object] = None
    vram_mb: float = 0.0
    _last_used: float = field(default_factory=time.monotonic, repr=False, compare=False)

    def touch(self) -> None:
        """Update the last-used timestamp (LRU tracking)."""
        self._last_used = time.monotonic()


class HotSwapManager:
    """
    Manages hot-swapping of specialist models within a fixed VRAM budget.

    Models are registered with a name, path, and VRAM cost. Loading a model
    deducts from the budget; unloading returns it. When budget is tight,
    auto_evict() removes LRU models to make room.

    Thread-safe: all mutations are protected by a reentrant lock.
    """

    def __init__(self, vram_budget_mb: float) -> None:
        """
        Args:
            vram_budget_mb: Total VRAM budget in MB. Loading is refused when
                            a model would exceed this limit (unless auto_evict
                            is called first).
        """
        self._budget_mb: float = vram_budget_mb
        self._used_mb: float = 0.0
        self._slots: dict[str, ModelSlot] = {}
        # Ordered by insertion/access time — front is LRU, back is MRU.
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, path: str, vram_mb: float) -> ModelSlot:
        """Register a model in the manager.

        Does not load the model. Safe to call multiple times — re-registration
        updates the path and VRAM estimate (unloads first if currently loaded).

        Args:
            name: Unique identifier for the model (e.g. "react").
            path: Filesystem path to the model checkpoint.
            vram_mb: Estimated VRAM cost when loaded, in MB.

        Returns:
            The ModelSlot created (or updated).
        """
        with self._lock:
            if name in self._slots and self._slots[name].loaded:
                self.unload(name)
            slot = ModelSlot(name=name, path=path, vram_mb=vram_mb)
            self._slots[name] = slot
            return slot

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    def load(self, name: str) -> ModelSlot:
        """Load a registered model into memory.

        If the model is already loaded, this is a no-op (returns existing slot).
        Raises ValueError if the model is not registered or if the budget would
        be exceeded even after attempting auto_evict.

        In a real implementation this would call into safetensors / PyTorch.
        For testing, loading is simulated by storing a sentinel object.

        Args:
            name: Name of the model to load.

        Returns:
            The ModelSlot after loading.
        """
        with self._lock:
            if name not in self._slots:
                raise ValueError(f"Model '{name}' is not registered. Call register() first.")

            slot = self._slots[name]

            if slot.loaded:
                slot.touch()
                self._lru_touch(name)
                return slot

            # Evict LRU models if needed to fit within budget.
            needed = slot.vram_mb
            if self._used_mb + needed > self._budget_mb:
                self.auto_evict(needed_mb=needed)

            if self._used_mb + needed > self._budget_mb:
                raise RuntimeError(
                    f"Cannot load '{name}' ({needed:.0f} MB): "
                    f"only {self.available_vram():.0f} MB available after eviction."
                )

            # Simulate loading (replace with real model loader in production).
            slot.model = _SimulatedModel(name=name, path=slot.path)
            slot.loaded = True
            slot.touch()
            self._used_mb += needed
            self._lru_touch(name)
            return slot

    def unload(self, name: str) -> None:
        """Unload a model and free its VRAM budget allocation.

        Safe to call on an already-unloaded or unregistered model (no-op).

        Args:
            name: Name of the model to unload.
        """
        with self._lock:
            if name not in self._slots:
                return
            slot = self._slots[name]
            if not slot.loaded:
                return

            # Release model reference (allows GC; real impl would call .cpu() / del).
            slot.model = None
            slot.loaded = False
            self._used_mb = max(0.0, self._used_mb - slot.vram_mb)
            self._lru.pop(name, None)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[object]:
        """Return the loaded model object, or None if not loaded.

        Also updates LRU order on a hit.

        Args:
            name: Name of the model.

        Returns:
            The model object, or None.
        """
        with self._lock:
            slot = self._slots.get(name)
            if slot is None or not slot.loaded:
                return None
            slot.touch()
            self._lru_touch(name)
            return slot.model

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def loaded_models(self) -> list[str]:
        """Return names of all currently loaded models (MRU order)."""
        with self._lock:
            return [name for name, slot in self._slots.items() if slot.loaded]

    def available_vram(self) -> float:
        """Return remaining VRAM budget in MB."""
        with self._lock:
            return self._budget_mb - self._used_mb

    def summary(self) -> dict:
        """Return a summary dict of manager state.

        Useful for logging, CLI status commands, and health checks.
        """
        with self._lock:
            return {
                "vram_budget_mb": self._budget_mb,
                "vram_used_mb": round(self._used_mb, 2),
                "vram_available_mb": round(self.available_vram(), 2),
                "registered": list(self._slots.keys()),
                "loaded": self.loaded_models(),
                "num_registered": len(self._slots),
                "num_loaded": sum(1 for s in self._slots.values() if s.loaded),
                "lru_order": list(self._lru.keys()),  # front = LRU, back = MRU
            }

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def auto_evict(self, needed_mb: float) -> list[str]:
        """Evict LRU models until at least `needed_mb` VRAM is free.

        Models are evicted in LRU order (least recently used first).
        Stops as soon as enough budget is freed — avoids over-eviction.

        Args:
            needed_mb: MB of VRAM that must be available after eviction.

        Returns:
            List of model names that were evicted.
        """
        with self._lock:
            evicted: list[str] = []
            # Iterate from front of _lru (least recently used).
            for name in list(self._lru.keys()):
                if self.available_vram() >= needed_mb:
                    break
                self.unload(name)
                evicted.append(name)
            return evicted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lru_touch(self, name: str) -> None:
        """Move `name` to the MRU end of the LRU tracker."""
        self._lru.pop(name, None)
        self._lru[name] = None  # insert at back (MRU)


# ---------------------------------------------------------------------------
# Internal: simulated model for CPU/test environments
# ---------------------------------------------------------------------------

class _SimulatedModel:
    """Stand-in model object used when no real checkpoint exists.

    In production this would be a torch.nn.Module. For tests and CI
    environments without CUDA/safetensors, this placeholder allows the
    full manager API to be exercised.
    """

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path

    def __repr__(self) -> str:
        return f"<SimulatedModel name={self.name!r} path={self.path!r}>"
