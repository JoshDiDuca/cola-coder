"""Model Fingerprint: generate a unique fingerprint for a model checkpoint.

The fingerprint combines:
  - Architecture metadata (layer count, hidden size, heads, etc.)
  - Weight statistics per layer (mean, std, min, max)
  - A SHA-256 hash of the concatenated statistics

This lets you verify model identity across format conversions (e.g. PyTorch →
safetensors → ONNX) or confirm that two checkpoints are identical without
loading all weights into memory.

For a TS dev: like a content hash for a bundle — if any weight changes even
slightly, the fingerprint changes.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Weight statistics for a single named layer."""

    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    min_val: float
    max_val: float
    num_params: int

    def to_bytes(self) -> bytes:
        """Pack stats into bytes for hashing (deterministic)."""
        floats = (self.mean, self.std, self.min_val, self.max_val)
        return struct.pack(f"4d{len(self.shape)}i", *floats, *self.shape)


@dataclass
class ModelFingerprint:
    """Fingerprint of a model checkpoint."""

    architecture_hash: str  # SHA-256 of architecture metadata
    weights_hash: str  # SHA-256 of all weight statistics
    combined_hash: str  # SHA-256(architecture_hash + weights_hash)
    total_params: int
    layer_count: int
    layer_stats: list[LayerStats] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 12 characters of combined hash — human-friendly ID."""
        return self.combined_hash[:12]

    def matches(self, other: "ModelFingerprint") -> bool:
        """Return True if both fingerprints represent identical models."""
        return self.combined_hash == other.combined_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_id": self.short_id,
            "combined_hash": self.combined_hash,
            "architecture_hash": self.architecture_hash,
            "weights_hash": self.weights_hash,
            "total_params": self.total_params,
            "layer_count": self.layer_count,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ModelFingerprintGenerator:
    """Generate fingerprints from model state dicts or plain weight dicts.

    Works with any dict-like ``{name: array}`` mapping where arrays expose
    numpy-compatible attributes (``shape``, ``mean()``, ``std()``, ``min()``,
    ``max()``).  This design means we can use numpy arrays, torch tensors, or
    mock objects in tests without importing torch.
    """

    def from_state_dict(
        self,
        state_dict: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ModelFingerprint:
        """Generate a fingerprint from a ``{name: tensor}`` state dict.

        Parameters
        ----------
        state_dict:
            Mapping of parameter names to arrays/tensors.
        metadata:
            Optional architecture metadata (num_layers, hidden_size, …).
        """
        meta = metadata or {}
        arch_hash = self._hash_architecture(meta)

        layer_stats: list[LayerStats] = []
        for name, tensor in sorted(state_dict.items()):
            stats = self._compute_stats(name, tensor)
            layer_stats.append(stats)

        weights_hash = self._hash_weights(layer_stats)
        combined = hashlib.sha256(
            (arch_hash + weights_hash).encode()
        ).hexdigest()

        total_params = sum(ls.num_params for ls in layer_stats)

        return ModelFingerprint(
            architecture_hash=arch_hash,
            weights_hash=weights_hash,
            combined_hash=combined,
            total_params=total_params,
            layer_count=len(layer_stats),
            layer_stats=layer_stats,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_architecture(metadata: dict[str, Any]) -> str:
        canonical = json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _hash_weights(layer_stats: list[LayerStats]) -> str:
        h = hashlib.sha256()
        for ls in layer_stats:
            h.update(ls.name.encode())
            h.update(ls.to_bytes())
        return h.hexdigest()

    @staticmethod
    def _compute_stats(name: str, tensor: Any) -> LayerStats:
        """Extract statistics from a tensor-like object."""
        shape: tuple[int, ...]
        mean_val: float
        std_val: float
        min_val: float
        max_val: float
        num_params: int

        # Support numpy arrays, torch tensors, and plain lists
        if hasattr(tensor, "shape"):
            shape = tuple(int(d) for d in tensor.shape)
        elif hasattr(tensor, "__len__"):
            shape = (len(tensor),)
        else:
            shape = (1,)

        num_params = 1
        for d in shape:
            num_params *= d

        if hasattr(tensor, "float"):
            # torch tensor
            t = tensor.float()
            mean_val = float(t.mean())
            std_val = float(t.std()) if num_params > 1 else 0.0
            min_val = float(t.min())
            max_val = float(t.max())
        elif hasattr(tensor, "mean"):
            # numpy array
            mean_val = float(tensor.mean())
            std_val = float(tensor.std()) if num_params > 1 else 0.0
            min_val = float(tensor.min())
            max_val = float(tensor.max())
        else:
            # Plain Python list / scalar
            flat = list(tensor) if hasattr(tensor, "__iter__") else [tensor]
            mean_val = sum(flat) / len(flat)
            variance = sum((x - mean_val) ** 2 for x in flat) / max(len(flat), 1)
            std_val = variance ** 0.5
            min_val = min(flat)
            max_val = max(flat)

        return LayerStats(
            name=name,
            shape=shape,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            num_params=num_params,
        )
