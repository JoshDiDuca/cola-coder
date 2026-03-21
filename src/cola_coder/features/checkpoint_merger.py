"""Checkpoint Merger — feature 49.

Merges/averages multiple model checkpoints with configurable strategies:

1. **linear**: Weighted arithmetic mean of parameters.
   ``merged[k] = sum(w_i * params_i[k]) / sum(w_i)``

2. **slerp**: Spherical Linear Interpolation between exactly two parameter
   vectors.  Preserves vector magnitude while interpolating direction.
   (Extended to N checkpoints by sequentially applying SLERP.)

3. **task_arithmetic**: Task Arithmetic (Ilharco et al. 2023) — compute a
   "task vector" as the difference from a base model, scale it, then add
   back to the base.
   ``merged = base + scale * sum(w_i * (params_i - base))``

Works with plain Python dicts of {param_name: List[float]} so tests need no
PyTorch.  Pass real ``torch.Tensor`` values when used in production.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → merger returns the first checkpoint unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if checkpoint merging is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MergeMethod(str, Enum):
    """Merging strategy."""

    LINEAR = "linear"
    SLERP = "slerp"
    TASK_ARITHMETIC = "task_arithmetic"


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

# A "state dict" in this module is just Dict[str, list[float]] for purity.
# Production code should pass torch.Tensor values — the arithmetic helpers
# fall back gracefully via duck typing.

StateDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MergeConfig:
    """Configuration for a checkpoint merge operation."""

    method: MergeMethod = MergeMethod.LINEAR
    weights: Optional[List[float]] = None
    """Per-checkpoint weights.  None → equal weights."""
    slerp_t: float = 0.5
    """Interpolation factor for SLERP (only used with two checkpoints)."""
    task_arithmetic_scale: float = 1.0
    """Scale applied to task vectors in task_arithmetic mode."""
    base_checkpoint: Optional[StateDict] = None
    """Base model required for task_arithmetic mode."""

    def __post_init__(self):
        if self.method == MergeMethod.TASK_ARITHMETIC and self.base_checkpoint is None:
            raise ValueError("task_arithmetic requires a base_checkpoint")


@dataclass
class MergeReport:
    """Summary of a merge operation."""

    method: str
    num_checkpoints: int
    weights: List[float]
    param_names: List[str]
    num_params: int
    """Total parameter count across all keys."""

    def summary(self) -> str:
        return (
            f"MergeReport: method={self.method} "
            f"checkpoints={self.num_checkpoints} "
            f"params={len(self.param_names)} "
            f"total_values={self.num_params}"
        )


# ---------------------------------------------------------------------------
# Checkpoint Merger
# ---------------------------------------------------------------------------


class CheckpointMerger:
    """Merges model checkpoints using configurable strategies.

    Example (linear average)::

        merger = CheckpointMerger()
        merged, report = merger.merge(
            checkpoints=[ckpt_a, ckpt_b, ckpt_c],
            config=MergeConfig(method=MergeMethod.LINEAR),
        )

    Example (task arithmetic)::

        config = MergeConfig(
            method=MergeMethod.TASK_ARITHMETIC,
            base_checkpoint=base_model,
            task_arithmetic_scale=0.7,
        )
        merged, report = merger.merge([finetuned_a, finetuned_b], config)
    """

    def merge(
        self,
        checkpoints: Sequence[StateDict],
        config: Optional[MergeConfig] = None,
    ) -> Tuple[StateDict, MergeReport]:
        """Merge checkpoints according to config.

        Args:
            checkpoints: Sequence of state dicts to merge.
            config: Merge configuration.  Defaults to equal-weight LINEAR.

        Returns:
            (merged_state_dict, MergeReport)

        Raises:
            ValueError: If checkpoints are empty or have mismatched keys.
        """
        if not checkpoints:
            raise ValueError("checkpoints must be non-empty")

        if config is None:
            config = MergeConfig(method=MergeMethod.LINEAR)

        if not FEATURE_ENABLED:
            return dict(checkpoints[0]), MergeReport(
                method=config.method.value,
                num_checkpoints=len(checkpoints),
                weights=[1.0],
                param_names=list(checkpoints[0].keys()),
                num_params=0,
            )

        # Validate keys match
        reference_keys = set(checkpoints[0].keys())
        for i, ckpt in enumerate(checkpoints[1:], 1):
            if set(ckpt.keys()) != reference_keys:
                missing = reference_keys - set(ckpt.keys())
                extra = set(ckpt.keys()) - reference_keys
                raise ValueError(
                    f"Checkpoint {i} key mismatch: missing={missing}, extra={extra}"
                )

        # Normalise weights
        n = len(checkpoints)
        raw_weights = config.weights if config.weights is not None else [1.0] * n
        if len(raw_weights) != n:
            raise ValueError(
                f"weights length {len(raw_weights)} != num checkpoints {n}"
            )
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]

        if config.method == MergeMethod.LINEAR:
            merged = _merge_linear(checkpoints, weights)
        elif config.method == MergeMethod.SLERP:
            merged = _merge_slerp(checkpoints, weights, config.slerp_t)
        elif config.method == MergeMethod.TASK_ARITHMETIC:
            assert config.base_checkpoint is not None
            merged = _merge_task_arithmetic(
                checkpoints,
                weights,
                config.base_checkpoint,
                config.task_arithmetic_scale,
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        param_names = list(merged.keys())
        num_params = sum(_param_count(v) for v in merged.values())

        report = MergeReport(
            method=config.method.value,
            num_checkpoints=n,
            weights=[w * weight_sum for w in weights],  # un-normalised for display
            param_names=param_names,
            num_params=num_params,
        )
        return merged, report


# ---------------------------------------------------------------------------
# Merge strategy implementations
# ---------------------------------------------------------------------------


def _merge_linear(
    checkpoints: Sequence[StateDict],
    weights: List[float],
) -> StateDict:
    """Weighted arithmetic average of all checkpoints."""
    merged: StateDict = {}
    for key in checkpoints[0]:
        merged[key] = _weighted_mean(
            [ckpt[key] for ckpt in checkpoints], weights
        )
    return merged


def _merge_slerp(
    checkpoints: Sequence[StateDict],
    weights: List[float],
    t: float,
) -> StateDict:
    """Sequential SLERP across checkpoints."""
    if len(checkpoints) == 1:
        return dict(checkpoints[0])

    # For 2 checkpoints: direct SLERP with t
    # For N checkpoints: chain SLERP pairwise using weights as cumulative t
    result = dict(checkpoints[0])
    for i in range(1, len(checkpoints)):
        # Compute interpolation ratio for this pair
        pair_t = t if len(checkpoints) == 2 else weights[i]
        new_result: StateDict = {}
        for key in result:
            new_result[key] = _slerp_param(result[key], checkpoints[i][key], pair_t)
        result = new_result
    return result


def _merge_task_arithmetic(
    checkpoints: Sequence[StateDict],
    weights: List[float],
    base: StateDict,
    scale: float,
) -> StateDict:
    """Task arithmetic: add scaled weighted task vectors to base."""
    merged: StateDict = {}
    for key in checkpoints[0]:
        base_val = base[key]
        # Compute task vectors
        task_vectors = [_subtract(ckpt[key], base_val) for ckpt in checkpoints]
        # Weighted sum of task vectors
        combined = _weighted_mean(task_vectors, weights)
        # Scale and add to base
        merged[key] = _add(_scale(combined, scale), base_val)
    return merged


# ---------------------------------------------------------------------------
# Arithmetic primitives (work for both lists and torch.Tensor)
# ---------------------------------------------------------------------------


def _param_count(val: Any) -> int:
    """Return total element count of a parameter."""
    if hasattr(val, "numel"):  # torch.Tensor
        return int(val.numel())
    if isinstance(val, (list, tuple)):
        return len(val)
    return 1


def _weighted_mean(values: List[Any], weights: List[float]) -> Any:
    """Compute weighted mean across a list of tensors/lists."""
    if hasattr(values[0], "__len__"):
        n = len(values[0])
        result = [0.0] * n
        for v, w in zip(values, weights):
            for i in range(n):
                result[i] += w * v[i]
        return result
    # Scalar
    return sum(v * w for v, w in zip(values, weights))


def _subtract(a: Any, b: Any) -> Any:
    """Element-wise a - b."""
    if hasattr(a, "__len__"):
        return [x - y for x, y in zip(a, b)]
    return a - b


def _add(a: Any, b: Any) -> Any:
    """Element-wise a + b."""
    if hasattr(a, "__len__"):
        return [x + y for x, y in zip(a, b)]
    return a + b


def _scale(a: Any, s: float) -> Any:
    """Element-wise a * s."""
    if hasattr(a, "__len__"):
        return [x * s for x in a]
    return a * s


def _dot(a: Any, b: Any) -> float:
    """Dot product."""
    if hasattr(a, "__len__"):
        return sum(float(x) * float(y) for x, y in zip(a, b))
    return float(a) * float(b)


def _norm(a: Any) -> float:
    """L2 norm."""
    if hasattr(a, "__len__"):
        return math.sqrt(sum(float(x) ** 2 for x in a))
    return abs(float(a))


def _slerp_param(a: Any, b: Any, t: float) -> Any:
    """SLERP between two parameter vectors."""
    na = _norm(a)
    nb = _norm(b)

    if na < 1e-12 or nb < 1e-12:
        # Degenerate case: fall back to linear
        return _add(_scale(a, 1 - t), _scale(b, t))

    # Normalise
    a_hat = _scale(a, 1.0 / na)
    b_hat = _scale(b, 1.0 / nb)

    # Clamp dot product to avoid numerical issues with acos
    raw_dot = max(-1.0, min(1.0, _dot(a_hat, b_hat)))
    omega = math.acos(raw_dot)

    if abs(omega) < 1e-6:
        # Essentially parallel — linear interpolation
        interp_hat = _add(_scale(a_hat, 1 - t), _scale(b_hat, t))
    else:
        sin_omega = math.sin(omega)
        coeff_a = math.sin((1 - t) * omega) / sin_omega
        coeff_b = math.sin(t * omega) / sin_omega
        interp_hat = _add(_scale(a_hat, coeff_a), _scale(b_hat, coeff_b))

    # Interpolate magnitude
    interp_norm = na * (1 - t) + nb * t
    return _scale(interp_hat, interp_norm)
