"""Batch Size Finder: binary search for maximum batch size that fits in VRAM.

Takes a model configuration (parameter counts, dtype, activation size) and a
memory budget, then returns the optimal batch size along with a detailed
memory breakdown.  No GPU required — all computations are analytical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the batch size finder feature is active."""
    return FEATURE_ENABLED


@dataclass
class ModelConfig:
    """Minimal model configuration for memory estimation."""

    n_params: int  # Total trainable parameter count
    dtype_bytes: int = 4  # 4 = float32, 2 = float16/bfloat16
    seq_len: int = 512  # Maximum sequence length
    hidden_size: int = 512  # Hidden / embedding dimension
    n_layers: int = 6  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 32000  # Vocabulary size
    # Approximate bytes for one token's activations across all layers
    # If None, estimated from hidden_size and n_layers automatically
    bytes_per_token_activation: Optional[int] = None


@dataclass
class MemoryBreakdown:
    """Detailed breakdown of VRAM usage for a given batch size."""

    batch_size: int
    parameters_mb: float
    gradients_mb: float  # Same size as params (optimizer stores grads)
    optimizer_state_mb: float  # Adam: 2× params for m/v states
    activations_mb: float  # Forward pass activations
    workspace_mb: float  # Misc buffers (typically 5% overhead)
    total_mb: float

    def summary(self) -> str:
        return (
            f"MemoryBreakdown(batch={self.batch_size}, "
            f"params={self.parameters_mb:.1f}MB, "
            f"grads={self.gradients_mb:.1f}MB, "
            f"optim={self.optimizer_state_mb:.1f}MB, "
            f"activations={self.activations_mb:.1f}MB, "
            f"workspace={self.workspace_mb:.1f}MB, "
            f"total={self.total_mb:.1f}MB)"
        )


@dataclass
class BatchSizeResult:
    """Result from the batch size finder."""

    optimal_batch_size: int
    memory_limit_mb: float
    breakdown: MemoryBreakdown
    candidate_sizes_tried: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _estimate_activation_bytes(config: ModelConfig, batch_size: int) -> float:
    """Estimate activation memory in bytes for a given batch size.

    Activations = batch × seq_len × hidden_size × n_layers × dtype_bytes
    Multiplied by ~4 for intermediate tensors (QKV projections, FFN, etc.).
    """
    if config.bytes_per_token_activation is not None:
        per_token = config.bytes_per_token_activation
    else:
        # Heuristic: each layer stores ~4 hidden-sized tensors per token
        per_token = config.hidden_size * config.n_layers * 4 * config.dtype_bytes

    return batch_size * config.seq_len * per_token


def _compute_breakdown(config: ModelConfig, batch_size: int) -> MemoryBreakdown:
    """Compute memory breakdown for a given batch size (bytes → MB)."""
    to_mb = 1 / (1024 * 1024)

    param_bytes = config.n_params * config.dtype_bytes
    grad_bytes = param_bytes  # One gradient tensor per parameter
    optim_bytes = 2 * config.n_params * 4  # Adam m+v in float32

    activation_bytes = _estimate_activation_bytes(config, batch_size)

    subtotal = param_bytes + grad_bytes + optim_bytes + activation_bytes
    workspace_bytes = subtotal * 0.05  # 5% overhead

    total_bytes = subtotal + workspace_bytes

    return MemoryBreakdown(
        batch_size=batch_size,
        parameters_mb=param_bytes * to_mb,
        gradients_mb=grad_bytes * to_mb,
        optimizer_state_mb=optim_bytes * to_mb,
        activations_mb=activation_bytes * to_mb,
        workspace_mb=workspace_bytes * to_mb,
        total_mb=total_bytes * to_mb,
    )


class BatchSizeFinder:
    """Find the largest batch size that fits within a VRAM budget."""

    def find(
        self,
        config: ModelConfig,
        memory_limit_mb: float,
        min_batch: int = 1,
        max_batch: int = 4096,
        power_of_two: bool = True,
    ) -> BatchSizeResult:
        """Binary search for the maximum feasible batch size.

        Parameters
        ----------
        config:
            Model configuration describing memory characteristics.
        memory_limit_mb:
            Available VRAM in megabytes.
        min_batch:
            Smallest batch size to consider.
        max_batch:
            Upper bound for the search.
        power_of_two:
            If True, restrict candidates to powers of two (common for GPU).

        Returns
        -------
        BatchSizeResult with the optimal batch size and memory breakdown.
        """
        notes: list[str] = []
        candidates_tried: list[int] = []

        # Check if even batch_size=1 fits
        bd1 = _compute_breakdown(config, min_batch)
        if bd1.total_mb > memory_limit_mb:
            notes.append(f"Even batch_size={min_batch} exceeds budget ({bd1.total_mb:.1f}MB > {memory_limit_mb:.1f}MB)")
            return BatchSizeResult(
                optimal_batch_size=0,
                memory_limit_mb=memory_limit_mb,
                breakdown=bd1,
                candidate_sizes_tried=[min_batch],
                notes=notes,
            )

        if power_of_two:
            # Build list of powers of two in range
            candidates = []
            b = 1
            while b <= max_batch:
                if b >= min_batch:
                    candidates.append(b)
                b *= 2
        else:
            candidates = list(range(min_batch, max_batch + 1))

        # Binary search over candidate list
        lo, hi = 0, len(candidates) - 1
        best_idx = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            batch = candidates[mid]
            bd = _compute_breakdown(config, batch)
            candidates_tried.append(batch)

            if bd.total_mb <= memory_limit_mb:
                best_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        optimal = candidates[best_idx]
        breakdown = _compute_breakdown(config, optimal)

        headroom = memory_limit_mb - breakdown.total_mb
        notes.append(f"Headroom: {headroom:.1f}MB remaining after allocation")
        if optimal == max_batch:
            notes.append("Upper bound reached; larger batches may also fit")

        return BatchSizeResult(
            optimal_batch_size=optimal,
            memory_limit_mb=memory_limit_mb,
            breakdown=breakdown,
            candidate_sizes_tried=sorted(set(candidates_tried)),
            notes=notes,
        )

    def estimate_memory(self, config: ModelConfig, batch_size: int) -> MemoryBreakdown:
        """Return the memory breakdown for a specific batch size without searching."""
        return _compute_breakdown(config, batch_size)

    def recommend_gradient_accumulation(
        self,
        config: ModelConfig,
        memory_limit_mb: float,
        target_effective_batch: int,
    ) -> dict:
        """Suggest micro-batch + gradient accumulation steps to hit an effective batch.

        Returns a dict with keys: micro_batch, accumulation_steps, effective_batch,
        and breakdown.
        """
        result = self.find(config, memory_limit_mb)
        micro = result.optimal_batch_size
        if micro <= 0:
            return {
                "micro_batch": 0,
                "accumulation_steps": 0,
                "effective_batch": 0,
                "breakdown": result.breakdown,
                "feasible": False,
            }
        steps = max(1, target_effective_batch // micro)
        effective = micro * steps
        return {
            "micro_batch": micro,
            "accumulation_steps": steps,
            "effective_batch": effective,
            "breakdown": result.breakdown,
            "feasible": True,
        }
