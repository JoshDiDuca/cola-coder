"""
Position interpolation for extending context length beyond what a model was trained on.

Implements three strategies:
  - linear: scale position indices directly (simple, degrades at high ratios)
  - ntk:    NTK-aware scaling — adjusts RoPE base frequency so high-freq dims
            stay sharp while low-freq dims absorb the stretch
  - yarn:   YaRN (yet another RoPE extension) — per-dimension interpolation factor
            that blends no-interpolation and linear based on wavelength
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InterpolationConfig:
    original_max_len: int = 2048
    target_max_len: int = 8192
    method: Literal["linear", "ntk", "yarn"] = "ntk"

    def __post_init__(self) -> None:
        if self.method not in ("linear", "ntk", "yarn"):
            raise ValueError(f"method must be 'linear', 'ntk', or 'yarn', got '{self.method}'")
        if self.target_max_len < self.original_max_len:
            raise ValueError("target_max_len must be >= original_max_len")


# ---------------------------------------------------------------------------
# Core frequency computation
# ---------------------------------------------------------------------------

def compute_rope_freqs(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Compute standard RoPE frequency table.

    Each row i corresponds to position i; each column pair (2j, 2j+1) uses
    theta_j = base^(-2j/dim).

    Returns shape (max_len, dim // 2) — the angles (position * theta_j) for
    every position and every frequency dimension.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    half = dim // 2
    # theta_j = 1 / base^(2j/dim),  j = 0 .. half-1
    j = torch.arange(half, dtype=torch.float32)
    thetas = 1.0 / (base ** (2.0 * j / dim))           # (half,)

    positions = torch.arange(max_len, dtype=torch.float32)  # (max_len,)
    freqs = torch.outer(positions, thetas)              # (max_len, half)
    return freqs


# ---------------------------------------------------------------------------
# Interpolation methods
# ---------------------------------------------------------------------------

def linear_position_interpolation(
    freqs: torch.Tensor,
    original_len: int,
    target_len: int,
) -> torch.Tensor:
    """
    Scale position indices linearly so that position target_len-1 maps to
    original_len-1.

    This is equivalent to multiplying every row (position) by the ratio
    original_len / target_len, then recomputing for target_len positions.

    Args:
        freqs:        (original_len, half_dim) — existing frequency table.
        original_len: context length the model was trained on.
        target_len:   desired extended context length.

    Returns:
        (target_len, half_dim) frequency table.
    """
    if target_len == original_len:
        return freqs

    scale = original_len / target_len

    # Derive theta values from the first row (position 1) of the original table.
    # freqs[1] = 1 * thetas  =>  thetas = freqs[1]
    thetas = freqs[1] if freqs.shape[0] > 1 else freqs[0]   # (half_dim,)

    positions = torch.arange(target_len, dtype=freqs.dtype, device=freqs.device)
    scaled_positions = positions * scale                       # (target_len,)
    extended = torch.outer(scaled_positions, thetas)           # (target_len, half_dim)
    return extended


def ntk_aware_interpolation(
    freqs: torch.Tensor,
    original_len: int,
    target_len: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    NTK-aware (Neural Tangent Kernel) scaling.

    Instead of scaling positions, rescale the RoPE base so that the effective
    maximum position in the new base equals target_len - 1:

        new_base = base * (target_len / original_len) ^ (dim / (dim - 2))

    High-frequency dimensions (small j) are nearly unaffected; low-frequency
    dimensions absorb the extension — exactly what NTK theory prescribes.

    Args:
        freqs:        (original_len, half_dim) — existing frequency table.
        original_len: context length the model was trained on.
        target_len:   desired extended context length.
        base:         original RoPE base (default 10000).

    Returns:
        (target_len, half_dim) frequency table.
    """
    if target_len == original_len:
        return freqs

    half_dim = freqs.shape[1]
    dim = half_dim * 2
    scale = target_len / original_len

    # Rescaled base
    new_base = base * (scale ** (dim / (dim - 2)))

    # Recompute thetas with the new base
    j = torch.arange(half_dim, dtype=freqs.dtype, device=freqs.device)
    thetas = 1.0 / (new_base ** (2.0 * j / dim))          # (half_dim,)

    positions = torch.arange(target_len, dtype=freqs.dtype, device=freqs.device)
    extended = torch.outer(positions, thetas)               # (target_len, half_dim)
    return extended


def _yarn_interpolation(
    freqs: torch.Tensor,
    original_len: int,
    target_len: int,
    base: float = 10000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """
    YaRN (Yet Another RoPE extensioN) per-dimension blending.

    Each dimension gets its own interpolation factor r_j in [0, 1]:
      - dims with wavelength << original_len  -> r_j = 1  (no interpolation)
      - dims with wavelength >> original_len  -> r_j = scale (full linear)
      - in between                            -> smooth ramp

    Args:
        freqs:        (original_len, half_dim) — existing frequency table.
        original_len: context length the model was trained on.
        target_len:   desired extended context length.
        base:         original RoPE base.
        beta_fast:    wavelength boundary below which dims are NOT interpolated.
        beta_slow:    wavelength boundary above which dims ARE fully interpolated.

    Returns:
        (target_len, half_dim) frequency table.
    """
    if target_len == original_len:
        return freqs

    half_dim = freqs.shape[1]
    dim = half_dim * 2
    scale = target_len / original_len

    j = torch.arange(half_dim, dtype=freqs.dtype, device=freqs.device)
    thetas = 1.0 / (base ** (2.0 * j / dim))              # (half_dim,) original thetas

    # Wavelength of each dimension: lambda_j = 2*pi / theta_j
    wavelengths = 2.0 * math.pi / thetas                  # (half_dim,)

    # Compute per-dimension interpolation ratio
    # r_j = 0  -> use original theta (no stretch needed, high-freq)
    # r_j = 1  -> apply linear scale (low-freq, needs extension)
    low = beta_fast                  # wavelength below which we don't interpolate
    high = beta_slow * original_len  # wavelength above which we fully interpolate

    # Clamp-ramp: ramp from 0 to 1 as wavelength goes from low to high
    ramp = (wavelengths - low) / (high - low + 1e-8)
    ramp = ramp.clamp(0.0, 1.0)                            # (half_dim,)

    # Blend: interpolated theta uses linear scaling; un-interpolated keeps original
    thetas_interp = thetas / scale                         # positions compressed
    thetas_blended = thetas * (1.0 - ramp) + thetas_interp * ramp  # (half_dim,)

    positions = torch.arange(target_len, dtype=freqs.dtype, device=freqs.device)
    extended = torch.outer(positions, thetas_blended)      # (target_len, half_dim)
    return extended


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def extend_rope_freqs(freqs: torch.Tensor, config: InterpolationConfig) -> torch.Tensor:
    """
    Apply the interpolation method specified in *config* to *freqs*.

    Args:
        freqs:  (original_max_len, half_dim) frequency table from compute_rope_freqs.
        config: InterpolationConfig describing original length, target length, method.

    Returns:
        (target_max_len, half_dim) extended frequency table.
    """
    if config.method == "linear":
        return linear_position_interpolation(
            freqs, config.original_max_len, config.target_max_len
        )
    elif config.method == "ntk":
        return ntk_aware_interpolation(
            freqs, config.original_max_len, config.target_max_len
        )
    elif config.method == "yarn":
        return _yarn_interpolation(
            freqs, config.original_max_len, config.target_max_len
        )
    else:
        raise ValueError(f"Unknown interpolation method: {config.method}")


# ---------------------------------------------------------------------------
# Quality impact estimator
# ---------------------------------------------------------------------------

def estimate_quality_impact(original_len: int, target_len: int) -> dict:
    """
    Heuristic estimate of quality degradation when extending context.

    Returns a dict with the following keys:
        scale_factor        (float)  — target / original
        perplexity_increase (float)  — rough % increase in perplexity
        coherence_score     (float)  — 0-1, estimated long-range coherence
        recommended_method  (str)    — suggested interpolation strategy
        notes               (list[str]) — human-readable observations
    """
    if target_len < original_len:
        return {
            "scale_factor": target_len / original_len,
            "perplexity_increase": 0.0,
            "coherence_score": 1.0,
            "recommended_method": "none",
            "notes": ["target_len < original_len — no extension needed"],
        }

    scale = target_len / original_len
    notes: list[str] = []

    # Perplexity increase grows roughly linearly with log of scale factor,
    # calibrated to empirical results from the LongRoPE / YaRN papers:
    #   scale=2 -> ~5-10% increase,  scale=4 -> ~15-25%,  scale=8 -> ~30-50%
    perplexity_increase = max(0.0, (math.log2(scale) ** 1.3) * 8.0)

    # Coherence degrades more steeply at high extension ratios
    coherence_score = max(0.0, 1.0 - 0.12 * math.log2(scale))

    if scale <= 2.0:
        recommended_method = "linear"
        notes.append("Low extension ratio — linear interpolation is sufficient.")
    elif scale <= 4.0:
        recommended_method = "ntk"
        notes.append("Moderate extension — NTK-aware scaling is recommended.")
    else:
        recommended_method = "yarn"
        notes.append("High extension ratio — YaRN blended interpolation is recommended.")

    if scale > 8.0:
        notes.append(
            f"WARNING: scale factor {scale:.1f}x is very high; "
            "fine-tuning on long sequences is strongly advised."
        )

    if perplexity_increase > 20.0:
        notes.append(
            f"Estimated perplexity increase of {perplexity_increase:.1f}% is significant; "
            "consider fine-tuning with LoRA on longer examples."
        )

    return {
        "scale_factor": scale,
        "perplexity_increase": round(perplexity_increase, 2),
        "coherence_score": round(coherence_score, 4),
        "recommended_method": recommended_method,
        "notes": notes,
    }
