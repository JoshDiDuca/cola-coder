"""Rotary Positional Encoding (RoPE).

The problem: attention computes similarity between tokens, but it has no
concept of ORDER. Without position info, the model can't tell the difference
between "x = y + z" and "z = y + x" — both have the same set of tokens.

The solution: RoPE encodes position by ROTATING the query and key vectors.
Each position gets a unique rotation angle. When we compute the dot product
(attention score) between two rotated vectors, the result naturally depends
on the DISTANCE between their positions.

Intuition for a TS dev:
Imagine each token's Q/K vector as an arrow in 2D space. RoPE rotates each
arrow by an angle proportional to its position. Token 0 gets rotated 0°,
token 1 gets rotated θ°, token 2 gets rotated 2θ°, etc. When you compute
dot products between these rotated arrows, nearby tokens naturally have
higher similarity because their arrows point in similar directions.

In practice, we work in pairs of dimensions and use different rotation
frequencies for each pair — like a clock with multiple hands spinning at
different speeds. The fast-spinning hands encode fine-grained position,
the slow-spinning hands encode coarse position.
"""

import torch


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE.

    This creates a table of rotation values that we look up during forward pass.
    Computed once and cached — no learned parameters.

    Args:
        dim: Head dimension (model dim / num heads). Must be even.
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base frequency. Higher = longer-range position patterns.
               10000.0 is the standard value used by LLaMA/Mistral.
        device: Where to store the tensor (CPU or GPU).

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2).
        Each entry is cos(angle) + i*sin(angle) for a specific (position, dimension) pair.
    """
    # Frequency for each pair of dimensions
    # Lower dimensions get higher frequencies (fast rotation),
    # higher dimensions get lower frequencies (slow rotation)
    # This is like having multiple clock hands spinning at different speeds:
    # freqs = [1/θ^0, 1/θ^(2/dim), 1/θ^(4/dim), ...]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Position indices: [0, 1, 2, ..., max_seq_len - 1]
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: every position × every frequency = all rotation angles
    # Shape: (max_seq_len, dim // 2)
    angles = torch.outer(positions, freqs)

    # Convert angles to complex numbers: e^(i*angle) = cos(angle) + i*sin(angle)
    # This is the mathematical way to represent a 2D rotation
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional encoding to query and key tensors.

    The trick: we treat consecutive pairs of dimensions as a complex number,
    multiply by the precomputed rotation (complex multiplication = rotation),
    then convert back to real numbers.

    Args:
        q: Query tensor, shape (batch, seq_len, n_heads, head_dim)
        k: Key tensor, shape (batch, seq_len, n_kv_heads, head_dim)
        freqs: Precomputed frequencies from precompute_rope_freqs.
        start_pos: Starting position in the sequence (used during inference
                    with KV-cache, where we're adding one token at a time).

    Returns:
        Rotated (q, k) tensors with the same shapes as input.
    """
    seq_len = q.shape[1]

    # Slice the frequency table for our positions
    # During training: start_pos=0, seq_len=full_sequence
    # During inference: start_pos=current_position, seq_len=1
    rope_freqs = freqs[start_pos : start_pos + seq_len]

    # Reshape for broadcasting: (seq_len, dim//2) -> (1, seq_len, 1, dim//2)
    rope_freqs = rope_freqs.unsqueeze(0).unsqueeze(2)

    # View the last dimension as pairs of real numbers → complex numbers
    # [a, b, c, d, ...] → [a+bi, c+di, ...]
    # This is the key insight: consecutive dimensions form 2D rotation planes
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Multiply by rotation (complex multiplication = 2D rotation)
    # This is where the actual position encoding happens
    q_rotated = q_complex * rope_freqs
    k_rotated = k_complex * rope_freqs

    # Convert back from complex to real: [a+bi, c+di, ...] → [a, b, c, d, ...]
    q_out = torch.view_as_real(q_rotated).reshape(q.shape)
    k_out = torch.view_as_real(k_rotated).reshape(k.shape)

    return q_out.type_as(q), k_out.type_as(k)


# ---------------------------------------------------------------------------
# YaRN / NTK-aware RoPE scaling for context extension
# ---------------------------------------------------------------------------


def precompute_rope_freqs_yarn(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    factor: float = 8.0,
    original_max_seq_len: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """YaRN: Yet another RoPE extensioN (Peng et al., 2023).

    Extends context window by scaling RoPE frequencies non-uniformly.
    Low frequencies (long-range dependencies) are scaled more than
    high frequencies (local patterns).

    Research: YaRN achieves 128K context with <5% additional compute.
    Used by Qwen2.5-Coder for context extension.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Target sequence length (after extension)
        theta: RoPE base frequency
        factor: Scaling factor (e.g., 8.0 for 4K→32K)
        original_max_seq_len: Original training sequence length
        beta_fast: Fast (high frequency) partition boundary
        beta_slow: Slow (low frequency) partition boundary
        device: Compute device

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2)
    """
    # Compute frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Wavelength thresholds for partitioning
    # Low frequencies (long wavelengths) → scale more
    # High frequencies (short wavelengths) → scale less
    low_freq_wavelen = original_max_seq_len / beta_slow
    high_freq_wavelen = original_max_seq_len / beta_fast

    scaled_freqs = []
    for freq in freqs:
        wavelen = 2 * 3.14159265 / freq.item()

        if wavelen < high_freq_wavelen:
            # High frequency: don't scale (local patterns)
            scaled_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            # Low frequency: full scale (long-range)
            scaled_freqs.append(freq / factor)
        else:
            # Interpolation region: smooth transition
            smooth = (low_freq_wavelen / wavelen - 1) / (beta_fast / beta_slow - 1)
            smooth = max(0, min(1, smooth))
            scaled_freq = (1 - smooth) * freq / factor + smooth * freq
            scaled_freqs.append(scaled_freq)

    freqs = torch.stack(scaled_freqs)

    # Compute positions and angles
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)

    # Convert to complex exponentials
    return torch.polar(torch.ones_like(angles), angles)


def precompute_rope_freqs_ntk(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    factor: float = 8.0,
    device: str = "cpu",
) -> torch.Tensor:
    """NTK-aware RoPE scaling (simpler alternative to YaRN).

    Scales the base theta by the factor, which uniformly extends
    all frequency bands. Simpler but less effective than YaRN.

    Args:
        dim: Head dimension
        max_seq_len: Target sequence length
        theta: Original RoPE base frequency
        factor: Scaling factor
        device: Compute device

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2)
    """
    # Scale theta by factor
    scaled_theta = theta * factor

    # Standard RoPE with scaled theta
    freqs = 1.0 / (scaled_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)

    return torch.polar(torch.ones_like(angles), angles)


def precompute_rope_freqs_linear(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    factor: float = 8.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Linear RoPE scaling (simplest, least effective).

    Simply divides all position indices by the factor.
    Equivalent to pretending the extended context is shorter.

    Args:
        dim: Head dimension
        max_seq_len: Target sequence length
        theta: RoPE base frequency
        factor: Scaling factor
        device: Compute device

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float() / factor
    angles = torch.outer(positions, freqs)

    return torch.polar(torch.ones_like(angles), angles)


def get_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_type: str = "none",
    scaling_factor: float = 1.0,
    original_max_seq_len: int = 4096,
    device: str = "cpu",
) -> torch.Tensor:
    """Get RoPE frequencies with optional scaling.

    Unified interface for all RoPE scaling methods.

    Args:
        dim: Head dimension
        max_seq_len: Target sequence length
        theta: Base frequency
        scaling_type: "none", "linear", "ntk", "yarn"
        scaling_factor: Extension factor
        original_max_seq_len: Original training length
        device: Compute device

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2)
    """
    if scaling_type == "yarn":
        return precompute_rope_freqs_yarn(
            dim,
            max_seq_len,
            theta,
            scaling_factor,
            original_max_seq_len,
            device=device,
        )
    elif scaling_type == "ntk":
        return precompute_rope_freqs_ntk(dim, max_seq_len, theta, scaling_factor, device=device)
    elif scaling_type == "linear":
        return precompute_rope_freqs_linear(dim, max_seq_len, theta, scaling_factor, device=device)
    else:
        # No scaling — use original function
        return precompute_rope_freqs(dim, max_seq_len, theta, device)
