"""Gradient Accumulation Calculator.

Recommends gradient_accumulation_steps given a target effective batch size,
micro batch size, and number of GPUs. Prints a formatted table showing all
combinations of micro_batch × grad_accum × num_gpu = effective_batch.

For a TS dev analogy: effective batch size is like the "logical" transaction size,
while micro batch is what actually fits in VRAM at one time. Gradient accumulation
sums gradients over N micro steps before updating weights — same math, less memory.
"""

from __future__ import annotations

from cola_coder.cli import cli

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return whether this feature is enabled."""
    return FEATURE_ENABLED


class GradAccumCalculator:
    """Recommends gradient_accumulation_steps for a desired effective batch size.

    Args:
        target_effective_batch: The total batch size you want across all steps.
        micro_batch: Per-GPU batch size that fits in VRAM.
        num_gpus: Number of GPUs participating in data-parallel training.
    """

    def __init__(
        self,
        target_effective_batch: int,
        micro_batch: int,
        num_gpus: int = 1,
    ) -> None:
        if target_effective_batch <= 0:
            raise ValueError(f"target_effective_batch must be positive, got {target_effective_batch}")
        if micro_batch <= 0:
            raise ValueError(f"micro_batch must be positive, got {micro_batch}")
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be positive, got {num_gpus}")

        self.target_effective_batch = target_effective_batch
        self.micro_batch = micro_batch
        self.num_gpus = num_gpus

    # ── Core calculation ──────────────────────────────────────────────────────

    def recommended_grad_accum(self) -> int:
        """Return the exact gradient_accumulation_steps to hit the target batch.

        effective_batch = micro_batch * grad_accum * num_gpus
        grad_accum = effective_batch / (micro_batch * num_gpus)

        If not exactly divisible, rounds up and warns.
        """
        tokens_per_accum = self.micro_batch * self.num_gpus
        if self.target_effective_batch % tokens_per_accum == 0:
            return self.target_effective_batch // tokens_per_accum
        # Round up to the nearest integer
        return (self.target_effective_batch + tokens_per_accum - 1) // tokens_per_accum

    def actual_effective_batch(self) -> int:
        """Return the actual effective batch after rounding grad_accum."""
        return self.micro_batch * self.recommended_grad_accum() * self.num_gpus

    def is_exact(self) -> bool:
        """Return True if target is exactly achievable with integer grad_accum."""
        tokens_per_accum = self.micro_batch * self.num_gpus
        return self.target_effective_batch % tokens_per_accum == 0

    # ── Table display ─────────────────────────────────────────────────────────

    def print_table(self, candidates: list[int] | None = None) -> None:
        """Print a table showing micro_batch × grad_accum × num_gpu = effective_batch.

        Args:
            candidates: Optional list of grad_accum values to show. If None,
                auto-generates a set of sensible candidates (powers of 2, etc.)
        """
        if candidates is None:
            candidates = self._generate_candidates()

        rec = self.recommended_grad_accum()
        cli.header(
            f"Gradient Accumulation Options  "
            f"(micro_batch={self.micro_batch}, num_gpus={self.num_gpus})"
        )
        rows: dict[str, str] = {}
        for ga in sorted(set(candidates)):
            if ga <= 0:
                continue
            eff = self.micro_batch * ga * self.num_gpus
            is_target = eff == self.target_effective_batch
            is_rec = ga == rec
            if is_target:
                match_str = "YES"
            elif is_rec:
                match_str = "closest"
            else:
                match_str = "-"
            label = f"grad_accum={ga}" + (" *" if is_rec else "")
            rows[label] = f"{eff:>12,}   {match_str}"
        cli.kv_table(rows, title="grad_accum → effective_batch  [matches target?]")
        cli.info(
            "Recommendation",
            f"gradient_accumulation_steps = {rec}  "
            f"-> effective_batch = {self.actual_effective_batch():,}",
        )
        if not self.is_exact():
            cli.warn(
                f"target={self.target_effective_batch:,} is not exactly achievable. "
                f"Actual will be {self.actual_effective_batch():,}."
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_candidates(self) -> list[int]:
        """Generate a sensible range of grad_accum candidates to display."""
        rec = self.recommended_grad_accum()
        candidates = set()
        # Powers of 2 up to 64
        for exp in range(7):
            candidates.add(2**exp)
        # Neighbourhood around recommendation
        for delta in range(-4, 8):
            v = rec + delta
            if v > 0:
                candidates.add(v)
        # Also include explicit factors of the target
        tokens_per_accum = self.micro_batch * self.num_gpus
        if tokens_per_accum > 0:
            for v in range(1, min(self.target_effective_batch // tokens_per_accum + 2, 65)):
                candidates.add(v)
        return sorted(candidates)


# ── Convenience function ──────────────────────────────────────────────────────

def recommend(
    target_effective_batch: int,
    micro_batch: int,
    num_gpus: int = 1,
    *,
    print_table: bool = True,
) -> int:
    """Quick helper: return recommended grad_accum and optionally print a table.

    Args:
        target_effective_batch: Desired total effective batch size.
        micro_batch: Per-GPU micro batch size.
        num_gpus: Number of GPUs (default 1).
        print_table: Whether to print the candidate table (default True).

    Returns:
        Recommended gradient_accumulation_steps (int).
    """
    calc = GradAccumCalculator(target_effective_batch, micro_batch, num_gpus)
    if print_table:
        calc.print_table()
    return calc.recommended_grad_accum()
