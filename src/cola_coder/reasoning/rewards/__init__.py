"""Reward functions for GRPO reinforcement learning on code generation.

This package provides reward signals that score generated TypeScript code
using the TypeScript compiler (tsc) as a fast, deterministic oracle.

Key classes:
- TypeCheckReward: Score a single TypeScript file with tsc --strict
- BatchTypeChecker: Score a batch of files with a single tsc invocation
- CombinedReward: Multi-signal reward (type check + syntax + completeness)

All reward functions are OPTIONAL — if tsc is not installed, they degrade
gracefully with a warning.
"""

from .type_check import TypeCheckReward
from .batch_type_check import BatchTypeChecker
from .combined import CombinedReward

__all__ = ["TypeCheckReward", "BatchTypeChecker", "CombinedReward"]
