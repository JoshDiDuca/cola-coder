"""Reward functions for GRPO reinforcement learning on code generation.

This package provides reward signals that score generated TypeScript code
using the TypeScript compiler (tsc) as a fast, deterministic oracle.

Key classes:
- TscRunner: Unified sandboxed tsc execution engine (used by all callers)
- TscError: Structured tsc diagnostic dataclass
- TypeCheckReward: Score a single TypeScript file with tsc --strict
- BatchTypeChecker: Score a batch of files with a single tsc invocation
- CombinedReward: Multi-signal reward (type check + syntax + completeness)

All tsc execution goes through TscRunner -> SandboxedRunner with hardened
tsconfig.json. All reward functions are OPTIONAL -- if tsc is not installed,
they degrade gracefully with a warning.
"""

from .tsc_runner import TscRunner, TscError
from .type_check import TypeCheckReward
from .batch_type_check import BatchTypeChecker
from .combined import CombinedReward

__all__ = [
    "TscRunner",
    "TscError",
    "TypeCheckReward",
    "BatchTypeChecker",
    "CombinedReward",
]
