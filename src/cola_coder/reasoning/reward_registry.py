"""Pluggable reward registry for GRPO training.

Provides a RewardRegistry that maps string names to reward callables,
making it easy to swap reward functions without changing training code.

Built-in rewards:
    "python_exec"  — existing Python subprocess execution reward (default)
    "typescript"   — TypeScript compiler reward via tsc --strict
    "combined"     — Multi-signal: type check + syntax + style + completeness

Usage::

    from cola_coder.reasoning.reward_registry import RewardRegistry

    reward = RewardRegistry.get("typescript")
    rewards, infos = reward(generations, test_code)

The feature is gated by FEATURE_ENABLED. When disabled, the registry still
works but typescript/combined rewards log a warning.

For a TS dev: think of this as a service locator / DI container for reward
functions — register by name, resolve at runtime.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

FEATURE_ENABLED: bool = True


def is_enabled() -> bool:
    """Return True if the TypeScript rewards feature is enabled."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for all GRPO reward functions.

    A reward function must be callable with (generations, test_code) and
    return (rewards, infos) matching the signature of compute_batch_rewards.

    The ``generations`` argument is a list of model-generated code strings.
    The ``test_code`` argument is used differently by each reward:
    - python_exec: Python test code that gets appended and executed
    - typescript / combined: ignored (TypeScript doesn't run test_code)

    Returns:
        rewards: list of float scores, one per generation
        infos: list of dicts with reward breakdown details
    """

    def __call__(
        self,
        generations: list[str],
        test_code: str,
        **kwargs: object,
    ) -> tuple[list[float], list[dict]]: ...


# ---------------------------------------------------------------------------
# Adapter: wrap TypeCheckReward → RewardFunction signature
# ---------------------------------------------------------------------------


def _make_typescript_reward() -> RewardFunction:
    """Build a RewardFunction wrapper around TypeCheckReward.

    Wraps the single-file tsc scorer so it accepts the same
    (generations, test_code, **kwargs) signature as the Python reward.
    """
    from cola_coder.reasoning.rewards.type_check import TypeCheckReward

    _checker = TypeCheckReward()

    def _typescript_reward(
        generations: list[str],
        test_code: str,
        **kwargs: object,
    ) -> tuple[list[float], list[dict]]:
        rewards: list[float] = []
        infos: list[dict] = []
        for code in generations:
            score = _checker.score(code)
            # Clamp to [0, 1] for GRPO stability (score can be -0.5 for syntax errors)
            clamped = max(0.0, score)
            detailed = _checker.detailed_score(code)
            rewards.append(clamped)
            infos.append({
                "correct": score == 1.0,
                "raw_score": score,
                "clamped_score": clamped,
                "num_errors": detailed.get("num_errors", -1),
                "has_syntax_errors": detailed.get("has_syntax_errors", False),
                "error_codes": detailed.get("error_codes", []),
                "reward_type": "typescript",
            })
        return rewards, infos

    return _typescript_reward  # type: ignore[return-value]


def _make_combined_reward() -> RewardFunction:
    """Build a RewardFunction wrapper around CombinedReward.

    Wraps the multi-signal scorer so it accepts the same
    (generations, test_code, **kwargs) signature as the Python reward.
    """
    from cola_coder.reasoning.rewards.combined import CombinedReward

    _combined = CombinedReward()

    def _combined_reward(
        generations: list[str],
        test_code: str,
        **kwargs: object,
    ) -> tuple[list[float], list[dict]]:
        rewards: list[float] = []
        infos: list[dict] = []
        for code in generations:
            detailed = _combined.detailed_score(code)
            score = detailed["combined_score"]
            clamped = max(0.0, score)
            rewards.append(clamped)
            infos.append({
                "correct": score >= 0.9,
                "raw_score": score,
                "clamped_score": clamped,
                "type_score": detailed.get("type_score"),
                "syntax_score": detailed.get("syntax_score"),
                "style_score": detailed.get("style_score"),
                "completeness_score": detailed.get("completeness_score"),
                "weights": detailed.get("weights", {}),
                "reward_type": "combined",
            })
        return rewards, infos

    return _combined_reward  # type: ignore[return-value]


def _make_python_reward() -> RewardFunction:
    """Build a RewardFunction wrapper around the existing Python exec reward.

    Forwards to compute_batch_rewards, preserving the original behaviour.
    """
    from cola_coder.reasoning.reward import compute_batch_rewards

    def _python_reward(
        generations: list[str],
        test_code: str,
        **kwargs: object,
    ) -> tuple[list[float], list[dict]]:
        max_thinking_tokens: int = int(kwargs.get("max_thinking_tokens", 512))
        return compute_batch_rewards(
            generations,
            test_code,
            max_thinking_tokens=max_thinking_tokens,
        )

    return _python_reward  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Lazy factory type: () -> RewardFunction
_RewardFactory = Callable[[], RewardFunction]

# Built-in reward names (for CLI choices)
BUILTIN_NAMES: tuple[str, ...] = ("python_exec", "typescript", "combined")


class RewardRegistry:
    """Registry of named reward functions for GRPO training.

    All built-in rewards are registered at import time.  Custom rewards
    can be added via :meth:`register`.

    Example::

        # Retrieve a built-in reward
        reward_fn = RewardRegistry.get("typescript")
        rewards, infos = reward_fn(generations, test_code)

        # Register a custom reward
        @RewardRegistry.register("my_reward")
        def my_reward(generations, test_code, **kwargs):
            return [0.5] * len(generations), [{} for _ in generations]
    """

    _factories: dict[str, _RewardFactory] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    @classmethod
    def register(
        cls,
        name: str,
        factory: _RewardFactory | None = None,
    ) -> Callable:
        """Register a reward factory under *name*.

        Can be used as a plain call or as a decorator::

            RewardRegistry.register("my_reward", lambda: my_fn)

            @RewardRegistry.register("my_reward")
            def my_fn(generations, test_code, **kwargs): ...

        Args:
            name: Unique string key for this reward.
            factory: Zero-argument callable that returns a RewardFunction.
                     When used as a decorator the decorated function is
                     treated as the RewardFunction itself (not as a factory).
        """
        if factory is not None:
            cls._factories[name] = factory
            return factory

        # Used as a decorator — the decorated object IS the reward function
        def _decorator(fn: RewardFunction) -> RewardFunction:
            cls._factories[name] = lambda: fn
            return fn

        return _decorator

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    @classmethod
    def get(cls, name: str) -> RewardFunction:
        """Return the reward function registered under *name*.

        Args:
            name: One of the registered reward names.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._factories:
            available = ", ".join(sorted(cls._factories))
            raise KeyError(
                f"Unknown reward '{name}'. Available rewards: {available}"
            )

        if not is_enabled() and name != "python_exec":
            logger.warning(
                "TypeScript rewards feature is disabled (FEATURE_ENABLED=False). "
                "Returning '%s' reward anyway — set FEATURE_ENABLED=True to suppress this warning.",
                name,
            )

        return cls._factories[name]()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @classmethod
    def list_available(cls) -> list[str]:
        """Return a sorted list of registered reward names."""
        return sorted(cls._factories)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return True if *name* is registered."""
        return name in cls._factories


# ---------------------------------------------------------------------------
# Auto-register built-ins
# ---------------------------------------------------------------------------

RewardRegistry.register("python_exec", _make_python_reward)
RewardRegistry.register("typescript", _make_typescript_reward)
RewardRegistry.register("combined", _make_combined_reward)
