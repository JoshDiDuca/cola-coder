"""Domain-aware reward function selection.

Maps domains to appropriate reward functions for GRPO training.
TypeScript domains use type-checking rewards, Python uses execution,
and combined rewards are used for general domains.

Research backing:
- DeepSeek-R1: Domain-appropriate rewards improve sample efficiency
- Using tsc for TypeScript, pytest for Python, combined for general
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


# Domain to reward function mapping
DOMAIN_REWARD_MAP: dict[str, str] = {
    "react": "combined",  # TypeCheck + Syntax + Completeness
    "nextjs": "combined",  # TypeCheck + Syntax + Completeness
    "typescript": "typescript",  # tsc --strict
    "graphql": "combined",  # Syntax + Completeness
    "prisma": "combined",  # Syntax + Completeness
    "zod": "typescript",  # Heavy type usage
    "testing": "python_exec",  # Test execution
    "general": "combined",  # Multi-signal
    "python": "python_exec",  # Test execution
}


def get_reward_for_domain(domain: str) -> str:
    """Get the recommended reward function name for a domain.

    Args:
        domain: Domain name (e.g. "react", "typescript")

    Returns:
        Reward function name (e.g. "combined", "typescript", "python_exec")
    """
    return DOMAIN_REWARD_MAP.get(domain, "combined")


def select_reward_fn(
    domain: str,
    available_rewards: dict[str, Callable] | None = None,
) -> Callable | None:
    """Select the appropriate reward function for a domain.

    Args:
        domain: Domain name
        available_rewards: Dict of {name: reward_fn}. If None, returns None.

    Returns:
        Reward function or None
    """
    if available_rewards is None:
        return None

    reward_name = get_reward_for_domain(domain)
    fn = available_rewards.get(reward_name)
    if fn is None:
        logger.warning(
            "Reward '%s' not found for domain '%s'; falling back to 'combined'.",
            reward_name,
            domain,
        )
        fn = available_rewards.get("combined")
    return fn


def tag_problems_with_domains(
    problems: list[dict],
    router: object | None = None,
    tokenizer: object | None = None,
) -> list[dict]:
    """Add domain tags to problems using heuristic or router.

    If a router is available, uses it. Otherwise uses heuristic detection.

    Args:
        problems: List of problem dicts with 'prompt' key
        router: Optional SemanticRouter or DomainRouter
        tokenizer: Required if router is provided

    Returns:
        Same problems with 'domain' key added/updated
    """
    for problem in problems:
        if problem.get("domain"):
            continue  # Already tagged

        prompt = problem.get("prompt", "")

        if router is not None and tokenizer is not None:
            try:
                domain, confidence = router.route(prompt)
                problem["domain"] = domain
                problem["routing_confidence"] = confidence
                continue
            except Exception:
                logger.debug("Router failed for problem; falling back to heuristic.")

        # Heuristic fallback
        problem["domain"] = _heuristic_domain(prompt)

    return problems


def _heuristic_domain(prompt: str) -> str:
    """Detect domain from prompt using keyword matching."""
    prompt_lower = prompt.lower()

    keywords: dict[str, list[str]] = {
        "react": ["react", "component", "jsx", "tsx", "usestate", "useeffect", "props"],
        "nextjs": ["next", "getserversideprops", "getstaticprops", "next/", "app router"],
        "typescript": ["typescript", "type ", "interface ", "generic", "keyof"],
        "graphql": ["graphql", "query", "mutation", "resolver", "schema"],
        "prisma": ["prisma", "findmany", "findunique", "createmany", "pris"],
        "zod": ["zod", "z.object", "z.string", "z.number", "safeParse"],
        "testing": ["test", "expect", "describe", "jest", "vitest", "assert"],
        "python": ["python", "def ", "import ", "class ", "self."],
    }

    best_domain = "general"
    best_score = 0

    for domain, kws in keywords.items():
        score = sum(1 for kw in kws if kw in prompt_lower)
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain
