"""Quick action bar for Cola-Coder.

Common actions with keyboard shortcuts, triggerable from a menu or CLI.
Keeps a registry of named actions with descriptions, shortcuts, and callbacks.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

# Feature toggle - this feature is OPTIONAL
FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Check if this feature is enabled."""
    return FEATURE_ENABLED


@dataclass
class Action:
    """A single quick action."""

    name: str
    description: str
    shortcut: str
    callback: Callable | None
    category: str = "general"


class QuickActionsBar:
    """Registry and display for quick actions with keyboard shortcuts."""

    def __init__(self):
        self._actions: dict[str, Action] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    # Default actions
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        defaults = [
            ("train",     "Start / resume model training",          "t", None, "training"),
            ("evaluate",  "Run evaluation on a checkpoint",         "e", None, "training"),
            ("generate",  "Generate code from a prompt",            "g", None, "inference"),
            ("benchmark", "Run the nano-benchmark suite",           "b", None, "evaluation"),
            ("status",    "Show training status and GPU utilization","s", None, "general"),
            ("help",      "Show help and available commands",        "?", None, "general"),
        ]
        for name, description, shortcut, callback, category in defaults:
            self.register(name, description, shortcut, callback, category)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        shortcut: str,
        callback: Callable | None,
        category: str = "general",
    ) -> None:
        """Register an action (overwrites if name already exists)."""
        self._actions[name] = Action(
            name=name,
            description=description,
            shortcut=shortcut,
            callback=callback,
            category=category,
        )

    def get_actions(self, category: str | None = None) -> list[Action]:
        """Return all actions, optionally filtered by category."""
        actions = list(self._actions.values())
        if category is not None:
            actions = [a for a in actions if a.category == category]
        return actions

    def execute(self, name: str, **kwargs) -> Any:
        """Run an action by name; raises KeyError if not found."""
        action = self._actions.get(name)
        if action is None:
            raise KeyError(f"Unknown action: {name!r}")
        if action.callback is None:
            return None
        return action.callback(**kwargs)

    def get_by_shortcut(self, shortcut: str) -> Action | None:
        """Return the action bound to the given shortcut key, or None."""
        for action in self._actions.values():
            if action.shortcut == shortcut:
                return action
        return None

    def format_bar(self) -> str:
        """Return a formatted one-line action bar string."""
        parts = []
        for action in self._actions.values():
            parts.append(f"[{action.shortcut}] {action.name}")
        return "  ".join(parts)

    def categories(self) -> list[str]:
        """Return a sorted, deduplicated list of all registered categories."""
        return sorted({a.category for a in self._actions.values()})

    def summary(self) -> dict:
        """Return a summary dict with counts and action names per category."""
        result: dict[str, Any] = {
            "total": len(self._actions),
            "categories": {},
        }
        for action in self._actions.values():
            result["categories"].setdefault(action.category, []).append(action.name)
        return result
