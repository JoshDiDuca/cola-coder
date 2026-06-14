"""Read-only view of the semantic domain router subsystem for the local UI.

Pure library module (no Rich, no CLI) that summarizes the router checkpoint
directory (``checkpoints/router/``) and reports the domain labels the router
classifies. The domain list is sourced from the project's real source of truth
(``cola_coder.features.router_model.DEFAULT_DOMAINS``) with a static fallback.

All functions are best-effort and never raise — they return JSON-serializable
results or an ``{"error": ...}`` dict instead.
"""

from __future__ import annotations

from pathlib import Path

# Static fallback mirroring features.router_model.DEFAULT_DOMAINS — used only
# when the canonical constant cannot be imported.
_FALLBACK_DOMAINS = ["react", "nextjs", "graphql", "prisma", "zod", "testing", "general"]


def _step_from_name(name: str) -> int | None:
    """Parse the integer step out of a ``step_<n>`` directory name."""
    if not name.startswith("step_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (ValueError, IndexError):
        return None


def _resolve_domains() -> list[str]:
    """Return the router's domain labels from the canonical source of truth."""
    try:
        from cola_coder.features.router_model import DEFAULT_DOMAINS

        domains = list(DEFAULT_DOMAINS)
        if domains:
            return domains
    except Exception:
        pass
    return list(_FALLBACK_DOMAINS)


def router_overview(root: str = ".") -> dict:
    """Summarize the semantic router. Returns:
      {"has_router": bool,                          # any router checkpoint dir present
       "checkpoints": [ {"path": str, "name": str, "step": int | None} ],  # newest-first
       "domains": list[str]}                         # the domain labels the router classifies
    Resolve `checkpoints/router/` under root (and any step_* subdirs, like status.list_checkpoints).
    `domains` comes from the real domain source-of-truth you found (fallback to a sensible
    default list if none is locatable). On any failure return {"error": "..."}. Never raise.
    """
    try:
        domains = _resolve_domains()
        router_dir = Path(root) / "checkpoints" / "router"

        if not router_dir.is_dir():
            return {"has_router": False, "checkpoints": [], "domains": domains}

        checkpoints: list[dict] = []
        try:
            entries = [d for d in router_dir.iterdir() if d.is_dir()]
        except OSError:
            entries = []

        for entry in entries:
            step = _step_from_name(entry.name)
            if step is None:
                continue
            checkpoints.append(
                {
                    "path": str(entry),
                    "name": entry.name,
                    "step": step,
                }
            )

        # Newest-first by step number.
        checkpoints.sort(key=lambda c: c["step"], reverse=True)

        return {
            "has_router": True,
            "checkpoints": checkpoints,
            "domains": domains,
        }
    except Exception as exc:  # never raise
        return {"error": str(exc)}
