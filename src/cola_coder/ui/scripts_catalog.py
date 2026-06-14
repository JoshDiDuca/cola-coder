"""CLI scripts catalog helpers for the local UI/dashboard.

Lightweight, read-only inventory of the project's CLI entry points, so the UI can
render a "CLI parity coverage" view (which scripts exist, in which category, and
their one-line purpose). The canonical source is the markdown reference doc at
``.claude/rules/scripts-reference.md`` — a series of ``## <Category>`` headings,
each followed by a ``| script.py | Purpose |`` table. Scripts present on disk in
``scripts/`` but absent from the doc are surfaced as category ``"Uncategorized"``.

All functions are robust to missing or malformed inputs and never raise — they
return an ``{"error": ...}`` dict instead.
"""

from __future__ import annotations

import re
from pathlib import Path

# Path to the canonical reference doc, relative to ``root``.
_REFERENCE_REL = Path(".claude") / "rules" / "scripts-reference.md"

# Directory (relative to ``root``) holding the actual script files.
_SCRIPTS_DIRNAME = "scripts"

# ``## Heading`` (level-2 markdown header) marks a new category.
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")

# A table row: ``| `name.py` | purpose text | (optional extra cols) |``.
# We capture the first two cells; extra cells/whitespace are tolerated.
_ROW_RE = re.compile(r"^\|(.+)$")

_UNCATEGORIZED = "Uncategorized"


def list_scripts(root: str = ".") -> dict:
    """Parse the scripts catalog from ``.claude/rules/scripts-reference.md`` (under root).

    Returns::

        {"scripts": [ {"name": str, "category": str, "purpose": str, "exists": bool} ],
         "categories": list[str],     # distinct categories in document order
         "count": int,
         "on_disk": int}              # number of *.py actually present in scripts/

    ``exists`` = whether ``scripts/<name>`` is present on disk. Include any
    ``scripts/*.py`` NOT in the doc as category ``"Uncategorized"`` + purpose ``""``.
    On any failure return ``{"error": "..."}``. Never raises. If the reference doc is
    missing, fall back to a disk-only catalog (all ``"Uncategorized"``).
    """
    try:
        root_path = Path(root)

        disk_names = _scan_disk(root_path)
        on_disk = len(disk_names)

        doc_path = root_path / _REFERENCE_REL
        if not doc_path.is_file():
            return _disk_only_catalog(disk_names, on_disk)

        try:
            raw = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"error": str(exc)}

        parsed, category_order = _parse_doc(raw)

        # Build the scripts list from the doc, marking on-disk presence.
        scripts: list[dict] = []
        documented: set[str] = set()
        for name, category, purpose in parsed:
            documented.add(name)
            scripts.append(
                {
                    "name": name,
                    "category": category,
                    "purpose": purpose,
                    "exists": name in disk_names,
                }
            )

        # Surface any on-disk scripts that the doc does not mention.
        extra = sorted(disk_names - documented)
        if extra:
            if _UNCATEGORIZED not in category_order:
                category_order.append(_UNCATEGORIZED)
            for name in extra:
                scripts.append(
                    {
                        "name": name,
                        "category": _UNCATEGORIZED,
                        "purpose": "",
                        "exists": True,
                    }
                )

        scripts.sort(key=lambda entry: (entry["category"], entry["name"]))

        return {
            "scripts": scripts,
            "categories": category_order,
            "count": len(scripts),
            "on_disk": on_disk,
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise.
        return {"error": str(exc)}


# ── Internals ───────────────────────────────────────────────────────────────


def _scan_disk(root_path: Path) -> set[str]:
    """Return the set of ``*.py`` filenames in ``<root>/scripts`` (empty if absent)."""
    scripts_dir = root_path / _SCRIPTS_DIRNAME
    if not scripts_dir.is_dir():
        return set()
    names: set[str] = set()
    try:
        for entry in scripts_dir.iterdir():
            if entry.is_file() and entry.suffix == ".py":
                names.add(entry.name)
    except OSError:
        return names
    return names


def _parse_doc(raw: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Parse the reference markdown.

    Returns ``(rows, category_order)`` where ``rows`` is a list of
    ``(name, category, purpose)`` triples in document order and ``category_order``
    is the distinct categories in the order their headings first appear.
    """
    rows: list[tuple[str, str, str]] = []
    category_order: list[str] = []
    current_category = _UNCATEGORIZED

    for line in raw.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            current_category = heading.group(1).strip()
            if current_category not in category_order:
                category_order.append(current_category)
            continue

        parsed_row = _parse_row(line)
        if parsed_row is None:
            continue
        name, purpose = parsed_row

        # Ensure the category is registered even if a table precedes any heading.
        if current_category not in category_order:
            category_order.append(current_category)

        rows.append((name, current_category, purpose))

    return rows, category_order


def _parse_row(line: str) -> tuple[str, str] | None:
    """Parse one table row into ``(name, purpose)`` or ``None`` if not a data row.

    Skips the header row (``| Script | Purpose |``) and separator rows
    (``|---|---|``). Tolerates extra columns and surrounding whitespace.
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None

    # Split on pipes; drop the empty leading/trailing cells produced by the
    # outer borders.
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if len(cells) < 2:
        return None

    first, second = cells[0], cells[1]

    # Separator row: cells are made only of dashes/colons/spaces.
    if first and all(ch in "-: " for ch in first):
        return None

    name = first.strip().strip("`").strip()
    if not name.endswith(".py"):
        return None

    purpose = second.strip()
    return name, purpose


def _disk_only_catalog(disk_names: set[str], on_disk: int) -> dict:
    """Build a catalog from disk alone (all ``Uncategorized``) when the doc is absent."""
    scripts = [
        {
            "name": name,
            "category": _UNCATEGORIZED,
            "purpose": "",
            "exists": True,
        }
        for name in sorted(disk_names)
    ]
    return {
        "scripts": scripts,
        "categories": [_UNCATEGORIZED] if scripts else [],
        "count": len(scripts),
        "on_disk": on_disk,
    }
