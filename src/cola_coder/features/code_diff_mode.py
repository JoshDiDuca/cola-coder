"""Code Diff Mode: compare original and generated code side by side.

Shows what the model changed, added, or removed using unified diff format.
Useful for code review, refactoring validation, and edit suggestions.

For a TS dev: like running `git diff` but comparing your original code
with what the AI generated.
"""

import difflib
from dataclasses import dataclass

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class DiffStats:
    """Statistics about a code diff."""
    additions: int = 0
    deletions: int = 0
    modifications: int = 0
    unchanged: int = 0

    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions + self.modifications

    @property
    def change_ratio(self) -> float:
        total = self.additions + self.deletions + self.modifications + self.unchanged
        if total == 0:
            return 0.0
        return self.total_changes / total

    def summary(self) -> str:
        parts = []
        if self.additions:
            parts.append(f"+{self.additions}")
        if self.deletions:
            parts.append(f"-{self.deletions}")
        if self.modifications:
            parts.append(f"~{self.modifications}")
        return ", ".join(parts) if parts else "no changes"


class CodeDiff:
    """Generate and display diffs between code versions."""

    def __init__(self, context_lines: int = 3):
        """
        Args:
            context_lines: Number of unchanged lines to show around changes
        """
        self.context_lines = context_lines

    def unified_diff(
        self,
        original: str,
        modified: str,
        original_label: str = "original",
        modified_label: str = "modified",
    ) -> str:
        """Generate a unified diff between two code strings.

        Args:
            original: The original code
            modified: The modified/generated code
            original_label: Label for the original file
            modified_label: Label for the modified file

        Returns:
            Unified diff string (like `git diff` output)
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=original_label,
            tofile=modified_label,
            n=self.context_lines,
        )
        return "".join(diff)

    def compute_stats(self, original: str, modified: str) -> DiffStats:
        """Compute diff statistics between two code strings.

        Args:
            original: The original code
            modified: The modified code

        Returns:
            DiffStats with counts of additions, deletions, modifications
        """
        stats = DiffStats()
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                stats.unchanged += (i2 - i1)
            elif op == "insert":
                stats.additions += (j2 - j1)
            elif op == "delete":
                stats.deletions += (i2 - i1)
            elif op == "replace":
                stats.modifications += max(i2 - i1, j2 - j1)

        return stats

    def side_by_side(
        self,
        original: str,
        modified: str,
        width: int = 80,
    ) -> str:
        """Generate a side-by-side diff view.

        Args:
            original: The original code
            modified: The modified code
            width: Total width of the output (each side gets width/2)

        Returns:
            Side-by-side diff as a string
        """
        half_width = (width - 3) // 2  # -3 for " | " separator
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
        output_lines = []

        # Header
        orig_header = "Original".center(half_width)
        mod_header = "Modified".center(half_width)
        output_lines.append(f"{orig_header} | {mod_header}")
        output_lines.append("-" * width)

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                for i in range(i1, i2):
                    left = original_lines[i][:half_width].ljust(half_width)
                    right = modified_lines[i - i1 + j1][:half_width].ljust(half_width)
                    output_lines.append(f"{left} | {right}")
            elif op == "replace":
                max_lines = max(i2 - i1, j2 - j1)
                for k in range(max_lines):
                    left_idx = i1 + k
                    right_idx = j1 + k
                    left = (
                        original_lines[left_idx][:half_width].ljust(half_width)
                        if left_idx < i2 else " " * half_width
                    )
                    right = (
                        modified_lines[right_idx][:half_width].ljust(half_width)
                        if right_idx < j2 else " " * half_width
                    )
                    marker = "!"
                    output_lines.append(f"{left} {marker} {right}")
            elif op == "delete":
                for i in range(i1, i2):
                    left = original_lines[i][:half_width].ljust(half_width)
                    right = " " * half_width
                    output_lines.append(f"{left} < {right}")
            elif op == "insert":
                for j in range(j1, j2):
                    left = " " * half_width
                    right = modified_lines[j][:half_width].ljust(half_width)
                    output_lines.append(f"{left} > {right}")

        return "\n".join(output_lines)

    def inline_diff(self, original: str, modified: str) -> str:
        """Generate an inline diff with +/- markers.

        Similar to git diff but without the @@ headers.
        """
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
        output_lines = []

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                for i in range(i1, i2):
                    output_lines.append(f"  {original_lines[i]}")
            elif op == "replace":
                for i in range(i1, i2):
                    output_lines.append(f"- {original_lines[i]}")
                for j in range(j1, j2):
                    output_lines.append(f"+ {modified_lines[j]}")
            elif op == "delete":
                for i in range(i1, i2):
                    output_lines.append(f"- {original_lines[i]}")
            elif op == "insert":
                for j in range(j1, j2):
                    output_lines.append(f"+ {modified_lines[j]}")

        return "\n".join(output_lines)

    def print_diff(
        self,
        original: str,
        modified: str,
        mode: str = "unified",
        original_label: str = "original",
        modified_label: str = "modified",
    ) -> None:
        """Print a formatted diff to the console.

        Args:
            original: Original code
            modified: Modified code
            mode: "unified", "side_by_side", or "inline"
            original_label: Label for original
            modified_label: Label for modified
        """
        try:
            from rich.console import Console
            from rich.syntax import Syntax
            from rich.panel import Panel
            console = Console()
            has_rich = True
        except ImportError:
            has_rich = False

        stats = self.compute_stats(original, modified)

        if mode == "unified":
            diff_text = self.unified_diff(original, modified, original_label, modified_label)
        elif mode == "side_by_side":
            diff_text = self.side_by_side(original, modified)
        elif mode == "inline":
            diff_text = self.inline_diff(original, modified)
        else:
            diff_text = self.unified_diff(original, modified, original_label, modified_label)

        if not diff_text.strip():
            if has_rich:
                console.print("[dim]No changes detected[/dim]")
            else:
                print("No changes detected")
            return

        if has_rich:
            syntax = Syntax(diff_text, "diff", theme="monokai")
            console.print(Panel(
                syntax,
                title=f"[bold]Diff[/bold] ({stats.summary()})",
                subtitle=f"[dim]{original_label} -> {modified_label}[/dim]",
            ))
        else:
            print(f"--- Diff ({stats.summary()}) ---")
            print(diff_text)


def apply_edit_suggestion(
    original: str,
    start_line: int,
    end_line: int,
    replacement: str,
) -> str:
    """Apply an edit suggestion to code.

    Args:
        original: The original code
        start_line: First line to replace (1-indexed)
        end_line: Last line to replace (1-indexed, inclusive)
        replacement: New code to insert

    Returns:
        Modified code with the replacement applied
    """
    lines = original.splitlines(keepends=True)
    # Convert to 0-indexed
    start = max(0, start_line - 1)
    end = min(len(lines), end_line)

    replacement_lines = replacement.splitlines(keepends=True)
    # Ensure last line has newline
    if replacement_lines and not replacement_lines[-1].endswith("\n"):
        replacement_lines[-1] += "\n"

    result_lines = lines[:start] + replacement_lines + lines[end:]
    return "".join(result_lines)
