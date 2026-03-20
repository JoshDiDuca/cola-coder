"""Multi-File Context: include related files in the generation context.

When generating code for a file, the model benefits from seeing related files:
- Imported modules and their type definitions
- Test files for the target module
- Configuration files
- Shared types and interfaces

This feature finds and formats related files to include in the prompt context.

For a TS dev: like how your IDE resolves imports to show you type info —
this does the same thing to give the AI model more context.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ContextFile:
    """A file included in the generation context."""
    path: str
    content: str
    relationship: str  # "import", "test", "config", "sibling", "type_def"
    priority: int = 0  # Higher = more important
    truncated: bool = False

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (~4 chars per token)."""
        return len(self.content) // 4


@dataclass
class ContextWindow:
    """A collection of context files that fit within a token budget."""
    target_file: str
    files: list[ContextFile] = field(default_factory=list)
    max_tokens: int = 2048
    separator: str = "\n// ─── {path} ───\n"

    @property
    def total_tokens(self) -> int:
        return sum(f.token_estimate for f in self.files)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    def add_file(self, ctx_file: ContextFile) -> bool:
        """Add a file if it fits within the token budget.

        Returns:
            True if file was added, False if it would exceed the budget
        """
        if ctx_file.token_estimate > self.remaining_tokens:
            # Try truncating to fit
            available_chars = self.remaining_tokens * 4
            if available_chars > 100:  # Only add if meaningful amount fits
                ctx_file.content = ctx_file.content[:available_chars] + "\n// ... truncated"
                ctx_file.truncated = True
            else:
                return False
        self.files.append(ctx_file)
        return True

    def format_prompt(self) -> str:
        """Format all context files into a single prompt string."""
        parts = []
        # Sort by priority (higher first)
        sorted_files = sorted(self.files, key=lambda f: f.priority, reverse=True)

        for f in sorted_files:
            header = self.separator.format(path=f.path)
            parts.append(f"{header}{f.content}")

        return "\n".join(parts)

    def summary(self) -> str:
        lines = [f"Context for {self.target_file}:"]
        for f in self.files:
            trunc = " (truncated)" if f.truncated else ""
            lines.append(f"  [{f.relationship}] {f.path} (~{f.token_estimate} tokens){trunc}")
        lines.append(f"Total: ~{self.total_tokens}/{self.max_tokens} tokens")
        return "\n".join(lines)


class ContextBuilder:
    """Build multi-file context for code generation."""

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)

    def find_imports(self, file_path: str) -> list[str]:
        """Find imported file paths from a source file.

        Resolves relative imports to actual file paths.
        """
        source = Path(file_path)
        if not source.exists():
            return []

        content = source.read_text(errors="replace")
        imports = []

        # TypeScript/JavaScript imports
        patterns = [
            r'import\s+.*?\s+from\s+["\'](\.[^"\']+)["\']',  # import X from './foo'
            r'import\s*\(\s*["\'](\.[^"\']+)["\']\s*\)',       # import('./foo')
            r'require\s*\(\s*["\'](\.[^"\']+)["\']\s*\)',       # require('./foo')
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                import_path = match.group(1)
                resolved = self._resolve_import(source.parent, import_path)
                if resolved:
                    imports.append(str(resolved))

        # Python imports (relative)
        py_patterns = [
            r'from\s+\.(\w+)\s+import',  # from .module import X
            r'from\s+\.\.\s*(\w+)\s+import',  # from ..module import X
        ]
        for pattern in py_patterns:
            for match in re.finditer(pattern, content):
                module = match.group(1)
                resolved = source.parent / f"{module}.py"
                if resolved.exists():
                    imports.append(str(resolved))

        return imports

    def find_test_file(self, file_path: str) -> str | None:
        """Find the test file for a given source file."""
        source = Path(file_path)
        stem = source.stem
        parent = source.parent

        # Common test file patterns
        test_patterns = [
            parent / f"test_{stem}{source.suffix}",
            parent / f"{stem}.test{source.suffix}",
            parent / f"{stem}.spec{source.suffix}",
            parent / "__tests__" / f"{stem}{source.suffix}",
            parent / "__tests__" / f"{stem}.test{source.suffix}",
            parent.parent / "tests" / f"test_{stem}{source.suffix}",
            parent.parent / "tests" / f"{stem}.test{source.suffix}",
        ]

        for test_path in test_patterns:
            if test_path.exists():
                return str(test_path)
        return None

    def find_siblings(self, file_path: str, max_siblings: int = 3) -> list[str]:
        """Find sibling files in the same directory."""
        source = Path(file_path)
        siblings = []

        if source.parent.exists():
            for f in sorted(source.parent.iterdir()):
                if f.is_file() and f != source and f.suffix == source.suffix:
                    siblings.append(str(f))
                    if len(siblings) >= max_siblings:
                        break
        return siblings

    def find_type_definitions(self, file_path: str) -> list[str]:
        """Find TypeScript type definition files related to a source file."""
        source = Path(file_path)
        type_files = []

        # Look for .d.ts files
        parent = source.parent
        for f in parent.glob("*.d.ts"):
            type_files.append(str(f))

        # Look for types/ directory
        types_dir = parent / "types"
        if types_dir.exists():
            for f in types_dir.glob("*.ts"):
                type_files.append(str(f))

        return type_files[:5]  # Limit

    def build_context(
        self,
        file_path: str,
        max_tokens: int = 2048,
        include_imports: bool = True,
        include_tests: bool = True,
        include_siblings: bool = False,
        include_types: bool = True,
    ) -> ContextWindow:
        """Build a context window for a target file.

        Args:
            file_path: The file being generated/completed
            max_tokens: Maximum tokens for the context window
            include_imports: Include imported files
            include_tests: Include test files
            include_siblings: Include sibling files
            include_types: Include type definition files

        Returns:
            ContextWindow with prioritized context files
        """
        ctx = ContextWindow(target_file=file_path, max_tokens=max_tokens)

        # Priority order: imports > types > tests > siblings
        if include_imports:
            for imp_path in self.find_imports(file_path):
                content = self._read_file(imp_path)
                if content:
                    ctx.add_file(ContextFile(
                        path=imp_path, content=content,
                        relationship="import", priority=10,
                    ))

        if include_types:
            for type_path in self.find_type_definitions(file_path):
                content = self._read_file(type_path)
                if content:
                    ctx.add_file(ContextFile(
                        path=type_path, content=content,
                        relationship="type_def", priority=8,
                    ))

        if include_tests:
            test_path = self.find_test_file(file_path)
            if test_path:
                content = self._read_file(test_path)
                if content:
                    ctx.add_file(ContextFile(
                        path=test_path, content=content,
                        relationship="test", priority=5,
                    ))

        if include_siblings:
            for sib_path in self.find_siblings(file_path):
                content = self._read_file(sib_path)
                if content:
                    ctx.add_file(ContextFile(
                        path=sib_path, content=content,
                        relationship="sibling", priority=2,
                    ))

        return ctx

    def _resolve_import(self, base_dir: Path, import_path: str) -> Path | None:
        """Resolve a relative import to an actual file path."""
        # Try with various extensions
        extensions = ["", ".ts", ".tsx", ".js", ".jsx", ".py", "/index.ts", "/index.js"]

        for ext in extensions:
            resolved = base_dir / f"{import_path}{ext}"
            if resolved.exists() and resolved.is_file():
                return resolved
        return None

    def _read_file(self, path: str, max_chars: int = 8000) -> str | None:
        """Read a file, returning None if it doesn't exist or fails."""
        try:
            p = Path(path)
            if not p.exists():
                return None
            content = p.read_text(errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n// ... truncated"
            return content
        except Exception:
            return None
