"""Import Graph Enrichment: analyze and visualize code dependencies.

Parses import/require statements to build a dependency graph of a codebase.
Useful for:
- Understanding code structure
- Enriching training data with dependency context
- Finding related files for multi-file context
- Detecting circular dependencies

For a TS dev: like the dependency graph that bundlers (webpack, esbuild) build
internally, but exposed as a tool for analysis and data enrichment.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ImportInfo:
    """Information about a single import statement."""
    source_file: str
    imported_path: str
    imported_names: list[str] = field(default_factory=list)
    is_relative: bool = False
    is_type_only: bool = False
    line_number: int = 0


@dataclass
class FileNode:
    """A node in the import graph representing a file."""
    path: str
    imports: list[str] = field(default_factory=list)  # Files this file imports
    imported_by: list[str] = field(default_factory=list)  # Files that import this file
    import_details: list[ImportInfo] = field(default_factory=list)

    @property
    def dependency_count(self) -> int:
        return len(self.imports)

    @property
    def dependant_count(self) -> int:
        return len(self.imported_by)


class ImportGraph:
    """Build and analyze import dependency graphs."""

    def __init__(self):
        self.nodes: dict[str, FileNode] = {}
        self._circular_deps: list[tuple[str, str]] = []

    def add_file(self, file_path: str, content: str | None = None) -> FileNode:
        """Add a file to the graph and parse its imports.

        Args:
            file_path: Path to the file
            content: File content (if None, reads from disk)

        Returns:
            FileNode for this file
        """
        file_path = str(Path(file_path))

        if file_path not in self.nodes:
            self.nodes[file_path] = FileNode(path=file_path)

        node = self.nodes[file_path]

        if content is None:
            try:
                content = Path(file_path).read_text(errors="replace")
            except Exception:
                return node

        # Parse imports
        imports = self._parse_imports(file_path, content)
        node.import_details = imports

        for imp in imports:
            resolved = imp.imported_path
            node.imports.append(resolved)

            # Create node for imported file if it doesn't exist
            if resolved not in self.nodes:
                self.nodes[resolved] = FileNode(path=resolved)
            self.nodes[resolved].imported_by.append(file_path)

        return node

    def build_from_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> int:
        """Scan a directory and build the import graph.

        Args:
            directory: Root directory to scan
            extensions: File extensions to include (default: .ts, .tsx, .js, .jsx, .py)
            exclude_patterns: Directory patterns to exclude

        Returns:
            Number of files processed
        """
        if extensions is None:
            extensions = [".ts", ".tsx", ".js", ".jsx", ".py"]
        if exclude_patterns is None:
            exclude_patterns = ["node_modules", ".git", "__pycache__", ".venv", "dist", "build"]

        root = Path(directory)
        count = 0

        for ext in extensions:
            for file_path in root.rglob(f"*{ext}"):
                # Skip excluded directories
                if any(excl in file_path.parts for excl in exclude_patterns):
                    continue
                self.add_file(str(file_path))
                count += 1

        self._detect_circular_deps()
        return count

    def get_dependencies(self, file_path: str, depth: int = 1) -> set[str]:
        """Get all dependencies of a file up to a given depth.

        Args:
            file_path: The file to get dependencies for
            depth: How deep to follow the dependency chain

        Returns:
            Set of file paths that this file depends on
        """
        deps = set()
        self._collect_deps(file_path, depth, deps, set())
        return deps

    def get_dependants(self, file_path: str) -> set[str]:
        """Get all files that depend on this file."""
        node = self.nodes.get(str(file_path))
        if not node:
            return set()
        return set(node.imported_by)

    def find_circular_dependencies(self) -> list[tuple[str, str]]:
        """Find all circular dependency pairs."""
        self._detect_circular_deps()
        return self._circular_deps

    def most_imported(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Get the most-imported files (highest dependant count)."""
        ranked = [
            (path, node.dependant_count)
            for path, node in self.nodes.items()
            if node.dependant_count > 0
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def most_dependencies(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Get files with the most dependencies."""
        ranked = [
            (path, node.dependency_count)
            for path, node in self.nodes.items()
            if node.dependency_count > 0
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def leaf_files(self) -> list[str]:
        """Get files with no dependencies (leaf nodes)."""
        return [
            path for path, node in self.nodes.items()
            if node.dependency_count == 0 and node.dependant_count > 0
        ]

    def isolated_files(self) -> list[str]:
        """Get files with no imports and not imported by anything."""
        return [
            path for path, node in self.nodes.items()
            if node.dependency_count == 0 and node.dependant_count == 0
        ]

    def stats(self) -> dict:
        """Get graph statistics."""
        total_edges = sum(len(n.imports) for n in self.nodes.values())
        return {
            "total_files": len(self.nodes),
            "total_edges": total_edges,
            "circular_deps": len(self._circular_deps),
            "leaf_files": len(self.leaf_files()),
            "isolated_files": len(self.isolated_files()),
            "avg_dependencies": total_edges / max(1, len(self.nodes)),
        }

    def to_dot(self) -> str:
        """Export graph in DOT format (for Graphviz visualization)."""
        lines = ["digraph imports {", "  rankdir=LR;"]
        for path, node in self.nodes.items():
            short_name = Path(path).name
            for imp in node.imports:
                imp_short = Path(imp).name
                lines.append(f'  "{short_name}" -> "{imp_short}";')
        lines.append("}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a formatted summary."""
        from cola_coder.cli import cli

        s = self.stats()
        cli.header("Import Graph", f"{s['total_files']} files")
        cli.info("Files", s["total_files"])
        cli.info("Import edges", s["total_edges"])
        cli.info("Avg dependencies", f"{s['avg_dependencies']:.1f}")
        cli.info("Circular deps", s["circular_deps"])
        cli.info("Leaf files", s["leaf_files"])
        cli.info("Isolated files", s["isolated_files"])

        top = self.most_imported(5)
        if top:
            cli.rule("Most Imported")
            for path, count in top:
                cli.info(Path(path).name, f"imported by {count} files")

    # ── Internal ─────────────────────────────────────────────────────

    def _parse_imports(self, file_path: str, content: str) -> list[ImportInfo]:
        """Parse import statements from file content."""
        imports = []
        source = Path(file_path)

        # TypeScript/JavaScript patterns
        ts_patterns = [
            # import X from './path'
            (r'import\s+(?:type\s+)?(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+["\']([^"\']+)["\']', False),
            # import './path'  (side-effect import)
            (r'import\s+["\']([^"\']+)["\']', False),
            # require('./path')
            (r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', False),
        ]

        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern, _ in ts_patterns:
                for match in re.finditer(pattern, line):
                    raw_path = match.group(1)
                    is_relative = raw_path.startswith(".")
                    is_type = "import type" in line

                    if is_relative:
                        resolved = self._resolve_path(source.parent, raw_path)
                    else:
                        resolved = raw_path  # External package

                    imports.append(ImportInfo(
                        source_file=file_path,
                        imported_path=resolved,
                        is_relative=is_relative,
                        is_type_only=is_type,
                        line_number=line_num,
                    ))

        # Python imports
        py_patterns = [
            (r'from\s+(\.\w+)\s+import\s+(.+)', True),
            (r'import\s+(\w[\w.]*)', False),
        ]
        if file_path.endswith(".py"):
            for line_num, line in enumerate(content.splitlines(), 1):
                for pattern, is_from in py_patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        if is_from:
                            module = match.group(1)
                            names = [n.strip() for n in match.group(2).split(",")]
                        else:
                            module = match.group(1)
                            names = []

                        is_relative = module.startswith(".")
                        if is_relative:
                            resolved = self._resolve_path(source.parent, module.replace(".", "/"))
                        else:
                            resolved = module

                        imports.append(ImportInfo(
                            source_file=file_path,
                            imported_path=resolved,
                            imported_names=names,
                            is_relative=is_relative,
                            line_number=line_num,
                        ))

        return imports

    def _resolve_path(self, base_dir: Path, import_path: str) -> str:
        """Resolve a relative import path."""
        extensions = ["", ".ts", ".tsx", ".js", ".jsx", ".py", "/index.ts", "/index.js"]
        for ext in extensions:
            resolved = base_dir / f"{import_path}{ext}"
            if resolved.exists():
                return str(resolved.resolve())
        # Return unresolved path
        return str((base_dir / import_path).resolve())

    def _collect_deps(self, file_path: str, depth: int, result: set, visited: set):
        """Recursively collect dependencies."""
        if depth <= 0 or file_path in visited:
            return
        visited.add(file_path)
        node = self.nodes.get(str(file_path))
        if not node:
            return
        for dep in node.imports:
            result.add(dep)
            self._collect_deps(dep, depth - 1, result, visited)

    def _detect_circular_deps(self):
        """Detect circular dependencies in the graph."""
        self._circular_deps = []
        for path, node in self.nodes.items():
            for imp in node.imports:
                imp_node = self.nodes.get(imp)
                if imp_node and path in imp_node.imports:
                    pair = tuple(sorted([path, imp]))
                    if pair not in self._circular_deps:
                        self._circular_deps.append(pair)
