"""Repository indexer for building searchable code indices.

Chunks code by function/class boundaries using regex heuristics
(no tree-sitter dependency). Each chunk gets metadata including
file path, line range, type (function/class/module), and language.

Research backing:
- Cursor: syntax-aware chunking by function boundaries
- CodeXEmbed: AST-based chunking with metadata improves retrieval
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeChunk:
    """A chunk of code for indexing."""

    id: str                     # Unique ID: "file.ts:10-30"
    content: str                # The code text
    file_path: str              # Relative file path
    start_line: int             # Start line number
    end_line: int               # End line number
    chunk_type: str             # "function", "class", "module", "doc"
    language: str               # Programming language
    name: str = ""              # Function/class name if detected
    metadata: dict = field(default_factory=dict)


# Language to file extensions mapping
LANGUAGE_MAP = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".next", "dist", "build",
    "coverage", ".cache", ".venv", "venv", ".tox", "target",
}


class RepoIndexer:
    """Index a repository into searchable code chunks.

    Chunks code by function/class boundaries using regex-based
    detection. Falls back to fixed-size chunking for files without
    clear boundaries.
    """

    def __init__(
        self,
        max_chunk_chars: int = 2000,
        min_chunk_chars: int = 50,
        overlap_lines: int = 2,
        languages: list[str] | None = None,
    ):
        """
        Args:
            max_chunk_chars: Maximum characters per chunk
            min_chunk_chars: Minimum characters (skip tiny chunks)
            overlap_lines: Context lines between chunks
            languages: Filter to specific languages (None = all supported)
        """
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_lines = overlap_lines
        self.languages = languages

    def index_repo(self, repo_path: str | Path) -> list[CodeChunk]:
        """Index all code files in a repository.

        Args:
            repo_path: Path to repository root

        Returns:
            List of CodeChunk objects
        """
        repo_path = Path(repo_path)
        chunks = []

        allowed_exts = set(LANGUAGE_MAP.keys())
        if self.languages:
            lang_set = {lang.lower() for lang in self.languages}
            allowed_exts = {ext for ext, lang in LANGUAGE_MAP.items() if lang in lang_set}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in allowed_exts:
                    continue

                full_path = Path(root) / fname
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    rel_path = str(full_path.relative_to(repo_path)).replace("\\", "/")
                    language = LANGUAGE_MAP.get(ext, "unknown")

                    file_chunks = self._chunk_file(content, rel_path, language)
                    chunks.extend(file_chunks)
                except (OSError, UnicodeDecodeError):
                    continue

        return chunks

    def index_documents(
        self,
        doc_dir: str | Path,
        extensions: list[str] | None = None,
    ) -> list[CodeChunk]:
        """Index documentation files (markdown, text).

        Args:
            doc_dir: Directory containing docs
            extensions: File extensions to include (default: .md)

        Returns:
            List of CodeChunk objects
        """
        doc_dir = Path(doc_dir)
        extensions = extensions or [".md"]
        chunks = []

        for ext in extensions:
            for file_path in doc_dir.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    rel_path = str(file_path.relative_to(doc_dir)).replace("\\", "/")

                    # Chunk markdown by headers
                    sections = re.split(r"\n(?=#{1,3} )", content)
                    for i, section in enumerate(sections):
                        section = section.strip()
                        if len(section) < self.min_chunk_chars:
                            continue
                        if len(section) > self.max_chunk_chars:
                            section = section[: self.max_chunk_chars]

                        # Extract heading
                        heading_match = re.match(r"^(#{1,3})\s+(.+)", section)
                        name = heading_match.group(2) if heading_match else f"Section {i}"

                        chunks.append(
                            CodeChunk(
                                id=f"{rel_path}:section-{i}",
                                content=section,
                                file_path=rel_path,
                                start_line=0,
                                end_line=section.count("\n"),
                                chunk_type="doc",
                                language="markdown",
                                name=name,
                            )
                        )
                except (OSError, UnicodeDecodeError):
                    continue

        return chunks

    def _chunk_file(self, content: str, file_path: str, language: str) -> list[CodeChunk]:
        """Chunk a single file by function/class boundaries."""
        lines = content.split("\n")

        if language in ("typescript", "javascript"):
            boundaries = self._find_ts_boundaries(lines)
        elif language == "python":
            boundaries = self._find_python_boundaries(lines)
        else:
            boundaries = []

        if not boundaries:
            # Fallback: fixed-size chunks
            return self._fixed_chunks(content, file_path, language)

        chunks = []
        for start, end, name, chunk_type in boundaries:
            chunk_content = "\n".join(lines[start : end + 1])

            if len(chunk_content) < self.min_chunk_chars:
                continue
            if len(chunk_content) > self.max_chunk_chars:
                chunk_content = chunk_content[: self.max_chunk_chars]

            chunks.append(
                CodeChunk(
                    id=f"{file_path}:{start + 1}-{end + 1}",
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end + 1,
                    chunk_type=chunk_type,
                    language=language,
                    name=name,
                    metadata={"file_path": file_path},
                )
            )

        return chunks

    def _find_ts_boundaries(
        self, lines: list[str]
    ) -> list[tuple[int, int, str, str]]:
        """Find function/class boundaries in TypeScript/JavaScript.

        Returns list of (start_line, end_line, name, type) tuples.
        """
        boundaries = []
        patterns = [
            (r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            (r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", "function"),
            (
                r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?"
                r"(?:\([^)]*\)|[^=])\s*=>",
                "function",
            ),
            (r"^(?:export\s+)?class\s+(\w+)", "class"),
            (r"^(?:export\s+)?interface\s+(\w+)", "interface"),
            (r"^(?:export\s+)?type\s+(\w+)", "type"),
        ]

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            for pattern, chunk_type in patterns:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1)
                    end = self._find_block_end(lines, i)
                    boundaries.append((i, end, name, chunk_type))
                    i = end + 1
                    break
            else:
                i += 1

        return boundaries

    def _find_python_boundaries(
        self, lines: list[str]
    ) -> list[tuple[int, int, str, str]]:
        """Find function/class boundaries in Python."""
        boundaries = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect function/class definitions
            func_match = re.match(r"^(\s*)(?:async\s+)?def\s+(\w+)", line)
            class_match = re.match(r"^(\s*)class\s+(\w+)", line)

            if func_match or class_match:
                match = func_match or class_match
                indent = len(match.group(1))  # type: ignore[union-attr]
                name = match.group(2)  # type: ignore[union-attr]
                chunk_type = "function" if func_match else "class"

                # Find end: next line with same or less indentation
                end = i + 1
                while end < len(lines):
                    next_line = lines[end]
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent and not next_line.strip().startswith("#"):
                            break
                    end += 1

                boundaries.append((i, end - 1, name, chunk_type))
                i = end
            else:
                i += 1

        return boundaries

    def _find_block_end(self, lines: list[str], start: int) -> int:
        """Find the end of a brace-delimited block."""
        depth = 0
        found_open = False

        for i in range(start, len(lines)):
            for char in lines[i]:
                if char == "{":
                    depth += 1
                    found_open = True
                elif char == "}":
                    depth -= 1
                    if found_open and depth == 0:
                        return i

        # If no braces found, take until next blank line or end
        for i in range(start + 1, min(start + 50, len(lines))):
            if not lines[i].strip():
                return i

        return min(start + 30, len(lines) - 1)

    def _fixed_chunks(self, content: str, file_path: str, language: str) -> list[CodeChunk]:
        """Fallback: split into fixed-size chunks."""
        lines = content.split("\n")
        chunks = []
        chunk_lines = self.max_chunk_chars // 40  # ~40 chars per line

        for i in range(0, len(lines), chunk_lines - self.overlap_lines):
            chunk_content = "\n".join(lines[i : i + chunk_lines])
            if len(chunk_content) < self.min_chunk_chars:
                continue

            chunks.append(
                CodeChunk(
                    id=f"{file_path}:{i + 1}-{i + chunk_lines}",
                    content=chunk_content,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=min(i + chunk_lines, len(lines)),
                    chunk_type="module",
                    language=language,
                    name=Path(file_path).stem,
                    metadata={"file_path": file_path},
                )
            )

        return chunks
