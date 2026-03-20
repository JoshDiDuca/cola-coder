"""AST-Aware Chunking: split code at meaningful boundaries.

Instead of splitting code files at arbitrary positions (which can cut a
function in half), this splits at semantic boundaries like function
definitions, class definitions, and import blocks.

For a TS dev: like how a linter understands the AST structure — we use
regex-based heuristics to find function/class boundaries and split there.
"""

import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class CodeChunk:
    """A single chunk of code at a meaningful boundary."""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "import_block", "module_level", "mixed"
    name: str = ""  # Function/class name if applicable

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def token_estimate(self) -> int:
        return len(self.content) // 4


@dataclass
class ChunkingResult:
    """Result of chunking a file."""
    chunks: list[CodeChunk] = field(default_factory=list)
    source_file: str = ""
    total_lines: int = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def summary(self) -> str:
        types = {}
        for c in self.chunks:
            types[c.chunk_type] = types.get(c.chunk_type, 0) + 1
        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(types.items()))
        return (
            f"Chunks: {self.chunk_count} from {self.total_lines} lines\n"
            f"Types: {type_str}"
        )


# ── Boundary detection patterns ──────────────────────────────────────

# TypeScript/JavaScript boundaries
TS_BOUNDARIES = [
    # Function declarations
    (r"^(?:export\s+)?(?:async\s+)?function\s+\w+", "function"),
    # Arrow function assignments
    (r"^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(", "function"),
    # Class declarations
    (r"^(?:export\s+)?(?:abstract\s+)?class\s+\w+", "class"),
    # Interface/type declarations
    (r"^(?:export\s+)?(?:interface|type)\s+\w+", "type"),
    # Import block start
    (r"^import\s+", "import"),
]

# Python boundaries
PY_BOUNDARIES = [
    (r"^(?:async\s+)?def\s+\w+", "function"),
    (r"^class\s+\w+", "class"),
    (r"^(?:from|import)\s+", "import"),
]


def detect_boundaries(
    code: str,
    language: str = "typescript",
) -> list[tuple[int, str, str]]:
    """Find semantic boundaries in code.

    Args:
        code: Source code string
        language: "typescript", "javascript", or "python"

    Returns:
        List of (line_number, boundary_type, name) tuples
    """
    patterns = PY_BOUNDARIES if language == "python" else TS_BOUNDARIES
    boundaries = []
    lines = code.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        for pattern, btype in patterns:
            if re.match(pattern, stripped):
                # Try to extract name
                name = _extract_name(stripped, btype)
                boundaries.append((i, btype, name))
                break

    return boundaries


def _extract_name(line: str, btype: str) -> str:
    """Extract function/class name from a declaration line."""
    if btype in ("function", "class", "type"):
        # Look for the name after the keyword
        patterns = [
            r"(?:function|class|interface|type)\s+(\w+)",
            r"(?:const|let|var)\s+(\w+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    return ""


class ASTChunker:
    """Split code into semantically meaningful chunks."""

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 32,
        overlap_lines: int = 2,
    ):
        """
        Args:
            max_chunk_tokens: Maximum tokens per chunk
            min_chunk_tokens: Minimum tokens per chunk (merge small ones)
            overlap_lines: Lines of overlap between adjacent chunks
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_lines = overlap_lines

    def chunk_code(
        self,
        code: str,
        language: str = "typescript",
        source_file: str = "",
    ) -> ChunkingResult:
        """Split code into AST-aware chunks.

        Strategy:
        1. Find all semantic boundaries
        2. Split at boundaries, respecting max chunk size
        3. Merge small chunks with neighbors
        4. Add overlap for context continuity
        """
        lines = code.splitlines()
        if not lines:
            return ChunkingResult(source_file=source_file, total_lines=0)

        boundaries = detect_boundaries(code, language)
        result = ChunkingResult(source_file=source_file, total_lines=len(lines))

        if not boundaries:
            # No boundaries found — chunk by size
            result.chunks = self._chunk_by_size(lines)
            return result

        # Split at boundaries
        raw_chunks = self._split_at_boundaries(lines, boundaries)

        # Merge small chunks
        merged = self._merge_small_chunks(raw_chunks)

        # Split oversized chunks
        final = []
        for chunk in merged:
            if chunk.token_estimate > self.max_chunk_tokens:
                sub_chunks = self._split_large_chunk(chunk)
                final.extend(sub_chunks)
            else:
                final.append(chunk)

        result.chunks = final
        return result

    def chunk_file(self, file_path: str, language: str | None = None) -> ChunkingResult:
        """Chunk a file from disk.

        Args:
            file_path: Path to the source file
            language: Override language detection
        """
        from pathlib import Path
        p = Path(file_path)

        if not p.exists():
            return ChunkingResult(source_file=file_path)

        content = p.read_text(errors="replace")

        if language is None:
            ext = p.suffix.lower()
            lang_map = {
                ".ts": "typescript", ".tsx": "typescript",
                ".js": "javascript", ".jsx": "javascript",
                ".py": "python",
            }
            language = lang_map.get(ext, "typescript")

        return self.chunk_code(content, language, str(p))

    def _split_at_boundaries(
        self,
        lines: list[str],
        boundaries: list[tuple[int, str, str]],
    ) -> list[CodeChunk]:
        """Split lines at detected boundaries."""
        chunks = []

        for i, (line_num, btype, name) in enumerate(boundaries):
            # Chunk extends from this boundary to the next (or end of file)
            start = line_num
            if i + 1 < len(boundaries):
                end = boundaries[i + 1][0] - 1
            else:
                end = len(lines) - 1

            content = "\n".join(lines[start:end + 1])
            chunks.append(CodeChunk(
                content=content,
                start_line=start + 1,
                end_line=end + 1,
                chunk_type=btype,
                name=name,
            ))

        # Handle lines before the first boundary
        if boundaries and boundaries[0][0] > 0:
            pre_content = "\n".join(lines[:boundaries[0][0]])
            if pre_content.strip():
                chunks.insert(0, CodeChunk(
                    content=pre_content,
                    start_line=1,
                    end_line=boundaries[0][0],
                    chunk_type="module_level",
                ))

        return chunks

    def _merge_small_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return []

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            if prev.token_estimate < self.min_chunk_tokens:
                # Merge with previous
                merged[-1] = CodeChunk(
                    content=prev.content + "\n" + chunk.content,
                    start_line=prev.start_line,
                    end_line=chunk.end_line,
                    chunk_type="mixed" if prev.chunk_type != chunk.chunk_type else prev.chunk_type,
                    name=prev.name or chunk.name,
                )
            else:
                merged.append(chunk)

        return merged

    def _split_large_chunk(self, chunk: CodeChunk) -> list[CodeChunk]:
        """Split an oversized chunk into smaller pieces."""
        lines = chunk.content.splitlines()
        max_lines = max(10, self.max_chunk_tokens // 4)  # Rough estimate

        sub_chunks = []
        for i in range(0, len(lines), max_lines):
            end = min(i + max_lines, len(lines))
            content = "\n".join(lines[i:end])
            sub_chunks.append(CodeChunk(
                content=content,
                start_line=chunk.start_line + i,
                end_line=chunk.start_line + end - 1,
                chunk_type=chunk.chunk_type,
                name=chunk.name if i == 0 else "",
            ))

        return sub_chunks

    def _chunk_by_size(self, lines: list[str]) -> list[CodeChunk]:
        """Fallback: chunk by size when no boundaries are found."""
        max_lines = max(10, self.max_chunk_tokens // 4)
        chunks = []

        for i in range(0, len(lines), max_lines):
            end = min(i + max_lines, len(lines))
            content = "\n".join(lines[i:end])
            chunks.append(CodeChunk(
                content=content,
                start_line=i + 1,
                end_line=end,
                chunk_type="mixed",
            ))

        return chunks
