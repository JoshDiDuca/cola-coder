"""Memory manager for persistent project knowledge.

Stores project context, patterns, errors, decisions, and domain knowledge
in structured markdown files under .cola/memory/. Supports retrieval
via TF-IDF similarity (with optional embedding-based upgrade).

Inspired by:
- Cursor's .cursorrules project context
- Claude Code's CLAUDE.md project memory
- Mem0's structured memory management (2024)
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from cola_coder.memory.config import MemoryConfig


@dataclass
class MemoryChunk:
    """A single retrievable chunk of memory."""

    content: str
    source_file: str  # Which memory file it came from
    section: str  # Section header (## heading)
    relevance_score: float = 0.0
    timestamp: str = ""  # When it was added


class MemoryManager:
    """Manages persistent project memory in .cola/memory/.

    Memory is organized into themed markdown files:
    - project.md: Tech stack, architecture, conventions
    - patterns.md: Recurring code patterns, idioms
    - errors.md: Common errors and their fixes
    - decisions.md: Architectural decisions and rationale
    - domain_knowledge.md: Domain-specific facts
    - session_log.md: Recent interaction summaries

    Retrieval uses TF-IDF similarity by default, with optional
    embedding-based retrieval when a model is available.
    """

    def __init__(
        self,
        project_root: Path | str,
        config: MemoryConfig | None = None,
    ):
        self.project_root = Path(project_root)
        self.config = config or MemoryConfig()
        self.memory_path = self.config.get_memory_path(self.project_root)
        self._retriever = None  # Lazy-loaded

    @property
    def is_initialized(self) -> bool:
        """Check if memory has been initialized for this project."""
        return self.memory_path.exists()

    # ----- Initialization -----

    def init_project(
        self,
        tech_stack: dict[str, str] | None = None,
        description: str = "",
        conventions: list[str] | None = None,
    ) -> Path:
        """Initialize memory for a project.

        Creates .cola/memory/ with template files.

        Args:
            tech_stack: Dict of technology names and versions
                e.g. {"framework": "Next.js 14", "language": "TypeScript"}
            description: Short project description
            conventions: List of coding conventions

        Returns:
            Path to the memory directory
        """
        self.memory_path.mkdir(parents=True, exist_ok=True)

        # Create project.md
        project_content = "# Project Context\n\n"
        if description:
            project_content += f"## Description\n\n{description}\n\n"
        if tech_stack:
            project_content += "## Tech Stack\n\n"
            for key, value in tech_stack.items():
                project_content += f"- **{key}**: {value}\n"
            project_content += "\n"
        if conventions:
            project_content += "## Conventions\n\n"
            for conv in conventions:
                project_content += f"- {conv}\n"
            project_content += "\n"

        self._write_file("project", project_content)

        # Create empty template files
        templates = {
            "patterns": "# Code Patterns\n\nRecurring patterns and idioms.\n\n",
            "errors": "# Common Errors\n\nErrors encountered and their fixes.\n\n",
            "decisions": "# Architectural Decisions\n\nKey decisions and rationale.\n\n",
            "domain_knowledge": "# Domain Knowledge\n\nDomain-specific facts and context.\n\n",
            "session_log": "# Session Log\n\nRecent interaction summaries.\n\n",
        }

        for key, template in templates.items():
            file_path = self.memory_path / self.config.files[key]
            if not file_path.exists():
                self._write_file(key, template)

        return self.memory_path

    # ----- Read Operations -----

    def get_project_context(self) -> str:
        """Get the always-included project summary.

        Returns the content of project.md, which should always be
        included in the model's context window.
        """
        return self._read_file("project")

    def retrieve(
        self,
        query: str,
        max_chunks: int | None = None,
        exclude_files: list[str] | None = None,
    ) -> list[MemoryChunk]:
        """Retrieve relevant memory chunks for a query.

        Uses TF-IDF similarity to find the most relevant chunks
        across all memory files.

        Args:
            query: The search query (code snippet, question, etc.)
            max_chunks: Max chunks to return (default from config)
            exclude_files: Memory file keys to exclude

        Returns:
            List of MemoryChunk sorted by relevance (highest first)
        """
        max_chunks = max_chunks or self.config.max_chunks_per_query
        exclude_files = exclude_files or []

        if not self.is_initialized:
            return []

        # Collect all chunks from all memory files
        all_chunks = list(self._iter_chunks(exclude_files))

        if not all_chunks:
            return []

        # Score by TF-IDF similarity
        scored = _tfidf_rank(query, all_chunks)

        # Return top-k
        return scored[:max_chunks]

    def get_relevant_memories(
        self,
        code: str = "",
        file_path: str = "",
        query: str = "",
    ) -> str:
        """Get relevant memories formatted for context injection.

        Combines project context + retrieved memories into a string
        ready to prepend to a prompt.

        Args:
            code: Code being worked on
            file_path: Path of the file being edited
            query: Additional query text

        Returns:
            Formatted memory context string
        """
        parts = []

        # Always include project context
        project = self.get_project_context()
        if project.strip():
            parts.append(project.strip())

        # Retrieve relevant chunks
        search_query = f"{query} {code[:500]} {file_path}".strip()
        if search_query:
            chunks = self.retrieve(search_query, max_chunks=3)
            for chunk in chunks:
                if chunk.relevance_score > 0.1:
                    parts.append(
                        f"## {chunk.section} ({chunk.source_file})\n{chunk.content}"
                    )

        return "\n\n".join(parts) if parts else ""

    # ----- Write Operations -----

    def add_pattern(self, pattern: str, example: str = "") -> None:
        """Record a code pattern.

        Args:
            pattern: Description of the pattern
            example: Optional code example
        """
        entry = f"\n## {pattern}\n\n"
        if example:
            entry += f"```\n{example}\n```\n"
        entry += f"\n_Added: {_now()}_\n"

        self._append_file("patterns", entry)

    def add_error(self, error: str, fix: str = "") -> None:
        """Record an error and its fix.

        Args:
            error: Error description or message
            fix: How to fix it
        """
        entry = f"\n## {error[:100]}\n\n"
        entry += f"**Error**: {error}\n\n"
        if fix:
            entry += f"**Fix**: {fix}\n\n"
        entry += f"_Added: {_now()}_\n"

        self._append_file("errors", entry)

    def add_decision(self, decision: str, rationale: str = "") -> None:
        """Record an architectural decision.

        Args:
            decision: What was decided
            rationale: Why it was decided
        """
        entry = f"\n## {decision[:100]}\n\n"
        entry += f"**Decision**: {decision}\n\n"
        if rationale:
            entry += f"**Rationale**: {rationale}\n\n"
        entry += f"_Added: {_now()}_\n"

        self._append_file("decisions", entry)

    def add_domain_knowledge(self, topic: str, content: str) -> None:
        """Record domain-specific knowledge.

        Args:
            topic: Topic heading
            content: The knowledge to record
        """
        entry = f"\n## {topic}\n\n{content}\n\n_Added: {_now()}_\n"
        self._append_file("domain_knowledge", entry)

    def log_session(self, summary: str, domain: str = "") -> None:
        """Add a session log entry.

        Maintains a rolling window of recent interactions.

        Args:
            summary: Brief summary of the interaction
            domain: Domain of the interaction (e.g. "react")
        """
        domain_tag = f" [{domain}]" if domain else ""
        entry = f"\n## {_now()}{domain_tag}\n\n{summary}\n"

        self._append_file("session_log", entry)

        # Trim to rolling window
        self._trim_session_log()

    def update_from_interaction(
        self, prompt: str, response: str, domain: str = ""
    ) -> None:
        """Auto-extract and store knowledge from an interaction.

        Heuristically extracts patterns, errors, and facts from the
        conversation and stores them in appropriate memory files.

        Args:
            prompt: User's prompt
            response: Model's response
            domain: Detected domain
        """
        # Log the session
        summary = prompt[:200] if len(prompt) > 200 else prompt
        self.log_session(summary, domain)

        # Extract error patterns
        error_patterns = re.findall(
            r"(?:error|Error|ERROR|exception|Exception)[\s:]+(.+?)(?:\n|$)",
            response,
        )
        for err in error_patterns[:3]:  # Cap at 3 per interaction
            self.add_error(err.strip()[:200])

    # ----- Management -----

    def compact(self) -> dict[str, int]:
        """Compact memory files by removing duplicates and old entries.

        Returns:
            Dict of {file_key: entries_removed}
        """
        results = {}

        for key in self.config.files:
            if key == "project":
                continue  # Don't compact project.md

            content = self._read_file(key)
            if not content:
                continue

            sections = _split_sections(content)
            original_count = len(sections)

            # Remove exact duplicates (by content, ignoring timestamps)
            seen: set[str] = set()
            unique_sections = []
            for section in sections:
                # Normalize: remove timestamps for comparison
                normalized = re.sub(r"_Added: .*?_", "", section).strip()
                if normalized not in seen:
                    seen.add(normalized)
                    unique_sections.append(section)

            removed = original_count - len(unique_sections)
            if removed > 0:
                # Keep the header (first line) and rejoin sections
                header_match = re.match(r"^# .+\n", content)
                header = header_match.group(0) if header_match else ""
                new_content = header + "\n" + "\n".join(unique_sections)
                self._write_file(key, new_content)

            results[key] = removed

        return results

    def export(self) -> dict[str, str]:
        """Export all memory as a dict.

        Returns:
            Dict of {file_key: file_content}
        """
        result = {}
        for key in self.config.files:
            content = self._read_file(key)
            if content:
                result[key] = content
        return result

    def stats(self) -> dict[str, dict]:
        """Get memory statistics.

        Returns:
            Dict of {file_key: {chunks, chars, last_modified}}
        """
        result = {}
        for key, filename in self.config.files.items():
            file_path = self.memory_path / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                chunks = list(_iter_sections(content))
                stat = file_path.stat()
                result[key] = {
                    "chunks": len(chunks),
                    "chars": len(content),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
            else:
                result[key] = {"chunks": 0, "chars": 0, "last_modified": "N/A"}
        return result

    # ----- Internal helpers -----

    def _read_file(self, key: str) -> str:
        """Read a memory file by key."""
        filename = self.config.files.get(key, "")
        if not filename:
            return ""
        file_path = self.memory_path / filename
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8")

    def _write_file(self, key: str, content: str) -> None:
        """Write content to a memory file."""
        filename = self.config.files.get(key, "")
        if not filename:
            return
        file_path = self.memory_path / filename
        file_path.write_text(content, encoding="utf-8")

    def _append_file(self, key: str, content: str) -> None:
        """Append content to a memory file."""
        existing = self._read_file(key)
        self._write_file(key, existing + content)

    def _iter_chunks(
        self, exclude_files: list[str] | None = None
    ) -> Iterator[MemoryChunk]:
        """Iterate over all memory chunks from all files."""
        exclude = set(exclude_files or [])

        for key, filename in self.config.files.items():
            if key in exclude:
                continue

            content = self._read_file(key)
            if not content:
                continue

            for section_title, section_content in _iter_sections(content):
                if len(section_content) < self.config.chunk_min_length:
                    continue
                yield MemoryChunk(
                    content=section_content,
                    source_file=filename,
                    section=section_title,
                )

    def _trim_session_log(self) -> None:
        """Trim session log to rolling window."""
        content = self._read_file("session_log")
        if not content:
            return

        sections = list(_iter_sections(content))
        max_entries = self.config.session_log_max_entries

        if len(sections) > max_entries:
            # Keep header + last N entries
            header_match = re.match(r"^# .+\n\n.*?\n\n", content, re.DOTALL)
            header = header_match.group(0) if header_match else "# Session Log\n\n"
            kept_sections = sections[-max_entries:]
            new_content = header
            for title, body in kept_sections:
                new_content += f"\n## {title}\n\n{body}\n"
            self._write_file("session_log", new_content)


# ---------------------------------------------------------------------------
# TF-IDF Retrieval (no external dependencies)
# ---------------------------------------------------------------------------


def _tfidf_rank(query: str, chunks: list[MemoryChunk]) -> list[MemoryChunk]:
    """Rank chunks by TF-IDF cosine similarity to query.

    Simple but effective retrieval without external dependencies.
    Can be replaced with embedding-based retrieval later.
    """

    def tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r"\w+", text.lower())

    query_tokens = tokenize(query)
    if not query_tokens:
        return chunks

    # Document frequency
    doc_count = len(chunks) + 1
    df: dict[str, int] = Counter()
    chunk_tokens: list[list[str]] = []

    for chunk in chunks:
        tokens = tokenize(chunk.content)
        chunk_tokens.append(tokens)
        for unique_token in set(tokens):
            df[unique_token] += 1

    # IDF
    idf: dict[str, float] = {}
    for token, freq in df.items():
        idf[token] = math.log(doc_count / (freq + 1))

    # Query TF-IDF vector
    query_tf = Counter(query_tokens)
    query_vec: dict[str, float] = {}
    for token, count in query_tf.items():
        query_vec[token] = count * idf.get(token, 0)

    # Score each chunk
    for i, chunk in enumerate(chunks):
        tokens = chunk_tokens[i]
        if not tokens:
            chunk.relevance_score = 0.0
            continue

        doc_tf = Counter(tokens)
        doc_vec: dict[str, float] = {}
        for token, count in doc_tf.items():
            doc_vec[token] = count * idf.get(token, 0)

        # Cosine similarity
        dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)
        mag_q = math.sqrt(sum(v**2 for v in query_vec.values()))
        mag_d = math.sqrt(sum(v**2 for v in doc_vec.values()))

        if mag_q > 0 and mag_d > 0:
            chunk.relevance_score = dot / (mag_q * mag_d)
        else:
            chunk.relevance_score = 0.0

    return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def _split_sections(content: str) -> list[str]:
    """Split markdown content into sections by ## headers."""
    sections = re.split(r"\n(?=## )", content)
    return [s.strip() for s in sections if s.strip() and s.strip().startswith("## ")]


def _iter_sections(content: str) -> Iterator[tuple[str, str]]:
    """Iterate over (title, body) tuples from markdown sections."""
    sections = re.split(r"\n(?=## )", content)
    for section in sections:
        section = section.strip()
        if not section.startswith("## "):
            continue
        lines = section.split("\n", 1)
        title = lines[0].lstrip("# ").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if title:
            yield title, body


def _now() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")
