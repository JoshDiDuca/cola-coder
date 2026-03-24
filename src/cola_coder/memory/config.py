"""Configuration for the memory system."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryConfig:
    """Memory system configuration.

    Attributes:
        memory_dir: Name of the memory directory (inside .cola/)
        max_context_tokens: Max tokens of memory to inject into prompts
        max_chunks_per_query: Max memory chunks to retrieve per query
        chunk_min_length: Minimum chunk length (chars) to index
        compact_threshold: Number of entries before auto-compacting
        session_log_max_entries: Rolling window size for session log
    """

    memory_dir: str = "memory"
    max_context_tokens: int = 1024
    max_chunks_per_query: int = 5
    chunk_min_length: int = 20
    compact_threshold: int = 100
    session_log_max_entries: int = 50

    # Memory file names
    files: dict[str, str] = field(
        default_factory=lambda: {
            "project": "project.md",
            "patterns": "patterns.md",
            "errors": "errors.md",
            "decisions": "decisions.md",
            "domain_knowledge": "domain_knowledge.md",
            "session_log": "session_log.md",
        }
    )

    def get_memory_path(self, project_root: Path) -> Path:
        """Get the full path to the memory directory."""
        return project_root / ".cola" / self.memory_dir
