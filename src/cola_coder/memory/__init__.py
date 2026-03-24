"""Project memory system for cola-coder.

Stores persistent project knowledge in .cola/memory/ as markdown files.
Supports embedding-based retrieval for context-aware code generation.

Usage:
    from cola_coder.memory import MemoryManager

    mm = MemoryManager(project_root=Path("."))
    mm.init_project(tech_stack={"framework": "Next.js", "language": "TypeScript"})
    relevant = mm.retrieve("how to handle auth", max_chunks=3)
"""

from cola_coder.memory.manager import MemoryChunk, MemoryManager
from cola_coder.memory.config import MemoryConfig

__all__ = ["MemoryManager", "MemoryChunk", "MemoryConfig"]
