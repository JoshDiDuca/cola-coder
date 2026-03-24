"""Retrieval-Augmented Generation (RAG) for code.

Provides semantic code search via embedding-based vector index.
Retrieves relevant code snippets, docs, and past PRs to augment
the model's generation context.

Research backing:
- CodeXEmbed (2024): Code embedding SOTA, +20% over Voyage-Code
- Repoformer (ICML 2024): Selective retrieval for code (~100% speedup)
- Cursor: syntax-aware chunking + hybrid semantic+grep search

Usage:
    from cola_coder.retrieval import VectorStore, RepoIndexer, RAGPipeline
"""

from cola_coder.retrieval.vector_store import VectorStore, SearchResult
from cola_coder.retrieval.indexer import RepoIndexer, CodeChunk
from cola_coder.retrieval.rag import RAGPipeline

__all__ = [
    "VectorStore", "SearchResult",
    "RepoIndexer", "CodeChunk",
    "RAGPipeline",
]
