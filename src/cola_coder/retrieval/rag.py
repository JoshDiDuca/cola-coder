"""Retrieval-Augmented Generation (RAG) pipeline for code.

Retrieves relevant code snippets before generation to reduce
hallucination and improve accuracy.

Flow:
1. Receive query/prompt
2. Retrieve top-k relevant code chunks
3. Format retrieved context
4. Prepend to prompt and generate

Research backing:
- Repoformer (ICML 2024): Selective retrieval — skip when unnecessary
- RAFT (2024): Fine-tuning on retrieved context improves both
"""

from dataclasses import dataclass
from pathlib import Path

from cola_coder.retrieval.indexer import RepoIndexer
from cola_coder.retrieval.vector_store import SearchResult, VectorStore


@dataclass
class RAGContext:
    """Retrieved context for augmented generation."""

    chunks: list[SearchResult]
    formatted_context: str
    query: str
    retrieval_used: bool = True  # False if selective retrieval skipped


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Retrieves relevant code from the vector store and formats it
    for injection into the model's context window.

    Supports selective retrieval (Repoformer approach): skips
    retrieval for simple queries that don't benefit from context.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder=None,
        max_context_tokens: int = 2048,
        min_relevance: float = 0.15,
        selective: bool = True,
    ):
        """
        Args:
            vector_store: Vector store with indexed code
            embedder: Model embedder for query embedding
            max_context_tokens: Max tokens for retrieved context
            min_relevance: Minimum similarity score to include
            selective: Enable selective retrieval (skip for simple queries)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.max_context_tokens = max_context_tokens
        self.min_relevance = min_relevance
        self.selective = selective

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> RAGContext:
        """Retrieve relevant context for a query.

        Args:
            query: The query/prompt to find context for
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filter

        Returns:
            RAGContext with retrieved chunks and formatted text
        """
        # Selective retrieval: skip for very short/simple queries
        if self.selective and _is_simple_query(query):
            return RAGContext(
                chunks=[],
                formatted_context="",
                query=query,
                retrieval_used=False,
            )

        # Embed query
        if self.embedder is None:
            return RAGContext(
                chunks=[],
                formatted_context="",
                query=query,
                retrieval_used=False,
            )

        query_embedding = self.embedder.embed(query)

        # Search
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            min_score=self.min_relevance,
            filter_metadata=filter_metadata,
        )

        # Format context
        formatted = self._format_context(results)

        return RAGContext(
            chunks=results,
            formatted_context=formatted,
            query=query,
            retrieval_used=True,
        )

    def augment_prompt(self, prompt: str, context: RAGContext) -> str:
        """Prepend retrieved context to a prompt.

        Args:
            prompt: Original prompt
            context: Retrieved context from retrieve()

        Returns:
            Augmented prompt with context prepended
        """
        if not context.formatted_context:
            return prompt

        return f"{context.formatted_context}\n\n{prompt}"

    def index_repo(
        self,
        repo_path: str | Path,
        languages: list[str] | None = None,
    ) -> int:
        """Index a repository into the vector store.

        Args:
            repo_path: Path to repository
            languages: Filter to specific languages

        Returns:
            Number of chunks indexed
        """
        indexer = RepoIndexer(languages=languages)
        chunks = indexer.index_repo(repo_path)

        if not chunks or self.embedder is None:
            return 0

        # Embed and store
        for chunk in chunks:
            embedding = self.embedder.embed(chunk.content)
            self.vector_store.add(
                id=chunk.id,
                text=chunk.content,
                embedding=embedding,
                metadata={
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "name": chunk.name,
                    "source": "code",
                },
            )

        return len(chunks)

    def index_documents(
        self,
        doc_dir: str | Path,
        extensions: list[str] | None = None,
    ) -> int:
        """Index documentation files."""
        indexer = RepoIndexer()
        chunks = indexer.index_documents(doc_dir, extensions)

        if not chunks or self.embedder is None:
            return 0

        for chunk in chunks:
            embedding = self.embedder.embed(chunk.content)
            self.vector_store.add(
                id=chunk.id,
                text=chunk.content,
                embedding=embedding,
                metadata={
                    "file_path": chunk.file_path,
                    "chunk_type": "doc",
                    "language": "markdown",
                    "name": chunk.name,
                    "source": "documentation",
                },
            )

        return len(chunks)

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results as context text.

        Respects the token budget (max_context_tokens).
        """
        if not results:
            return ""

        parts = ["# Retrieved Context\n"]
        total_chars = len(parts[0])
        # Rough estimate: 4 chars per token
        char_budget = self.max_context_tokens * 4

        for result in results:
            file_path = result.metadata.get("file_path", "unknown")
            chunk_type = result.metadata.get("chunk_type", "code")
            name = result.metadata.get("name", "")

            header = f"\n## {file_path}"
            if name:
                header += f" — {name}"
            header += f" ({chunk_type}, score={result.score:.2f})\n"

            content = f"```\n{result.text}\n```\n"

            needed = len(header) + len(content)
            if total_chars + needed > char_budget:
                # Try to fit with truncation
                remaining = char_budget - total_chars - len(header) - 20
                if remaining > 100:
                    content = f"```\n{result.text[:remaining]}...\n```\n"
                    parts.append(header + content)
                break

            parts.append(header + content)
            total_chars += needed

        return "".join(parts)


def _is_simple_query(query: str) -> bool:
    """Detect simple queries that don't need retrieval.

    Short queries, single-word queries, or greetings don't
    benefit from code retrieval.
    """
    query = query.strip()

    # Very short queries
    if len(query.split()) < 3:
        return True

    # Common non-code queries
    simple_patterns = [
        "hello", "hi", "hey", "thanks", "thank you",
        "what is", "how are", "help",
    ]
    lower = query.lower()
    if any(lower.startswith(p) for p in simple_patterns):
        return True

    return False
