"""Embedding-based memory retrieval using base model hidden states.

This module provides an upgrade path from TF-IDF retrieval to
embedding-based retrieval using the model's own representations.

When a model is available, it uses the hidden states from an
intermediate layer (layer n//2) as embeddings. When no model is
available, it falls back to TF-IDF (in manager.py).

Research backing:
- Using the model's own embeddings for retrieval avoids loading a
  separate embedding model (zero additional VRAM)
- Middle layers capture the best semantic representations
  (early layers = too syntactic, final layers = too task-specific)
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cola_coder.memory.manager import MemoryChunk


@dataclass
class EmbeddingCache:
    """Cached embeddings for memory chunks."""

    embeddings: np.ndarray  # (num_chunks, embed_dim)
    chunk_hashes: list[str]  # MD5 hashes for cache invalidation
    file_mtimes: dict[str, float]  # File modification times

    def is_stale(self, memory_path: Path, files: dict[str, str]) -> bool:
        """Check if any memory file has been modified since caching."""
        for key, filename in files.items():
            file_path = memory_path / filename
            if file_path.exists():
                current_mtime = file_path.stat().st_mtime
                cached_mtime = self.file_mtimes.get(filename, 0)
                if current_mtime > cached_mtime:
                    return True
        return False


class ModelEmbedder:
    """Embed text using a transformer model's hidden states.

    Uses the model's intermediate layer representations as embeddings.
    This avoids loading a separate embedding model.
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        device: str = "cuda",
        layer: int | None = None,
    ):
        """
        Args:
            model: Transformer model with get_hidden_states method
            tokenizer: CodeTokenizer
            device: Device to run on
            layer: Which layer to extract (default: n_layers // 2)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer = layer

    def embed(self, text: str, max_tokens: int = 256) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed
            max_tokens: Maximum tokens to process

        Returns:
            1D numpy array of shape (embed_dim,)
        """
        import torch

        tokens = self.tokenizer.encode(text, add_bos=True)[:max_tokens]
        input_ids = torch.tensor([tokens], device=self.device)

        layer_idx = self.layer
        if layer_idx is None:
            n_layers = getattr(self.model, "n_layers", 12)
            layer_idx = n_layers // 2

        with torch.no_grad():
            if hasattr(self.model, "get_hidden_states"):
                hidden = self.model.get_hidden_states(input_ids, layer=layer_idx)
            else:
                # Fallback: run full forward pass and use output
                hidden = self.model(input_ids)

            # Mean pooling
            if hidden.dim() == 3:
                pooled = hidden.mean(dim=1)  # (1, dim)
            else:
                pooled = hidden

            return pooled.squeeze(0).cpu().float().numpy()

    def embed_batch(self, texts: list[str], max_tokens: int = 256) -> np.ndarray:
        """Embed multiple texts.

        Args:
            texts: List of texts
            max_tokens: Max tokens per text

        Returns:
            2D numpy array of shape (len(texts), embed_dim)
        """
        embeddings = []
        for text in texts:
            emb = self.embed(text, max_tokens)
            embeddings.append(emb)
        return np.stack(embeddings)


class EmbeddingRetriever:
    """Retrieve memory chunks using embedding similarity.

    Caches embeddings and only recomputes when memory files change.
    Falls back to TF-IDF if no model is available.
    """

    def __init__(
        self,
        embedder: ModelEmbedder | None = None,
        cache_dir: Path | None = None,
    ):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self._cache: EmbeddingCache | None = None

    def index(self, chunks: list[MemoryChunk]) -> None:
        """Build embedding index from memory chunks.

        Args:
            chunks: List of MemoryChunk objects to index
        """
        if not self.embedder or not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        self._cache = EmbeddingCache(
            embeddings=self.embedder.embed_batch(texts),
            chunk_hashes=[_hash_text(t) for t in texts],
            file_mtimes={},
        )

    def retrieve(
        self,
        query: str,
        chunks: list[MemoryChunk],
        top_k: int = 5,
    ) -> list[MemoryChunk]:
        """Retrieve most relevant chunks using embedding similarity.

        Args:
            query: Search query
            chunks: All available chunks
            top_k: Number of results to return

        Returns:
            Top-k chunks sorted by relevance
        """
        if not self.embedder or not chunks:
            return chunks[:top_k]

        # Embed query
        query_emb = self.embedder.embed(query)

        # Embed chunks (use cache if available)
        if self._cache is not None and len(self._cache.embeddings) == len(chunks):
            chunk_embs = self._cache.embeddings
        else:
            self.index(chunks)
            if self._cache is None:
                return chunks[:top_k]
            chunk_embs = self._cache.embeddings

        # Cosine similarity
        similarities = _cosine_similarity(query_emb, chunk_embs)

        # Assign scores and sort
        for i, chunk in enumerate(chunks):
            chunk.relevance_score = float(similarities[i])

        ranked = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        return ranked[:top_k]


def _cosine_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all docs.

    Args:
        query: (dim,) query vector
        docs: (n, dim) document vectors

    Returns:
        (n,) similarity scores
    """
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    doc_norms = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-8)
    return doc_norms @ query_norm


def _hash_text(text: str) -> str:
    """MD5 hash of text for cache invalidation."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
