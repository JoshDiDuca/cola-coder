"""In-memory vector store for semantic code search.

Uses numpy for similarity computation — no external vector DB needed.
Supports save/load to disk for persistence across sessions.

For a TS dev: think of this as a Map<string, {embedding, metadata}>
with a similarity-based search method instead of key lookup.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """In-memory vector store with numpy-based similarity search.

    Stores text chunks with their embeddings and metadata.
    Supports cosine similarity search, save/load, and incremental updates.
    """

    def __init__(self, embed_dim: int = 768):
        """
        Args:
            embed_dim: Dimension of embedding vectors
        """
        self.embed_dim = embed_dim
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._embeddings: np.ndarray | None = None  # (n, embed_dim)
        self._id_to_idx: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(
        self,
        id: str,
        text: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Add a single item to the store.

        Args:
            id: Unique identifier (e.g. "file.ts:10-20")
            text: The text content
            embedding: The embedding vector (1D numpy array)
            metadata: Optional metadata dict
        """
        if id in self._id_to_idx:
            # Update existing
            idx = self._id_to_idx[id]
            self._texts[idx] = text
            self._metadata[idx] = metadata or {}
            if self._embeddings is not None:
                self._embeddings[idx] = embedding
            return

        self._ids.append(id)
        self._texts.append(text)
        self._metadata.append(metadata or {})
        self._id_to_idx[id] = len(self._ids) - 1

        # Append to embeddings matrix
        emb = embedding.reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = emb
        else:
            self._embeddings = np.vstack([self._embeddings, emb])

    def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        """Add multiple items at once.

        Args:
            ids: List of unique identifiers
            texts: List of text contents
            embeddings: (n, embed_dim) embedding matrix
            metadata: Optional list of metadata dicts
        """
        if metadata is None:
            metadata = [{}] * len(ids)

        for i, (id_, text, meta) in enumerate(zip(ids, texts, metadata)):
            self._ids.append(id_)
            self._texts.append(text)
            self._metadata.append(meta)
            self._id_to_idx[id_] = len(self._ids) - 1

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar items by cosine similarity.

        Args:
            query_embedding: Query vector (1D numpy array)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            filter_metadata: Only return items matching these metadata fields

        Returns:
            List of SearchResult sorted by score (highest first)
        """
        if self._embeddings is None or len(self._ids) == 0:
            return []

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        )
        scores = doc_norms @ query_norm

        # Build results with optional filtering
        results = []
        for idx in np.argsort(scores)[::-1]:
            score = float(scores[idx])

            if score < min_score:
                break

            # Apply metadata filter
            if filter_metadata:
                meta = self._metadata[idx]
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(
                SearchResult(
                    id=self._ids[idx],
                    text=self._texts[idx],
                    score=score,
                    metadata=self._metadata[idx],
                )
            )

            if len(results) >= top_k:
                break

        return results

    def remove(self, id: str) -> bool:
        """Remove an item by ID.

        Note: This is O(n) — for frequent deletions, rebuild the index.

        Returns:
            True if the item was found and removed
        """
        if id not in self._id_to_idx:
            return False

        idx = self._id_to_idx[id]

        self._ids.pop(idx)
        self._texts.pop(idx)
        self._metadata.pop(idx)

        if self._embeddings is not None:
            self._embeddings = np.delete(self._embeddings, idx, axis=0)
            if len(self._embeddings) == 0:
                self._embeddings = None

        # Rebuild index mapping
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

        return True

    def clear(self) -> None:
        """Remove all items."""
        self._ids.clear()
        self._texts.clear()
        self._metadata.clear()
        self._embeddings = None
        self._id_to_idx.clear()

    def save(self, path: str | Path) -> None:
        """Save the vector store to disk.

        Saves as two files:
        - {path}.npz: embeddings matrix
        - {path}.json: ids, texts, metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self._embeddings is not None:
            np.savez_compressed(
                str(path) + ".npz",
                embeddings=self._embeddings,
            )

        # Save metadata
        meta = {
            "ids": self._ids,
            "texts": self._texts,
            "metadata": self._metadata,
            "embed_dim": self.embed_dim,
        }
        with open(str(path) + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, path: str | Path) -> None:
        """Load the vector store from disk."""
        path = Path(path)

        # Load metadata
        json_path = str(path) + ".json"
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        self._ids = meta["ids"]
        self._texts = meta["texts"]
        self._metadata = meta["metadata"]
        self.embed_dim = meta["embed_dim"]
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

        # Load embeddings
        npz_path = str(path) + ".npz"
        data = np.load(npz_path)
        self._embeddings = data["embeddings"]

    def stats(self) -> dict:
        """Get store statistics."""
        return {
            "total_items": len(self._ids),
            "embed_dim": self.embed_dim,
            "memory_mb": (
                self._embeddings.nbytes / 1024 / 1024 if self._embeddings is not None else 0
            ),
            "unique_sources": len(set(m.get("source", "unknown") for m in self._metadata)),
        }
