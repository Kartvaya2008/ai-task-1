"""
FAISS-based vector store with metadata management.

Design decisions:
- IndexFlatIP (inner-product / cosine after L2-normalisation) is chosen
  over IndexFlatL2 because cosine similarity is more robust to
  embedding magnitude differences across different document types.
- Metadata is persisted as JSON alongside the FAISS binary index so
  the store survives server restarts without a DB dependency.
- Thread safety: a threading.Lock protects all write operations so
  background ingestion jobs can't corrupt concurrent queries.

METRIC TRACKED – Similarity Score:
  Every retrieval returns the cosine similarity for each chunk (0–1 range).
  Typical good retrievals score 0.75+.  Scores below 0.45 usually indicate
  the document set doesn't contain relevant information for the query.
  We log the top-1 score to detect "hallucination risk" situations.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from utils.logger import get_logger

logger = get_logger("rag.vector_store")


@dataclass
class ChunkMeta:
    document_id: str
    filename:    str
    chunk_id:    int
    text:        str       # full chunk text kept for LLM context


class FAISSStore:
    """
    Thread-safe FAISS vector store with JSON metadata sidecar.

    Lifecycle:
        store = FAISSStore(dim=384)
        store.load()                          # from disk if available
        store.add(embeddings, metas)          # during ingestion
        results = store.search(query_vec, k)  # at query time
        store.save()                          # persisted automatically after add
    """

    def __init__(self, index_path: Path, meta_path: Path, dim: int = 384):
        self.index_path = index_path
        self.meta_path  = meta_path
        self.dim        = dim
        self._lock      = threading.Lock()
        self._index: Optional[faiss.Index] = None
        self._meta:  list[ChunkMeta]       = []

    # ── Persistence ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load index + metadata from disk (no-op if files don't exist)."""
        idx_file  = Path(str(self.index_path) + ".index")
        meta_file = self.meta_path

        if idx_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(idx_file))
            with open(meta_file) as f:
                raw = json.load(f)
            self._meta = [ChunkMeta(**m) for m in raw]
            logger.info("Loaded FAISS index: %d vectors", self._index.ntotal)
        else:
            self._init_empty_index()
            logger.info("Initialised fresh FAISS index (dim=%d)", self.dim)

    def save(self) -> None:
        """Persist index + metadata to disk."""
        idx_file  = Path(str(self.index_path) + ".index")
        meta_file = self.meta_path
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(idx_file))
        with open(meta_file, "w") as f:
            json.dump([asdict(m) for m in self._meta], f, indent=2)

    def _init_empty_index(self) -> None:
        # IndexFlatIP = exact cosine search (after L2 normalisation of vectors)
        self._index = faiss.IndexFlatIP(self.dim)

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, metas: list[ChunkMeta]) -> None:
        """
        Add a batch of (embedding, metadata) pairs.
        Vectors are L2-normalised in-place before insertion.
        """
        if len(embeddings) != len(metas):
            raise ValueError("embeddings and metas must have equal length")

        # Normalise to unit sphere → inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = (embeddings / norms).astype(np.float32)

        with self._lock:
            self._index.add(normed)
            self._meta.extend(metas)
        self.save()
        logger.info("Added %d chunks; total=%d", len(metas), self._index.ntotal)

    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document.
        FAISS IndexFlatIP doesn't support deletion natively,
        so we rebuild the index from the surviving metadata.
        Returns number of chunks removed.
        """
        with self._lock:
            surviving = [m for m in self._meta if m.document_id != document_id]
            removed   = len(self._meta) - len(surviving)

            if removed == 0:
                return 0

            # Re-embed is expensive; we store vectors separately in a parallel
            # list.  For simplicity here we rebuild from scratch using texts
            # (acceptable for the scale of this system).
            # In production, use faiss.IndexIDMap for O(1) deletion.
            self._init_empty_index()
            self._meta = surviving
            if surviving:
                from services.embedding_service import embed_texts
                texts = [m.text for m in surviving]
                embeddings = embed_texts(texts)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-9, norms)
                normed = (embeddings / norms).astype(np.float32)
                self._index.add(normed)
            self.save()

        logger.info("Deleted %d chunks for document %s", removed, document_id)
        return removed

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[tuple[ChunkMeta, float]]:
        """
        Find the top-k most similar chunks.

        Returns list of (ChunkMeta, similarity_score) sorted by similarity desc.
        similarity_score ∈ [0, 1] where 1 = identical.

        document_id: if given, restrict results to that document only.
        """
        if self._index.ntotal == 0:
            return []

        # Normalise query
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_normed = (query_vec / norm).astype(np.float32).reshape(1, -1)

        # Retrieve more candidates when filtering by document
        fetch_k = min(k * 10 if document_id else k * 2, self._index.ntotal)
        scores, indices = self._index.search(query_normed, fetch_k)

        results: list[tuple[ChunkMeta, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:                    # FAISS padding
                continue
            meta = self._meta[idx]
            if document_id and meta.document_id != document_id:
                continue
            # Clip to [0, 1] – inner product of unit vectors is cosine similarity
            sim = float(np.clip(score, 0.0, 1.0))
            results.append((meta, sim))
            if len(results) == k:
                break

        return results

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal if self._index else 0

    def chunks_for_document(self, document_id: str) -> int:
        return sum(1 for m in self._meta if m.document_id == document_id)

    def document_ids(self) -> list[str]:
        return list({m.document_id for m in self._meta})
