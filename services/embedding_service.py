"""
Embedding service using fastembed (lightweight ONNX runtime, no PyTorch).

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dim embeddings
  - ~90 MB download, ultra-low RAM & CPU usage
  - 512 token limit

The model is loaded lazily on first use (singleton pattern).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from fastembed import TextEmbedding

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger("rag.embedding")

_model: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    global _model
    if _model is None:
        cfg = get_settings()
        model_name = cfg.embedding_model
        if model_name == "all-MiniLM-L6-v2":
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Loading fastembed model: %s", model_name)
        _model = TextEmbedding(model_name=model_name)
        logger.info("Fastembed model ready ✓")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using fastembed in small batches (16).
    Returns float32 array of shape (N, dim), L2-normalised.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    # model.embed returns a generator of numpy arrays per batch
    raw_generator = model.embed(texts, batch_size=16)
    all_batches = [np.array(batch, dtype=np.float32) for batch in raw_generator]
    
    if not all_batches:
        return np.empty((0, 384), dtype=np.float32)

    embeddings = np.vstack(all_batches).astype(np.float32)
    
    # L2-normalise vectors for IndexFlatIP cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = (embeddings / norms).astype(np.float32)
    
    return normed


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (dim,)."""
    return embed_texts([query])[0]

