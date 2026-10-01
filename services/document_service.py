"""
Document ingestion service.

Pipeline (runs as a background task):
  1. Read file from disk
  2. Extract text (PDF or TXT)
  3. Chunk text using sliding-window strategy
  4. Generate embeddings for each chunk
  5. Store embeddings + metadata in FAISS

Document registry is kept in a simple in-memory dict backed by JSON.
This avoids a DB dependency while still surviving restarts.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from models.schemas import DocumentInfo, DocumentStatus
from utils.chunker import chunk_text
from utils.config import get_settings
from utils.extractors import extract_text
from utils.logger import get_logger
from vector_store.faiss_store import ChunkMeta

if TYPE_CHECKING:
    from vector_store.faiss_store import FAISSStore
    from services.embedding_service import embed_texts as EmbedFn

logger = get_logger("rag.document")

# ── Document registry ──────────────────────────────────────────────────────────
# Stored as {document_id: DocumentInfo dict} in a JSON sidecar file.
_registry: dict[str, dict] = {}


def _get_registry_file() -> Path:
    return get_settings().docs_path


def _load_registry() -> None:
    global _registry
    reg_file = _get_registry_file()
    if reg_file.exists():
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                _registry = json.load(f)
            logger.info("Loaded document registry: %d documents", len(_registry))
        except Exception as e:
            logger.warning("Failed to load document registry from %s: %s", reg_file, e)
            _registry = {}


def _save_registry() -> None:
    reg_file = _get_registry_file()
    reg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(_registry, f, indent=2, default=str)



def get_document(document_id: str) -> dict | None:
    return _registry.get(document_id)


def list_documents() -> list[dict]:
    return list(_registry.values())


def register_document(document_id: str, filename: str, file_path: str) -> None:
    _registry[document_id] = {
        "document_id":  document_id,
        "filename":     filename,
        "file_path":    file_path,
        "status":       DocumentStatus.PENDING,
        "chunk_count":  None,
        "error_detail": None,
        "uploaded_at":  datetime.now(timezone.utc).isoformat(),
        "processed_at": None,
    }
    _save_registry()


def delete_document_registry(document_id: str) -> bool:
    global _registry
    if document_id in _registry:
        del _registry[document_id]
        _save_registry()
        logger.info("Removed document %s from registry", document_id)
        return True
    return False


def _update_status(
    document_id: str,
    status: DocumentStatus,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    if document_id not in _registry:
        return
    rec = _registry[document_id]
    rec["status"]       = status
    rec["chunk_count"]  = chunk_count
    rec["error_detail"] = error
    if status in (DocumentStatus.READY, DocumentStatus.FAILED):
        rec["processed_at"] = datetime.now(timezone.utc).isoformat()
    _save_registry()


# ── Ingestion pipeline ─────────────────────────────────────────────────────────

def ingest_document(
    document_id: str,
    file_path: Path,
    store: "FAISSStore",
    embed_fn=None,
) -> None:
    """
    Full ingestion pipeline executed as a background task.
    Errors are caught and recorded in the registry so the API
    can report them without crashing the worker.
    """
    if embed_fn is None:
        from services.embedding_service import embed_texts
        embed_fn = embed_texts

    cfg = get_settings()
    logger.info("Starting ingestion: %s (%s)", document_id, file_path.name)
    _update_status(document_id, DocumentStatus.PROCESSING)

    try:
        # Step 1: Extract text
        t0   = time.perf_counter()
        text = extract_text(file_path)

        if not text.strip():
            raise ValueError("No readable text found in document")

        logger.info("Extracted %d chars in %.0f ms", len(text),
                    (time.perf_counter() - t0) * 1000)

        # Step 2: Chunk
        chunks = list(chunk_text(
            text,
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
        ))
        logger.info("Split into %d chunks", len(chunks))

        if not chunks:
            raise ValueError("Chunking produced zero chunks")

        # Step 3: Embed
        t1 = time.perf_counter()
        embeddings = embed_fn(chunks)
        logger.info("Embedded %d chunks in %.0f ms", len(chunks),
                    (time.perf_counter() - t1) * 1000)

        # Step 4: Store in FAISS
        filename = file_path.name
        metas = [
            ChunkMeta(
                document_id=document_id,
                filename=filename,
                chunk_id=i,
                text=chunk,
            )
            for i, chunk in enumerate(chunks)
        ]
        store.add(embeddings, metas)

        _update_status(document_id, DocumentStatus.READY, chunk_count=len(chunks))
        logger.info("Ingestion complete: %s → %d chunks", document_id, len(chunks))

    except Exception as exc:
        logger.exception("Ingestion failed for %s: %s", document_id, exc)
        _update_status(document_id, DocumentStatus.FAILED, error=str(exc))


# Initialise registry on module load
_load_registry()
