"""
/health        – full system health check (FAISS stats, document counts).
/api/health    – instant liveness probe (no FAISS, no embedding model).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models.schemas import DocumentStatus, HealthResponse
from services.document_service import list_documents
from utils.config import get_settings

router = APIRouter(tags=["Health"])


def _get_store():
    from main import store
    return store


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns system health including model info and index stats."""
    cfg   = get_settings()
    store = _get_store()
    docs  = list_documents()
    ready = sum(1 for d in docs if d.get("status") == DocumentStatus.READY)

    return HealthResponse(
        status="ok",
        embedding_model=cfg.embedding_model,
        llm_model=cfg.llm_model,
        indexed_chunks=store.total_chunks,
        documents_ready=ready,
    )


@router.get("/api/health", include_in_schema=True, tags=["Health"])
async def api_health():
    """
    Instant liveness probe — safe to call before the embedding model is loaded.
    Does NOT touch FAISS or the embedding service.
    Returns the current LLM provider and model name from config.
    """
    cfg = get_settings()
    return JSONResponse(
        content={
            "status": "online",
            "provider": "Groq",
            "model": cfg.llm_model,
        }
    )

