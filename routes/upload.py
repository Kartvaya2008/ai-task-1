"""
/documents – upload and management endpoints.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi import File as FastAPIFile

from models.schemas import (
    DocumentInfo,
    DocumentListResponse,
    DocumentStatus,
    DocumentStatusResponse,
    UploadResponse,
)
from services.document_service import (
    get_document,
    ingest_document,
    list_documents,
    register_document,
    delete_document_registry,
)
from utils.config import Settings, get_settings
from utils.logger import get_logger

logger = get_logger("rag.routes.upload")

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_TYPES  = {".pdf", ".txt"}
ALLOWED_MIMES  = {
    "application/pdf",
    "text/plain",
    "application/octet-stream",  # some clients send this for txt
}


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_store():
    """Import app-level FAISS store (avoids circular imports)."""
    from main import store
    return store


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = FastAPIFile(...),
    cfg: Settings    = Depends(get_settings),
):
    """
    Upload a PDF or TXT document for ingestion.
    Processing happens asynchronously — poll /documents/{id}/status to track progress.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_TYPES)}",
        )

    # Read file data (with size guard)
    data = await file.read(cfg.max_file_size + 1)
    if len(data) > cfg.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {cfg.max_file_size // 1_048_576} MB.",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Persist to disk
    doc_id    = str(uuid.uuid4())
    safe_name = f"{doc_id}{suffix}"
    dest      = cfg.upload_path / safe_name
    dest.write_bytes(data)

    logger.info("Saved upload: %s → %s (%d bytes)", file.filename, dest, len(data))

    # Register in document registry
    register_document(doc_id, file.filename, str(dest))

    # Queue background ingestion
    background_tasks.add_task(ingest_document, doc_id, dest, _get_store())

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
        message="Document received and queued for processing.",
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Check the processing status of an uploaded document."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentStatusResponse(**doc)


@router.get("/", response_model=DocumentListResponse)
async def list_all_documents():
    """List all uploaded documents with their status."""
    docs = list_documents()
    items = [DocumentInfo(**d) for d in docs]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str):
    """Delete a document and remove its vectors from the index."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    store = _get_store()
    removed = store.delete_document(document_id)

    # Remove file from disk
    fp = Path(doc.get("file_path", ""))
    if fp.exists():
        fp.unlink(missing_ok=True)

    # Remove from registry
    delete_document_registry(document_id)

    logger.info("Deleted document %s (%d chunks removed)", document_id, removed)
    return None   # 204 No Content
