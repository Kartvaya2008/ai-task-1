"""
Pydantic schemas for request/response validation.
All API contracts are defined here for strict typing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    READY      = "ready"
    FAILED     = "failed"


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: str          = Field(..., description="Unique document identifier")
    filename:    str          = Field(..., description="Original filename")
    status:      DocumentStatus = Field(default=DocumentStatus.PENDING)
    message:     str          = Field(..., description="Human-readable status message")
    uploaded_at: datetime     = Field(default_factory=datetime.utcnow)


class DocumentStatusResponse(BaseModel):
    document_id:  str
    filename:     str
    status:       DocumentStatus
    chunk_count:  Optional[int]   = None
    error_detail: Optional[str]   = None
    processed_at: Optional[datetime] = None


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:    str           = Field(..., min_length=3, max_length=2000,
                                       description="User's natural-language question")
    document_id: Optional[str] = Field(None,
                                       description="Restrict search to one document (optional)")
    top_k:       int           = Field(default=5, ge=1, le=20,
                                       description="Number of chunks to retrieve")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace")
        return v.strip()


class SourceChunk(BaseModel):
    document_id:  str
    filename:     str
    chunk_id:     int
    text_preview: str   = Field(..., description="First 200 chars of the chunk")
    similarity:   float = Field(..., description="Cosine similarity score (0–1)")


class QueryResponse(BaseModel):
    answer:           str
    sources:          list[SourceChunk]
    latency_ms:       float  = Field(..., description="End-to-end query latency in ms")
    retrieval_ms:     float  = Field(..., description="FAISS retrieval latency in ms")
    llm_ms:           float  = Field(..., description="LLM generation latency in ms")
    top_similarity:   float  = Field(..., description="Highest similarity score in retrieved set")
    question:         str
    provider_used:    Optional[str] = Field(
        None, description="LLM provider that generated the answer: groq or openrouter"
    )


# ── Listing ────────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    document_id:  str
    filename:     str
    status:       DocumentStatus
    chunk_count:  Optional[int] = None
    uploaded_at:  datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total:     int


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:          str
    version:         str  = "1.0.0"
    embedding_model: str
    llm_model:       str
    indexed_chunks:  int
    documents_ready: int


# ── Error ──────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail:  str
    code:    Optional[str] = None
