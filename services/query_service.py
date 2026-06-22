"""
Query service: orchestrates the full RAG retrieval + generation + evaluation pipeline.

Pipeline:
  1. Embed user question
  2. Similarity search in FAISS      [retrieval_ms tracked]
  3. Pass top-k chunks to LLM        [llm_ms tracked]
  4. Evaluate answer with 2nd LLM    [eval_ms tracked]
  5. Return structured response
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from models.schemas import QueryResponse, SourceChunk
from services.llm_service import generate_answer, evaluate_answer
from utils.config import get_settings
from utils.logger import get_logger

if TYPE_CHECKING:
    from vector_store.faiss_store import FAISSStore

logger = get_logger("rag.query")


def run_query(
    question: str,
    store: "FAISSStore",
    document_id: str | None = None,
    top_k: int | None = None,
) -> QueryResponse:
    """
    Execute the full RAG pipeline and return a structured response.

    Args:
        question:    User's natural-language question
        store:       Initialised FAISSStore
        document_id: If given, search only within this document
        top_k:       Override default number of chunks to retrieve
    """
    cfg   = get_settings()
    k     = top_k or cfg.top_k_results
    t_all = time.perf_counter()

    # ── Step 1: Embed query ──────────────────────────────────────────────────
    from services.embedding_service import embed_query
    query_vec = embed_query(question)

    # ── Step 2: Retrieve from FAISS ──────────────────────────────────────────
    t_ret = time.perf_counter()
    results = store.search(query_vec, k=k, document_id=document_id)
    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    if not results:
        logger.warning("No chunks retrieved for question: %r", question[:80])
        return QueryResponse(
            answer=(
                "No relevant documents found. "
                "Please upload documents before asking questions."
            ),
            sources=[],
            latency_ms=(time.perf_counter() - t_all) * 1000,
            retrieval_ms=retrieval_ms,
            llm_ms=0.0,
            top_similarity=0.0,
            question=question,
            provider_used=None,
        )

    top_sim = results[0][1]
    logger.info(
        "Retrieved %d chunks | top_sim=%.3f | retrieval_ms=%.1f",
        len(results), top_sim, retrieval_ms,
    )

    if top_sim < 0.40:
        logger.warning(
            "Low similarity (%.3f) for query %r — answer may be unreliable",
            top_sim, question[:60],
        )

    # ── Step 3: Build context ────────────────────────────────────────────────
    context_chunks = [
        (meta.text, meta.filename, sim)
        for meta, sim in results
    ]

    # ── Step 4: Generate answer (LLM call 1) ─────────────────────────────────
    answer, llm_ms, provider_used = generate_answer(question, context_chunks)

    # ── Step 5: Evaluate answer (LLM call 2) ─────────────────────────────────
    evaluation, eval_ms = evaluate_answer(question, answer, context_chunks)
    logger.info("Evaluation completed in %.1f ms", eval_ms)

    latency_ms = (time.perf_counter() - t_all) * 1000

    logger.info(
        "Query complete | latency=%.1f ms | llm=%.1f ms | eval=%.1f ms | top_sim=%.3f",
        latency_ms, llm_ms, eval_ms, top_sim,
    )

    # ── Build response ────────────────────────────────────────────────────────
    sources = [
        SourceChunk(
            document_id=meta.document_id,
            filename=meta.filename,
            chunk_id=meta.chunk_id,
            text_preview=meta.text[:200],
            similarity=round(sim, 4),
        )
        for meta, sim in results
    ]

    # Append evaluation block to answer
    full_answer = f"{answer}\n\n---\n\n📊 **RAG Evaluation**\n\n{evaluation}"

    return QueryResponse(
        answer=full_answer,
        sources=sources,
        latency_ms=round(latency_ms, 2),
        retrieval_ms=round(retrieval_ms, 2),
        llm_ms=round(llm_ms + eval_ms, 2),
        top_similarity=round(top_sim, 4),
        question=question,
        provider_used=provider_used,
    )