"""
LLM service with Groq (primary) and OpenRouter (fallback).

Two LLM calls per query:
  1. generate_answer  — answers the question from context
  2. evaluate_answer  — strictly evaluates the answer against context
"""

from __future__ import annotations

import time
import traceback

import httpx
from groq import Groq

from utils.logger import get_logger
from utils.config import get_settings

logger = get_logger("rag.llm")

_groq_client: Groq | None = None
_groq_client_key: str | None = None

_PLACEHOLDER_KEYS = frozenset({
    "ADD_YOUR_API",
    "your_actual_groq_api_key_here",
    "your_api_key_here",
    "",
})


def _is_valid_key(key: str | None) -> bool:
    return bool(key and key not in _PLACEHOLDER_KEYS)


def get_groq_client() -> Groq:
    global _groq_client, _groq_client_key
    settings = get_settings()
    if _groq_client is None or _groq_client_key != settings.groq_api_key:
        if not _is_valid_key(settings.groq_api_key):
            logger.warning("GROQ_API_KEY is missing or placeholder")
        elif not settings.groq_api_key.startswith("gsk_"):
            logger.warning(
                "GROQ_API_KEY does not look like a Groq key (expected gsk_ prefix, got %s...)",
                settings.groq_api_key[:8],
            )
        else:
            logger.info("Initializing Groq client (key length=%d)", len(settings.groq_api_key))
        _groq_client = Groq(api_key=settings.groq_api_key)
        _groq_client_key = settings.groq_api_key
    return _groq_client


def _groq_chat(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    response = get_groq_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


def _openrouter_chat(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    settings = get_settings()
    if not _is_valid_key(settings.openrouter_api_key):
        raise ValueError("OpenRouter API key not configured")

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RAG Document Q&A",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _chat_with_fallback(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    timeout: float = 30.0,
    call_type: str = "completion",
) -> tuple[str, float, str]:
    """
    Try Groq first; on any failure fall back to OpenRouter.
    Returns (content, latency_ms, provider_used).
    """
    settings = get_settings()
    t0 = time.perf_counter()

    # ── Primary: Groq ────────────────────────────────────────────────────────
    logger.info("[LLM] Trying Groq (%s)", call_type)

    groq_error: Exception | None = None
    if _is_valid_key(settings.groq_api_key):
        try:
            content = _groq_chat(
                messages, settings.llm_model, temperature, max_tokens, timeout
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info("[LLM] Groq Success (%.1f ms)", latency_ms)
            return content, latency_ms, "groq"
        except Exception as exc:
            groq_error = exc
            logger.error("[LLM] Groq Failed: %s: %s", type(exc).__name__, exc)
    else:
        groq_error = ValueError("Groq API key not configured")
        logger.warning("[LLM] Groq Failed: API key not configured")

    # ── Fallback: OpenRouter ─────────────────────────────────────────────────
    logger.info("[LLM] Switching To OpenRouter (%s)", call_type)

    if not _is_valid_key(settings.openrouter_api_key):
        raise RuntimeError(
            f"All LLM providers failed. Groq: {groq_error}. OpenRouter: API key not configured."
        ) from groq_error

    try:
        content = _openrouter_chat(
            messages, settings.openrouter_model, temperature, max_tokens, timeout
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("[LLM] OpenRouter Success (%.1f ms)", latency_ms)
        return content, latency_ms, "openrouter"
    except Exception as or_exc:
        logger.error("[LLM] OpenRouter Failed: %s: %s", type(or_exc).__name__, or_exc)
        raise RuntimeError(
            f"All LLM providers failed. Groq: {groq_error}. OpenRouter: {or_exc}"
        ) from or_exc


# ── Prompt: Answer generation ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise document question-answering assistant.

RULES:
1. Answer the user's question using ONLY the context chunks provided below.
2. If the context does not contain sufficient information, respond with:
   "I don't have enough information in the provided documents to answer this question."
3. Be concise and factual. Do not add information not present in the context.
4. Reference the source document name(s) when possible (e.g., "According to report.pdf").
5. Use bullet points or numbered lists only when they improve readability.
"""

EVALUATOR_SYSTEM_PROMPT = """You are an expert evaluator of RAG-based QA systems.
Strictly evaluate the system answer ONLY based on the given document context.
Do NOT assume or add any external knowledge.

Follow these rules:
- If information is not explicitly present in the context, mark it as hallucination.
- Be strict and critical in evaluation.
- Keep explanation clear, concise, and factual.

Return output in this EXACT format (no extra text before or after):
Relevance: (High / Medium / Low)
Accuracy: (Correct / Partially Correct / Incorrect)
Completeness: (Complete / Partial / Poor)
Retrieval Quality: (Good / Average / Bad)
Hallucination: (Yes / No)
Final Verdict: (Good / Needs Improvement / Poor)
Explanation:
- 3-4 short lines
- Clearly mention:
  what is correct
  what is missing
  what is hallucinated (if any)
  how well retrieval matches the question

Important:
- Do not generalize
- Do not overpraise
- Do not add assumptions
- Be objective like a reviewer
"""


def _build_context_str(context_chunks: list[tuple[str, str, float]]) -> str:
    context_parts = []
    for i, (text, filename, sim) in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | Source: {filename} | Relevance: {sim:.2f}]\n{text}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    question: str,
    context_chunks: list[tuple[str, str, float]],
) -> tuple[str, float, str | None]:
    """
    Generate an answer given retrieved context chunks.
    Returns: (answer, latency_ms, provider_used or None if all providers failed)
    """
    context_str = _build_context_str(context_chunks)
    user_message = f"""CONTEXT DOCUMENTS:
{context_str}

---

QUESTION: {question}

Please answer based on the context above."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    try:
        answer, latency_ms, provider = _chat_with_fallback(
            messages, temperature=0.1, max_tokens=1024, call_type="generate_answer"
        )
        logger.info("LLM answer generated via %s in %.1f ms", provider, latency_ms)
        return answer, latency_ms, provider
    except Exception as exc:
        tb_str = traceback.format_exc()
        logger.error(
            "=== ALL LLM PROVIDERS FAILED (generate_answer) ===\n%s\n%s",
            exc, tb_str,
        )
        answer = (
            "LLM service is currently unavailable. "
            "Here are the most relevant excerpts from your documents:\n\n"
            + "\n\n".join(f"- [{fn}] {txt[:300]}…" for txt, fn, _ in context_chunks[:3])
        )
        return answer, 0.0, None


def evaluate_answer(
    question: str,
    answer: str,
    context_chunks: list[tuple[str, str, float]],
) -> tuple[str, float]:
    """
    Second LLM call: strictly evaluates the generated answer.
    Returns: (evaluation, latency_ms)
    """
    context_str = _build_context_str(context_chunks)
    eval_user_message = f"""DOCUMENT CONTEXT:
{context_str}

---

QUESTION ASKED:
{question}

---

SYSTEM ANSWER TO EVALUATE:
{answer}

---

Now strictly evaluate the system answer using ONLY the document context above.
"""

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user",   "content": eval_user_message},
    ]

    try:
        evaluation, latency_ms, provider = _chat_with_fallback(
            messages, temperature=0.0, max_tokens=512, call_type="evaluate_answer"
        )
        logger.info("RAG evaluation completed via %s in %.1f ms", provider, latency_ms)
        return evaluation, latency_ms
    except Exception as exc:
        logger.error("Evaluation failed on all providers: %s", exc)
        return "Evaluation unavailable — all LLM providers failed.", 0.0
