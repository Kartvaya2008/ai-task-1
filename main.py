"""
RAG Question-Answering API
==========================
Entry point: FastAPI application with rate limiting, CORS, and lifespan management.

Start with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from routes import health_router, query_router, upload_router
from utils.config import get_settings
from utils.logger import setup_logger
from vector_store.faiss_store import FAISSStore

# ── Config & logging ───────────────────────────────────────────────────────────
cfg    = get_settings()
logger = setup_logger(cfg.log_path)

# ── Global FAISS store (shared across requests) ────────────────────────────────
store = FAISSStore(
    index_path=cfg.faiss_path,
    meta_path=cfg.meta_path,
    dim=384,   # all-MiniLM-L6-v2 output dimension
)

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{cfg.rate_limit}/minute"])


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load FAISS index safely."""
    logger.info("=======================================")
    logger.info("  RAG API starting up")
    logger.info("=======================================")

    # Load vector store safely
    store.load()
    logger.info("Vector store ready (%d chunks)", store.total_chunks)

    yield   # ← application runs here

    logger.info("RAG API shutting down")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Question-Answering API",
    description=(
        "Upload PDF/TXT documents and ask natural-language questions. "
        "Uses FAISS vector search + Llama 3 (via Groq) for context-aware answers."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
origins = [
    origin.strip()
    for origin in cfg.allowed_origins.split(",")
    if origin.strip() and origin.strip() != "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["http://localhost:3000", "http://localhost:8000"],
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:.*|http://127\.0\.0\.1:.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Rate-limited middleware ────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, and response status."""
    response = await call_next(request)
    logger.info(
        "%s %s -> %d",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response



# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(upload_router, prefix="/api/v1")
app.include_router(query_router,  prefix="/api/v1")


# ── Root landing page ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>RAG API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 2rem;
    }
    .card {
      background: #1a1f2e;
      border: 1px solid #2d3748;
      border-radius: 16px;
      padding: 3rem;
      max-width: 640px;
      width: 100%;
      text-align: center;
      box-shadow: 0 25px 50px rgba(0,0,0,0.5);
    }
    .badge {
      display: inline-block;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: white;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 0.3rem 0.9rem;
      border-radius: 99px;
      margin-bottom: 1.5rem;
    }
    h1 { font-size: 2rem; font-weight: 800; margin-bottom: 0.75rem; color: #f8fafc; }
    p  { color: #94a3b8; line-height: 1.7; margin-bottom: 2rem; }
    .links { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
    a {
      display: inline-block;
      padding: 0.65rem 1.5rem;
      border-radius: 8px;
      font-weight: 600;
      text-decoration: none;
      font-size: 0.9rem;
      transition: opacity 0.2s;
    }
    a:hover { opacity: 0.85; }
    .primary   { background: #6366f1; color: white; }
    .secondary { background: #2d3748; color: #e2e8f0; }
    .endpoints {
      margin-top: 2rem;
      text-align: left;
      background: #0f1117;
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      font-family: monospace;
      font-size: 0.85rem;
      color: #7dd3fc;
      line-height: 2;
    }
    .method { color: #86efac; font-weight: 700; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">v1.0.0</div>
    <h1>📄 RAG API</h1>
    <p>Upload documents, ask questions — powered by FAISS retrieval and Llama 3 via Groq.</p>
    <div class="links">
      <a href="/docs"  class="primary">Swagger UI</a>
      <a href="/redoc" class="secondary">ReDoc</a>
      <a href="/health" class="secondary">Health</a>
    </div>
    <div class="endpoints">
      <div><span class="method">POST</span>  /api/v1/documents/upload</div>
      <div><span class="method">GET &nbsp;</span> /api/v1/documents/</div>
      <div><span class="method">GET &nbsp;</span> /api/v1/documents/{id}/status</div>
      <div><span class="method">POST</span>  /api/v1/query/</div>
      <div><span class="method">GET &nbsp;</span> /health</div>
    </div>
  </div>
</body>
</html>
"""


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
