"""
Centralised settings loaded from environment variables / .env file.
Uses pydantic-settings for type-safe configuration.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM — Groq (primary) + OpenRouter (fallback)
    groq_api_key:        str = "ADD_YOUR_API"
    llm_model:           str = "llama-3.3-70b-versatile"
    openrouter_api_key:  str = ""
    openrouter_model:    str = "meta-llama/llama-3.3-70b-instruct:free"

    # Embeddings (local, no key needed)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size:    int = 400
    chunk_overlap: int = 80
    top_k_results: int = 5

    # Rate limiting
    rate_limit: int = 20   # requests / minute / IP

    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"

    # File limits
    max_file_size: int = 10_485_760  # 10 MB

    # Paths & Data Directory
    data_dir:          str = ""
    upload_dir:        str = "uploads"
    faiss_index_path:  str = "vector_store/faiss_index"
    metadata_path:     str = "vector_store/metadata.json"
    documents_path:    str = "vector_store/documents.json"
    log_file:          str = "logs/rag.log"

    @property
    def base_data_path(self) -> Path:
        if self.data_dir and self.data_dir.strip():
            p = Path(self.data_dir.strip())
            p.mkdir(parents=True, exist_ok=True)
            return p
        return Path(".")

    @property
    def upload_path(self) -> Path:
        p = self.base_data_path / self.upload_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def faiss_path(self) -> Path:
        p = self.base_data_path / self.faiss_index_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def meta_path(self) -> Path:
        p = self.base_data_path / self.metadata_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def docs_path(self) -> Path:
        p = self.base_data_path / self.documents_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_path(self) -> Path:
        p = self.base_data_path / self.log_file
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()