"""Environment-driven configuration (pydantic-settings).

Every tunable of the retrieval pipeline lives here so that the SQL, the chunker, and the provider
clients never hard-code magic numbers. Values are read from the process environment first and then
from a local ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ application
    app_name: str = "hybrid-retrieval-engine"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    api_key: SecretStr | None = Field(
        default=None,
        description="When set, /api/v1/* requests must send it in the X-API-Key header.",
    )

    # --------------------------------------------------------------------- database
    database_url: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/rag",
        description="SQLAlchemy async URL using the asyncpg driver.",
    )
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_connect_timeout_s: float = Field(default=10.0, gt=0, le=120)
    health_timeout_s: float = Field(
        default=5.0, gt=0, le=60, description="Deadline for each /health database probe."
    )
    db_echo: bool = False
    auto_create_schema: bool = Field(
        default=True,
        description="Run CREATE EXTENSION / CREATE TABLE / CREATE INDEX on startup.",
    )

    # ------------------------------------------------------------------- embeddings
    embedding_provider: Literal["openai", "fastembed", "hashing"] = "openai"
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(
        default=1536,
        ge=1,
        le=2000,
        description="Width of the pgvector column. pgvector's `vector` type supports up to 2000 dims.",
    )
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    embedding_concurrency: int = Field(
        default=4, ge=1, le=32, description="Concurrent embedding requests during ingestion."
    )
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    # ------------------------------------------------------------------- generation
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    generation_model: str = "claude-opus-5-5"
    generation_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="low",
        description="Claude 5.x reasoning effort. Low keeps answers fast and deterministic-leaning.",
    )
    generation_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16_000,
        description=(
            "Output budget for the grounded answer (thinking counts toward it on Claude 5.x). "
            "Capped at 16K because non-streaming Anthropic requests are limited to ~21K tokens."
        ),
    )
    generation_temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for providers that accept it (OpenAI). Claude 5.x models reject "
            "non-default sampling parameters, so determinism there comes from constrained JSON "
            "decoding plus low effort. Set to an empty value to omit the parameter entirely."
        ),
    )
    openai_generation_model: str = "gpt-4.1"

    # --------------------------------------------------------------- contextualizer
    contextualizer_enabled: bool = True
    contextualizer_model: str = "claude-haiku-5-5"
    contextualizer_max_tokens: int = Field(
        default=2048,
        ge=256,
        le=8192,
        description="Must leave room for Haiku's adaptive thinking plus the ~100-token preamble.",
    )
    contextualizer_concurrency: int = Field(default=8, ge=1, le=64)
    contextualizer_doc_max_chars: int = Field(default=120_000, ge=1_000)

    # -------------------------------------------------------------------- reranking
    rerank_provider: Literal["cohere", "none"] = "cohere"
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    # -------------------------------------------------------------------- retrieval
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant k.")
    top_k_retrieval: int = Field(default=50, ge=1, le=1000, description="Fused candidates kept.")
    top_k_rerank: int = Field(
        default=3, ge=1, description="Chunks passed to the LLM after re-ranking."
    )
    hnsw_m: int = Field(default=16, ge=2, le=100)
    hnsw_ef_construction: int = Field(default=64, ge=4, le=1000)
    hnsw_ef_search: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Dynamic candidate list at query time. Must be >= top_k_retrieval.",
    )
    hnsw_iterative_scan: Literal["off", "relaxed_order", "strict_order"] = Field(
        default="relaxed_order",
        description="pgvector >= 0.8 iterative scans keep filtered searches from returning too few rows.",
    )

    # --------------------------------------------------------------------- chunking
    chunk_size_tokens: int = Field(default=400, ge=50, le=4000)
    chunk_overlap_tokens: int = Field(default=60, ge=0, le=1000)
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    max_pdf_pages: int = Field(
        default=500, ge=1, description="Uploads with more pages are rejected before extraction."
    )
    max_chunks_per_document: int = Field(
        default=2000,
        ge=1,
        description="Caps LLM contextualizer calls and embedding requests per ingested document.",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("generation_temperature", "api_key", mode="before")
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        """``GENERATION_TEMPERATURE=`` / ``API_KEY=`` in .env mean "unset", not an empty value."""
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    @model_validator(mode="after")
    def _validate_relationships(self) -> Settings:
        if self.top_k_rerank > self.top_k_retrieval:
            raise ValueError("TOP_K_RERANK must be <= TOP_K_RETRIEVAL")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.hnsw_ef_construction < 2 * self.hnsw_m:
            raise ValueError("HNSW_EF_CONSTRUCTION must be >= 2 * HNSW_M (pgvector requirement)")
        return self

    # ------------------------------------------------------------- derived helpers
    @property
    def effective_hnsw_ef_search(self) -> int:
        """HNSW returns at most ``ef_search`` rows, so it can never be below the candidate limit."""
        return max(self.hnsw_ef_search, self.top_k_retrieval)

    @property
    def anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def cohere_configured(self) -> bool:
        return bool(self.cohere_api_key)

    @property
    def generator_configured(self) -> bool:
        if self.llm_provider == "anthropic":
            return self.anthropic_configured
        return self.openai_configured

    @property
    def embedder_configured(self) -> bool:
        if self.embedding_provider == "openai":
            return self.openai_configured
        return True

    @property
    def reranker_configured(self) -> bool:
        if self.rerank_provider == "cohere":
            return self.cohere_configured
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
