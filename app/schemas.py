"""Pydantic v2 schemas: API contracts plus the internal chunk records shared across stages."""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- API: query


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=2000, description="Natural-language question.")
    filter_document_id: str | None = Field(
        default=None,
        description="Restrict retrieval to a single ingested document (UUID).",
    )

    @field_validator("filter_document_id")
    @classmethod
    def _validate_document_id(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        try:
            return str(uuid.UUID(value))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("filter_document_id must be a valid UUID") from exc


class Citation(BaseModel):
    document_title: str
    page_number: int = Field(ge=1)
    chunk_id: str
    verbatim_quote: str = Field(min_length=1)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    REFUSED = "refused"


class GroundedAnswer(BaseModel):
    query: str
    answer: str
    is_grounded: bool = Field(description="False when the retrieved context was insufficient.")
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]
    retrieval_latency_ms: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    status: AnswerStatus = Field(
        default=AnswerStatus.ANSWERED,
        description="Explicit negative-out signal: answered, insufficient_context, or refused.",
    )
    generation_model: str | None = None


# -------------------------------------------------------------------------- API: ingest


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    title: str
    chunks_count: int = Field(ge=0)
    pages_count: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0.0)


# -------------------------------------------------------------------------- API: health


class ProviderStatus(BaseModel):
    provider: str
    configured: bool
    model: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    database_connected: bool
    pgvector_installed: bool
    pgvector_version: str | None = None
    providers: dict[str, ProviderStatus]


# ----------------------------------------------------------------- internal: retrieval


class RetrievedChunk(BaseModel):
    """A fused hybrid-search candidate (output of the RRF CTE)."""

    id: str
    document_id: str
    document_title: str
    chunk_index: int
    page_number: int
    raw_content: str
    contextualized_content: str
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class RerankedChunk(RetrievedChunk):
    """A candidate after cross-encoder re-ranking."""

    relevance_score: float
    rerank_position: int
