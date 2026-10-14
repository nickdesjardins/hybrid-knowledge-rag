"""Pydantic v2 schemas: API contracts plus the internal chunk records shared across stages."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.textnorm import fold_compat, has_control_characters

# --------------------------------------------------------------------------- API: query


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=2000, description="Natural-language question.")
    filter_document_id: str | None = Field(
        default=None,
        description="Restrict retrieval to a single ingested document (UUID).",
    )

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        """Refuse control characters, then fold the question the same way documents are folded.

        Ingestion turns "ﬁ" into "fi" and non-breaking spaces into spaces; the question gets the
        same treatment so keyword search matches what is stored.
        """
        if has_control_characters(value):
            raise ValueError("query must not contain control characters")
        folded = fold_compat(value).strip()
        if len(folded) < 3:
            raise ValueError("query must contain at least 3 characters")
        return folded

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


class RetrievedPassage(BaseModel):
    """One of the passages handed to the model, with how the retriever found it."""

    chunk_id: str
    document_title: str
    page_number: int
    dense_rank: int | None = Field(default=None, description="Rank in the pgvector arm, if found.")
    sparse_rank: int | None = Field(
        default=None, description="Rank in the full-text arm, if found."
    )
    rrf_score: float
    fused_rank: int | None = Field(
        default=None,
        description="Position in the fused (RRF) candidate list, before re-ranking (1 = best).",
    )
    rerank_position: int
    relevance_score: float
    cited: bool = Field(default=False, description="True when a verified citation quotes it.")
    preview: str = Field(default="", description="First characters of the passage text.")


class StageTimings(BaseModel):
    """Milliseconds spent in each stage of this answer."""

    embed_ms: float = Field(ge=0.0, description="Turning the question into a vector.")
    search_ms: float = Field(ge=0.0, description="The hybrid SQL statement (both arms + RRF).")
    rerank_ms: float = Field(ge=0.0, description="Cross-encoder scoring of the fused candidates.")
    generate_ms: float | None = Field(
        default=None, ge=0.0, description="The language model call plus quote verification."
    )


class PipelineTrace(BaseModel):
    """How this answer was produced, stage by stage: what the playground walkthrough explains."""

    embedding_model: str
    embedding_dimensions: int
    reranker_model: str
    rrf_k: int
    candidate_limit: int = Field(description="Chunks each search arm may return (TOP_K_RETRIEVAL).")
    candidates: int = Field(description="Distinct chunks after fusing both arms.")
    found_by_both: int
    found_by_vector_only: int
    found_by_keyword_only: int
    passages_to_model: int = Field(description="Chunks kept after re-ranking (TOP_K_RERANK).")
    quotes_proposed: int | None = Field(
        default=None, description="Quotes the model returned before verification."
    )
    quotes_verified: int | None = Field(
        default=None, description="Quotes found character for character in the stored chunk."
    )
    timings: StageTimings


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
    retrieved_passages: list[RetrievedPassage] = Field(
        default_factory=list,
        description="The re-ranked passages the answer was generated from (retrieval trace).",
    )
    pipeline: PipelineTrace | None = Field(
        default=None, description="Stage-by-stage counts and timings of this answer."
    )


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
    demo_mode: bool = False
    uploads_enabled: bool = True


# ----------------------------------------------------------------------- API: corpus


class DocumentSummary(BaseModel):
    id: str
    title: str
    filename: str
    pages_count: int = Field(ge=0)
    chunks_count: int = Field(ge=0)
    created_at: datetime | None = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]


class PlaygroundConfig(BaseModel):
    demo_mode: bool
    uploads_enabled: bool
    generator: ProviderStatus
    retrieval: dict[str, int]
    limits: dict[str, int | None]
    documents: list[DocumentSummary]
    sample_questions: list[str]
    models: dict[str, str | int | None] = Field(
        default_factory=dict, description="Embedding and re-ranking models, for the walkthrough."
    )
    system_prompt: str = Field(
        default="", description="The rules the language model answers under (shown to learners)."
    )


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
    page_map: list[tuple[int, int]] | None = Field(
        default=None,
        description="(offset in raw_content, page): where each page begins inside the chunk.",
    )


class RerankedChunk(RetrievedChunk):
    """A candidate after cross-encoder re-ranking."""

    relevance_score: float
    rerank_position: int
