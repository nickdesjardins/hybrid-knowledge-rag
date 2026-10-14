"""The ingestion pipeline as one function, shared by ``POST /api/v1/ingest`` and the demo seeder.

parse (worker thread) -> chunk (worker thread) -> ceilings -> contextualize -> embed -> store.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.database import ChunkRecord
from app.errors import PDFTooLargeError, RetrievalEngineError
from app.ingestion.chunker import chunk_document
from app.ingestion.contextualizer import Contextualizer, apply_context
from app.ingestion.pdf_parser import parse_pdf_bytes_async
from app.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


class DocumentStore(Protocol):
    async def store_document(
        self,
        *,
        title: str,
        filename: str,
        chunks: Sequence[ChunkRecord],
        page_count: int | None = None,
    ) -> uuid.UUID: ...


class IngestComponents(Protocol):
    settings: Settings
    store: DocumentStore
    embedder: Embedder
    contextualizer: Contextualizer


class EmptyDocumentError(RetrievalEngineError):
    """The PDF parsed but produced no chunks."""


@dataclass(slots=True)
class IngestResult:
    document_id: uuid.UUID
    title: str
    filename: str
    pages_count: int
    chunks_count: int
    processing_time_ms: float


async def ingest_pdf(
    components: IngestComponents, data: bytes, filename: str, title: str | None = None
) -> IngestResult:
    """Run the whole ingestion pipeline for one PDF.

    Raises ``PDFParseError`` / ``PDFTooLargeError`` (parse stage), ``EmptyDocumentError`` (no
    text chunks) and ``ProviderError`` (embeddings); the caller maps them to HTTP codes.
    """
    settings = components.settings
    started = time.perf_counter()

    parsed = await parse_pdf_bytes_async(data, filename, title, max_pages=settings.max_pdf_pages)

    # Pure-Python tokenisation over a whole document is CPU-bound; keep it off the event loop.
    chunks = await asyncio.to_thread(
        chunk_document,
        parsed,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise EmptyDocumentError("No text chunks could be produced")
    if len(chunks) > settings.max_chunks_per_document:
        raise PDFTooLargeError(
            f"Document produces {len(chunks)} chunks; limit is MAX_CHUNKS_PER_DOCUMENT="
            f"{settings.max_chunks_per_document}"
        )

    contexts = await components.contextualizer.contextualize(parsed, chunks)
    contextualized = [
        apply_context(ctx, chunk.text) for ctx, chunk in zip(contexts, chunks, strict=True)
    ]
    embeddings = await components.embedder.embed_documents(contextualized)

    records = [
        ChunkRecord(
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            raw_content=chunk.text,
            contextualized_content=text,
            embedding=vector,
            page_map=chunk.page_map,
        )
        for chunk, text, vector in zip(chunks, contextualized, embeddings, strict=True)
    ]
    document_id = await components.store.store_document(
        title=parsed.title, filename=filename, chunks=records, page_count=parsed.page_count
    )
    # Every write goes through here, so this is where the cached corpus listing and the vector
    # map's projection of the corpus go stale.
    for cache in ("catalogue", "vector_map"):
        invalidate = getattr(getattr(components, cache, None), "invalidate", None)
        if callable(invalidate):
            invalidate()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "Ingested %s: %d pages, %d chunks in %.0f ms",
        filename,
        parsed.page_count,
        len(records),
        elapsed_ms,
    )
    return IngestResult(
        document_id=document_id,
        title=parsed.title,
        filename=filename,
        pages_count=parsed.page_count,
        chunks_count=len(records),
        processing_time_ms=round(elapsed_ms, 2),
    )
