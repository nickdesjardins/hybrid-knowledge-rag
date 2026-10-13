"""Async PostgreSQL layer: engine, sessions, schema bootstrap, and document/chunk writes.

Raw SQL is used deliberately. The hybrid-search CTE (see ``app/retrieval/hybrid_search.py``) and the
pgvector/tsvector DDL are easier to reason about, EXPLAIN, and tune as literal SQL than through an ORM.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.errors import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRecord:
    """Everything needed to persist one chunk row."""

    chunk_index: int
    page_number: int
    raw_content: str
    contextualized_content: str
    embedding: Sequence[float]


def vector_literal(values: Sequence[float]) -> str:
    """Serialise floats into pgvector's text input format, e.g. ``[0.1,0.2,0.3]``.

    The literal is sent as ``text`` and cast server-side (``CAST(CAST(:v AS text) AS vector)``), which
    works with asyncpg without registering a custom codec.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def build_schema_statements(settings: Settings) -> list[str]:
    """DDL executed on startup. Idempotent: every statement uses IF NOT EXISTS."""
    dims = int(settings.embedding_dimensions)
    m = int(settings.hnsw_m)
    ef_construction = int(settings.hnsw_ef_construction)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
        CREATE TABLE IF NOT EXISTS documents (
            id         UUID PRIMARY KEY,
            title      TEXT NOT NULL,
            filename   TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id                     UUID PRIMARY KEY,
            document_id            UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index            INTEGER NOT NULL,
            page_number            INTEGER NOT NULL,
            raw_content            TEXT NOT NULL,
            contextualized_content TEXT NOT NULL,
            embedding              vector({dims}) NOT NULL,
            tsv_content            tsvector GENERATED ALWAYS AS
                                   (to_tsvector('english', contextualized_content)) STORED,
            created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        );
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
        """,
        "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING gin (tsv_content);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks (document_id);",
    ]


_INSERT_DOCUMENT_SQL = text(
    """
    INSERT INTO documents (id, title, filename)
    VALUES (:id, :title, :filename)
    """
)

_INSERT_CHUNK_SQL = text(
    """
    INSERT INTO document_chunks
        (id, document_id, chunk_index, page_number, raw_content, contextualized_content, embedding)
    VALUES
        (:id, :document_id, :chunk_index, :page_number, :raw_content, :contextualized_content,
         CAST(CAST(:embedding AS text) AS vector))
    """
)


class Database:
    """Owns the async engine and exposes the handful of operations the engine needs."""

    def __init__(self, settings: Settings, engine: AsyncEngine | None = None) -> None:
        self.settings = settings
        self.engine: AsyncEngine = engine or create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            echo=settings.db_echo,
            # asyncpg's default connect timeout is 60 s; fail fast on unreachable hosts.
            connect_args={"timeout": settings.db_connect_timeout_s},
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self.pgvector_version: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def init_schema(self) -> None:
        """Create the extension, tables, and indexes (idempotent), then verify the vector width.

        ``CREATE TABLE IF NOT EXISTS`` silently keeps an existing table, so a changed
        ``EMBEDDING_DIMENSIONS`` would otherwise only surface as insert/query failures later.
        """
        async with self.engine.begin() as conn:
            for statement in build_schema_statements(self.settings):
                await conn.execute(text(statement))
        self.pgvector_version = await self.get_pgvector_version()
        actual_dims = await self.embedding_column_dimensions()
        if actual_dims is not None and actual_dims != self.settings.embedding_dimensions:
            raise ConfigurationError(
                f"document_chunks.embedding is vector({actual_dims}) but EMBEDDING_DIMENSIONS="
                f"{self.settings.embedding_dimensions}; re-create the table or change the setting"
            )
        logger.info(
            "Schema ready (pgvector %s, dims=%d, hnsw m=%d ef_construction=%d)",
            self.pgvector_version,
            self.settings.embedding_dimensions,
            self.settings.hnsw_m,
            self.settings.hnsw_ef_construction,
        )

    async def dispose(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    # --------------------------------------------------------------------- health
    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except (SQLAlchemyError, OSError) as exc:
            logger.warning("Database ping failed: %s", exc)
            return False

    async def get_pgvector_version(self) -> str | None:
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
                row = result.first()
                return str(row[0]) if row else None
        except (SQLAlchemyError, OSError) as exc:
            logger.warning("pgvector version lookup failed: %s", exc)
            return None

    async def embedding_column_dimensions(self) -> int | None:
        """Width of the existing ``embedding`` column (pgvector stores it in the typmod)."""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT a.atttypmod FROM pg_attribute AS a "
                    "WHERE a.attrelid = 'document_chunks'::regclass AND a.attname = 'embedding'"
                )
            )
            row = result.first()
        if row is None or row[0] is None or int(row[0]) <= 0:
            return None
        return int(row[0])

    async def pgvector_installed(self) -> bool:
        version = await self.get_pgvector_version()
        if version is not None:
            self.pgvector_version = version
        return version is not None

    @property
    def supports_iterative_scan(self) -> bool:
        """``hnsw.iterative_scan`` shipped in pgvector 0.8.0."""
        version = self.pgvector_version
        if not version:
            return False
        try:
            major, minor = (int(part) for part in version.split(".")[:2])
        except ValueError:
            return False
        return (major, minor) >= (0, 8)

    # --------------------------------------------------------------------- writes
    async def store_document(
        self,
        *,
        title: str,
        filename: str,
        chunks: Sequence[ChunkRecord],
        document_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Insert a document and all of its chunks in one transaction."""
        doc_id = document_id or uuid.uuid4()
        async with self.session() as session:
            async with session.begin():
                await session.execute(
                    _INSERT_DOCUMENT_SQL, {"id": doc_id, "title": title, "filename": filename}
                )
                if chunks:
                    await session.execute(
                        _INSERT_CHUNK_SQL,
                        [
                            {
                                "id": uuid.uuid4(),
                                "document_id": doc_id,
                                "chunk_index": chunk.chunk_index,
                                "page_number": chunk.page_number,
                                "raw_content": chunk.raw_content,
                                "contextualized_content": chunk.contextualized_content,
                                "embedding": vector_literal(chunk.embedding),
                            }
                            for chunk in chunks
                        ],
                    )
        logger.info("Stored document %s (%s) with %d chunks", doc_id, filename, len(chunks))
        return doc_id
