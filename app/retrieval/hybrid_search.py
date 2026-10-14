"""Hybrid retrieval: pgvector cosine search + tsvector full-text search fused with RRF in one CTE.

Why one statement: both retrievers run inside a single round-trip and a single snapshot, and the
fusion happens in SQL. Each arm returns at most ``limit`` chunks, so the fused union holds at most
``2 * limit``; the statement returns all of it in fusion order and the Python side keeps the first
``limit`` as candidates. The rest are "cut after fusion": only their arm ranks are kept, so the
vector map can show that an arm did return them.

RRF(d) = sum over retrievers m of 1 / (k + rank_m(d)), k = 60 by default.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text

from app.config import Settings
from app.database import Database, vector_literal
from app.schemas import CutChunk, RetrievedChunk

logger = logging.getLogger(__name__)

# NOTE on parameters: every bind is wrapped in an explicit CAST so asyncpg can infer a stable
# parameter type even though the same name is reused several times. The query vector is sent as a
# pgvector text literal and cast server-side, so no custom asyncpg codec is required.
#
# NOTE on the sparse arm: websearch_to_tsquery('english', 'How long must batch records be retained?')
# yields 'long' & 'batch' & 'record' & 'retain'. The governing clause contains no "long", so a strict
# AND match returns nothing. The CTE therefore *matches* on the OR-relaxed query and *ranks* by the
# strict score plus the relaxed score: full matches outrank partial ones, partial ones still surface.
HYBRID_SEARCH_SQL = text(
    """
    WITH dense_raw AS MATERIALIZED (
        SELECT c.id,
               c.embedding <=> CAST(CAST(:query_vector AS text) AS vector) AS distance
        FROM document_chunks AS c
        WHERE CAST(:document_id AS uuid) IS NULL
           OR c.document_id = CAST(:document_id AS uuid)
        ORDER BY c.embedding <=> CAST(CAST(:query_vector AS text) AS vector) ASC
        LIMIT CAST(:limit_n AS integer)
    ),
    dense_search AS (
        -- Re-sort explicitly (MATERIALIZED + "distance + 0"): with hnsw.iterative_scan =
        -- relaxed_order the index may emit rows slightly out of distance order, and the planner
        -- would otherwise trust the index order and skip the sort.
        SELECT id,
               ROW_NUMBER() OVER (ORDER BY distance + 0 ASC, id ASC) AS rank
        FROM dense_raw
    ),
    sparse_search AS (
        SELECT sub.id,
               ROW_NUMBER() OVER (ORDER BY sub.score DESC, sub.id ASC) AS rank
        FROM (
            SELECT c.id,
                   -- Strict query (all terms, websearch AND semantics) scores 0 on partial matches,
                   -- so chunks containing every term rank above chunks matching only some terms.
                   ts_rank_cd(c.tsv_content,
                              websearch_to_tsquery('english', CAST(:query_text AS text)))
                 + ts_rank_cd(c.tsv_content,
                              CAST(replace(CAST(websearch_to_tsquery('english', CAST(:query_text AS text))
                                                AS text), ' & ', ' | ') AS tsquery)) AS score
            FROM document_chunks AS c
            -- Match on the OR-relaxed form: natural-language questions carry words ("how",
            -- "long", "must") that never appear in the governing clause, and a strict AND
            -- would return nothing. Exact identifiers (SOP-QA-017, 21 CFR Part 11) still match.
            WHERE c.tsv_content @@ CAST(replace(CAST(websearch_to_tsquery('english', CAST(:query_text AS text))
                                                     AS text), ' & ', ' | ') AS tsquery)
              AND (CAST(:document_id AS uuid) IS NULL
                   OR c.document_id = CAST(:document_id AS uuid))
            ORDER BY score DESC
            LIMIT CAST(:limit_n AS integer)
        ) AS sub
    ),
    rrf_fusion AS (
        SELECT COALESCE(d.id, s.id) AS id,
               COALESCE(1.0 / (CAST(:rrf_k AS double precision) + d.rank), 0.0)
             + COALESCE(1.0 / (CAST(:rrf_k AS double precision) + s.rank), 0.0) AS rrf_score,
               d.rank AS dense_rank,
               s.rank AS sparse_rank
        FROM dense_search AS d
        FULL OUTER JOIN sparse_search AS s ON d.id = s.id
    )
    -- No LIMIT here: each arm has its own, so the union is at most twice that many rows. The
    -- caller keeps the first limit_n rows as candidates; the ORDER BY ends on the unique id, so
    -- that cut is deterministic.
    SELECT c.id,
           c.document_id,
           doc.title AS document_title,
           c.chunk_index,
           c.page_number,
           c.raw_content,
           c.contextualized_content,
           c.page_map,
           f.rrf_score,
           f.dense_rank,
           f.sparse_rank
    FROM rrf_fusion AS f
    JOIN document_chunks AS c ON c.id = f.id
    JOIN documents AS doc ON doc.id = c.document_id
    ORDER BY f.rrf_score DESC,
             f.dense_rank ASC NULLS LAST,
             f.sparse_rank ASC NULLS LAST,
             c.id ASC
    """
)


def rrf_score(ranks: Sequence[int | None], k: int = 60) -> float:
    """RRF contribution for one document given its rank (1-based) in each retriever, or None."""
    return sum(1.0 / (k + rank) for rank in ranks if rank is not None)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Pure-Python reference implementation of the SQL ``rrf_fusion`` CTE.

    ``ranked_lists`` holds one ordered id list per retriever (best first). Returns ``(id, score)``
    pairs ordered by score descending. Ties follow the SQL statement's ORDER BY exactly: better
    rank in the first retriever (dense), then in the second (sparse), then id.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    retriever_count = len(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, list[int | None]] = {}
    for retriever, ranking in enumerate(ranked_lists):
        seen: set[str] = set()
        for position, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # a retriever never contributes twice for one document
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + position)
            ranks.setdefault(doc_id, [None] * retriever_count)[retriever] = position

    def sort_key(item: tuple[str, float]) -> tuple:
        doc_id, score = item
        rank_keys = tuple(r if r is not None else math.inf for r in ranks[doc_id])
        return (-score, *rank_keys, doc_id)

    fused = sorted(scores.items(), key=sort_key)
    return fused[:limit] if limit is not None else fused


def _decode_page_map(value: object) -> list[tuple[int, int]] | None:
    """JSONB arrives as text through a raw statement; tolerate both text and decoded lists."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    try:
        return [(int(start), int(page)) for start, page in value]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class HybridResult:
    """One hybrid search: the fused candidates, and the chunks fusion cut below the limit."""

    #: The best ``limit`` chunks in fusion (RRF) order: what re-ranking sees.
    candidates: list[RetrievedChunk]
    #: Every other chunk an arm returned, in fusion order, with its arm ranks only.
    cut: list[CutChunk]

    @property
    def union_size(self) -> int:
        """Distinct chunks the two arms returned together."""
        return len(self.candidates) + len(self.cut)


class HybridSearcher(Protocol):
    async def search_with_arms(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        document_id: str | None = None,
        limit: int | None = None,
    ) -> HybridResult: ...

    async def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        document_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]: ...


class PostgresHybridSearcher:
    """Executes the RRF CTE against PostgreSQL."""

    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    async def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        document_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """The fused candidates only (see :meth:`search_with_arms`)."""
        result = await self.search_with_arms(
            query_text=query_text, query_vector=query_vector, document_id=document_id, limit=limit
        )
        return result.candidates

    async def search_with_arms(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        document_id: str | None = None,
        limit: int | None = None,
    ) -> HybridResult:
        limit_n = int(limit or self.settings.top_k_retrieval)
        params = {
            "query_vector": vector_literal(query_vector),
            "query_text": query_text,
            "document_id": uuid.UUID(document_id) if document_id else None,
            "limit_n": limit_n,
            "rrf_k": float(self.settings.rrf_k),
        }
        ef_search = max(self.settings.effective_hnsw_ef_search, limit_n)

        async with self.db.session() as session:
            async with session.begin():
                # SET LOCAL scopes the GUCs to this transaction only.
                await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
                if self.db.supports_iterative_scan and self.settings.hnsw_iterative_scan != "off":
                    await session.execute(
                        text(f"SET LOCAL hnsw.iterative_scan = {self.settings.hnsw_iterative_scan}")
                    )
                result = await session.execute(HYBRID_SEARCH_SQL, params)
                rows = result.mappings().all()

        cut = [
            CutChunk(
                id=str(row["id"]),
                rrf_score=float(row["rrf_score"]),
                dense_rank=int(row["dense_rank"]) if row["dense_rank"] is not None else None,
                sparse_rank=int(row["sparse_rank"]) if row["sparse_rank"] is not None else None,
            )
            for row in rows[limit_n:]
        ]
        chunks = [
            RetrievedChunk(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                document_title=row["document_title"],
                chunk_index=int(row["chunk_index"]),
                page_number=int(row["page_number"]),
                raw_content=row["raw_content"],
                contextualized_content=row["contextualized_content"],
                page_map=_decode_page_map(row["page_map"]),
                rrf_score=float(row["rrf_score"]),
                dense_rank=int(row["dense_rank"]) if row["dense_rank"] is not None else None,
                sparse_rank=int(row["sparse_rank"]) if row["sparse_rank"] is not None else None,
            )
            for row in rows[:limit_n]
        ]
        logger.debug(
            "Hybrid search returned %d candidates (dense hits=%d, sparse hits=%d), %d cut",
            len(chunks),
            sum(1 for c in chunks if c.dense_rank is not None),
            sum(1 for c in chunks if c.sparse_rank is not None),
            len(cut),
        )
        return HybridResult(candidates=chunks, cut=cut)
