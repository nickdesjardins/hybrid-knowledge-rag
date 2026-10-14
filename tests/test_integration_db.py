"""Live PostgreSQL + pgvector integration test (opt-in).

Skipped unless ``TEST_DATABASE_URL`` is set, e.g.::

    TEST_DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag pytest tests/test_integration_db.py

or, without Docker, through the embedded server: ``python scripts/run_live_db_tests.py``.

The test works inside its own schema (``rag_test``) so it never touches production tables, and it
verifies that the SQL CTE really computes RRF the way the pure-Python reference does, that it
reports the chunks fusion cut, and that the vector map's read (``Database.chunk_vectors``) returns
every chunk in the order, with the labels and the embeddings, that the hermetic fake assumes.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings
from app.database import ChunkRecord, Database, vector_literal
from app.errors import ConfigurationError
from app.retrieval.embedder import HashingEmbedder
from app.retrieval.hybrid_search import (
    PostgresHybridSearcher,
    reciprocal_rank_fusion,
    rrf_score,
)
from app.retrieval.vector_map import PREVIEW_CHARS, VectorMapper, preview_of

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")

DIMS = 64
SCHEMA = "rag_test"

PASSAGES = [
    "Batch records must be retained for a minimum of seven years after the product expiry date.",
    "The Quality Assurance manager approves all deviation reports within 48 hours of submission.",
    "Cleaning validation demonstrates that residues are reduced below the acceptance limit of 10 ppm.",
    "Electronic records require an audit trail compliant with 21 CFR Part 11 and weekly backups.",
    "Visitors must sign the logbook at reception and wear a badge at all times.",
]


def _settings(**overrides) -> Settings:
    base = dict(
        database_url=DATABASE_URL,
        embedding_provider="hashing",
        embedding_dimensions=DIMS,
        rerank_provider="none",
        contextualizer_enabled=False,
        top_k_retrieval=50,
    )
    base.update(overrides)
    return Settings(_env_file=None, _env_prefix="HERMETIC_TEST_", **base)


@pytest.fixture
async def db():
    engine = create_async_engine(DATABASE_URL)

    # A per-connection SET rather than a startup parameter: hosted providers such as Neon
    # drop unknown startup options, which would silently run the tests in `public`.
    @event.listens_for(engine.sync_engine, "connect")
    def _set_search_path(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
        dbapi_connection.run_async(
            lambda conn: conn.execute(f"SET search_path TO {SCHEMA}, public")
        )

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        await conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    database = Database(_settings(), engine=engine)
    await database.init_schema()
    try:
        yield database
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        await engine.dispose()


async def _seed(database: Database) -> tuple[uuid.UUID, uuid.UUID]:
    embedder = HashingEmbedder(DIMS)
    vectors = await embedder.embed_documents(PASSAGES)
    records = [
        ChunkRecord(
            chunk_index=i,
            page_number=i + 1,
            raw_content=p,
            contextualized_content=f"Document: Quality Manual. Page {i + 1} of 5.\n\n{p}",
            embedding=v,
        )
        for i, (p, v) in enumerate(zip(PASSAGES, vectors, strict=True))
    ]
    doc_a = await database.store_document(
        title="Quality Manual", filename="qm.pdf", chunks=records[:4]
    )
    doc_b = await database.store_document(
        title="Visitor Policy",
        filename="vp.pdf",
        chunks=[
            ChunkRecord(
                0, 1, PASSAGES[4], f"Document: Visitor Policy.\n\n{PASSAGES[4]}", vectors[4]
            )
        ],
    )
    return doc_a, doc_b


async def test_schema_and_indexes_exist(db: Database) -> None:
    assert db.pgvector_version is not None
    assert await db.embedding_column_dimensions() == DIMS
    async with db.engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE schemaname = :schema AND tablename = 'document_chunks'"
            ),
            {"schema": SCHEMA},
        )
        indexes = {name: definition for name, definition in rows.all()}
    assert "idx_chunks_hnsw" in indexes
    assert "USING hnsw" in indexes["idx_chunks_hnsw"]
    assert "vector_cosine_ops" in indexes["idx_chunks_hnsw"]
    assert "m='16'" in indexes["idx_chunks_hnsw"]
    assert "ef_construction='64'" in indexes["idx_chunks_hnsw"]
    assert "idx_chunks_tsv" in indexes
    assert "USING gin" in indexes["idx_chunks_tsv"]


async def test_init_schema_is_idempotent_and_detects_dimension_drift(db: Database) -> None:
    await db.init_schema()  # second run: every statement is IF NOT EXISTS
    drifted = Database(_settings(embedding_dimensions=32), engine=db.engine)
    with pytest.raises(ConfigurationError, match="vector\\(64\\)"):
        await drifted.init_schema()


async def test_hybrid_search_fuses_dense_and_sparse_arms(db: Database) -> None:
    doc_a, _ = await _seed(db)
    searcher = PostgresHybridSearcher(db, db.settings)
    embedder = HashingEmbedder(DIMS)
    query = "How long must batch records be retained?"
    results = await searcher.search(
        query_text=query, query_vector=await embedder.embed_query(query)
    )
    assert results, "expected fused candidates"
    assert results[0].raw_content.startswith("Batch records must be retained")
    assert results[0].document_id == str(doc_a)
    assert results[0].dense_rank == 1 and results[0].sparse_rank == 1
    for chunk in results:
        assert chunk.rrf_score == pytest.approx(
            rrf_score([chunk.dense_rank, chunk.sparse_rank], k=60), rel=1e-9
        )
    scores = [c.rrf_score for c in results]
    assert scores == sorted(scores, reverse=True)

    # The SQL ORDER BY and the Python reference agree on the full order, ties included.
    dense_ids = [
        c.id
        for c in sorted(
            (c for c in results if c.dense_rank is not None), key=lambda c: c.dense_rank
        )
    ]
    sparse_ids = [
        c.id
        for c in sorted(
            (c for c in results if c.sparse_rank is not None), key=lambda c: c.sparse_rank
        )
    ]
    expected = [doc for doc, _ in reciprocal_rank_fusion([dense_ids, sparse_ids], k=60)]
    assert [c.id for c in results] == expected


async def test_hybrid_search_reports_the_chunks_fusion_cut(db: Database) -> None:
    """Each arm returns ``limit`` chunks and fusion keeps ``limit`` of their union: the rest come
    back with their arm ranks, in the same order the Python reference gives."""
    await _seed(db)
    searcher = PostgresHybridSearcher(db, db.settings)
    embedder = HashingEmbedder(DIMS)
    # The text and the vector point at different passages, so the two arms disagree.
    query = "visitor badge"
    vector = await embedder.embed_query(PASSAGES[2])
    result = await searcher.search_with_arms(query_text=query, query_vector=vector, limit=2)
    assert result.cut, "the two arms must return more than 2 distinct chunks here"
    assert len(result.candidates) == 2 and result.union_size <= 4
    plain = await searcher.search(query_text=query, query_vector=vector, limit=2)
    assert [c.id for c in plain] == [c.id for c in result.candidates]  # retrieval is unchanged

    union = [*result.candidates, *result.cut]
    assert all(c.dense_rank is not None or c.sparse_rank is not None for c in result.cut)
    assert all(rank <= 2 for c in union for rank in (c.dense_rank, c.sparse_rank) if rank)
    dense_ids = [c.id for c in sorted(union, key=lambda c: c.dense_rank or 99) if c.dense_rank]
    sparse_ids = [c.id for c in sorted(union, key=lambda c: c.sparse_rank or 99) if c.sparse_rank]
    expected = reciprocal_rank_fusion([dense_ids, sparse_ids], k=60)
    assert [c.id for c in union] == [doc for doc, _ in expected]
    assert [c.rrf_score for c in union] == pytest.approx([score for _, score in expected])

    # On the map, a cut chunk keeps its arm rank and has no fused rank, and every similarity is
    # pgvector's own 1 - <=>.
    vmap = await VectorMapper(db, db.settings).build(
        query_vector=vector, candidates=result.candidates, cut=result.cut, reranked=[]
    )
    assert vmap is not None
    points = {p.chunk_id: p for p in vmap.points}
    for chunk in result.cut:
        point = points[chunk.id]
        assert (point.dense_rank, point.sparse_rank, point.fused_rank) == (
            chunk.dense_rank,
            chunk.sparse_rank,
            None,
        )
    async with db.engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT id, 1 - (embedding <=> CAST(CAST(:v AS text) AS vector)) "
                "FROM document_chunks"
            ),
            {"v": vector_literal(vector)},
        )
        similarity = {str(chunk_id): value for chunk_id, value in rows.all()}
    assert len(similarity) == len(points)
    for chunk_id, point in points.items():
        assert point.similarity == pytest.approx(similarity[chunk_id], abs=1e-4)


async def test_chunk_vectors_read_every_chunk_in_corpus_order(db: Database) -> None:
    embedder = HashingEmbedder(DIMS)
    # Longer than the 600 characters the statement reads, with runs of whitespace to collapse.
    long_text = "Batch   records\n\nmust be retained.  " + "  ".join(PASSAGES) * 4
    assert len(long_text) > 600
    texts = [long_text, PASSAGES[1], PASSAGES[2], PASSAGES[4]]
    vectors = await embedder.embed_documents(texts)
    manual = [
        # Stored page 2, but the text begins on page 1 (overlap carried from the page before).
        ChunkRecord(0, 2, texts[0], texts[0], vectors[0], page_map=[[0, 1], [40, 2]]),
        ChunkRecord(1, 2, texts[1], texts[1], vectors[1]),  # no page map: the stored page
        ChunkRecord(2, 3, texts[2], texts[2], vectors[2], page_map=[[0, 3]]),
    ]
    # Inserted out of chunk order: the read must sort by chunk_index, not by insertion.
    doc_a = await db.store_document(
        title="Quality Manual", filename="qm.pdf", chunks=[manual[2], manual[0], manual[1]]
    )
    doc_b = await db.store_document(
        title="Visitor Policy",
        filename="vp.pdf",
        chunks=[ChunkRecord(0, 1, texts[3], texts[3], vectors[3])],
    )

    rows = await db.chunk_vectors(limit=100)
    assert [(r.document_id, r.chunk_index) for r in rows] == [
        (str(doc_a), 0),
        (str(doc_a), 1),
        (str(doc_a), 2),
        (str(doc_b), 0),
    ]
    assert [d.id for d in await db.list_documents()] == [str(doc_a), str(doc_b)]
    # The "too large" check reads max_chunks + 1 rows: a smaller limit is a prefix of the order.
    assert [r.id for r in await db.chunk_vectors(limit=3)] == [r.id for r in rows[:3]]

    for row, stored, full_text in zip(rows, vectors, texts, strict=True):
        assert len(row.vector) == DIMS
        assert row.vector == pytest.approx(stored, abs=1e-6)  # pgvector stores float32
        assert row.preview == preview_of(full_text)
    assert [r.page_number for r in rows] == [1, 2, 3, 1]
    assert len(rows[0].preview) == PREVIEW_CHARS
    assert [r.document_title for r in rows] == ["Quality Manual"] * 3 + ["Visitor Policy"]

    # A corpus above VECTOR_MAP_MAX_CHUNKS gets no map, through the real statement.
    small = VectorMapper(db, _settings(vector_map_max_chunks=3))
    assert await small.build(query_vector=vectors[0], candidates=[], reranked=[]) is None
    full = await VectorMapper(db, db.settings).build(
        query_vector=vectors[0], candidates=[], reranked=[]
    )
    assert full is not None and [p.chunk_id for p in full.points] == [r.id for r in rows]


async def test_generated_tsvector_and_sparse_only_hits(db: Database) -> None:
    await _seed(db)
    searcher = PostgresHybridSearcher(db, db.settings)
    # A query vector orthogonal to everything (all-equal components) still returns sparse hits.
    results = await searcher.search(query_text='"21 CFR Part 11"', query_vector=[1.0] * DIMS)
    sparse_hits = [c for c in results if c.sparse_rank is not None]
    assert sparse_hits
    assert any("21 CFR Part 11" in c.raw_content for c in sparse_hits)
    assert sparse_hits[0].sparse_rank == 1


async def test_document_filter_and_unknown_document(db: Database) -> None:
    doc_a, doc_b = await _seed(db)
    searcher = PostgresHybridSearcher(db, db.settings)
    embedder = HashingEmbedder(DIMS)
    query = "visitor badge logbook"
    vector = await embedder.embed_query(query)

    only_b = await searcher.search(query_text=query, query_vector=vector, document_id=str(doc_b))
    assert only_b and {c.document_id for c in only_b} == {str(doc_b)}

    only_a = await searcher.search(query_text=query, query_vector=vector, document_id=str(doc_a))
    assert {c.document_id for c in only_a} <= {str(doc_a)}

    nothing = await searcher.search(
        query_text=query, query_vector=vector, document_id=str(uuid.uuid4())
    )
    assert nothing == []


async def test_dense_arm_uses_hnsw_index(db: Database) -> None:
    await _seed(db)
    embedder = HashingEmbedder(DIMS)
    vector = await embedder.embed_query("deviation reports")

    async with db.engine.connect() as conn:
        await conn.execute(text("SET enable_seqscan = off"))
        plan = await conn.execute(
            text(
                "EXPLAIN SELECT id FROM document_chunks "
                "ORDER BY embedding <=> CAST(CAST(:v AS text) AS vector) LIMIT 5"
            ),
            {"v": vector_literal(vector)},
        )
        plan_text = "\n".join(row[0] for row in plan.all())
    assert "idx_chunks_hnsw" in plan_text, plan_text
