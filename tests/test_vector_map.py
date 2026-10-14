"""The vector map: PCA on data with known answers, the map of one question, and the cache."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pytest

from app.config import Settings
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.vector_map import (
    ChunkVector,
    VectorMapper,
    fit_projection,
    map_question,
)
from app.schemas import CutChunk, RerankedChunk, RetrievedChunk

RANKS = ("dense_rank", "sparse_rank", "fused_rank", "rerank_position")


def _row(
    chunk_id: str, vector: Sequence[float], document: str = "D1", index: int = 0
) -> ChunkVector:
    return ChunkVector(
        id=chunk_id,
        document_id=document,
        document_title=f"Document {document}",
        chunk_index=index,
        page_number=1,
        preview=f"text of {chunk_id}",
        vector=list(vector),
    )


def _scaled(rows: Sequence[ChunkVector], factors: Sequence[float]) -> list[ChunkVector]:
    return [
        _row(r.id, [factor * v for v in r.vector], r.document_id, r.chunk_index)
        for r, factor in zip(rows, factors, strict=True)
    ]


#: Four length-1 embeddings shaped like a cross: +-0.6 along x, +-0.8 along y, and the z part
#: that keeps each at length 1 (mean z = 0.7). Centred, they spread 2 x 0.8^2 = 1.28 along y,
#: 2 x 0.6^2 = 0.72 along x and 4 x 0.1^2 = 0.04 along z: 62.7 %, 35.3 % and 2 % of 2.04.
CROSS = [
    _row("a", [0.6, 0.0, 0.8]),
    _row("b", [-0.6, 0.0, 0.8]),
    _row("c", [0.0, 0.8, 0.6]),
    _row("d", [0.0, -0.8, 0.6]),
]

#: Six chunks of two documents with arbitrary, unrounded 4-dimensional embeddings.
CORPUS = [
    _row("c1", [0.91234567, 0.1, -0.2, 0.05], "D1", 0),
    _row("c2", [0.7, 0.3123456, 0.1, -0.4], "D1", 1),
    _row("c3", [0.2, 0.9, 0.3333333, 0.1], "D1", 2),
    _row("c4", [-0.5, 0.6, 0.2, 0.55555], "D2", 0),
    _row("c5", [-0.8, -0.1, 0.4, 0.3], "D2", 1),
    _row("c6", [0.1, -0.9, -0.3, 0.2], "D2", 2),
]
QUESTION = [0.81234, 0.2, 0.1, -0.05]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _candidate(chunk_id: str, dense: int | None, sparse: int | None) -> RetrievedChunk:
    row = next(r for r in CORPUS if r.id == chunk_id)
    return RetrievedChunk(
        id=chunk_id,
        document_id=row.document_id,
        document_title=row.document_title,
        chunk_index=row.chunk_index,
        page_number=1,
        raw_content="text",
        contextualized_content="text",
        rrf_score=sum(1 / (60 + r) for r in (dense, sparse) if r is not None),
        dense_rank=dense,
        sparse_rank=sparse,
    )


def _reranked(candidate: RetrievedChunk, position: int) -> RerankedChunk:
    return RerankedChunk(**candidate.model_dump(), relevance_score=1.0, rerank_position=position)


# Fused (RRF) order: c3, c1, c5, c4, c2; c6 is found by neither arm. Re-ranking keeps c5, c3.
CANDIDATES = [
    _candidate("c3", 2, 1),
    _candidate("c1", 1, None),
    _candidate("c5", None, 2),
    _candidate("c4", 3, 3),
    _candidate("c2", 4, None),
]
RERANKED = [_reranked(CANDIDATES[2], 1), _reranked(CANDIDATES[0], 2)]


# ---------------------------------------------------------------------------------- PCA


def test_pca_finds_the_two_directions_of_largest_spread() -> None:
    # Lengths are ignored (the rows are scaled to length 1 first), and the mean is removed.
    projection = fit_projection(_scaled(CROSS, [2.0, 0.5, 3.0, 10.0]))
    assert projection is not None
    assert projection.explained_variance == pytest.approx((1.28 / 2.04, 0.72 / 2.04))
    assert np.allclose(projection.axes, [[0, 1, 0], [1, 0, 0]])
    assert np.allclose(projection.coords, [[0, 0.6], [0, -0.6], [0.8, 0], [-0.8, 0]])
    assert projection.dimensions == 3


def test_the_question_is_placed_with_the_corpus_axes_and_loses_the_dropped_dimension() -> None:
    projection = fit_projection(CROSS)
    assert projection is not None
    # The question is scaled to length 1 too: ten times chunk c is still exactly on c.
    assert projection.project([0.0, 8.0, 6.0]) == pytest.approx((0.8, 0.0))
    # Two questions that differ only along z, the direction the map drops, land on the same
    # spot, although their cosine similarity to each other is only -0.28.
    assert projection.project([0.36, 0.48, 0.8]) == pytest.approx((0.48, 0.36))
    assert projection.project([0.36, 0.48, -0.8]) == pytest.approx((0.48, 0.36))
    with pytest.raises(ValueError):
        projection.project([1.0, 2.0])


def test_the_map_follows_cosine_so_a_vector_length_never_moves_a_dot() -> None:
    rows = [
        _row("c0", [1.0, 0.0, 0.0]),
        _row("c1", [3.0, 0.0, 0.0]),
        _row("c2", [0.0, 1.0, 0.0]),
        _row("c3", [0.0, 0.0, 1.0]),
    ]
    projection = fit_projection(rows)
    assert projection is not None
    # c0 and c1 point the same way (cosine 1): one dot, and a question along them lands on it.
    assert projection.coords[0] == pytest.approx(projection.coords[1])
    assert projection.project([2.0, 0.0, 0.0]) == pytest.approx(tuple(projection.coords[0]))
    assert list(projection.similarities([2.0, 0.0, 0.0])) == pytest.approx([1, 1, 0, 0])
    # Scaling any chunk or the question by a positive factor changes nothing on the map.
    rescaled = fit_projection(_scaled(rows, [0.5, 7.0, 2.0, 0.1]))
    assert rescaled is not None
    assert np.allclose(rescaled.coords, projection.coords)
    assert rescaled.project([0.4, 0.0, 0.0]) == pytest.approx(projection.project([2.0, 0, 0]))
    # A question with no direction at all still gets a finite place, and similarity 0.
    assert all(math.isfinite(v) for v in projection.project([0.0, 0.0, 0.0]))
    assert list(projection.similarities([0.0, 0.0, 0.0])) == [0.0] * 4


def test_axis_signs_follow_the_largest_loading_so_the_map_never_mirrors() -> None:
    u = np.array([-0.6, 0.8, 0.0])  # largest |loading| (0.8) is already positive
    v = np.array([0.8, 0.6, 0.0])
    w = np.array([0.0, 0.0, 1.0])
    # Length-1 rows (u, v, w are orthonormal) that spread 1.28 along u, 0.72 along v.
    rows = [
        _row(name, (scale * axis + lift * w).tolist())
        for name, scale, axis, lift in (
            ("a", 0.8, u, 0.6),
            ("b", -0.8, u, 0.6),
            ("c", 0.6, v, 0.8),
            ("d", -0.6, v, 0.8),
        )
    ]
    mirrored = [_row(r.id, [-x for x in r.vector]) for r in rows]
    first, second = fit_projection(rows), fit_projection(mirrored)
    assert first is not None and second is not None
    for projection in (first, second):
        assert np.allclose(projection.axes, [u, v])
        for axis in projection.axes:
            assert axis[int(np.argmax(np.abs(axis)))] > 0
    assert np.allclose(second.coords, -first.coords)  # mirrored data, mirrored dots, same axes


def test_fitting_is_deterministic() -> None:
    first, second = fit_projection(CORPUS), fit_projection(CORPUS)
    assert first is not None and second is not None
    assert np.array_equal(first.axes, second.axes)
    assert np.array_equal(first.coords, second.coords)
    assert first.explained_variance == second.explained_variance
    assert 1 > first.explained_variance[0] >= first.explained_variance[1] > 0


def test_no_map_without_three_chunks_or_without_spread() -> None:
    assert fit_projection([]) is None
    assert fit_projection(CROSS[:2]) is None
    assert fit_projection([_row(f"r{i}", [0.3, 0.1, 0.2]) for i in range(5)]) is None
    # One direction at different lengths: cosine sees a single direction, and so does the map.
    assert fit_projection([_row(f"s{i}", [0.3 * i, 0.1 * i, 0.2 * i]) for i in (1, 2, 5)]) is None
    assert fit_projection([_row(f"z{i}", [0.0, 0.0, 0.0]) for i in range(3)]) is None


def test_no_map_when_the_chunks_spread_along_a_single_line() -> None:
    """Only two distinct directions (e.g. a small file ingested twice): every chunk sits on one
    line, so the second axis would be an arbitrary direction the corpus does not spread along."""
    e1, e2 = [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
    assert fit_projection([_row("a", e1), _row("b", e1), _row("c", e2)]) is None
    assert fit_projection([_row("a", e1), _row("b", e2), _row("c", e1), _row("d", e2)]) is None
    # Re-embedding noise far below any real difference is not a direction either.
    noisy = [1.0, 0.0, 1e-7, 0.0]
    assert fit_projection([_row("a", e1), _row("b", noisy), _row("c", e2)]) is None
    # A third distinct direction is a real second axis.
    third = _row("c", [0.0, 0.0, 1.0, 0.0])
    assert fit_projection([_row("a", e1), _row("b", e2), third]) is not None


def test_mismatched_vector_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        fit_projection([_row("a", [1.0, 2.0]), _row("b", [1.0, 2.0, 3.0]), _row("c", [0.0, 1.0])])


# ------------------------------------------------------------------------ one question


def _map(document_id: str | None = None):  # type: ignore[no-untyped-def]
    projection = fit_projection(CORPUS)
    assert projection is not None
    return projection, map_question(
        projection,
        query_vector=QUESTION,
        candidates=CANDIDATES,
        reranked=RERANKED,
        document_id=document_id,
    )


def test_every_chunk_is_a_point_in_corpus_order_with_its_cosine_similarity() -> None:
    projection, vmap = _map()
    assert [p.chunk_id for p in vmap.points] == [r.id for r in CORPUS]
    assert vmap.corpus_chunks == 6 and vmap.dimensions == 4 and vmap.method == "pca"
    for point, row in zip(vmap.points, CORPUS, strict=True):
        assert point.similarity == pytest.approx(_cosine(QUESTION, row.vector), abs=5e-5)
        assert (point.x, point.y) == pytest.approx(projection.project(row.vector), abs=5e-5)
        assert (point.document_id, point.chunk_index, point.preview) == (
            row.document_id,
            row.chunk_index,
            row.preview,
        )
    assert (vmap.query.x, vmap.query.y) == pytest.approx(projection.project(QUESTION), abs=5e-5)


def test_points_carry_each_stage_rank_from_the_candidates() -> None:
    _, vmap = _map()
    ranks = {p.chunk_id: tuple(getattr(p, key) for key in RANKS) for p in vmap.points}
    assert ranks == {
        "c1": (1, None, 2, None),
        "c2": (4, None, 5, None),
        "c3": (2, 1, 1, 2),
        "c4": (3, 3, 4, None),
        "c5": (None, 2, 3, 1),
        "c6": (None, None, None, None),
    }
    assert all(p.searched for p in vmap.points)


def test_chunks_cut_after_fusion_keep_their_arm_ranks_but_get_no_fused_rank() -> None:
    """Each arm returns 3 and fusion keeps 3. Fused (RRF) order: c1 (vector #1), c4 (keyword #1,
    a tie with c1 that the vector rank breaks), c2, then the cut: c5, c3 (vector #3), c6."""
    projection = fit_projection(CORPUS)
    assert projection is not None
    candidates = [_candidate("c1", 1, None), _candidate("c4", None, 1), _candidate("c2", 2, None)]
    cut = [
        CutChunk(id="c5", rrf_score=1 / 62, sparse_rank=2),
        CutChunk(id="c3", rrf_score=1 / 63, dense_rank=3),
        CutChunk(id="c6", rrf_score=1 / 63, sparse_rank=3),
    ]
    fused = reciprocal_rank_fusion([["c1", "c2", "c3"], ["c4", "c5", "c6"]])
    assert [doc for doc, _ in fused] == [c.id for c in candidates] + [c.id for c in cut]
    reranked = [_reranked(candidates[1], 1)]
    vmap = map_question(
        projection, query_vector=QUESTION, candidates=candidates, reranked=reranked, cut=cut
    )
    ranks = {p.chunk_id: tuple(getattr(p, key) for key in RANKS) for p in vmap.points}
    assert ranks == {
        "c1": (1, None, 1, None),
        "c2": (2, None, 3, None),
        "c3": (3, None, None, None),  # the vector arm did return it: fusion cut it
        "c4": (None, 1, 2, 1),
        "c5": (None, 2, None, None),
        "c6": (None, 3, None, None),
    }
    # The spotlight holds the vector arm's own top 3, including the one fusion cut.
    assert list(vmap.vectors) == ["c1", "c2", "c3", "c4"]


def test_spotlight_vectors_are_the_vector_top_three_and_the_reranked_passages() -> None:
    _, vmap = _map()
    assert list(vmap.vectors) == ["c1", "c3", "c4", "c5"]
    for chunk_id, vector in vmap.vectors.items():
        row = next(r for r in CORPUS if r.id == chunk_id)
        assert vector == [round(v, 4) for v in row.vector]


def test_a_document_filter_marks_the_other_documents_as_not_searched() -> None:
    _, vmap = _map(document_id="d2")  # ids compare case-insensitively, as UUIDs do
    assert {p.chunk_id for p in vmap.points if p.searched} == {"c4", "c5", "c6"}


def test_every_number_is_rounded_to_four_decimals() -> None:
    _, vmap = _map()
    numbers = [*vmap.explained_variance, vmap.query.x, vmap.query.y, *vmap.query.vector]
    numbers += [n for p in vmap.points for n in (p.x, p.y, p.similarity)]
    numbers += [n for vector in vmap.vectors.values() for n in vector]
    assert all(round(n, 4) == n for n in numbers)
    assert vmap.query.vector == [0.8123, 0.2, 0.1, -0.05]
    assert vmap.vectors["c1"][0] == 0.9123
    assert not any(math.copysign(1.0, n) < 0 and n == 0 for n in numbers), "no -0.0 in the JSON"


# ------------------------------------------------------------------------------- cache


class FakeSource:
    def __init__(self, rows: Sequence[ChunkVector]) -> None:
        self.rows = list(rows)
        self.reads: list[int] = []
        self.during_read = None  # called inside the read, e.g. to simulate a concurrent ingest

    async def chunk_vectors(self, *, limit: int) -> list[ChunkVector]:
        self.reads.append(limit)
        if self.during_read:
            self.during_read()
        return self.rows[:limit]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _build(mapper: VectorMapper, candidates: Sequence[RetrievedChunk] = ()):  # type: ignore[no-untyped-def]
    return await mapper.build(query_vector=QUESTION, candidates=candidates, reranked=[])


async def test_the_projection_is_reused_until_the_ttl_expires(settings: Settings) -> None:
    source, clock = FakeSource(CORPUS), FakeClock()
    mapper = VectorMapper(source, settings, clock=clock)
    assert (await _build(mapper)) is not None
    clock.now = settings.vector_map_ttl_s - 1
    await _build(mapper)
    assert len(source.reads) == 1
    clock.now = settings.vector_map_ttl_s
    await _build(mapper)
    assert source.reads == [settings.vector_map_max_chunks + 1] * 2


async def test_invalidate_forces_a_fresh_read(settings: Settings) -> None:
    source = FakeSource(CORPUS[:3])
    mapper = VectorMapper(source, settings, clock=FakeClock())
    assert len((await _build(mapper)).points) == 3  # type: ignore[union-attr]
    source.rows = list(CORPUS)
    assert len((await _build(mapper)).points) == 3  # type: ignore[union-attr]
    mapper.invalidate()
    assert len((await _build(mapper)).points) == 6  # type: ignore[union-attr]
    assert len(source.reads) == 2


async def test_a_candidate_missing_from_the_projection_triggers_one_refit(
    settings: Settings,
) -> None:
    source = FakeSource(CORPUS[:5])
    mapper = VectorMapper(source, settings, clock=FakeClock())
    await _build(mapper)
    source.rows = list(CORPUS)  # another process ingested c6 since the fit
    newcomer = [_candidate("c6", 1, None)]
    vmap = await _build(mapper, newcomer)
    assert vmap is not None and len(vmap.points) == 6 and len(source.reads) == 2
    assert vmap.points[-1].dense_rank == 1
    await _build(mapper, newcomer)
    assert len(source.reads) == 2  # known now: no further read
    # A candidate that is in no read at all: one refit, then no map rather than a wrong one.
    ghost = [_candidate("c1", 1, None).model_copy(update={"id": "ghost"})]
    assert await _build(mapper, ghost) is None
    assert len(source.reads) == 3


async def test_a_cut_chunk_missing_from_the_projection_also_triggers_a_refit(
    settings: Settings,
) -> None:
    source = FakeSource(CORPUS[:5])
    mapper = VectorMapper(source, settings, clock=FakeClock())
    await _build(mapper)
    source.rows = list(CORPUS)  # c6 arrived since the fit; the vector arm returned it
    cut = [CutChunk(id="c6", rrf_score=1 / 70, dense_rank=10)]
    vmap = await mapper.build(query_vector=QUESTION, candidates=[], reranked=[], cut=cut)
    assert vmap is not None and len(source.reads) == 2
    assert (vmap.points[-1].dense_rank, vmap.points[-1].fused_rank) == (10, None)


async def test_a_corpus_above_the_limit_gets_no_map_and_the_verdict_is_cached(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    settings.vector_map_max_chunks = 5
    source, clock = FakeSource(CORPUS), FakeClock()
    mapper = VectorMapper(source, settings, clock=clock)
    with caplog.at_level(logging.INFO, logger="app.retrieval.vector_map"):
        assert await _build(mapper) is None
        assert await _build(mapper, CANDIDATES) is None  # no refit chase while too large
        assert source.reads == [6]
        clock.now = settings.vector_map_ttl_s
        assert await _build(mapper) is None
        assert len(source.reads) == 2
    assert caplog.text.count("VECTOR_MAP_MAX_CHUNKS=5") == 1  # said once, not per question
    source.rows = CORPUS[:5]
    mapper.invalidate()
    assert await _build(mapper) is not None


async def test_a_fit_read_while_a_document_was_stored_is_used_once_but_not_kept(
    settings: Settings,
) -> None:
    source = FakeSource(CORPUS)
    mapper = VectorMapper(source, settings, clock=FakeClock())
    source.during_read = mapper.invalidate  # an ingestion finishes during the read
    assert await _build(mapper) is not None
    source.during_read = None
    await _build(mapper)
    assert len(source.reads) == 2
    await _build(mapper)
    assert len(source.reads) == 2


async def test_an_empty_corpus_has_no_map(settings: Settings) -> None:
    mapper = VectorMapper(FakeSource([]), settings, clock=FakeClock())
    assert await _build(mapper) is None
