"""Vector map: where the question landed among the stored chunks, flattened to two dimensions.

Why: the vector arm ranks chunks by cosine similarity in ``EMBEDDING_DIMENSIONS`` dimensions, which
nobody can picture. A principal component analysis (PCA) of the corpus embeddings finds the two
directions in which the stored chunks differ most; projecting every chunk and the question onto
them gives a map on which the question lands among the chunks. Chunks that share a subject tend
to land together, so in the sample corpus each document forms its own region (a property of
these documents, not a rule: two documents on one subject would overlap). Next
to each dot the map carries the chunk's exact similarity and its rank in every stage, which is
what shows that the nearest vector is not always the passage that answers.

The PCA is fitted on the embeddings scaled to length 1, and the question is scaled the same way:
cosine similarity ignores length, so the map does too, and two vectors pointing the same way land
on the same dot whatever their lengths (most embedding models already return length-1 vectors;
this keeps the map honest for those that do not).

The map is a teaching aid, not part of retrieval, and it is lossy: two directions keep only part
of the spread (``explained_variance``), so dots that look close can be far apart in the dropped
dimensions. ``similarity`` is always computed over all dimensions.

The projection depends only on the corpus, so it is fitted once and cached; per question only the
question is projected and the similarities computed. PCA axes have no inherent sign, so each axis
is turned to make its largest loading positive (scikit-learn's ``svd_flip`` convention): the map
does not mirror itself between two fits of the same corpus.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import Settings
from app.schemas import CutChunk, MapPoint, MapQuery, RerankedChunk, RetrievedChunk, VectorMap

logger = logging.getLogger(__name__)

#: Characters of chunk text shown per point (the length of a retrieved passage's preview).
PREVIEW_CHARS = 180

#: Spotlight vectors per question: the vector arm's top N, plus every re-ranked passage.
SPOTLIGHT_DENSE_TOP = 3


@dataclass(frozen=True, slots=True)
class ChunkVector:
    """One stored chunk as the map needs it: its embedding and what a learner reads next to it."""

    id: str
    document_id: str
    document_title: str
    chunk_index: int
    #: The page the chunk text begins on (``page_for_offset(page_map, 0, page_number)``).
    page_number: int
    preview: str
    vector: Sequence[float]


class ChunkVectorSource(Protocol):
    async def chunk_vectors(self, *, limit: int) -> list[ChunkVector]:
        """Every chunk in stable corpus order (documents by created_at then title, then
        chunk_index), at most ``limit`` rows."""
        ...


def preview_of(text: str) -> str:
    """The first ``PREVIEW_CHARS`` characters of a chunk's text, whitespace collapsed."""
    return " ".join(text.split())[:PREVIEW_CHARS]


def _round4(value: float) -> float:
    return round(float(value), 4) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def _unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` scaled to length 1 (a zero vector stays zero: it has no direction)."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0.0 else vector


@dataclass(frozen=True, slots=True, eq=False)
class Projection:
    """A PCA fitted to the corpus's length-1 embeddings: two axes and every chunk's position."""

    #: Chunk metadata in corpus order; the embeddings themselves live in ``vectors``.
    chunks: tuple[ChunkVector, ...]
    index: dict[str, int]
    vectors: np.ndarray  # (n, d) as stored: the numbers the page multiplies out
    unit: np.ndarray  # (n, d) rows scaled to length 1: what the PCA and cosine use
    mean: np.ndarray  # (d,) mean of the unit rows
    axes: np.ndarray  # (2, d)
    coords: np.ndarray  # (n, 2)
    explained_variance: tuple[float, float]

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])

    def project(self, vector: Sequence[float]) -> tuple[float, float]:
        """Place a vector (the question) with the corpus's two axes; its length does not count."""
        x, y = (_unit(self._check(vector)) - self.mean) @ self.axes.T
        return float(x), float(y)

    def similarities(self, vector: Sequence[float]) -> np.ndarray:
        """Cosine similarity of every chunk to ``vector``: the score the vector arm ranks by."""
        return self.unit @ _unit(self._check(vector))

    def _check(self, vector: Sequence[float]) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (self.dimensions,):
            raise ValueError(f"expected a vector of {self.dimensions} numbers, got {array.shape}")
        return array


#: Below this share of the spread, the second direction is rounding noise, not a direction.
_MIN_SECOND_SPREAD = 1e-9


def fit_projection(rows: Sequence[ChunkVector]) -> Projection | None:
    """Fit the two-direction PCA of the corpus, or None when there is nothing to flatten.

    The rows are scaled to length 1 first, as cosine similarity does. None for fewer than 3
    chunks, or when the chunks spread along fewer than two directions: all pointing the same way,
    or only two distinct directions (two points always lie on a line, e.g. a small file ingested
    twice). The second axis would then be an arbitrary direction the corpus does not spread along.
    Pure and CPU-bound: callers use a worker thread.
    """
    if len(rows) < 3:
        return None
    matrix = np.asarray([row.vector for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("every chunk vector must have the same length, at least 2")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0.0, 1.0, norms)
    mean = unit.mean(axis=0)
    centred = unit - mean
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    spread = singular**2
    total = float(spread.sum())
    # Identical directions leave only rounding noise after centring: no spread at all. A second
    # singular value at noise level means every chunk sits on one line through the mean.
    if (
        not np.isfinite(total)
        or total <= 1e-12 * float(np.square(unit).sum())
        or spread[1] <= _MIN_SECOND_SPREAD * total
    ):
        return None
    axes = vt[:2].copy()
    for axis in axes:
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis *= -1.0
    explained = spread[:2] / total
    return Projection(
        chunks=tuple(dataclasses.replace(row, vector=()) for row in rows),
        index={row.id: i for i, row in enumerate(rows)},
        vectors=matrix,
        unit=unit,
        mean=mean,
        axes=axes,
        coords=centred @ axes.T,
        explained_variance=(float(explained[0]), float(explained[1])),
    )


def map_question(
    projection: Projection,
    *,
    query_vector: Sequence[float],
    candidates: Sequence[RetrievedChunk],
    reranked: Sequence[RerankedChunk],
    document_id: str | None = None,
    cut: Sequence[CutChunk] = (),
) -> VectorMap:
    """The map of one answer: every chunk with its position, similarity and stage ranks.

    ``candidates`` is the fused list in RRF order (its position is ``fused_rank``), ``cut`` the
    chunks an arm returned that fusion ranked below them (arm ranks, no ``fused_rank``), and
    ``reranked`` the passages sent to the model. Pure: no I/O, no shared state.
    """
    qx, qy = projection.project(query_vector)
    similarity = projection.similarities(query_vector)
    fused = {chunk.id: position for position, chunk in enumerate(candidates, start=1)}
    arm_ranks = {chunk.id: (chunk.dense_rank, chunk.sparse_rank) for chunk in (*cut, *candidates)}
    rerank_position = {chunk.id: chunk.rerank_position for chunk in reranked}
    wanted = document_id.lower() if document_id else None

    points = []
    for i, chunk in enumerate(projection.chunks):
        dense_rank, sparse_rank = arm_ranks.get(chunk.id, (None, None))
        points.append(
            MapPoint(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                preview=chunk.preview,
                x=_round4(projection.coords[i, 0]),
                y=_round4(projection.coords[i, 1]),
                similarity=_round4(similarity[i]),
                searched=wanted is None or chunk.document_id.lower() == wanted,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                fused_rank=fused.get(chunk.id),
                rerank_position=rerank_position.get(chunk.id),
            )
        )

    # The vector arm's own top 3, whether or not fusion kept them.
    dense_top = sorted(
        (rank, chunk_id) for chunk_id, (rank, _) in arm_ranks.items() if rank is not None
    )[:SPOTLIGHT_DENSE_TOP]
    spotlight = dict.fromkeys([chunk_id for _, chunk_id in dense_top] + [c.id for c in reranked])
    return VectorMap(
        dimensions=projection.dimensions,
        explained_variance=[_round4(v) for v in projection.explained_variance],
        corpus_chunks=len(projection.chunks),
        query=MapQuery(x=_round4(qx), y=_round4(qy), vector=[_round4(v) for v in query_vector]),
        points=points,
        vectors={
            chunk_id: [_round4(v) for v in projection.vectors[projection.index[chunk_id]]]
            for chunk_id in spotlight
            if chunk_id in projection.index
        },
    )


@dataclass(frozen=True, slots=True)
class _Fit:
    """One read of the corpus: the projection (None when there is no map) and when it was made."""

    projection: Projection | None
    ids: frozenset[str]
    too_large: bool
    at: float


class VectorMapper:
    """Builds the vector map of each answer from a cached projection of the corpus.

    The projection is refitted after ``VECTOR_MAP_TTL_S``, on the first question after
    :meth:`invalidate` (ingestion calls it, as it does for the cached corpus listing, so that
    question's ``map_ms`` includes the read and the SVD), and once per question when a chunk the
    search returned is missing from it: another process ingested since the last fit. A corpus
    larger than ``VECTOR_MAP_MAX_CHUNKS`` gets no map, and that verdict is cached for the TTL too.
    """

    def __init__(
        self,
        source: ChunkVectorSource,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._max_chunks = settings.vector_map_max_chunks
        self._ttl_s = settings.vector_map_ttl_s
        self._clock = clock
        self._fit: _Fit | None = None
        self._generation = 0  # bumped by invalidate(), so a fit read before a write is not kept
        self._said_too_large = False
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._fit = None
        self._generation += 1

    async def build(
        self,
        *,
        query_vector: Sequence[float],
        candidates: Sequence[RetrievedChunk],
        reranked: Sequence[RerankedChunk],
        document_id: str | None = None,
        cut: Sequence[CutChunk] = (),
    ) -> VectorMap | None:
        returned = {chunk.id for chunk in candidates} | {chunk.id for chunk in cut}
        fit = await self._current(returned)
        if fit.projection is None:
            return None
        if not returned <= fit.ids:
            # Still missing after a fresh read: a map without them would contradict the trace.
            logger.info("Vector map skipped: a returned chunk is not in the corpus read")
            return None
        return await asyncio.to_thread(
            map_question,
            fit.projection,
            query_vector=query_vector,
            candidates=candidates,
            reranked=reranked,
            document_id=document_id,
            cut=cut,
        )

    async def _current(self, candidate_ids: set[str]) -> _Fit:
        async with self._lock:
            fit = self._fit
            if fit is None or self._clock() - fit.at >= self._ttl_s:
                return await self._refit()
            if not fit.too_large and not candidate_ids <= fit.ids:
                return await self._refit()
            return fit

    async def _refit(self) -> _Fit:
        generation = self._generation
        rows = await self._source.chunk_vectors(limit=self._max_chunks + 1)
        too_large = len(rows) > self._max_chunks
        if too_large:
            projection = None
            if not self._said_too_large:
                logger.info(
                    "Vector map off: the corpus holds more than VECTOR_MAP_MAX_CHUNKS=%d chunks",
                    self._max_chunks,
                )
                self._said_too_large = True
        else:
            projection = await asyncio.to_thread(fit_projection, rows)
            self._said_too_large = False
        fit = _Fit(
            projection=projection,
            ids=frozenset(row.id for row in rows),
            too_large=too_large,
            at=self._clock(),
        )
        # An ingestion that finished while the rows were being read may be missing from them:
        # use this fit for the question that asked, but do not keep it.
        self._fit = fit if generation == self._generation else None
        return fit
