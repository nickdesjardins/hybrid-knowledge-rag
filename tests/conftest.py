"""Hermetic fixtures: an in-memory store/searcher, deterministic embeddings, and fake providers.

Nothing in this module touches the network or PostgreSQL. The in-memory searcher mirrors the SQL
pipeline: a dense ranking (cosine over feature-hashed vectors), a sparse ranking (lexical overlap),
and the same Reciprocal Rank Fusion as the ``rrf_fusion`` CTE.
"""

from __future__ import annotations

import math
import os
import re
import uuid
from collections.abc import AsyncIterator, Sequence

from app.config import Settings

# The suite must not depend on whatever the host exports (API keys, LOG_LEVEL, TOP_K_*, ...).
# Strip every Settings-related variable from the test process before any app module loads.
_SETTINGS_ENV_NAMES = {name.upper() for name in Settings.model_fields}
for _name in list(os.environ):
    if _name.upper() in _SETTINGS_ENV_NAMES:
        os.environ.pop(_name)
os.environ["ENVIRONMENT"] = "test"

import pymupdf  # noqa: E402
import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.database import ChunkRecord  # noqa: E402
from app.generator.grounded_llm import (  # noqa: E402
    GenerationResult,
    LLMCitation,
    LLMGroundedOutput,
)
from app.ingestion.contextualizer import HeuristicContextualizer  # noqa: E402
from app.ingestion.pdf_parser import ParsedDocument, parse_pdf_bytes  # noqa: E402
from app.main import Components, create_app  # noqa: E402
from app.retrieval.embedder import HashingEmbedder  # noqa: E402
from app.retrieval.hybrid_search import reciprocal_rank_fusion  # noqa: E402
from app.schemas import RerankedChunk, RetrievedChunk  # noqa: E402

EMBEDDING_DIMS = 256

SAMPLE_PAGES: list[list[str]] = [
    [
        "QUALITY MANUAL",
        "1. Purpose",
        "This Standard Operating Procedure SOP-QA-017 defines the cleaning validation requirements "
        "for all product-contact manufacturing equipment. Cleaning validation demonstrates that "
        "residues are reduced below the acceptance limit of 10 ppm before the next batch.",
    ],
    [
        "2. Responsibilities",
        "The Quality Assurance manager approves all deviation reports within 48 hours of "
        "submission. Production supervisors must document every cleaning cycle in the equipment "
        "logbook before releasing the line.",
    ],
    [
        "3. Records",
        "Batch records must be retained for a minimum of seven years after the product expiry "
        "date. Electronic records require an audit trail compliant with 21 CFR Part 11 and must be "
        "backed up weekly.",
    ],
]

SECOND_DOC_PAGES: list[list[str]] = [
    [
        "TRAVEL POLICY",
        "1. Scope",
        "Employees must submit travel expense claims within 30 days of returning. Claims submitted "
        "later than 30 days are rejected unless the finance director grants an exception.",
    ]
]

_WORD_RE = re.compile(r"\w+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are", "be", "must",
    "how", "what", "when", "who", "which", "does", "do", "long", "with", "by", "at", "it",
}  # fmt: skip


def make_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a small multi-page PDF; each paragraph becomes its own text box (layout block)."""
    document = pymupdf.open()
    for paragraphs in pages:
        page = document.new_page(width=595, height=842)
        y = 72
        for paragraph in paragraphs:
            rect = pymupdf.Rect(72, y, 523, y + 140)
            page.insert_textbox(rect, paragraph, fontsize=11, fontname="helv")
            y += 160
    data = document.tobytes()
    document.close()
    return data


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text) if len(t) > 2 and t.lower() not in _STOPWORDS}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryStore:
    """Fake database: document store + hybrid searcher + health probe, all in memory.

    Ids are derived deterministically (uuid5) so tie-breaks are stable between runs.
    """

    pgvector_version = "0.8.0"

    def __init__(self, rrf_k: int = 60, healthy: bool = True) -> None:
        self.rrf_k = rrf_k
        self.healthy = healthy
        self.documents: dict[str, dict[str, str]] = {}
        self.chunks: list[dict[str, object]] = []
        self.search_calls: list[str | None] = []

    async def ping(self) -> bool:
        return self.healthy

    async def pgvector_installed(self) -> bool:
        return self.healthy

    async def store_document(
        self, *, title: str, filename: str, chunks: Sequence[ChunkRecord]
    ) -> uuid.UUID:
        document_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{title}:{len(self.documents)}")
        self.documents[str(document_id)] = {"title": title, "filename": filename}
        for chunk in chunks:
            self.chunks.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk.chunk_index}")),
                    "document_id": str(document_id),
                    "document_title": title,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "raw_content": chunk.raw_content,
                    "contextualized_content": chunk.contextualized_content,
                    "embedding": list(chunk.embedding),
                }
            )
        return document_id

    async def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        document_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        self.search_calls.append(document_id)
        limit = limit or 50
        pool = [c for c in self.chunks if document_id is None or c["document_id"] == document_id]
        dense = sorted(
            pool,
            key=lambda c: (-_cosine(query_vector, c["embedding"]), c["id"]),  # type: ignore[arg-type]
        )[:limit]
        query_tokens = _tokens(query_text)
        sparse_scored = [
            (len(query_tokens & _tokens(str(c["contextualized_content"]))), c) for c in pool
        ]
        sparse = [
            c
            for score, c in sorted(
                (s for s in sparse_scored if s[0] > 0), key=lambda s: (-s[0], s[1]["id"])
            )
        ][:limit]
        dense_rank = {c["id"]: i for i, c in enumerate(dense, start=1)}
        sparse_rank = {c["id"]: i for i, c in enumerate(sparse, start=1)}
        fused = reciprocal_rank_fusion(
            [[str(c["id"]) for c in dense], [str(c["id"]) for c in sparse]],
            k=self.rrf_k,
            limit=limit,
        )
        by_id = {c["id"]: c for c in pool}
        return [
            RetrievedChunk(
                id=str(cid),
                document_id=str(by_id[cid]["document_id"]),
                document_title=str(by_id[cid]["document_title"]),
                chunk_index=int(by_id[cid]["chunk_index"]),  # type: ignore[call-overload]
                page_number=int(by_id[cid]["page_number"]),  # type: ignore[call-overload]
                raw_content=str(by_id[cid]["raw_content"]),
                contextualized_content=str(by_id[cid]["contextualized_content"]),
                rrf_score=score,
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
            )
            for cid, score in fused
        ]


class FakeReranker:
    """Deterministic lexical cross-encoder stand-in: overlap ratio between query and passage."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RerankedChunk]:
        self.calls.append((query, len(candidates), top_n))
        query_tokens = _tokens(query) or {query.lower()}
        scored = []
        for chunk in candidates:
            overlap = len(query_tokens & _tokens(chunk.raw_content)) / len(query_tokens)
            scored.append((overlap, chunk))
        scored.sort(key=lambda item: (-item[0], -item[1].rrf_score, item[1].id))
        return [
            RerankedChunk(**chunk.model_dump(), relevance_score=score, rerank_position=position)
            for position, (score, chunk) in enumerate(scored[:top_n], start=1)
        ]


class FakeGenerator:
    """Fake LLM. ``mode`` selects grounded / hallucinate / insufficient / refused / error."""

    name = "fake"
    model = "fake-grounded-model"

    def __init__(self, mode: str = "grounded") -> None:
        self.mode = mode
        self.calls: list[tuple[str, list[str]]] = []

    async def generate(self, query: str, chunks: Sequence[RerankedChunk]) -> GenerationResult:
        from app.errors import ProviderError

        self.calls.append((query, [c.id for c in chunks]))
        if self.mode == "error":
            raise ProviderError("fake", "simulated outage", retryable=True)
        if self.mode == "fatal":
            raise ProviderError("fake", "bad request", retryable=False)
        if self.mode == "refused":
            return GenerationResult(
                output=LLMGroundedOutput(
                    answer="Declined.", is_grounded=False, confidence_score=0.0, citations=[]
                ),
                model=self.model,
                refused=True,
            )
        if self.mode == "insufficient" or not chunks:
            return GenerationResult(
                output=LLMGroundedOutput(
                    answer="The passages do not state this.",
                    is_grounded=False,
                    confidence_score=0.0,
                    citations=[],
                ),
                model=self.model,
            )
        top = chunks[0]
        sentences = re.split(r"(?<=[.!?])\s+", top.raw_content.strip())
        query_tokens = _tokens(query)
        # Like a real model, quote the sentence most relevant to the question (overlap, then length).
        quote = max(sentences, key=lambda s: (len(query_tokens & _tokens(s)), len(s)))
        if self.mode == "hallucinate":
            quote = "This sentence does not appear anywhere in the ingested document."
        return GenerationResult(
            output=LLMGroundedOutput(
                answer=f"According to {top.document_title} (page {top.page_number}): {quote}",
                is_grounded=True,
                confidence_score=0.93,
                citations=[LLMCitation(chunk_id=top.id, verbatim_quote=quote)],
            ),
            model=self.model,
        )


# ------------------------------------------------------------------------------ fixtures


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        _env_prefix="HERMETIC_TEST_",  # ignore ambient variables entirely
        environment="test",
        embedding_provider="hashing",
        embedding_dimensions=EMBEDDING_DIMS,
        rerank_provider="none",
        contextualizer_enabled=False,
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
        top_k_retrieval=50,
        top_k_rerank=3,
        auto_create_schema=False,
        log_level="WARNING",
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def components(
    settings: Settings, store: InMemoryStore, generator: FakeGenerator, reranker: FakeReranker
) -> Components:
    return Components(
        settings=settings,
        db=store,
        store=store,
        searcher=store,
        embedder=HashingEmbedder(EMBEDDING_DIMS),
        reranker=reranker,
        contextualizer=HeuristicContextualizer(),
        generator=generator,
    )


@pytest.fixture
def app(settings: Settings, components: Components):
    return create_app(settings=settings, components=components)


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    return make_pdf(SAMPLE_PAGES)


@pytest.fixture(scope="session")
def second_pdf_bytes() -> bytes:
    return make_pdf(SECOND_DOC_PAGES)


@pytest.fixture
def parsed_document(sample_pdf_bytes: bytes) -> ParsedDocument:
    return parse_pdf_bytes(sample_pdf_bytes, "sop-qa-017.pdf")
