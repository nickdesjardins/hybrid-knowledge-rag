"""Protections added after the security review: body limits before parsing, the upload key
checked before the upload is read, trusted client addresses, liveness without the database,
the corpus-listing cache, database failures as 503s, the shared local-inference gate, and the
text folding that keeps quantities intact."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError
from sqlalchemy.exc import OperationalError
from starlette.requests import Request

from app.budget import Allowance, BudgetExceeded, QueryBudget
from app.config import Settings
from app.errors import ConfigurationError
from app.ingestion.chunker import (
    TextUnit,
    _tail_with_pages,
    count_tokens,
    pack_units,
    page_for_offset,
)
from app.ingestion.contextualizer import HeuristicContextualizer
from app.ingestion.pdf_parser import PageText, ParsedDocument, clean_block_text
from app.main import (
    CachedCatalogue,
    Components,
    RequestGuardMiddleware,
    _init_database,
    _normalise_ip,
    client_id_for,
    create_app,
)
from app.retrieval.embedder import FastEmbedEmbedder
from app.retrieval.hybrid_search import HybridResult
from app.retrieval.reranker import FastEmbedReranker
from app.schemas import DocumentSummary, QueryRequest, RetrievedChunk
from app.textnorm import fold_compat, has_control_characters

from .conftest import InMemoryStore


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://guard")


QUESTION = {"query": "How long must batch records be retained?"}


# ------------------------------------------------------------------------- liveness


async def test_livez_never_touches_the_database(
    settings: Settings, components: Components, store: InMemoryStore
) -> None:
    store.healthy = False
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        live = await http.get("/livez")
        assert live.status_code == 200 and live.json()["status"] == "alive"
        assert (await http.head("/livez")).status_code == 200
        assert (await http.get("/health")).status_code == 503  # the deep check still reports


async def test_playground_files_are_revalidated_after_a_deploy(
    settings: Settings, components: Components
) -> None:
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        for path in ("/", "/playground/assets/app.js", "/playground/assets/explain.js"):
            response = await http.get(path)
            assert response.status_code == 200, path
            assert response.headers["cache-control"] == "no-cache", path
        etag = (await http.get("/playground/assets/explain.js")).headers["etag"]
        again = await http.get("/playground/assets/explain.js", headers={"If-None-Match": etag})
        assert again.status_code == 304  # revalidation is cheap when nothing changed


# ----------------------------------------------------------------- body limits, early


async def test_json_bodies_are_limited_by_content_length(
    settings: Settings, components: Components
) -> None:
    settings.max_json_body_bytes = 1024
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        big = await http.post(
            "/api/v1/query",
            content=b'{"query": "' + b"a" * 4000 + b'"}',
            headers={"Content-Type": "application/json"},
        )
        assert big.status_code == 413
        assert "MAX_JSON_BODY_BYTES" in big.json()["detail"]
        assert (await http.post("/api/v1/query", json=QUESTION)).status_code == 200


async def test_streamed_bodies_without_content_length_are_cut_off(
    settings: Settings, components: Components
) -> None:
    settings.max_json_body_bytes = 1024
    app = create_app(settings=settings, components=components)

    async def chunks():
        yield b'{"query": "'
        for _ in range(10):
            yield b"a" * 512
        yield b'"}'

    async with _client(app) as http:
        response = await http.post(
            "/api/v1/query", content=chunks(), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413


def _upload_scope(headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/ingest",
        "headers": headers,
        "client": ("203.0.113.9", 5000),
    }


async def _run_guard(settings: Settings, headers: list[tuple[bytes, bytes]]) -> list[dict]:
    """Call the guard with a ``receive`` that fails the test if the body is ever read."""
    sent: list[dict] = []
    reached = {"app": False}

    async def downstream(scope, receive, send) -> None:
        reached["app"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> dict:
        raise AssertionError("the upload body must not be read")

    async def send(message: dict) -> None:
        sent.append(message)

    guard = RequestGuardMiddleware(downstream, settings=settings)
    await guard(_upload_scope(headers), receive, send)
    sent.append({"reached_app": reached["app"]})
    return sent


async def test_uploads_are_refused_before_a_byte_of_the_body_is_read(settings: Settings) -> None:
    huge = [(b"content-length", b"999999999999"), (b"content-type", b"multipart/form-data")]

    settings.demo_mode = True  # demo without API_KEY: uploads closed
    sent = await _run_guard(settings, huge)
    assert sent[0]["status"] == 403 and sent[-1]["reached_app"] is False

    settings.api_key = SecretStr("operator")  # wrong or missing key: 401 before the size check
    sent = await _run_guard(settings, huge)
    assert sent[0]["status"] == 401 and (b"www-authenticate", b"ApiKey") in sent[0]["headers"]
    sent = await _run_guard(settings, [*huge, (b"x-api-key", b"nope")])
    assert sent[0]["status"] == 401

    sent = await _run_guard(settings, [*huge, (b"x-api-key", b"operator")])
    assert sent[0]["status"] == 413  # the right key, but far over MAX_UPLOAD_BYTES


async def test_validation_errors_never_echo_the_payload(
    settings: Settings, components: Components
) -> None:
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        response = await http.post("/api/v1/query", json={"query": "MARKER" * 400})
        assert response.status_code == 422
        assert "MARKER" not in response.text
        assert all("input" not in error for error in response.json()["detail"])


# --------------------------------------------------------------- question validation


def test_query_rejects_control_characters_and_folds_like_documents() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="What is\x00 the limit?")
    with pytest.raises(ValidationError):
        QueryRequest(query="bell\x07 character here")
    folded = QueryRequest(query="  What is the ﬁre limit, 10³ cm² or 1½?\n ")
    assert folded.query == "What is the fire limit, 10³ cm² or 1½?"
    assert QueryRequest(query="line one\nline two").query == "line one\nline two"


async def test_demo_mode_caps_question_length_without_spending_budget(
    settings: Settings, components: Components
) -> None:
    settings.demo_mode = True
    settings.demo_seed_on_startup = False
    settings.demo_warm_models = False
    components.budget = QueryBudget(settings, allowance_reader=None)
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        long_question = await http.post("/api/v1/query", json={"query": "why " * 150})
        assert long_question.status_code == 422
        assert "500 characters" in long_question.json()["detail"]
        assert components.budget.snapshot()["used_today"] == 0


async def test_demo_mode_sheds_load_beyond_the_inflight_limit(
    settings: Settings, components: Components
) -> None:
    settings.demo_mode = True
    settings.demo_seed_on_startup = False
    settings.demo_warm_models = False
    settings.query_max_inflight = 2
    components.budget = QueryBudget(settings, allowance_reader=None)
    components.inflight = 2  # two questions are already being answered
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        busy = await http.post("/api/v1/query", json=QUESTION)
        assert busy.status_code == 503 and busy.headers["retry-after"] == "5"
        assert components.budget.snapshot()["used_today"] == 0
        components.inflight = 0
        assert (await http.post("/api/v1/query", json=QUESTION)).status_code == 200
        assert components.inflight == 0  # released after the answer


# ------------------------------------------------------------------------- budget


class _Clock:
    def __init__(self) -> None:
        self.now, self.wall = 1_000.0, 1_700_000_000.0

    def mono(self) -> float:
        return self.now

    def time(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.wall += seconds


async def test_budget_caps_each_address_per_day() -> None:
    settings = Settings(
        _env_file=None,
        _env_prefix="HERMETIC_TEST_",
        query_rate_limit_per_minute=10,
        query_daily_limit_per_client=2,
    )
    clock = _Clock()
    budget = QueryBudget(settings, allowance_reader=None, clock=clock.mono, wall=clock.time)
    await budget.admit("a")
    clock.advance(61)
    await budget.admit("a")
    clock.advance(61)
    with pytest.raises(BudgetExceeded) as info:
        await budget.admit("a")
    assert "for today" in info.value.message
    await budget.admit("b")  # other visitors are unaffected
    clock.advance(86_400)
    await budget.admit("a")  # a new UTC day
    assert budget.snapshot()["per_client_per_day"] == 2


async def test_allowance_reserve_never_exceeds_a_fifth_of_the_day() -> None:
    settings = Settings(
        _env_file=None, _env_prefix="HERMETIC_TEST_", openrouter_allowance_reserve=50
    )
    clock = _Clock()
    remaining = {"value": 30}

    async def reader() -> Allowance | None:
        # An account without credits gets 50 free requests a day: a reserve of 50 would
        # close the demo all day; a fifth of the day (10) is kept instead.
        return Allowance(used=50 - remaining["value"], limit=50, remaining=remaining["value"])

    budget = QueryBudget(settings, allowance_reader=reader, clock=clock.mono, wall=clock.time)
    await budget.admit("a")
    clock.advance(61)
    remaining["value"] = 10
    with pytest.raises(BudgetExceeded):
        await budget.admit("b")


# --------------------------------------------------------------- who is asking


def _request(headers: dict[str, str], client: str = "10.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/query",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": (client, 40000),
        }
    )


def test_client_address_ignores_proxy_headers_unless_trusted(settings: Settings) -> None:
    forged = {"X-Forwarded-For": "6.6.6.6", "CF-Connecting-IP": "7.7.7.7"}
    assert client_id_for(_request(forged), settings) == "10.0.0.1"


def test_client_address_uses_the_rightmost_trusted_hop(settings: Settings) -> None:
    settings.trust_proxy_headers = True
    assert client_id_for(_request({"X-Forwarded-For": "6.6.6.6, 1.2.3.4"}), settings) == "1.2.3.4"
    settings.trusted_proxy_hops = 2
    assert (
        client_id_for(_request({"X-Forwarded-For": "6.6.6.6, 1.2.3.4, 10.1.1.1"}), settings)
        == "1.2.3.4"
    )
    # Fewer entries than hops, or garbage: fall back to the socket address.
    assert client_id_for(_request({"X-Forwarded-For": "1.2.3.4"}), settings) == "10.0.0.1"
    settings.trusted_proxy_hops = 1
    assert client_id_for(_request({"X-Forwarded-For": "not-an-ip"}), settings) == "10.0.0.1"


def test_client_address_prefers_the_edge_header(settings: Settings) -> None:
    settings.trust_proxy_headers = True
    settings.client_ip_header = "CF-Connecting-IP"
    headers = {"CF-Connecting-IP": "198.51.100.4", "X-Forwarded-For": "6.6.6.6, 1.2.3.4"}
    assert client_id_for(_request(headers), settings) == "198.51.100.4"
    headers["CF-Connecting-IP"] = "garbage"
    assert client_id_for(_request(headers), settings) == "1.2.3.4"


def test_ipv6_visitors_are_grouped_by_their_64_prefix() -> None:
    assert _normalise_ip("2001:db8:1:2:aaaa::1") == "2001:db8:1:2::/64"
    assert _normalise_ip("2001:db8:1:2:bbbb::9") == "2001:db8:1:2::/64"
    assert _normalise_ip("::ffff:203.0.113.7") == "203.0.113.7"
    assert _normalise_ip("[2001:db8::1]:443") == "2001:db8::/64"
    assert _normalise_ip("203.0.113.7:51234") == "203.0.113.7"
    assert _normalise_ip("nope") is None


# ---------------------------------------------------------------- corpus listing


class _CountingCatalogue:
    def __init__(self) -> None:
        self.calls = 0

    async def list_documents(self) -> list[DocumentSummary]:
        self.calls += 1
        return []


async def test_corpus_listing_is_cached_and_invalidated() -> None:
    inner = _CountingCatalogue()
    now = {"t": 0.0}
    cached = CachedCatalogue(inner, ttl_s=300, clock=lambda: now["t"])
    await cached.list_documents()
    await cached.list_documents()
    assert inner.calls == 1
    now["t"] = 301
    await cached.list_documents()
    assert inner.calls == 2
    cached.invalidate()
    await cached.list_documents()
    assert inner.calls == 3

    uncached = CachedCatalogue(inner, ttl_s=0)
    await uncached.list_documents()
    await uncached.list_documents()
    assert inner.calls == 5


async def test_ingest_invalidates_the_cached_listing(
    settings: Settings, components: Components, store: InMemoryStore, sample_pdf_bytes: bytes
) -> None:
    components.catalogue = CachedCatalogue(store, ttl_s=300)
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        assert (await http.get("/api/v1/documents")).json()["documents"] == []
        files = {"file": ("a.pdf", sample_pdf_bytes, "application/pdf")}
        assert (await http.post("/api/v1/ingest", files=files)).status_code == 200
        assert len((await http.get("/api/v1/documents")).json()["documents"]) == 1


async def test_seeding_refreshes_a_listing_cached_before_it(
    settings: Settings, components: Components, store: InMemoryStore, sample_pdf_bytes: bytes
) -> None:
    # Regression: seeding reads the listing (to skip present files) before it ingests, so the
    # cache held an empty corpus and the recording script reported "0 documents".
    from app.ingestion.pipeline import ingest_pdf

    components.catalogue = CachedCatalogue(store, ttl_s=300)
    assert await components.catalogue.list_documents() == []
    await ingest_pdf(components, sample_pdf_bytes, "a.pdf")
    assert len(await components.catalogue.list_documents()) == 1


async def test_listing_needs_the_key_outside_demo_mode(
    settings: Settings, components: Components
) -> None:
    settings.api_key = SecretStr("operator")
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        assert (await http.get("/api/v1/documents")).status_code == 401
        ok = await http.get("/api/v1/documents", headers={"X-API-Key": "operator"})
        assert ok.status_code == 200
        config = (await http.get("/playground/config")).json()
        assert config["documents"] == [] and config["sample_questions"] == []


# ------------------------------------------------------------------ database errors


class _BrokenSearcher:
    async def search_with_arms(self, **_: object) -> HybridResult:
        raise OperationalError("SELECT secret_column FROM t", {"p": "secret"}, Exception("down"))

    async def search(self, **_: object) -> list[RetrievedChunk]:
        return (await self.search_with_arms()).candidates


async def test_database_failures_become_503_without_sql(
    settings: Settings, components: Components
) -> None:
    components.searcher = _BrokenSearcher()

    class _BrokenCatalogue:
        async def list_documents(self) -> list[DocumentSummary]:
            raise OSError("connection refused")

    components.catalogue = _BrokenCatalogue()
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        query = await http.post("/api/v1/query", json=QUESTION)
        assert query.status_code == 503
        assert "database is unavailable" in query.json()["detail"]
        assert "secret" not in query.text and "SELECT" not in query.text
        listing = await http.get("/api/v1/documents")
        assert listing.status_code == 503 and listing.headers["retry-after"] == "10"


class _WakingDatabase:
    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    async def init_schema(self) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error

    async def pgvector_installed(self) -> bool:
        await self.init_schema()
        return True


async def test_start_up_retries_while_the_database_wakes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("app.main.asyncio.sleep", no_sleep)
    settings.auto_create_schema = True
    settings.db_startup_retries = 3

    waking = _WakingDatabase(failures=2, error=OSError("waking up"))
    await _init_database(waking, settings)  # type: ignore[arg-type]
    assert waking.calls == 3

    asleep = _WakingDatabase(failures=10, error=OSError("still asleep"))
    with pytest.raises(OSError):
        await _init_database(asleep, settings)  # type: ignore[arg-type]
    assert asleep.calls == 4

    misconfigured = _WakingDatabase(failures=10, error=ConfigurationError("dims drift"))
    with pytest.raises(ConfigurationError):
        await _init_database(misconfigured, settings)  # type: ignore[arg-type]
    assert misconfigured.calls == 1  # a wrong setting is not retried


# --------------------------------------------------------------- inference gate


class _Concurrency:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def work(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1


async def test_embedding_and_reranking_share_one_inference_gate(settings: Settings) -> None:
    settings.embedding_dimensions = 3
    meter = _Concurrency()

    class Model:
        def embed(self, texts):
            meter.work()
            return [[0.1, 0.2, 0.3] for _ in texts]

    class Encoder:
        def rerank(self, query, documents, batch_size):
            meter.work()
            return [1.0 for _ in documents]

    gate = asyncio.Semaphore(1)
    embedder = FastEmbedEmbedder(settings, model=Model(), gate=gate)
    reranker = FastEmbedReranker(settings, encoder=Encoder(), gate=gate)
    probe = RetrievedChunk(
        id="p",
        document_id="d",
        document_title="t",
        chunk_index=0,
        page_number=1,
        raw_content="x",
        contextualized_content="x",
        rrf_score=0.0,
    )
    await asyncio.gather(
        embedder.embed_query("a"),
        embedder.embed_documents(["b", "c"]),
        reranker.rerank("q", [probe], top_n=1),
        reranker.rerank("q", [probe], top_n=1),
    )
    assert meter.peak == 1


# ------------------------------------------------------------------- text folding


def test_folding_keeps_quantities_and_drops_ligatures() -> None:
    assert fold_compat("deﬁnes 10³ cm² H₂O 1½ ppm") == "defines 10³ cm² H₂O 1½ ppm"
    assert clean_block_text("Exposure limit: 10³ cfu/cm²,\nﬁltered") == (
        "Exposure limit: 10³ cfu/cm², filtered"
    )
    assert has_control_characters("a\x00b") and not has_control_characters("a\tb\nc\rd")


# ----------------------------------------------------------------- chunk pages


def test_overlap_carry_keeps_the_pages_it_came_from() -> None:
    units = [
        TextUnit(2, "alpha bravo charlie delta echo foxtrot golf hotel"),
        TextUnit(3, "india juliet kilo"),
        TextUnit(3, "lima mike november oscar"),
    ]
    first, second = pack_units(units, chunk_size=12, overlap=5)
    assert second.text.startswith("golf hotel india juliet kilo\n\nlima")
    # "golf hotel" sits on page 2, the rest of the carry on page 3.
    assert second.page_map == [(0, 2), (len("golf hotel "), 3)]
    assert page_for_offset(second.page_map, second.text.index("golf"), 0) == 2
    assert page_for_offset(second.page_map, second.text.index("india"), 0) == 3
    assert first.page_map == [(0, 2), (first.text.index("india"), 3)]

    tail, tail_map = _tail_with_pages(first.text, first.page_map, 5, count_tokens)
    assert tail == "golf hotel india juliet kilo" and tail_map == second.page_map


def test_units_deferred_by_an_overshoot_keep_document_order() -> None:
    def seam_penalty(text: str) -> int:
        # Charges nothing for one joiner (so packing budgets none) but 5 tokens once a chunk
        # holds three paragraphs: the flush-time re-count overshoots and defers a unit.
        return count_tokens(text) + (5 if text.count("\n\n") >= 2 else 0)

    units = [TextUnit(1 + i // 2, f"u{i} w w w w w") for i in range(6)]
    chunks = pack_units(units, chunk_size=20, overlap=0, counter=seam_penalty)
    labels = [word for chunk in chunks for word in chunk.text.split() if word.startswith("u")]
    assert labels == [f"u{i}" for i in range(6)]
    assert all(chunk.token_count <= 20 for chunk in chunks)


def test_heuristic_preamble_names_every_page_a_chunk_spans() -> None:
    document = ParsedDocument(
        title="SOP",
        filename="sop.pdf",
        pages=[PageText(n, "x") for n in range(1, 5)],
    )
    units = [TextUnit(2, "alpha bravo"), TextUnit(3, "charlie delta")]
    (chunk,) = pack_units(units, chunk_size=50, overlap=5)
    assert HeuristicContextualizer.context_for(document, chunk) == "Document: SOP. Pages 2-3 of 4."
