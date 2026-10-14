"""FastAPI application: ``GET /health``, ``POST /api/v1/ingest``, ``POST /api/v1/query``, plus the
read-only corpus listing, the browser playground, and the demo-mode protections.

All collaborators (database, embedder, re-ranker, contextualizer, generator, budget) are assembled
into a ``Components`` bundle at startup. Tests build the app with an in-memory bundle instead, so
the whole request pipeline runs without PostgreSQL or any external API.

``uvicorn app.main:app`` works as usual: the module-level ``app`` is created lazily on first
access (PEP 562), so merely importing this module never reads the environment.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import __version__
from app.budget import BudgetExceeded, QueryBudget
from app.config import Settings, get_settings
from app.database import ChunkRecord, Database
from app.demo import sample_questions, seed_demo_corpus
from app.errors import ConfigurationError, PDFParseError, PDFTooLargeError, ProviderError
from app.generator.grounded_llm import (
    SYSTEM_PROMPT,
    GroundedGenerator,
    build_generator,
    ground_answer,
)
from app.ingestion.chunker import page_for_offset
from app.ingestion.contextualizer import Contextualizer, build_contextualizer
from app.ingestion.pipeline import EmptyDocumentError, ingest_pdf
from app.retrieval.embedder import Embedder, build_embedder, local_inference_gate
from app.retrieval.hybrid_search import HybridSearcher, PostgresHybridSearcher
from app.retrieval.reranker import Reranker, build_reranker
from app.schemas import (
    AnswerStatus,
    DocumentsResponse,
    DocumentSummary,
    GroundedAnswer,
    HealthResponse,
    IngestResponse,
    PipelineTrace,
    PlaygroundConfig,
    ProviderStatus,
    QueryRequest,
    RetrievedChunk,
    RetrievedPassage,
    StageTimings,
)

logger = logging.getLogger(__name__)

MULTIPART_OVERHEAD_BYTES = 64 * 1024
PASSAGE_PREVIEW_CHARS = 180
PLAYGROUND_DIR = Path(__file__).resolve().parent / "playground" / "static"
UPLOAD_PATH = "/api/v1/ingest"
_BODYLESS_METHODS = {"GET", "HEAD", "OPTIONS"}


# ------------------------------------------------------------------------- components


class DocumentStore(Protocol):
    async def store_document(
        self,
        *,
        title: str,
        filename: str,
        chunks: Sequence[ChunkRecord],
        page_count: int | None = None,
    ) -> uuid.UUID: ...


class DocumentCatalogue(Protocol):
    async def list_documents(self) -> list[DocumentSummary]: ...


class HealthProbe(Protocol):
    pgvector_version: str | None

    async def ping(self) -> bool: ...

    async def pgvector_installed(self) -> bool: ...


class CachedCatalogue:
    """``list_documents`` with a time-to-live, so anonymous page loads do not keep Neon awake.

    The corpus changes only through ``ingest_pdf``, which calls :meth:`invalidate` after it
    stores a document. Anything that deletes rows directly must call it too.
    """

    def __init__(self, inner: DocumentCatalogue, ttl_s: float, clock=time.monotonic) -> None:  # type: ignore[no-untyped-def]
        self._inner = inner
        self._ttl_s = ttl_s
        self._clock = clock
        self._cached: list[DocumentSummary] | None = None
        self._at = 0.0
        self._lock = asyncio.Lock()

    async def list_documents(self) -> list[DocumentSummary]:
        if self._ttl_s <= 0:
            return await self._inner.list_documents()
        async with self._lock:
            if self._cached is None or self._clock() - self._at >= self._ttl_s:
                self._cached = await self._inner.list_documents()
                self._at = self._clock()
            return list(self._cached)

    def invalidate(self) -> None:
        self._cached = None


@dataclass
class Components:
    settings: Settings
    db: HealthProbe | None
    store: DocumentStore
    catalogue: DocumentCatalogue
    searcher: HybridSearcher
    embedder: Embedder
    reranker: Reranker
    contextualizer: Contextualizer
    generator: GroundedGenerator | None
    budget: QueryBudget | None = None
    inflight: int = 0

    async def close(self) -> None:
        if isinstance(self.db, Database):
            await self.db.dispose()


async def _init_database(db: Database, settings: Settings) -> None:
    """Create or check the schema, retrying while a serverless database wakes up."""
    attempts = max(1, settings.db_startup_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            if settings.auto_create_schema:
                await db.init_schema()
            else:
                await db.pgvector_installed()
            return
        except ConfigurationError:
            raise  # a wrong setting does not fix itself
        except (SQLAlchemyError, OSError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            delay = min(2.0 * attempt, 10.0)
            logger.warning(
                "Database not ready (%s), attempt %d of %d; retrying in %.0f s",
                type(exc).__name__,
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)


async def build_components(settings: Settings) -> Components:
    """Wire the production collaborators. Raises ``ConfigurationError`` on fatal misconfiguration."""
    db = Database(settings)
    await _init_database(db, settings)

    # One gate for both local models: questions, seeding and warm-up never run inference at
    # the same time, which is what keeps a 512 MB instance inside its memory.
    gate = local_inference_gate(settings)
    embedder = build_embedder(settings, gate=gate)
    if embedder.dimensions != settings.embedding_dimensions:
        raise ConfigurationError(
            f"Embedder produces {embedder.dimensions} dims but EMBEDDING_DIMENSIONS="
            f"{settings.embedding_dimensions}"
        )
    reranker = build_reranker(settings, gate=gate)
    contextualizer = build_contextualizer(settings)

    generator: GroundedGenerator | None = None
    if settings.generator_configured:
        generator = build_generator(settings)
    else:
        logger.warning(
            "No API key for LLM_PROVIDER=%s; /api/v1/query will return 503 until configured",
            settings.llm_provider,
        )

    budget = QueryBudget(settings) if settings.demo_mode else None
    if settings.demo_mode:
        logger.info(
            "Demo mode: uploads %s, %d questions/client/min, %d/client/day, %d/min, %d/day, "
            "%d in flight at most",
            "need API_KEY" if settings.uploads_enabled else "disabled",
            settings.query_rate_limit_per_minute,
            settings.query_daily_limit_per_client,
            settings.query_global_limit_per_minute,
            settings.query_daily_limit,
            settings.query_max_inflight,
        )

    return Components(
        settings=settings,
        db=db,
        store=db,
        catalogue=CachedCatalogue(db, settings.documents_cache_ttl_s),
        searcher=PostgresHybridSearcher(db, settings),
        embedder=embedder,
        reranker=reranker,
        contextualizer=contextualizer,
        generator=generator,
        budget=budget,
    )


def get_components(request: Request) -> Components:
    components = getattr(request.app.state, "components", None)
    if components is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Application is still starting")
    return components


# ---------------------------------------------------------------------- who is asking


def _normalise_ip(value: str) -> str | None:
    """A rate-limit key for an address: IPv4 as is, IPv6 by its /64 network; None if invalid."""
    value = value.strip()
    if not value:
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        host = value
        if value.startswith("["):  # "[2001:db8::1]:443"
            host = value[1:].split("]", 1)[0]
        elif value.count(":") == 1:  # "203.0.113.7:51234"
            host = value.split(":", 1)[0]
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return None
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return f"{ipaddress.ip_network(f'{address}/64', strict=False).network_address}/64"
    return str(address)


def client_id_for(request: Request, settings: Settings) -> str:
    """The visitor's address, as far as it can be trusted, for the per-visitor limits.

    The left-most ``X-Forwarded-For`` entry is whatever the client wrote, because proxies
    append to the header rather than replace it, so it is never used. Behind a proxy the
    address comes from ``CLIENT_IP_HEADER`` (a header the edge overwrites, such as Cloudflare's
    ``CF-Connecting-IP``), or else from the entry ``TRUSTED_PROXY_HOPS`` places from the right.
    A wrong hop count only ever merges visitors into one bucket (stricter), never splits one
    visitor into many.
    """
    if settings.trust_proxy_headers:
        if settings.client_ip_header:
            candidate = _normalise_ip(request.headers.get(settings.client_ip_header, ""))
            if candidate:
                return candidate
        hops = [
            hop.strip()
            for value in request.headers.getlist("x-forwarded-for")
            for hop in value.split(",")
            if hop.strip()
        ]
        if len(hops) >= settings.trusted_proxy_hops:
            candidate = _normalise_ip(hops[-settings.trusted_proxy_hops])
            if candidate:
                return candidate
    host = request.client.host if request.client else ""
    return _normalise_ip(host) or host or "unknown"


# ------------------------------------------------------------------------------- auth


def _key_matches(provided: str, expected: str) -> bool:
    return bool(provided) and secrets.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    )


async def require_api_key(
    request: Request, components: Components = Depends(get_components)
) -> None:
    """Optional shared-secret auth: enforced on /api/v1/* only when ``API_KEY`` is configured."""
    expected = components.settings.api_key
    if expected is None:
        return
    if not _key_matches(request.headers.get("x-api-key", ""), expected.get_secret_value()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )


UPLOADS_CLOSED = "Uploads are disabled on this demo; ask questions about the sample corpus instead."


async def require_upload_allowed(
    request: Request, components: Components = Depends(get_components)
) -> None:
    """In demo mode the upload box is closed: only the operator's API_KEY can ingest.

    ``RequestGuardMiddleware`` already applies this rule before the body is read; this is the
    second line of defence for anything that reaches the route another way.
    """
    settings = components.settings
    if settings.demo_mode and settings.api_key is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, UPLOADS_CLOSED)
    await require_api_key(request, components)


async def require_query_access(
    request: Request, components: Components = Depends(get_components)
) -> None:
    """Who may ask questions and list the corpus.

    In demo mode both are public on purpose (the budget protects questions) even though
    ``API_KEY`` is set, because that key exists there only to gate uploads. Outside demo mode
    ``API_KEY`` protects them like every other ``/api/v1`` route.
    """
    if components.settings.demo_mode:
        return
    await require_api_key(request, components)


# ----------------------------------------------------------------------- middleware


class _BodyTooLarge(HTTPException):
    """Raised from inside ``receive`` while the body is being parsed.

    It must be an ``HTTPException``: FastAPI re-raises those untouched, whereas any other
    exception raised during body parsing is converted into a generic 400.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(413, detail)


class RequestGuardMiddleware:
    """Pure-ASGI checks that must happen before FastAPI reads a request body.

    FastAPI reads and parses the whole body (JSON or multipart, spooled to disk) before any
    route dependency runs, so a size limit or an upload key checked in the route comes too late
    on a 512 MB host. This middleware:

    * refuses ``POST /api/v1/ingest`` without the right key (403 in demo mode without a key
      configured, 401 otherwise) before reading a byte of the upload;
    * limits every request body: ``MAX_UPLOAD_BYTES`` (plus multipart overhead) for uploads,
      ``MAX_JSON_BODY_BYTES`` for everything else, on ``Content-Length`` and on streamed bytes.
    """

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    def _upload_denial(
        self, headers: dict[str, str]
    ) -> tuple[int, str, list[tuple[bytes, bytes]]] | None:
        settings = self.settings
        if settings.demo_mode and settings.api_key is None:
            return 403, UPLOADS_CLOSED, []
        if settings.api_key is not None and not _key_matches(
            headers.get("x-api-key", ""), settings.api_key.get_secret_value()
        ):
            return 401, "Invalid or missing X-API-Key header", [(b"www-authenticate", b"ApiKey")]
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method", "GET") in _BODYLESS_METHODS:
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        is_upload = str(scope.get("path", "")).rstrip("/") == UPLOAD_PATH
        if is_upload:
            denial = self._upload_denial(headers)
            if denial is not None:
                await _send_json(send, denial[0], {"detail": denial[1]}, denial[2])
                return
            limit = self.settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
            detail = f"Request body exceeds MAX_UPLOAD_BYTES ({limit} bytes)"
        else:
            limit = self.settings.max_json_body_bytes
            detail = f"Request body exceeds MAX_JSON_BODY_BYTES ({limit} bytes)"

        content_length = headers.get("content-length", "")
        if content_length.isdigit() and int(content_length) > limit:
            await _send_json(send, 413, {"detail": detail})
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge(detail)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await _send_json(send, 413, {"detail": detail})


async def _send_json(
    send: Send, status_code: int, payload: dict[str, object], extra_headers: list | None = None
) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                *(extra_headers or []),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# ------------------------------------------------------------------------------ routes

router = APIRouter()


async def _bounded_probe(probe: Coroutine[object, object, bool], timeout_s: float) -> bool:
    """Run a health probe with a deadline so an unreachable host cannot stall ``/health``."""
    try:
        return await asyncio.wait_for(probe, timeout=timeout_s)
    except TimeoutError:
        logger.warning("Health probe timed out after %.1fs", timeout_s)
        return False
    except Exception as exc:  # noqa: BLE001 - a probe must never take the endpoint down
        logger.warning("Health probe failed: %s", type(exc).__name__)
        return False


def _provider_statuses(components: Components) -> dict[str, ProviderStatus]:
    settings = components.settings
    return {
        "embeddings": ProviderStatus(
            provider=components.embedder.name,
            configured=settings.embedder_configured,
            model=getattr(components.embedder, "model_name", components.embedder.name),
        ),
        "reranker": ProviderStatus(
            provider=components.reranker.name,
            configured=settings.reranker_configured,
            model=getattr(components.reranker, "model", None),
        ),
        "contextualizer": ProviderStatus(
            provider=components.contextualizer.name,
            configured=components.contextualizer.name != "heuristic"
            or not settings.contextualizer_enabled
            or settings.contextualizer_provider == "heuristic",
            model=getattr(components.contextualizer, "model", None),
        ),
        "generator": ProviderStatus(
            provider=settings.llm_provider,
            configured=components.generator is not None,
            model=components.generator.model if components.generator else None,
        ),
    }


@router.api_route("/livez", methods=["GET", "HEAD"], tags=["ops"])
async def livez() -> dict[str, str]:
    """Liveness for the host's health check and uptime monitors: never touches the database.

    Render probes its health-check path every few seconds while the service is up. Pointing
    that at ``/health`` would query Neon each time and keep its compute awake around the clock,
    which the free plan's monthly compute allowance does not cover.
    """
    return {"status": "alive", "version": __version__}


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unreachable"}},
    tags=["ops"],
)
async def health(components: Components = Depends(get_components)) -> JSONResponse:
    """Deep check for people: database, pgvector and providers. Monitors should use /livez."""
    settings = components.settings
    database_connected = (
        await _bounded_probe(components.db.ping(), settings.health_timeout_s)
        if components.db
        else False
    )
    pgvector_installed = (
        await _bounded_probe(components.db.pgvector_installed(), settings.health_timeout_s)
        if components.db and database_connected
        else False
    )
    healthy = database_connected and pgvector_installed and components.generator is not None
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        database_connected=database_connected,
        pgvector_installed=pgvector_installed,
        pgvector_version=components.db.pgvector_version if components.db else None,
        providers=_provider_statuses(components),
        demo_mode=settings.demo_mode,
        uploads_enabled=settings.uploads_enabled,
    )
    http_status = status.HTTP_200_OK if database_connected else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=body.model_dump(mode="json"))


@router.get(
    "/api/v1/documents",
    response_model=DocumentsResponse,
    tags=["retrieval"],
    dependencies=[Depends(require_query_access)],
)
async def list_documents(components: Components = Depends(get_components)) -> DocumentsResponse:
    """The ingested corpus (titles, pages, chunk counts): what a visitor can ask about."""
    try:
        documents = await components.catalogue.list_documents()
    except (SQLAlchemyError, OSError) as exc:
        raise _database_http_error(exc) from exc
    return DocumentsResponse(documents=documents)


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Second line of defence after ``RequestGuardMiddleware`` (exact per-file limit)."""
    buffer = bytearray()
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        buffer.extend(part)
        if len(buffer) > max_bytes:
            raise HTTPException(413, f"Upload exceeds MAX_UPLOAD_BYTES ({max_bytes} bytes)")
    return bytes(buffer)


@router.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    tags=["ingestion"],
    dependencies=[Depends(require_upload_allowed)],
)
async def ingest(
    file: UploadFile = File(..., description="PDF document"),
    title: str | None = Form(default=None, description="Optional human-readable title"),
    components: Components = Depends(get_components),
) -> IngestResponse:
    settings = components.settings

    filename = file.filename or "upload.pdf"
    is_pdf = filename.lower().endswith(".pdf") or (file.content_type or "") == "application/pdf"
    if not is_pdf:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF uploads are supported")

    data = await _read_upload(file, settings.max_upload_bytes)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")

    try:
        result = await ingest_pdf(components, data, filename, title)
    except PDFTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except (PDFParseError, EmptyDocumentError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc

    return IngestResponse(
        document_id=str(result.document_id),
        filename=result.filename,
        title=result.title,
        chunks_count=result.chunks_count,
        pages_count=result.pages_count,
        processing_time_ms=result.processing_time_ms,
    )


@router.post(
    "/api/v1/query",
    response_model=GroundedAnswer,
    tags=["retrieval"],
    dependencies=[Depends(require_query_access)],
)
async def query(
    payload: QueryRequest,
    request: Request,
    components: Components = Depends(get_components),
) -> GroundedAnswer:
    settings = components.settings
    started = time.perf_counter()

    if settings.demo_mode:
        # Cheap refusals first: none of these spends a budget slot.
        if len(payload.query) > settings.demo_max_query_chars:
            raise HTTPException(
                422,
                f"Questions are limited to {settings.demo_max_query_chars} characters on this demo.",
            )
        if components.inflight >= settings.query_max_inflight:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The demo is answering other questions; please try again in a few seconds.",
                headers={"Retry-After": "5"},
            )

    if components.budget is not None:
        try:
            await components.budget.admit(client_id_for(request, settings))
        except BudgetExceeded as exc:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                exc.message,
                headers={"Retry-After": str(exc.retry_after_s)},
            ) from exc

    components.inflight += 1
    try:
        return await _answer(payload, components, started)
    finally:
        components.inflight -= 1


async def _answer(payload: QueryRequest, components: Components, started: float) -> GroundedAnswer:
    settings = components.settings

    def lap(since: float) -> float:
        return round((time.perf_counter() - since) * 1000.0, 2)

    try:
        mark = time.perf_counter()
        query_vector = await components.embedder.embed_query(payload.query)
        embed_ms = lap(mark)
        mark = time.perf_counter()
        candidates = await components.searcher.search(
            query_text=payload.query,
            query_vector=query_vector,
            document_id=payload.filter_document_id,
            limit=settings.top_k_retrieval,
        )
        search_ms = lap(mark)
        mark = time.perf_counter()
        reranked = (
            await components.reranker.rerank(payload.query, candidates, top_n=settings.top_k_rerank)
            if candidates
            else []
        )
        rerank_ms = lap(mark)
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc
    except (SQLAlchemyError, OSError) as exc:
        raise _database_http_error(exc) from exc

    retrieval_ms = (time.perf_counter() - started) * 1000.0
    trace = _pipeline_trace(
        components,
        candidates,
        reranked,
        StageTimings(embed_ms=embed_ms, search_ms=search_ms, rerank_ms=rerank_ms),
    )

    if not reranked:
        total_ms = (time.perf_counter() - started) * 1000.0
        return GroundedAnswer(
            query=payload.query,
            answer=(
                "No relevant passages were found in the knowledge base for this question. "
                "Ingest the governing document or rephrase the query."
            ),
            is_grounded=False,
            confidence_score=0.0,
            citations=[],
            retrieval_latency_ms=round(retrieval_ms, 2),
            total_latency_ms=round(total_ms, 2),
            status=AnswerStatus.INSUFFICIENT_CONTEXT,
            generation_model=None,
            pipeline=trace,
        )

    if components.generator is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"LLM provider '{settings.llm_provider}' is not configured (missing API key)",
        )

    mark = time.perf_counter()
    try:
        result = await components.generator.generate(payload.query, reranked)
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc

    total_ms = (time.perf_counter() - started) * 1000.0
    answer = ground_answer(
        query=payload.query,
        result=result,
        chunks=reranked,
        retrieval_latency_ms=round(retrieval_ms, 2),
        total_latency_ms=round(total_ms, 2),
    )
    trace.timings.generate_ms = lap(mark)
    trace.quotes_proposed = len(result.output.citations)
    trace.quotes_verified = len(answer.citations)
    answer.pipeline = trace

    fused_rank = {chunk.id: position for position, chunk in enumerate(candidates, start=1)}
    cited_ids = {citation.chunk_id for citation in answer.citations}
    answer.retrieved_passages = [
        RetrievedPassage(
            chunk_id=chunk.id,
            document_title=chunk.document_title,
            # The page the passage text begins on, which is what its preview shows: a chunk
            # can open with overlap carried from the previous page.
            page_number=page_for_offset(chunk.page_map, 0, chunk.page_number),
            dense_rank=chunk.dense_rank,
            sparse_rank=chunk.sparse_rank,
            rrf_score=round(chunk.rrf_score, 6),
            fused_rank=fused_rank.get(chunk.id),
            rerank_position=chunk.rerank_position,
            relevance_score=round(chunk.relevance_score, 4),
            cited=chunk.id in cited_ids,
            preview=" ".join(chunk.raw_content.split())[:PASSAGE_PREVIEW_CHARS],
        )
        for chunk in reranked
    ]
    logger.info(
        "Query answered status=%s grounded=%s citations=%d retrieval=%.0fms total=%.0fms",
        answer.status.value,
        answer.is_grounded,
        len(answer.citations),
        retrieval_ms,
        total_ms,
    )
    return answer


def _model_names(components: Components) -> tuple[str, str]:
    embedder = components.embedder
    reranker = components.reranker
    return (
        str(getattr(embedder, "model_name", None) or embedder.name),
        str(getattr(reranker, "model", None) or reranker.name),
    )


def _pipeline_trace(
    components: Components,
    candidates: Sequence[RetrievedChunk],
    reranked: Sequence[RetrievedChunk],
    timings: StageTimings,
) -> PipelineTrace:
    """Counts the playground walkthrough explains: what each search arm contributed."""
    settings = components.settings
    both = sum(1 for c in candidates if c.dense_rank is not None and c.sparse_rank is not None)
    vector_only = sum(1 for c in candidates if c.dense_rank is not None and c.sparse_rank is None)
    keyword_only = sum(1 for c in candidates if c.dense_rank is None and c.sparse_rank is not None)
    embedding_model, reranker_model = _model_names(components)
    return PipelineTrace(
        embedding_model=embedding_model,
        embedding_dimensions=components.embedder.dimensions,
        reranker_model=reranker_model,
        rrf_k=settings.rrf_k,
        candidate_limit=settings.top_k_retrieval,
        candidates=len(candidates),
        found_by_both=both,
        found_by_vector_only=vector_only,
        found_by_keyword_only=keyword_only,
        passages_to_model=len(reranked),
        timings=timings,
    )


def _provider_http_error(exc: ProviderError) -> HTTPException:
    """Map a provider failure to 503 (retryable) or 502 (permanent) without echoing its body."""
    code = status.HTTP_503_SERVICE_UNAVAILABLE if exc.retryable else status.HTTP_502_BAD_GATEWAY
    incident = uuid.uuid4().hex[:12]
    logger.error(
        "Provider failure incident=%s provider=%s retryable=%s: %s",
        incident,
        exc.provider,
        exc.retryable,
        exc.message,
    )
    kind = "temporary failure, retry later" if exc.retryable else "permanent failure"
    return HTTPException(
        code, f"Upstream provider error ({exc.provider}): {kind}; incident {incident}"
    )


def _database_http_error(exc: BaseException) -> HTTPException:
    """A database failure becomes a 503 with an incident id; SQL and parameters stay out of it."""
    incident = uuid.uuid4().hex[:12]
    logger.error("Database failure incident=%s: %s", incident, type(exc).__name__)
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"The database is unavailable; retry later. Incident {incident}",
        headers={"Retry-After": "10"},
    )


# -------------------------------------------------------------------------- playground

playground_router = APIRouter()

#: Browsers keep a copy but ask before each use (an unchanged file costs a 304). Without it, a
#: visitor returning after a deploy can pair the new page with a cached script from the old one.
REVALIDATE = {"Cache-Control": "no-cache"}


class RevalidatingStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = super().file_response(*args, **kwargs)
        response.headers.update(REVALIDATE)
        return response


@playground_router.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def playground_index() -> FileResponse:
    return FileResponse(PLAYGROUND_DIR / "index.html", headers=REVALIDATE)


@playground_router.get("/playground/config", response_model=PlaygroundConfig, tags=["ops"])
async def playground_config(components: Components = Depends(get_components)) -> PlaygroundConfig:
    settings = components.settings
    limits: dict[str, int | None] = (
        {
            k: (int(v) if isinstance(v, int) else None)
            for k, v in components.budget.snapshot().items()
        }
        if components.budget
        else {}
    )
    # Outside demo mode an API_KEY protects the corpus listing; the page cannot send the key.
    listing_open = settings.demo_mode or settings.api_key is None
    documents: list[DocumentSummary] = []
    if listing_open:
        try:
            documents = await components.catalogue.list_documents()
        except (SQLAlchemyError, OSError) as exc:
            raise _database_http_error(exc) from exc
    return PlaygroundConfig(
        demo_mode=settings.demo_mode,
        uploads_enabled=settings.uploads_enabled,
        generator=_provider_statuses(components)["generator"],
        retrieval={
            "top_k_retrieval": settings.top_k_retrieval,
            "top_k_rerank": settings.top_k_rerank,
            "rrf_k": settings.rrf_k,
            "max_query_chars": settings.demo_max_query_chars if settings.demo_mode else 2000,
        },
        limits=limits,
        documents=documents,
        sample_questions=sample_questions(settings.demo_seed_dir) if listing_open else [],
        models={
            "embedding": _model_names(components)[0],
            "embedding_dimensions": components.embedder.dimensions,
            "reranker": _model_names(components)[1],
            "contextualizer": components.contextualizer.name,
        },
        # The repository is public, so the rules are no secret; showing them is the lesson.
        system_prompt=SYSTEM_PROMPT,
    )


# ------------------------------------------------------------------------- app factory


def _configure_logging(settings: Settings) -> None:
    """Install a root handler once (never override a host's logging) and set the app level."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logging.getLogger("app").setLevel(settings.log_level)


async def _prepare_demo(components: Components, settings: Settings) -> None:
    """Background start-up work of the showcase: seed the corpus, then warm the models."""
    try:
        if settings.demo_seed_on_startup:
            report = await seed_demo_corpus(components, settings)
            logger.info(
                "Demo corpus seeding finished: %d ingested, %d already present, %d failed",
                len(report.ingested),
                len(report.skipped),
                len(report.failed),
            )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - never take the service down over the sample corpus
        logger.exception("Demo corpus seeding crashed")
    if settings.demo_warm_models:
        await _warm_models(components)


async def _warm_models(components: Components) -> None:
    """Load the local models before the first visitor asks, so that question is not the slow one.

    The project page wakes the service as soon as it is opened; by the time a visitor reaches
    the question box the embedding model and the cross-encoder are already in memory.
    """
    started = time.perf_counter()
    try:
        await components.embedder.embed_query("warm up")
        probe = RetrievedChunk(
            id="warm-up",
            document_id="warm-up",
            document_title="warm-up",
            chunk_index=0,
            page_number=1,
            raw_content="warm up",
            contextualized_content="warm up",
            rrf_score=0.0,
        )
        await components.reranker.rerank("warm up", [probe], top_n=1)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - warming is an optimisation, never a failure
        logger.warning("Model warm-up skipped: %s", exc)
        return
    logger.info("Local models warmed in %.1f s", time.perf_counter() - started)


def create_app(settings: Settings | None = None, components: Components | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = False
        if getattr(app.state, "components", None) is None:
            app.state.components = await build_components(settings)
            owned = True
            logger.info("Startup complete: %s v%s", settings.app_name, __version__)
        seed_task: asyncio.Task[None] | None = None
        if settings.demo_mode and (settings.demo_seed_on_startup or settings.demo_warm_models):
            seed_task = asyncio.create_task(_prepare_demo(app.state.components, settings))
        app.state.seed_task = seed_task
        try:
            yield
        finally:
            if seed_task is not None and not seed_task.done():
                seed_task.cancel()
            if owned:
                await app.state.components.close()

    app = FastAPI(
        title="Context-Aware Hybrid Knowledge & Compliance Retrieval Engine",
        version=__version__,
        description=(
            "Hybrid pgvector + tsvector retrieval with Reciprocal Rank Fusion, cross-encoder "
            "re-ranking, and schema-constrained grounded answers with verbatim citations."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.components = components
    app.include_router(router)
    if settings.playground_enabled:
        app.include_router(playground_router)
        app.mount(
            "/playground/assets",
            RevalidatingStaticFiles(directory=PLAYGROUND_DIR / "assets"),
            name="playground-assets",
        )
    app.add_middleware(RequestGuardMiddleware, settings=settings)
    if settings.cors_origins:
        # Added last, so it is the outermost layer and the guard's refusals carry CORS headers.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
            max_age=600,
        )

    @app.exception_handler(ConfigurationError)
    async def _configuration_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default echoes each rejected value back ("input"); never reflect payloads.
        errors = [
            {"type": error.get("type"), "loc": list(error.get("loc", ())), "msg": error.get("msg")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    return app


_lazy_app: FastAPI | None = None


def __getattr__(name: str) -> object:
    """Expose ``app`` for ``uvicorn app.main:app`` without building it at import time."""
    global _lazy_app
    if name == "app":
        if _lazy_app is None:
            _lazy_app = create_app()
        return _lazy_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
