"""FastAPI application: ``GET /health``, ``POST /api/v1/ingest``, ``POST /api/v1/query``.

All collaborators (database, embedder, re-ranker, contextualizer, generator) are assembled into a
``Components`` bundle at startup. Tests build the app with an in-memory bundle instead, so the
whole request pipeline runs without PostgreSQL or any external API.

``uvicorn app.main:app`` works as usual: the module-level ``app`` is created lazily on first
access (PEP 562), so merely importing this module never reads the environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import __version__
from app.config import Settings, get_settings
from app.database import ChunkRecord, Database
from app.errors import ConfigurationError, PDFParseError, PDFTooLargeError, ProviderError
from app.generator.grounded_llm import GroundedGenerator, build_generator, ground_answer
from app.ingestion.chunker import chunk_document
from app.ingestion.contextualizer import Contextualizer, apply_context, build_contextualizer
from app.ingestion.pdf_parser import parse_pdf_bytes_async
from app.retrieval.embedder import Embedder, build_embedder
from app.retrieval.hybrid_search import HybridSearcher, PostgresHybridSearcher
from app.retrieval.reranker import Reranker, build_reranker
from app.schemas import (
    AnswerStatus,
    GroundedAnswer,
    HealthResponse,
    IngestResponse,
    ProviderStatus,
    QueryRequest,
)

logger = logging.getLogger(__name__)

MULTIPART_OVERHEAD_BYTES = 64 * 1024


# ------------------------------------------------------------------------- components


class DocumentStore(Protocol):
    async def store_document(
        self, *, title: str, filename: str, chunks: Sequence[ChunkRecord]
    ) -> uuid.UUID: ...


class HealthProbe(Protocol):
    pgvector_version: str | None

    async def ping(self) -> bool: ...

    async def pgvector_installed(self) -> bool: ...


@dataclass
class Components:
    settings: Settings
    db: HealthProbe | None
    store: DocumentStore
    searcher: HybridSearcher
    embedder: Embedder
    reranker: Reranker
    contextualizer: Contextualizer
    generator: GroundedGenerator | None

    async def close(self) -> None:
        if isinstance(self.db, Database):
            await self.db.dispose()


async def build_components(settings: Settings) -> Components:
    """Wire the production collaborators. Raises ``ConfigurationError`` on fatal misconfiguration."""
    db = Database(settings)
    if settings.auto_create_schema:
        await db.init_schema()
    else:
        await db.pgvector_installed()

    embedder = build_embedder(settings)
    if embedder.dimensions != settings.embedding_dimensions:
        raise ConfigurationError(
            f"Embedder produces {embedder.dimensions} dims but EMBEDDING_DIMENSIONS="
            f"{settings.embedding_dimensions}"
        )
    reranker = build_reranker(settings)
    contextualizer = build_contextualizer(settings)

    generator: GroundedGenerator | None = None
    if settings.generator_configured:
        generator = build_generator(settings)
    else:
        logger.warning(
            "No API key for LLM_PROVIDER=%s; /api/v1/query will return 503 until configured",
            settings.llm_provider,
        )

    return Components(
        settings=settings,
        db=db,
        store=db,
        searcher=PostgresHybridSearcher(db, settings),
        embedder=embedder,
        reranker=reranker,
        contextualizer=contextualizer,
        generator=generator,
    )


def get_components(request: Request) -> Components:
    components = getattr(request.app.state, "components", None)
    if components is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Application is still starting")
    return components


async def require_api_key(
    request: Request, components: Components = Depends(get_components)
) -> None:
    """Optional shared-secret auth: enforced on /api/v1/* only when ``API_KEY`` is configured."""
    expected = components.settings.api_key
    if expected is None:
        return
    provided = request.headers.get("x-api-key", "")
    if not provided or not secrets.compare_digest(
        provided.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ----------------------------------------------------------------------- middleware


class _BodyTooLarge(HTTPException):
    """Raised from inside ``receive`` while the body is being parsed.

    It must be an ``HTTPException``: FastAPI re-raises those untouched, whereas any other
    exception raised during body parsing is converted into a generic 400.
    """

    def __init__(self, max_bytes: int) -> None:
        super().__init__(413, f"Request body exceeds MAX_UPLOAD_BYTES ({max_bytes} bytes)")


class MaxBodySizeMiddleware:
    """Reject over-sized request bodies *before* Starlette parses and spools multipart uploads.

    ``UploadFile`` only becomes available after the whole body has been received, so checking
    the file size inside the endpoint is too late. This pure-ASGI middleware (a) rejects on the
    ``Content-Length`` header and (b) counts streamed bytes for chunked bodies.
    """

    def __init__(
        self, app: ASGIApp, *, max_bytes: int, path_prefix: str = "/api/v1/ingest"
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_prefix = path_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope.get("path", "")).startswith(self.path_prefix):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get("content-length", "")
        if content_length.isdigit() and int(content_length) > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge(self.max_bytes)
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
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(
            {"detail": f"Request body exceeds MAX_UPLOAD_BYTES ({self.max_bytes} bytes)"}
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
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
        logger.warning("Health probe failed: %s", exc)
        return False


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unreachable"}},
    tags=["ops"],
)
async def health(components: Components = Depends(get_components)) -> JSONResponse:
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
    providers = {
        "embeddings": ProviderStatus(
            provider=components.embedder.name,
            configured=settings.embedder_configured,
            model=settings.embedding_model
            if components.embedder.name == "openai"
            else getattr(components.embedder, "model_name", components.embedder.name),
        ),
        "reranker": ProviderStatus(
            provider=components.reranker.name,
            configured=settings.reranker_configured,
            model=settings.cohere_rerank_model if components.reranker.name == "cohere" else None,
        ),
        "contextualizer": ProviderStatus(
            provider=components.contextualizer.name,
            configured=components.contextualizer.name != "heuristic"
            or not settings.contextualizer_enabled,
            model=settings.contextualizer_model
            if components.contextualizer.name == "anthropic"
            else None,
        ),
        "generator": ProviderStatus(
            provider=settings.llm_provider,
            configured=components.generator is not None,
            model=components.generator.model if components.generator else None,
        ),
    }
    healthy = database_connected and pgvector_installed and components.generator is not None
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        database_connected=database_connected,
        pgvector_installed=pgvector_installed,
        pgvector_version=components.db.pgvector_version if components.db else None,
        providers=providers,
    )
    http_status = status.HTTP_200_OK if database_connected else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=body.model_dump(mode="json"))


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Second line of defence after ``MaxBodySizeMiddleware`` (exact per-file limit)."""
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
    dependencies=[Depends(require_api_key)],
)
async def ingest(
    file: UploadFile = File(..., description="PDF document"),
    title: str | None = Form(default=None, description="Optional human-readable title"),
    components: Components = Depends(get_components),
) -> IngestResponse:
    settings = components.settings
    started = time.perf_counter()

    filename = file.filename or "upload.pdf"
    is_pdf = filename.lower().endswith(".pdf") or (file.content_type or "") == "application/pdf"
    if not is_pdf:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF uploads are supported")

    data = await _read_upload(file, settings.max_upload_bytes)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")

    try:
        parsed = await parse_pdf_bytes_async(
            data, filename, title, max_pages=settings.max_pdf_pages
        )
    except PDFTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except PDFParseError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    # Pure-Python tokenisation over a whole document is CPU-bound; keep it off the event loop.
    chunks = await asyncio.to_thread(
        chunk_document,
        parsed,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No text chunks could be produced")
    if len(chunks) > settings.max_chunks_per_document:
        raise HTTPException(
            413,
            f"Document produces {len(chunks)} chunks; limit is MAX_CHUNKS_PER_DOCUMENT="
            f"{settings.max_chunks_per_document}",
        )

    contexts = await components.contextualizer.contextualize(parsed, chunks)
    contextualized = [
        apply_context(ctx, chunk.text) for ctx, chunk in zip(contexts, chunks, strict=True)
    ]

    try:
        embeddings = await components.embedder.embed_documents(contextualized)
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc

    records = [
        ChunkRecord(
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            raw_content=chunk.text,
            contextualized_content=text,
            embedding=vector,
        )
        for chunk, text, vector in zip(chunks, contextualized, embeddings, strict=True)
    ]
    document_id = await components.store.store_document(
        title=parsed.title, filename=filename, chunks=records
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "Ingested %s: %d pages, %d chunks in %.0f ms",
        filename,
        parsed.page_count,
        len(records),
        elapsed_ms,
    )
    return IngestResponse(
        document_id=str(document_id),
        filename=filename,
        title=parsed.title,
        chunks_count=len(records),
        pages_count=parsed.page_count,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/api/v1/query",
    response_model=GroundedAnswer,
    tags=["retrieval"],
    dependencies=[Depends(require_api_key)],
)
async def query(
    payload: QueryRequest, components: Components = Depends(get_components)
) -> GroundedAnswer:
    settings = components.settings
    started = time.perf_counter()

    try:
        query_vector = await components.embedder.embed_query(payload.query)
        candidates = await components.searcher.search(
            query_text=payload.query,
            query_vector=query_vector,
            document_id=payload.filter_document_id,
            limit=settings.top_k_retrieval,
        )
        reranked = (
            await components.reranker.rerank(payload.query, candidates, top_n=settings.top_k_rerank)
            if candidates
            else []
        )
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc

    retrieval_ms = (time.perf_counter() - started) * 1000.0

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
        )

    if components.generator is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"LLM provider '{settings.llm_provider}' is not configured (missing API key)",
        )

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
    logger.info(
        "Query answered status=%s grounded=%s citations=%d retrieval=%.0fms total=%.0fms",
        answer.status.value,
        answer.is_grounded,
        len(answer.citations),
        retrieval_ms,
        total_ms,
    )
    return answer


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
        try:
            yield
        finally:
            if owned:
                await app.state.components.close()

    app = FastAPI(
        title="Context-Aware Hybrid Knowledge & Compliance Retrieval Engine",
        version=__version__,
        description=(
            "Hybrid pgvector + tsvector retrieval with Reciprocal Rank Fusion, Cohere re-ranking, "
            "and schema-constrained grounded answers with verbatim citations."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.components = components
    app.include_router(router)
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )

    @app.exception_handler(ConfigurationError)
    async def _configuration_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

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
