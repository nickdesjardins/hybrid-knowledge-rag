"""Cross-encoder re-ranking of the fused candidate set (Cohere Rerank v3.5).

Bi-encoders (the embedding model) score query and passage independently. A cross-encoder reads
them *together*, so it can tell "retention period for batch records" apart from "retention of
cleaning validation samples" when both share most of their vocabulary. We run it on the 50 fused
candidates and keep only the top ``top_n`` (3 by default) for generation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

from app.config import Settings
from app.errors import ConfigurationError, ProviderError
from app.retrieval.embedder import local_inference_gate, local_model_kwargs
from app.schemas import RerankedChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RerankedChunk]: ...


def _to_reranked(chunk: RetrievedChunk, score: float, position: int) -> RerankedChunk:
    return RerankedChunk(**chunk.model_dump(), relevance_score=score, rerank_position=position)


class PassthroughReranker:
    """Keeps RRF order. Used when no re-rank provider is configured (degraded mode)."""

    name = "none"

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RerankedChunk]:
        ordered = sorted(candidates, key=lambda c: (-c.rrf_score, c.id))[:top_n]
        if not ordered:
            return []
        top_score = ordered[0].rrf_score
        return [
            _to_reranked(chunk, chunk.rrf_score / top_score if top_score else 0.0, position)
            for position, chunk in enumerate(ordered, start=1)
        ]


class CohereReranker:
    name = "cohere"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        import cohere

        if client is None and not settings.cohere_api_key:
            raise ConfigurationError("COHERE_API_KEY is required for RERANK_PROVIDER=cohere")
        self.model = settings.cohere_rerank_model
        self._client = client or cohere.AsyncClientV2(api_key=settings.cohere_api_key)

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RerankedChunk]:
        # Explicit submodule imports: `cohere.errors` is only an attribute of the package after a
        # client has been constructed, so attribute access would fail on the client-injection path.
        import httpx
        from cohere.core import ApiError
        from cohere.errors import TooManyRequestsError

        if not candidates:
            return []
        documents = [chunk.contextualized_content for chunk in candidates]
        try:
            response = await self._client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
        except TooManyRequestsError as exc:
            raise ProviderError("cohere-rerank", "rate limited", retryable=True) from exc
        except ApiError as exc:
            status = getattr(exc, "status_code", None)
            retryable = status is not None and status >= 500
            raise ProviderError(
                "cohere-rerank", f"HTTP {status}: {getattr(exc, 'body', exc)}", retryable=retryable
            ) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # The Cohere SDK re-raises transport errors bare once its retries are exhausted.
            raise ProviderError(
                "cohere-rerank", f"network error: {type(exc).__name__}", retryable=True
            ) from exc

        reranked: list[RerankedChunk] = []
        for position, item in enumerate(response.results, start=1):
            if item.index < 0 or item.index >= len(candidates):
                logger.warning("Cohere returned out-of-range index %s; skipping", item.index)
                continue
            reranked.append(
                _to_reranked(candidates[item.index], float(item.relevance_score), position)
            )
        return reranked[:top_n]


class FastEmbedReranker:
    """Local cross-encoder re-ranking (ONNX via fastembed). Free, offline after the first download.

    ``Xenova/ms-marco-MiniLM-L-6-v2`` (80 MB) is the default; ``BAAI/bge-reranker-v2-m3-int8``
    is stronger and multilingual at 570 MB. Scores are raw cross-encoder logits, higher is better.
    """

    name = "fastembed"

    def __init__(
        self,
        settings: Settings,
        encoder: object | None = None,
        *,
        gate: asyncio.Semaphore | None = None,
    ) -> None:
        self.model = settings.fastembed_rerank_model
        if encoder is None:
            try:
                import fastembed.rerank.cross_encoder  # noqa: F401 - fail fast when missing
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ConfigurationError(
                    "pip install -r requirements-local-models.txt to use RERANK_PROVIDER=fastembed"
                ) from exc
        # Loaded on first use: keeps startup fast and memory low until a query arrives.
        self._encoder = encoder
        self._load_lock = asyncio.Lock()
        self._gate = gate or local_inference_gate(settings)
        self._batch_size = settings.fastembed_rerank_batch_size
        self._model_kwargs = local_model_kwargs(settings)

    async def _get_encoder(self) -> object:
        if self._encoder is None:
            async with self._load_lock:
                if self._encoder is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    logger.info("Loading local cross-encoder %s", self.model)
                    async with self._gate:
                        self._encoder = await asyncio.to_thread(
                            lambda: TextCrossEncoder(model_name=self.model, **self._model_kwargs)
                        )
        return self._encoder

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RerankedChunk]:
        if not candidates:
            return []
        encoder = await self._get_encoder()
        documents = [chunk.contextualized_content for chunk in candidates]
        async with self._gate:
            scores = await asyncio.to_thread(
                lambda: [
                    float(s) for s in encoder.rerank(query, documents, batch_size=self._batch_size)
                ]
            )
        if len(scores) != len(candidates):
            raise ProviderError(
                "fastembed-rerank", f"got {len(scores)} scores for {len(candidates)} documents"
            )
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].id))
        return [
            _to_reranked(candidates[i], scores[i], position)
            for position, i in enumerate(order[:top_n], start=1)
        ]


def build_reranker(settings: Settings, *, gate: asyncio.Semaphore | None = None) -> Reranker:
    if settings.rerank_provider == "cohere":
        return CohereReranker(settings)
    if settings.rerank_provider == "fastembed":
        return FastEmbedReranker(settings, gate=gate)
    logger.warning(
        "RERANK_PROVIDER=none: candidates keep RRF order without cross-encoder re-ranking"
    )
    return PassthroughReranker()
