"""Cross-encoder re-ranking of the fused candidate set (Cohere Rerank v3.5).

Bi-encoders (the embedding model) score query and passage independently. A cross-encoder reads
them *together*, so it can tell "retention period for batch records" apart from "retention of
cleaning validation samples" when both share most of their vocabulary. We run it on the 50 fused
candidates and keep only the top ``top_n`` (3 by default) for generation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from app.config import Settings
from app.errors import ConfigurationError, ProviderError
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


def build_reranker(settings: Settings) -> Reranker:
    if settings.rerank_provider == "cohere":
        return CohereReranker(settings)
    logger.warning(
        "RERANK_PROVIDER=none: candidates keep RRF order without cross-encoder re-ranking"
    )
    return PassthroughReranker()
