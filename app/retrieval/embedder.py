"""Embedding clients with batch support.

Three interchangeable providers:

* ``openai``    - ``text-embedding-3-*`` via the official async SDK (default, 1536 dims).
* ``fastembed`` - local ONNX models (no network at query time; ``pip install fastembed``).
* ``hashing``   - deterministic feature-hashing bag-of-words vectors. Not semantic; exists so the
                  whole stack can be exercised offline (tests, demos, CI) without any API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from collections.abc import Sequence
from typing import Protocol

from app.config import Settings
from app.errors import ConfigurationError, ProviderError

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    name: str
    dimensions: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class HashingEmbedder:
    """Feature-hashed unigram+bigram vectors (deterministic, offline, L2-normalised)."""

    name = "hashing"
    model_name = "hashing"

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [t.lower() for t in _WORD_RE.findall(text)]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Hashing a whole document is CPU-bound; keep it off the event loop.
        return await asyncio.to_thread(lambda: [self._embed_one(t) for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAIEmbedder:
    name = "openai"
    provider_label = "openai-embeddings"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        from openai import AsyncOpenAI

        if client is None and not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        self.model = settings.embedding_model
        self.model_name = self.model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        # Bounds the number of in-flight embedding requests for a large ingestion.
        self._semaphore = asyncio.Semaphore(settings.embedding_concurrency)
        # Only the text-embedding-3 family supports Matryoshka truncation via `dimensions`.
        self._send_dimensions = "text-embedding-3" in self.model

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        import openai

        kwargs: dict[str, object] = {"model": self.model, "input": list(texts)}
        if self._send_dimensions:
            kwargs["dimensions"] = self.dimensions
        try:
            async with self._semaphore:
                response = await self._client.embeddings.create(**kwargs)
        except openai.RateLimitError as exc:
            raise ProviderError(self.provider_label, str(exc), retryable=True) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(
                self.provider_label,
                f"HTTP {exc.status_code}: {exc.message}",
                retryable=exc.status_code >= 500,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(self.provider_label, str(exc), retryable=True) from exc

        items = list(response.data)
        # OpenAI numbers each item; some OpenAI-compatible gateways omit `index` (order kept).
        if all(getattr(item, "index", None) is not None for item in items):
            items.sort(key=lambda d: d.index)
        vectors = [list(item.embedding) for item in items]
        if len(vectors) != len(texts):
            raise ProviderError(
                self.provider_label, f"got {len(vectors)} embeddings for {len(texts)} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ProviderError(
                    self.provider_label,
                    f"model returned {len(vector)} dims but EMBEDDING_DIMENSIONS={self.dimensions}",
                )
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        return [vector for batch in results for vector in batch]

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]


class OpenRouterEmbedder(OpenAIEmbedder):
    """Embeddings through OpenRouter's OpenAI-compatible endpoint (paid models only today)."""

    name = "openrouter"
    provider_label = "openrouter-embeddings"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        from openai import AsyncOpenAI

        if client is None and settings.openrouter_api_key is None:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for EMBEDDING_PROVIDER=openrouter"
            )
        self.model = settings.openrouter_embedding_model
        self.model_name = self.model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        headers = (
            {"X-Title": settings.openrouter_app_name} if settings.openrouter_app_name else None
        )
        self._client = client or AsyncOpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),  # type: ignore[union-attr]
            base_url=settings.openrouter_base_url,
            timeout=settings.openrouter_timeout_s,
            default_headers=headers,
        )
        self._semaphore = asyncio.Semaphore(settings.embedding_concurrency)
        self._send_dimensions = "text-embedding-3" in self.model


def local_inference_gate(settings: Settings) -> asyncio.Semaphore:
    """A semaphore bounding local model work (loading and inference) at once.

    ``build_components`` shares one gate between the embedding model and the cross-encoder, so
    a question, the corpus seeding and the warm-up never hold activations at the same time.
    """
    return asyncio.Semaphore(settings.local_inference_concurrency)


def local_model_kwargs(settings: Settings) -> dict[str, object]:
    """ONNX session options shared by the local embedding model and the local cross-encoder."""
    kwargs: dict[str, object] = {}
    if settings.fastembed_low_memory:
        # Without this the CPU arena grows in powers of two and never shrinks.
        kwargs["enable_cpu_mem_arena"] = False
    if settings.fastembed_threads:
        kwargs["threads"] = settings.fastembed_threads
    return kwargs


class FastEmbedEmbedder:
    """Local embeddings via fastembed (ONNX). Model download happens on first use."""

    name = "fastembed"

    def __init__(
        self,
        settings: Settings,
        model: object | None = None,
        *,
        gate: asyncio.Semaphore | None = None,
    ) -> None:
        self.model_name = settings.fastembed_model
        self.dimensions = settings.embedding_dimensions
        if model is None:
            try:
                import fastembed  # noqa: F401 - fail fast when missing
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ConfigurationError(
                    "pip install -r requirements-local-models.txt to use EMBEDDING_PROVIDER=fastembed"
                ) from exc
        # Loaded on first use (worker thread), so startup stays fast on a small host.
        self._model = model
        self._load_lock = asyncio.Lock()
        self._gate = gate or local_inference_gate(settings)
        self._model_kwargs = local_model_kwargs(settings)

    async def _get_model(self) -> object:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    logger.info("Loading local embedding model %s", self.model_name)
                    async with self._gate:
                        self._model = await asyncio.to_thread(
                            lambda: TextEmbedding(model_name=self.model_name, **self._model_kwargs)
                        )
        return self._model

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [list(map(float, v)) for v in self._model.embed(list(texts))]
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ConfigurationError(
                    f"{self.model_name} produces {len(vector)} dims; "
                    "set EMBEDDING_DIMENSIONS accordingly"
                )
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        await self._get_model()
        async with self._gate:
            return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        await self._get_model()
        async with self._gate:
            return (await asyncio.to_thread(self._embed_sync, [text]))[0]


def build_embedder(settings: Settings, *, gate: asyncio.Semaphore | None = None) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embedding_provider == "openrouter":
        return OpenRouterEmbedder(settings)
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbedder(settings, gate=gate)
    logger.warning(
        "EMBEDDING_PROVIDER=hashing is not semantic; use it for offline demos and tests only"
    )
    return HashingEmbedder(settings.embedding_dimensions)
