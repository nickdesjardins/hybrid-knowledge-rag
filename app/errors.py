"""Exception hierarchy shared across the engine.

Keeping provider SDK exceptions out of the API layer lets ``app.main`` map failures to HTTP
status codes without importing every vendor SDK.
"""

from __future__ import annotations


class RetrievalEngineError(Exception):
    """Base class for all engine errors."""


class ConfigurationError(RetrievalEngineError):
    """A required provider or setting is missing or inconsistent."""


class PDFParseError(RetrievalEngineError):
    """The uploaded file could not be parsed into text (corrupt, encrypted, or image-only)."""


class PDFTooLargeError(PDFParseError):
    """The document exceeds a configured ceiling (pages, chunks) and was rejected early."""


class ProviderError(RetrievalEngineError):
    """A remote provider (embeddings, re-ranker, LLM) failed.

    ``retryable`` distinguishes transient failures (rate limits, 5xx, network) from permanent ones
    (bad request, authentication) so the API layer can choose 503 vs 502.
    """

    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message
        self.retryable = retryable
