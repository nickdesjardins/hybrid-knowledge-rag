"""Adapter tests: the real provider classes driven by stub SDK clients (no network).

These exercise the request shapes sent to Anthropic / OpenAI / Cohere and the mapping of SDK
responses and exceptions onto the engine's own contracts (``GenerationResult``, ``ProviderError``,
heuristic fallbacks).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import openai
import pytest
from cohere.core import ApiError
from cohere.errors import TooManyRequestsError

from app.config import Settings
from app.errors import ProviderError
from app.generator.grounded_llm import (
    AnthropicGroundedGenerator,
    LLMGroundedOutput,
    OpenAIGroundedGenerator,
)
from app.ingestion.chunker import Chunk
from app.ingestion.contextualizer import AnthropicContextualizer
from app.ingestion.pdf_parser import PageText, ParsedDocument
from app.retrieval.embedder import OpenAIEmbedder
from app.retrieval.reranker import CohereReranker
from app.schemas import RerankedChunk, RetrievedChunk

# --------------------------------------------------------------------------------- helpers


def _settings(**overrides) -> Settings:
    base = {
        "embedding_provider": "hashing",
        "rerank_provider": "none",
        "contextualizer_enabled": False,
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _retrieved(i: int, text: str | None = None) -> RetrievedChunk:
    body = text or f"Chunk {i} body text."
    return RetrievedChunk(
        id=f"c{i}",
        document_id="d1",
        document_title="Doc",
        chunk_index=i,
        page_number=i + 1,
        raw_content=body,
        contextualized_content=f"Context {i}.\n\n{body}",
        rrf_score=1.0 / (61 + i),
        dense_rank=i + 1,
        sparse_rank=None,
    )


def _reranked(i: int, text: str | None = None) -> RerankedChunk:
    return RerankedChunk(
        **_retrieved(i, text).model_dump(), relevance_score=0.9 - i * 0.1, rerank_position=i + 1
    )


def _anthropic_message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    thinking = SimpleNamespace(type="thinking", thinking="")
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[thinking, block], stop_reason=stop_reason, stop_details=None)


def _anthropic_status_error(status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.test/v1"))
    cls = {
        400: anthropic.BadRequestError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
    }[status]
    return cls(f"status {status}", response=response, body=None)


class StubAnthropic:
    """Mimics ``AsyncAnthropic().messages.create``; outcomes are responses or exceptions."""

    def __init__(self, outcomes: list, delay: float = 0.0) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.events: list[tuple[str, int]] = []
        self.delay = delay
        self.messages = self

    async def create(self, **kwargs):
        index = len(self.calls) + 1
        self.calls.append(kwargs)
        self.events.append(("start", index))
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.outcomes.pop(0) if self.outcomes else _anthropic_message("Context.")
        self.events.append(("end", index))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


GOOD_JSON = json.dumps(
    {
        "answer": "Seven years.",
        "is_grounded": True,
        "confidence_score": 0.9,
        "citations": [{"chunk_id": "c0", "verbatim_quote": "Chunk 0 body text."}],
    }
)


# --------------------------------------------------------------------- Anthropic generator


async def test_anthropic_generator_sends_structured_output_without_sampling_params() -> None:
    client = StubAnthropic([_anthropic_message(GOOD_JSON)])
    generator = AnthropicGroundedGenerator(_settings(generation_effort="medium"), client=client)
    result = await generator.generate("How long?", [_reranked(0), _reranked(1)])

    assert result.refused is False
    assert result.model == "claude-opus-5-5"
    assert result.output.answer == "Seven years."
    assert result.output.citations[0].chunk_id == "c0"

    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 4096
    assert "temperature" not in call and "top_p" not in call and "thinking" not in call
    assert call["output_config"]["effort"] == "medium"
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["type"] == "object"
    assert set(fmt["schema"]["required"]) == {
        "answer",
        "is_grounded",
        "confidence_score",
        "citations",
    }
    assert fmt["schema"]["additionalProperties"] is False
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    user_text = call["messages"][0]["content"]
    assert call["messages"][0]["role"] == "user"
    assert "<passages-" in user_text and "<question-" in user_text
    assert "Chunk 0 body text." in user_text


async def test_anthropic_generator_refusal_and_truncation_are_detected_before_parsing() -> None:
    refused = StubAnthropic([_anthropic_message("", stop_reason="refusal")])
    result = await AnthropicGroundedGenerator(_settings(), client=refused).generate(
        "q?", [_reranked(0)]
    )
    assert result.refused is True
    assert result.output.is_grounded is False and result.output.citations == []

    truncated = StubAnthropic([_anthropic_message('{"answer": "Sev', stop_reason="max_tokens")])
    with pytest.raises(ProviderError) as info:
        await AnthropicGroundedGenerator(_settings(), client=truncated).generate(
            "q?", [_reranked(0)]
        )
    assert info.value.retryable is False
    assert "GENERATION_MAX_TOKENS" in info.value.message


async def test_anthropic_generator_rejects_invalid_json() -> None:
    client = StubAnthropic([_anthropic_message('{"answer": "x"}')])  # missing required fields
    with pytest.raises(ProviderError) as info:
        await AnthropicGroundedGenerator(_settings(), client=client).generate("q?", [_reranked(0)])
    assert "failed validation" in info.value.message
    assert "stop_reason=end_turn" in info.value.message


@pytest.mark.parametrize(
    "error, retryable",
    [
        (_anthropic_status_error(429), True),
        (_anthropic_status_error(500), True),
        (_anthropic_status_error(400), False),
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.test")), True),
    ],
)
async def test_anthropic_generator_maps_sdk_errors(error: Exception, retryable: bool) -> None:
    client = StubAnthropic([error])
    with pytest.raises(ProviderError) as info:
        await AnthropicGroundedGenerator(_settings(), client=client).generate("q?", [_reranked(0)])
    assert info.value.provider == "anthropic"
    assert info.value.retryable is retryable


# ------------------------------------------------------------------------- contextualizer


def _document(n_chunks: int) -> tuple[ParsedDocument, list[Chunk]]:
    pages = [PageText(page_number=i + 1, text=f"Page {i + 1} text.") for i in range(n_chunks)]
    document = ParsedDocument(title="SOP-1", filename="sop.pdf", pages=pages)
    chunks = [
        Chunk(chunk_index=i, page_number=i + 1, text=f"Page {i + 1} text.", token_count=4)
        for i in range(n_chunks)
    ]
    return document, chunks


async def test_contextualizer_warms_cache_with_first_chunk_before_fanning_out() -> None:
    client = StubAnthropic([], delay=0.01)
    contextualizer = AnthropicContextualizer(_settings(contextualizer_concurrency=8), client=client)
    document, chunks = _document(4)
    contexts = await contextualizer.contextualize(document, chunks)

    assert contexts == ["Context."] * 4
    assert client.events[0] == ("start", 1)
    assert client.events[1] == ("end", 1), "first request must finish before the fan-out starts"
    assert len(client.calls) == 4


async def test_contextualizer_request_shape_uses_cache_control_and_nonce_tags() -> None:
    client = StubAnthropic([_anthropic_message('"Situated context."')])
    settings = _settings(contextualizer_max_tokens=512)
    contextualizer = AnthropicContextualizer(settings, client=client)
    document, chunks = _document(1)
    contexts = await contextualizer.contextualize(document, chunks)

    assert contexts == ["Situated context."]  # surrounding quotes are stripped
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == 512
    assert call["output_config"] == {"effort": "low"}
    assert "temperature" not in call
    document_block, chunk_block = call["messages"][0]["content"]
    assert document_block["cache_control"] == {"type": "ephemeral"}
    assert document_block["text"].startswith("<document-")
    nonce = document_block["text"][len("<document-") : len("<document-") + 8]
    assert f"</document-{nonce}>" in document_block["text"]
    assert f"<chunk-{nonce}>" in chunk_block["text"]
    assert "Page 1 text." in chunk_block["text"]
    assert "untrusted" in call["system"]


async def test_contextualizer_falls_back_to_heuristic_on_errors_and_truncation() -> None:
    client = StubAnthropic(
        [
            _anthropic_message("Good context."),
            _anthropic_status_error(500),
            _anthropic_message("Cut off mid", stop_reason="max_tokens"),
            _anthropic_message("", stop_reason="refusal"),
            _anthropic_message("   "),
        ]
    )
    contextualizer = AnthropicContextualizer(_settings(contextualizer_concurrency=1), client=client)
    document, chunks = _document(5)
    contexts = await contextualizer.contextualize(document, chunks)

    assert contexts[0] == "Good context."
    for index in range(1, 5):
        assert contexts[index] == f"Document: SOP-1. Page {index + 1} of 5."


# --------------------------------------------------------------------------- Cohere rerank


class StubCohere:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    async def rerank(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _cohere_results(*pairs: tuple[int, float]) -> SimpleNamespace:
    return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in pairs])


async def test_cohere_reranker_maps_results_positions_and_scores() -> None:
    client = StubCohere(_cohere_results((2, 0.91), (0, 0.42), (9, 0.1)))
    reranker = CohereReranker(_settings(cohere_rerank_model="rerank-v3.5"), client=client)
    candidates = [_retrieved(i) for i in range(4)]
    ranked = await reranker.rerank("query", candidates, top_n=3)

    assert [c.id for c in ranked] == ["c2", "c0"]  # out-of-range index 9 is skipped
    assert [c.rerank_position for c in ranked] == [1, 2]
    assert [c.relevance_score for c in ranked] == [0.91, 0.42]
    call = client.calls[0]
    assert call["model"] == "rerank-v3.5"
    assert call["query"] == "query"
    assert call["top_n"] == 3
    assert call["documents"] == [c.contextualized_content for c in candidates]


async def test_cohere_reranker_top_n_never_exceeds_candidate_count_and_skips_empty() -> None:
    client = StubCohere(_cohere_results((0, 0.5)))
    reranker = CohereReranker(_settings(), client=client)
    assert await reranker.rerank("q", [], top_n=3) == []
    assert client.calls == []
    await reranker.rerank("q", [_retrieved(0)], top_n=50)
    assert client.calls[0]["top_n"] == 1


@pytest.mark.parametrize(
    "error, retryable",
    [
        (TooManyRequestsError(body={"message": "slow down"}), True),
        (ApiError(status_code=503, body="unavailable"), True),
        (ApiError(status_code=400, body="bad request"), False),
        (httpx.ReadTimeout("timed out"), True),
        (httpx.ConnectError("refused"), True),
    ],
)
async def test_cohere_reranker_maps_sdk_and_transport_errors(
    error: Exception, retryable: bool
) -> None:
    reranker = CohereReranker(_settings(), client=StubCohere(error))
    with pytest.raises(ProviderError) as info:
        await reranker.rerank("q", [_retrieved(0)], top_n=1)
    assert info.value.provider == "cohere-rerank"
    assert info.value.retryable is retryable


# ------------------------------------------------------------------------ OpenAI generator


class StubOpenAIChat:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _openai_completion(parsed, refusal=None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=refusal))]
    )


async def test_openai_generator_sends_temperature_and_schema() -> None:
    parsed = LLMGroundedOutput.model_validate_json(GOOD_JSON)
    client = StubOpenAIChat(_openai_completion(parsed))
    generator = OpenAIGroundedGenerator(
        _settings(llm_provider="openai", openai_generation_model="gpt-4.1"), client=client
    )
    result = await generator.generate("q?", [_reranked(0)])
    assert result.output is parsed and result.refused is False and result.model == "gpt-4.1"
    call = client.calls[0]
    assert call["temperature"] == 0.0
    assert call["response_format"] is LLMGroundedOutput
    assert call["max_completion_tokens"] == 4096
    assert call["messages"][0]["role"] == "system"
    assert "<passages-" in call["messages"][1]["content"]


async def test_openai_generator_omits_temperature_when_unset() -> None:
    parsed = LLMGroundedOutput.model_validate_json(GOOD_JSON)
    client = StubOpenAIChat(_openai_completion(parsed))
    generator = OpenAIGroundedGenerator(
        _settings(llm_provider="openai", generation_temperature=None), client=client
    )
    await generator.generate("q?", [_reranked(0)])
    assert "temperature" not in client.calls[0]


async def test_openai_generator_refusal_paths() -> None:
    client = StubOpenAIChat(_openai_completion(None, refusal="I cannot help with that."))
    result = await OpenAIGroundedGenerator(
        _settings(llm_provider="openai"), client=client
    ).generate("q?", [_reranked(0)])
    assert result.refused is True

    filtered = StubOpenAIChat(openai.ContentFilterFinishReasonError())
    result = await OpenAIGroundedGenerator(
        _settings(llm_provider="openai"), client=filtered
    ).generate("q?", [_reranked(0)])
    assert result.refused is True and "content filter" in result.output.answer


async def test_openai_generator_length_error_and_status_errors() -> None:
    truncated = StubOpenAIChat(
        openai.LengthFinishReasonError(completion=SimpleNamespace(usage=None))
    )
    with pytest.raises(ProviderError) as info:
        await OpenAIGroundedGenerator(_settings(llm_provider="openai"), client=truncated).generate(
            "q?", [_reranked(0)]
        )
    assert "GENERATION_MAX_TOKENS" in info.value.message and info.value.retryable is False

    response = httpx2.Response(429, request=httpx2.Request("POST", "https://api.test"))
    limited = StubOpenAIChat(openai.RateLimitError("rl", response=response, body=None))
    with pytest.raises(ProviderError) as info:
        await OpenAIGroundedGenerator(_settings(llm_provider="openai"), client=limited).generate(
            "q?", [_reranked(0)]
        )
    assert info.value.retryable is True


# ------------------------------------------------------------------------- OpenAI embedder


class StubOpenAIEmbeddings:
    def __init__(self, dims: int = 8, outcome: Exception | None = None) -> None:
        self.dims = dims
        self.outcome = outcome
        self.calls: list[dict] = []
        self.embeddings = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.outcome is not None:
            raise self.outcome
        texts = kwargs["input"]
        # Return items deliberately out of order to prove the adapter sorts by index.
        items = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dims) for i in range(len(texts))
        ]
        return SimpleNamespace(data=list(reversed(items)))


async def test_openai_embedder_batches_and_restores_order() -> None:
    client = StubOpenAIEmbeddings(dims=8)
    embedder = OpenAIEmbedder(
        _settings(embedding_provider="openai", embedding_dimensions=8, embedding_batch_size=2),
        client=client,
    )
    vectors = await embedder.embed_documents([f"t{i}" for i in range(5)])
    assert len(vectors) == 5
    assert [v[0] for v in vectors] == [0.0, 1.0, 0.0, 1.0, 0.0]  # per-batch index order restored
    assert len(client.calls) == 3
    assert [len(c["input"]) for c in client.calls] == [2, 2, 1]
    assert all(c["dimensions"] == 8 for c in client.calls)
    assert all(c["model"] == "text-embedding-3-small" for c in client.calls)
    assert await embedder.embed_query("q") == [0.0] * 8
    assert await embedder.embed_documents([]) == []


async def test_openai_embedder_omits_dimensions_for_legacy_models_and_checks_width() -> None:
    client = StubOpenAIEmbeddings(dims=8)
    legacy = OpenAIEmbedder(
        _settings(
            embedding_provider="openai",
            embedding_model="text-embedding-ada-002",
            embedding_dimensions=8,
        ),
        client=client,
    )
    await legacy.embed_query("q")
    assert "dimensions" not in client.calls[0]

    mismatched = OpenAIEmbedder(
        _settings(embedding_provider="openai", embedding_dimensions=16),
        client=StubOpenAIEmbeddings(dims=8),
    )
    with pytest.raises(ProviderError) as info:
        await mismatched.embed_query("q")
    assert "EMBEDDING_DIMENSIONS=16" in info.value.message


@pytest.mark.parametrize(
    "error, retryable",
    [
        (
            openai.RateLimitError(
                "rl",
                response=httpx2.Response(429, request=httpx2.Request("POST", "https://api.test")),
                body=None,
            ),
            True,
        ),
        (
            openai.BadRequestError(
                "bad",
                response=httpx2.Response(400, request=httpx2.Request("POST", "https://api.test")),
                body=None,
            ),
            False,
        ),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://api.test")), True),
    ],
)
async def test_openai_embedder_maps_sdk_errors(error: Exception, retryable: bool) -> None:
    embedder = OpenAIEmbedder(
        _settings(embedding_provider="openai", embedding_dimensions=8),
        client=StubOpenAIEmbeddings(dims=8, outcome=error),
    )
    with pytest.raises(ProviderError) as info:
        await embedder.embed_query("q")
    assert info.value.provider == "openai-embeddings"
    assert info.value.retryable is retryable
