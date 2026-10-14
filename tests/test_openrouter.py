"""Zero-cost stack: OpenRouter chain fallback, local re-ranking, local/OpenRouter embeddings.

Everything runs against stub clients; no network, no model downloads.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import ValidationError

from app.config import Settings, parse_model_chain
from app.errors import ConfigurationError, ProviderError
from app.generator.grounded_llm import (
    LLMGroundedOutput,
    OpenRouterGroundedGenerator,
    build_generator,
)
from app.ingestion.chunker import Chunk
from app.ingestion.contextualizer import (
    HeuristicContextualizer,
    OpenRouterContextualizer,
    build_contextualizer,
)
from app.ingestion.pdf_parser import PageText, ParsedDocument
from app.openrouter import (
    OpenRouterChat,
    RateLimiter,
    extract_json_object,
    strict_json_schema,
)
from app.retrieval.embedder import FastEmbedEmbedder, OpenRouterEmbedder, build_embedder
from app.retrieval.reranker import FastEmbedReranker
from app.schemas import RerankedChunk, RetrievedChunk

CHAIN = "vendor/alpha:free,vendor/beta:free"


def _settings(**overrides) -> Settings:
    base = {
        "llm_provider": "openrouter",
        "openrouter_api_key": "sk-or-test",
        "openrouter_models": CHAIN,
        "openrouter_contextualizer_models": CHAIN,
        "openrouter_requests_per_minute": 0,
        "embedding_provider": "hashing",
        "rerank_provider": "none",
        "contextualizer_enabled": True,
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(_env_file=None, _env_prefix="HERMETIC_TEST_", **base)


GOOD = {
    "answer": "Seven years.",
    "is_grounded": True,
    "confidence_score": 0.9,
    "citations": [{"chunk_id": "c0", "verbatim_quote": "Chunk 0 body text."}],
}


def _completion(content: str | None, finish: str = "stop", error: dict | None = None):
    message = SimpleNamespace(content=content)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])
    if error is not None:
        completion.error = error
        completion.choices = []
    return completion


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://or.test/v1"))
    cls = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
        502: openai.InternalServerError,
    }[status]
    return cls(f"status {status}", response=response, body={"error": {"message": f"s{status}"}})


class StubChat:
    """Stub for ``AsyncOpenAI().chat.completions.create``: outcomes consumed in order."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _chunk(i: int) -> RerankedChunk:
    body = f"Chunk {i} body text."
    return RerankedChunk(
        id=f"c{i}",
        document_id="d1",
        document_title="Doc",
        chunk_index=i,
        page_number=i + 1,
        raw_content=body,
        contextualized_content=f"Context {i}.\n\n{body}",
        rrf_score=0.03,
        relevance_score=0.9,
        rerank_position=i + 1,
    )


# ------------------------------------------------------------------------ pure helpers


def test_parse_model_chain_trims_and_dedupes() -> None:
    assert parse_model_chain(" a:free , b:free,a:free,, ") == ["a:free", "b:free"]
    assert parse_model_chain("") == []


def test_strict_schema_requires_every_property_everywhere() -> None:
    schema = strict_json_schema(LLMGroundedOutput)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "is_grounded", "confidence_score", "citations"}
    citation = schema["$defs"]["LLMCitation"]
    assert citation["additionalProperties"] is False
    assert set(citation["required"]) == {"chunk_id", "verbatim_quote"}


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Sure! Here is the answer:\n" + json.dumps(GOOD) + "\nHope this helps.",
        "[" + json.dumps(GOOD) + "]",
    ],
)
def test_extract_json_object_tolerates_fences_prose_and_lists(raw: str) -> None:
    assert extract_json_object(raw) == GOOD


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[1, 2]", "42", None])
def test_extract_json_object_rejects_garbage(raw) -> None:
    with pytest.raises(ValueError):
        extract_json_object(raw)


async def test_rate_limiter_spaces_request_starts() -> None:
    limiter = RateLimiter(requests_per_minute=600)  # 100 ms apart
    started = time.perf_counter()
    for _ in range(3):
        await limiter.wait()
    assert time.perf_counter() - started >= 0.19
    unlimited = RateLimiter(requests_per_minute=0)
    started = time.perf_counter()
    await asyncio.gather(*(unlimited.wait() for _ in range(50)))
    assert time.perf_counter() - started < 0.1


# ------------------------------------------------------------------------- chain walker


async def test_chain_uses_strict_schema_and_reports_the_answering_model() -> None:
    client = StubChat([_completion(json.dumps(GOOD))])
    chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
    outcome = await chat.complete(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        max_tokens=512,
        temperature=0.0,
        schema=strict_json_schema(LLMGroundedOutput),
        schema_name="grounded_answer",
        validate=lambda text: LLMGroundedOutput.model_validate(extract_json_object(text)),
    )
    assert outcome.model == "vendor/alpha:free"
    assert outcome.parsed.answer == "Seven years."
    assert outcome.json_mode is False
    call = client.calls[0]
    assert call["model"] == "vendor/alpha:free"
    assert call["temperature"] == 0.0 and call["max_tokens"] == 512 and call["stream"] is False
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["name"] == "grounded_answer"
    assert call["messages"][0]["content"] == "sys"  # no schema suffix in strict mode


async def test_chain_falls_back_to_next_model_on_429_in_band_errors_and_truncation() -> None:
    client = StubChat(
        [
            _status_error(429),
            _completion(json.dumps(GOOD)),
        ]
    )
    chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
    outcome = await chat.complete(
        messages=[{"role": "user", "content": "q"}], max_tokens=64, schema={"type": "object"}
    )
    assert outcome.model == "vendor/beta:free"
    assert [c["model"] for c in client.calls] == ["vendor/alpha:free", "vendor/beta:free"]

    for first in (
        _completion(None, error={"code": 503, "message": "upstream overloaded"}),
        _completion("partial", finish="length"),
        _status_error(404),
        _status_error(502),
        _completion("   "),
        openai.APIConnectionError(request=httpx2.Request("POST", "https://or.test")),
    ):
        client = StubChat([first, _completion("ok")])
        chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
        outcome = await chat.complete(messages=[{"role": "user", "content": "q"}], max_tokens=64)
        assert outcome.model == "vendor/beta:free", first


async def test_chain_retries_same_model_in_json_mode_when_strict_schema_is_rejected() -> None:
    rejected = openai.BadRequestError(
        "bad",
        response=httpx2.Response(400, request=httpx2.Request("POST", "https://or.test")),
        body={"error": {"message": "response_format json_schema is not supported by this model"}},
    )
    client = StubChat([rejected, _completion("```json\n" + json.dumps(GOOD) + "\n```")])
    chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
    outcome = await chat.complete(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        max_tokens=64,
        schema={"type": "object"},
        json_mode_system_suffix="\n\nSCHEMA HERE",
        validate=extract_json_object,
    )
    assert outcome.model == "vendor/alpha:free" and outcome.json_mode is True
    assert outcome.parsed == GOOD
    assert len(client.calls) == 2
    assert client.calls[1]["response_format"] == {"type": "json_object"}
    assert client.calls[1]["messages"][0]["content"].endswith("SCHEMA HERE")
    assert client.calls[0]["messages"][0]["content"] == "sys"  # the original list is untouched


async def test_chain_cascades_on_invalid_output_and_fails_when_every_model_fails() -> None:
    client = StubChat([_completion("not json at all"), _completion('{"answer": 1}')])
    chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
    with pytest.raises(ProviderError) as info:
        await chat.complete(
            messages=[{"role": "user", "content": "q"}],
            max_tokens=64,
            schema={"type": "object"},
            validate=lambda text: LLMGroundedOutput.model_validate(extract_json_object(text)),
        )
    assert info.value.provider == "openrouter"
    assert "vendor/alpha:free" in info.value.message and "vendor/beta:free" in info.value.message
    assert info.value.retryable is False

    client = StubChat([_status_error(429), _status_error(502)])
    chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
    with pytest.raises(ProviderError) as info:
        await chat.complete(messages=[{"role": "user", "content": "q"}], max_tokens=64)
    assert info.value.retryable is True


async def test_chain_stops_immediately_on_a_rejected_key() -> None:
    client = StubChat([_status_error(401), _completion("never reached")])
    chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
    with pytest.raises(ProviderError) as info:
        await chat.complete(messages=[{"role": "user", "content": "q"}], max_tokens=64)
    assert info.value.provider == "openrouter-auth" and info.value.retryable is False
    assert len(client.calls) == 1


async def test_chain_reports_content_filter_as_refusal() -> None:
    client = StubChat([_completion("", finish="content_filter")])
    chat = OpenRouterChat(_settings(), models=parse_model_chain(CHAIN), client=client)
    outcome = await chat.complete(messages=[{"role": "user", "content": "q"}], max_tokens=64)
    assert outcome.refused is True and outcome.model == "vendor/alpha:free"


def test_chain_requires_a_key_unless_a_client_is_injected() -> None:
    with pytest.raises(ConfigurationError):
        OpenRouterChat(_settings(openrouter_api_key=None), models=["a:free"])
    chat = OpenRouterChat(
        _settings(openrouter_api_key=None), models=["a:free"], client=StubChat([])
    )
    assert chat.models == ["a:free"]


# --------------------------------------------------------------------------- generator


async def test_openrouter_generator_end_to_end_with_fallback() -> None:
    client = StubChat([_status_error(429), _completion(json.dumps(GOOD))])
    generator = OpenRouterGroundedGenerator(_settings(), client=client)
    assert generator.model == CHAIN
    result = await generator.generate("How long?", [_chunk(0)])
    assert result.model == "vendor/beta:free"
    assert result.refused is False
    assert result.output.citations[0].verbatim_quote == "Chunk 0 body text."
    user_message = client.calls[1]["messages"][1]["content"]
    assert "<passages-" in user_message and "Chunk 0 body text." in user_message
    assert (
        client.calls[1]["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    )


async def test_openrouter_generator_refusal_and_total_failure() -> None:
    refused = OpenRouterGroundedGenerator(
        _settings(), client=StubChat([_completion("", finish="content_filter")])
    )
    result = await refused.generate("q?", [_chunk(0)])
    assert result.refused is True and result.output.is_grounded is False

    broken = OpenRouterGroundedGenerator(
        _settings(), client=StubChat([_completion("nope"), _completion("nope")])
    )
    with pytest.raises(ProviderError):
        await broken.generate("q?", [_chunk(0)])


def test_build_generator_selects_openrouter() -> None:
    assert isinstance(build_generator(_settings()), OpenRouterGroundedGenerator)


# ---------------------------------------------------------------------- contextualizer


def _document(n: int) -> tuple[ParsedDocument, list[Chunk]]:
    pages = [PageText(page_number=i + 1, text=f"Page {i + 1} text.") for i in range(n)]
    document = ParsedDocument(title="SOP-1", filename="sop.pdf", pages=pages)
    chunks = [
        Chunk(chunk_index=i, page_number=i + 1, text=f"Page {i + 1} text.", token_count=4)
        for i in range(n)
    ]
    return document, chunks


async def test_openrouter_contextualizer_contexts_fallbacks_and_cap() -> None:
    client = StubChat(
        [
            _completion('"Situated context one."'),
            _status_error(429),  # alpha fails for chunk 2 ...
            _completion("Context two via beta."),  # ... beta answers it
            _completion("", finish="content_filter"),
        ]
    )
    settings = _settings(contextualizer_concurrency=1)
    contextualizer = OpenRouterContextualizer(settings, client=client)
    assert contextualizer.model == CHAIN
    document, chunks = _document(3)
    contexts = await contextualizer.contextualize(document, chunks)
    assert contexts == [
        "Situated context one.",
        "Context two via beta.",
        "Document: SOP-1. Page 3 of 3.",
    ]
    first = client.calls[0]
    assert first["temperature"] == 0.0 and "response_format" not in first
    assert "untrusted" in first["messages"][0]["content"]
    assert "<document-" in first["messages"][1]["content"]
    assert "Page 1 text." in first["messages"][1]["content"]

    capped = OpenRouterContextualizer(
        _settings(contextualizer_max_llm_calls_per_document=2), client=StubChat([])
    )
    document, chunks = _document(3)
    assert await capped.contextualize(document, chunks) == [
        f"Document: SOP-1. Page {i} of 3." for i in (1, 2, 3)
    ]


def test_build_contextualizer_resolution() -> None:
    assert isinstance(build_contextualizer(_settings()), OpenRouterContextualizer)
    assert isinstance(
        build_contextualizer(_settings(contextualizer_provider="heuristic")),
        HeuristicContextualizer,
    )
    assert isinstance(
        build_contextualizer(_settings(openrouter_api_key=None)), HeuristicContextualizer
    )
    assert isinstance(
        build_contextualizer(_settings(contextualizer_enabled=False)), HeuristicContextualizer
    )
    assert (
        _settings(llm_provider="anthropic", anthropic_api_key="a").resolved_contextualizer_provider
        == "anthropic"
    )
    assert _settings(llm_provider="anthropic").resolved_contextualizer_provider == "heuristic"


# ------------------------------------------------------------------------- re-ranking


class StubCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents, batch_size: int = 64):
        docs = list(documents)
        self.calls.append((query, docs))
        self.batch_sizes = [*getattr(self, "batch_sizes", []), batch_size]
        return iter(self.scores[: len(docs)])


def _retrieved(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"c{i}",
        document_id="d1",
        document_title="Doc",
        chunk_index=i,
        page_number=i + 1,
        raw_content=f"Chunk {i}.",
        contextualized_content=f"Context {i}.\n\nChunk {i}.",
        rrf_score=1.0 / (61 + i),
        dense_rank=i + 1,
    )


async def test_fastembed_reranker_orders_by_cross_encoder_score() -> None:
    encoder = StubCrossEncoder([0.1, 2.5, -1.0, 2.5])
    reranker = FastEmbedReranker(_settings(rerank_provider="fastembed"), encoder=encoder)
    assert reranker.name == "fastembed" and reranker.model == "Xenova/ms-marco-MiniLM-L-6-v2"
    ranked = await reranker.rerank("query", [_retrieved(i) for i in range(4)], top_n=3)
    assert [c.id for c in ranked] == ["c1", "c3", "c0"]  # ties broken by id
    assert [c.rerank_position for c in ranked] == [1, 2, 3]
    assert ranked[0].relevance_score == 2.5
    assert encoder.calls[0][1] == [f"Context {i}.\n\nChunk {i}." for i in range(4)]
    assert encoder.batch_sizes == [8]  # FASTEMBED_RERANK_BATCH_SIZE default
    assert await reranker.rerank("query", [], top_n=3) == []


def test_local_model_kwargs_low_memory_and_threads() -> None:
    from app.retrieval.embedder import local_model_kwargs

    # Low-memory mode is the default: ONNX's CPU arena off (843 MB -> 385 MB measured).
    assert local_model_kwargs(_settings()) == {"enable_cpu_mem_arena": False}
    tuned = _settings(fastembed_low_memory=False, fastembed_threads=2)
    assert local_model_kwargs(tuned) == {"threads": 2}
    assert _settings(fastembed_threads="").fastembed_threads is None
    custom = FastEmbedReranker(
        _settings(fastembed_rerank_batch_size=4), encoder=StubCrossEncoder([0.1])
    )
    assert custom._batch_size == 4


async def test_fastembed_reranker_rejects_score_count_mismatch() -> None:
    reranker = FastEmbedReranker(_settings(), encoder=StubCrossEncoder([0.5]))
    with pytest.raises(ProviderError):
        await reranker.rerank("q", [_retrieved(0), _retrieved(1)], top_n=2)


async def test_build_reranker_selects_fastembed_and_loads_lazily(monkeypatch) -> None:
    import app.retrieval.reranker as module

    created: list[str] = []
    options: list[dict] = []

    class FakeEncoder:
        def __init__(self, model_name: str, **kwargs) -> None:
            created.append(model_name)
            options.append(kwargs)

        def rerank(self, query: str, documents, batch_size: int = 64):
            return iter([0.5 for _ in documents])

    # Fake the whole package chain: importing "fastembed.rerank.cross_encoder" imports its
    # parents first, so the test must not depend on fastembed being installed.
    import sys
    import types

    for name in ("fastembed", "fastembed.rerank"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    fake_pkg = SimpleNamespace(TextCrossEncoder=FakeEncoder)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", fake_pkg)
    reranker = module.build_reranker(_settings(rerank_provider="fastembed"))
    assert isinstance(reranker, FastEmbedReranker)
    assert created == []  # nothing loaded at startup
    ranked = await reranker.rerank("q", [_retrieved(0)], top_n=1)
    assert created == ["Xenova/ms-marco-MiniLM-L-6-v2"]  # loaded on first use, once
    assert options == [{"enable_cpu_mem_arena": False}]  # low-memory session by default
    await reranker.rerank("q", [_retrieved(0)], top_n=1)
    assert created == ["Xenova/ms-marco-MiniLM-L-6-v2"]
    assert ranked[0].relevance_score == 0.5


# ------------------------------------------------------------------------- embeddings


class StubEmbeddings:
    def __init__(self, dims: int, with_index: bool = True) -> None:
        self.dims = dims
        self.with_index = with_index
        self.calls: list[dict] = []
        self.embeddings = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        items = []
        for i, _ in enumerate(kwargs["input"]):
            item = SimpleNamespace(embedding=[float(i)] * self.dims)
            if self.with_index:
                item.index = i
            items.append(item)
        return SimpleNamespace(data=items)


async def test_openrouter_embedder_uses_openrouter_model_and_tolerates_missing_index() -> None:
    client = StubEmbeddings(dims=4, with_index=False)
    embedder = OpenRouterEmbedder(
        _settings(embedding_provider="openrouter", embedding_dimensions=4), client=client
    )
    assert embedder.name == "openrouter" and embedder.model_name == "openai/text-embedding-3-small"
    vectors = await embedder.embed_documents(["a", "b", "c"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]
    assert client.calls[0]["model"] == "openai/text-embedding-3-small"
    assert client.calls[0]["dimensions"] == 4  # text-embedding-3 family supports truncation
    with pytest.raises(ConfigurationError):
        OpenRouterEmbedder(_settings(embedding_provider="openrouter", openrouter_api_key=None))


async def test_fastembed_embedder_with_injected_model() -> None:
    class FakeModel:
        def embed(self, texts):
            for i, _ in enumerate(texts):
                yield [float(i)] * 8

    embedder = FastEmbedEmbedder(
        _settings(embedding_provider="fastembed", embedding_dimensions=8), model=FakeModel()
    )
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert await embedder.embed_query("q") == [0.0] * 8
    assert len(await embedder.embed_documents(["a", "b"])) == 2
    wrong = FastEmbedEmbedder(
        _settings(embedding_provider="fastembed", embedding_dimensions=16), model=FakeModel()
    )
    with pytest.raises(ConfigurationError):
        await wrong.embed_query("q")


def test_build_embedder_selects_openrouter() -> None:
    embedder = build_embedder(_settings(embedding_provider="openrouter"))
    assert isinstance(embedder, OpenRouterEmbedder)


# ------------------------------------------------------------------------------ config


def test_free_only_guard_rejects_paid_models() -> None:
    with pytest.raises(ValidationError, match="non-free"):
        _settings(openrouter_models="vendor/alpha:free,openai/gpt-4.1")
    paid = _settings(openrouter_models="openai/gpt-4.1", openrouter_free_only=False)
    assert paid.openrouter_model_chain == ["openai/gpt-4.1"]
    with pytest.raises(ValidationError, match="at least one"):
        _settings(openrouter_models=" , ")


def test_provider_configuration_flags_for_openrouter() -> None:
    settings = _settings(embedding_provider="openrouter")
    assert settings.generator_configured and settings.embedder_configured
    unset = _settings(openrouter_api_key="", embedding_provider="openrouter")
    assert unset.openrouter_api_key is None
    assert not unset.generator_configured and not unset.embedder_configured
    assert _settings(rerank_provider="fastembed").reranker_configured
