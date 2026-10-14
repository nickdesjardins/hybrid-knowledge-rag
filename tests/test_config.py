"""Settings validation: env parsing quirks and cross-field rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_match_the_spec() -> None:
    settings = _settings()
    assert settings.rrf_k == 60
    assert settings.top_k_retrieval == 50
    assert settings.top_k_rerank == 3
    assert settings.hnsw_m == 16 and settings.hnsw_ef_construction == 64
    assert settings.embedding_dimensions == 1536
    assert settings.generation_model == "claude-opus-5-5"
    assert settings.contextualizer_model == "claude-haiku-5-5"
    assert settings.cohere_rerank_model == "rerank-v3.5"
    assert settings.database_url.startswith("postgresql+asyncpg://")


@pytest.mark.parametrize("raw", ["", "  ", "none", "null"])
def test_empty_temperature_and_api_key_mean_unset(raw: str) -> None:
    settings = _settings(generation_temperature=raw, api_key=raw, openrouter_api_key=raw)
    assert settings.generation_temperature is None
    assert settings.api_key is None
    assert settings.openrouter_api_key is None


def test_api_key_is_inbound_auth_not_a_provider_key() -> None:
    settings = _settings(api_key="shared-secret", llm_provider="openrouter")
    assert settings.api_key is not None
    assert settings.openrouter_configured is False  # API_KEY never configures a provider
    assert settings.generator_configured is False


def test_temperature_is_parsed_and_bounded() -> None:
    assert _settings(generation_temperature="0.2").generation_temperature == 0.2
    with pytest.raises(ValidationError):
        _settings(generation_temperature="3")


def test_log_level_is_case_insensitive_but_validated() -> None:
    assert _settings(log_level="debug").log_level == "DEBUG"
    assert _settings(log_level=" warning ").log_level == "WARNING"
    with pytest.raises(ValidationError):
        _settings(log_level="verbose")


def test_cross_field_rules() -> None:
    with pytest.raises(ValidationError, match="TOP_K_RERANK"):
        _settings(top_k_rerank=10, top_k_retrieval=5)
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        _settings(chunk_size_tokens=100, chunk_overlap_tokens=100)
    with pytest.raises(ValidationError, match="HNSW_EF_CONSTRUCTION"):
        _settings(hnsw_m=40, hnsw_ef_construction=64)
    assert _settings(hnsw_m=32, hnsw_ef_construction=64).hnsw_m == 32


def test_generation_max_tokens_is_capped_for_non_streaming_requests() -> None:
    with pytest.raises(ValidationError):
        _settings(generation_max_tokens=100_000)
    assert _settings(generation_max_tokens=16_000).generation_max_tokens == 16_000


def test_effective_ef_search_never_below_candidate_limit() -> None:
    assert _settings(hnsw_ef_search=20, top_k_retrieval=50).effective_hnsw_ef_search == 50
    assert _settings(hnsw_ef_search=200, top_k_retrieval=50).effective_hnsw_ef_search == 200


def test_provider_configuration_flags() -> None:
    settings = _settings(
        anthropic_api_key=None,
        openai_api_key=None,
        cohere_api_key=None,
        embedding_provider="openai",
        rerank_provider="cohere",
    )
    assert not settings.generator_configured
    assert not settings.embedder_configured
    assert not settings.reranker_configured
    configured = _settings(
        anthropic_api_key="a", openai_api_key="o", cohere_api_key="c", embedding_provider="openai"
    )
    assert configured.generator_configured and configured.embedder_configured
    assert configured.reranker_configured
    assert _settings(llm_provider="openai", openai_api_key="o").generator_configured
