"""Environment-driven configuration (pydantic-settings).

Every tunable of the retrieval pipeline lives here so that the SQL, the chunker, and the provider
clients never hard-code magic numbers. Values are read from the process environment first and then
from a local ``.env`` file (see ``.env.example``).

Two ways to run the stack:

* **Paid APIs** (the defaults): Claude for generation and contextualization, OpenAI embeddings,
  Cohere re-ranking.
* **Zero cost**: ``LLM_PROVIDER=openrouter`` with a chain of ``:free`` models (the same pattern as
  the Phase 1 extraction service), ``EMBEDDING_PROVIDER=fastembed`` and ``RERANK_PROVIDER=fastembed``
  for local ONNX models, and PostgreSQL in Docker or through ``pgserver``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Ordered default chain of free OpenRouter models. Inherited from the Phase 1 extraction
#: service, where it was measured on 7 October 2026: the model with the fewest wrong answers
#: first, a fallback from a different upstream provider second. Both support strict JSON-schema
#: output. Free models are withdrawn often; ``scripts/openrouter_smoke.py`` reports which ids
#: the public catalogue still lists.
DEFAULT_OPENROUTER_CHAIN = "nvidia/nemotron-3-super-120b-a12b:free,liquid/lfm-2.5-2.6b:free"


#: Matches docker-compose.yml.
DEFAULT_DATABASE_URL = "postgresql+asyncpg://rag:rag@localhost:5432/rag"


def parse_model_chain(raw: str) -> list[str]:
    """Split an ordered, comma-separated chain of model ids (trimmed, de-duplicated, non-empty)."""
    models: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in models:
            models.append(model)
    return models


def is_free_model(model_id: str) -> bool:
    return model_id.endswith(":free")


#: Query parameters asyncpg does not accept as keyword arguments (Neon/Vercel add them).
_UNSUPPORTED_URL_PARAMS = {"channel_binding"}


#: What often comes along when a connection string is copied from a dashboard or a .env file.
_PASTED_PREFIXES = ("psql ", "database_url_unpooled=", "database_url=")


def _strip_pasted(url: str) -> str:
    """``psql 'postgresql://...'``, ``"postgresql://..."`` or ``DATABASE_URL=...`` -> the URL."""
    text = url.strip()
    for prefix in _PASTED_PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def normalize_database_url(url: str) -> str:
    """Make a hosted-Postgres URL (Neon, Vercel, Heroku, Render) usable by the asyncpg driver.

    * Quotes, a ``psql`` prefix or a ``DATABASE_URL=`` prefix copied along with it are removed.
    * ``postgres://`` and ``postgresql://`` become ``postgresql+asyncpg://``.
    * ``sslmode=<mode>`` becomes ``ssl=<mode>`` (asyncpg's spelling, same values).
    * ``channel_binding=...`` is dropped: asyncpg has no such option and would fail to connect.

    Error messages never include the URL: it carries the database password.
    """
    parts = urlsplit(_strip_pasted(url))
    scheme = parts.scheme.lower()
    if scheme in {"postgres", "postgresql"}:
        scheme = "postgresql+asyncpg"
    elif scheme.startswith("postgresql+") and scheme != "postgresql+asyncpg":
        raise ValueError("DATABASE_URL must use the asyncpg driver (postgresql+asyncpg://...)")
    elif scheme != "postgresql+asyncpg":
        raise ValueError(
            "DATABASE_URL must be a PostgreSQL connection string starting with postgresql:// "
            "(paste the string itself, without quotes)"
        )
    query: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":
            query.append(("ssl", value))
        elif key in _UNSUPPORTED_URL_PARAMS:
            continue
        else:
            query.append((key, value))
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Both are read; values in .env.local override .env. Neither is committed.
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # A rejected value (a database URL, a key) must never be echoed into a start-up log.
        hide_input_in_errors=True,
    )

    # ------------------------------------------------------------------ application
    app_name: str = "hybrid-retrieval-engine"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Inbound shared secret for THIS service. When set, /api/v1/* requests must send it "
            "in the X-API-Key header. It is not a provider key."
        ),
    )

    # --------------------------------------------------------------------- showcase
    demo_mode: bool = Field(
        default=False,
        description=(
            "Public showcase mode: uploads need API_KEY (and are disabled when it is unset), "
            "queries are rate limited and budgeted, and the corpus is seeded from DEMO_SEED_DIR."
        ),
    )
    demo_seed_dir: str = Field(default="demo", description="Folder with corpus.json and PDFs.")
    demo_seed_on_startup: bool = True
    demo_warm_models: bool = Field(
        default=True,
        description="Demo mode: load the local models in the background right after start-up.",
    )
    playground_enabled: bool = Field(
        default=True, description="Serve the browser playground at / and /playground/*."
    )
    cors_allow_origins: str = Field(
        default="",
        description=(
            "Comma-separated origins allowed to call the API from a browser (the GitHub Pages "
            "site). Empty disables CORS headers; * allows any origin."
        ),
    )
    trust_proxy_headers: bool = Field(
        default=False,
        description=(
            "Behind a reverse proxy (Render), read the visitor address from CLIENT_IP_HEADER or "
            "X-Forwarded-For. Leave off when clients reach the app directly: they could forge it."
        ),
    )
    client_ip_header: str | None = Field(
        default=None,
        description=(
            "Header the edge proxy overwrites with the visitor address (CF-Connecting-IP on "
            "Render, which sits behind Cloudflare). Used before X-Forwarded-For when set."
        ),
    )
    trusted_proxy_hops: int = Field(
        default=1,
        ge=1,
        le=10,
        description="X-Forwarded-For entry to trust, counted from the right (one per proxy).",
    )
    query_rate_limit_per_minute: int = Field(
        default=10, ge=1, description="Questions per client address per minute (demo mode)."
    )
    query_daily_limit_per_client: int = Field(
        default=60, ge=1, description="Questions per client address per UTC day (demo mode)."
    )
    query_global_limit_per_minute: int = Field(
        default=20, ge=1, description="Questions per minute for everyone together (demo mode)."
    )
    query_daily_limit: int = Field(
        default=800, ge=1, description="Questions per UTC day for everyone together (demo mode)."
    )
    query_max_inflight: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Demo mode: questions answered at once; more get 503 + Retry-After.",
    )
    demo_max_query_chars: int = Field(
        default=500,
        ge=3,
        le=2000,
        description="Demo mode: longest question accepted (the API itself allows 2000).",
    )
    max_json_body_bytes: int = Field(
        default=32 * 1024,
        ge=1024,
        description="Largest body accepted on any POST except uploads (checked before parsing).",
    )
    documents_cache_ttl_s: float = Field(
        default=300.0,
        ge=0,
        description="Cache the corpus listing this long, so page views do not wake the database.",
    )
    openrouter_allowance_check: bool = Field(
        default=True,
        description="Ask OpenRouter (GET /key) how many free requests are left today, once a minute.",
    )
    openrouter_allowance_reserve: int = Field(
        default=50,
        ge=0,
        description="Refuse public questions when fewer free requests than this remain today.",
    )

    # --------------------------------------------------------------------- database
    database_url: str = Field(
        default=DEFAULT_DATABASE_URL,
        description=(
            "SQLAlchemy async URL using the asyncpg driver. Neon/Vercel/Heroku-style "
            "postgresql://...?sslmode=require&channel_binding=require URLs are accepted and "
            "normalised automatically."
        ),
    )
    database_url_unpooled: str | None = Field(
        default=None,
        description=(
            "Direct (non-PgBouncer) URL as Vercel/Neon provide it (DATABASE_URL_UNPOOLED). "
            "Preferred over DATABASE_URL when set: the app runs its own connection pool."
        ),
    )
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_connect_timeout_s: float = Field(default=10.0, gt=0, le=120)
    db_pool_recycle_s: int = Field(
        default=240,
        ge=-1,
        description="Recycle pooled connections older than this (Neon suspends after 5 min idle).",
    )
    health_timeout_s: float = Field(
        default=5.0, gt=0, le=60, description="Deadline for each /health database probe."
    )
    db_startup_retries: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Retries of the start-up schema check while a serverless database wakes up.",
    )
    db_echo: bool = False
    auto_create_schema: bool = Field(
        default=True,
        description="Run CREATE EXTENSION / CREATE TABLE / CREATE INDEX on startup.",
    )

    # ------------------------------------------------------------------- embeddings
    embedding_provider: Literal["openai", "openrouter", "fastembed", "hashing"] = "openai"
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(
        default=1536,
        ge=1,
        le=2000,
        description=(
            "Width of the pgvector column (max 2000). text-embedding-3-small: 1536; "
            "fastembed BAAI/bge-small-en-v1.5: 384; nomic-embed-text-v1.5: 768."
        ),
    )
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    embedding_concurrency: int = Field(
        default=4, ge=1, le=32, description="Concurrent embedding requests during ingestion."
    )
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_low_memory: bool = Field(
        default=True,
        description=(
            "Disable ONNX Runtime's CPU memory arena for the local models. Measured on this "
            "project: steady memory 843 MB -> 385 MB for ~15% slower inference. Keep it on for "
            "small hosts (512 MB)."
        ),
    )
    fastembed_threads: int | None = Field(
        default=None, ge=1, le=64, description="ONNX intra-op threads (unset = library default)."
    )
    fastembed_rerank_batch_size: int = Field(
        default=8, ge=1, le=256, description="Passages scored per cross-encoder batch."
    )
    local_inference_concurrency: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "Local model calls (embedding + re-ranking) that may run at once. Each one holds "
            "its own activations, so 1 is what keeps a 512 MB host inside its memory."
        ),
    )

    # ------------------------------------------------------------------- generation
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    anthropic_api_key: str | None = None
    generation_model: str = "claude-opus-5-5"
    generation_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="low",
        description="Claude 5.x reasoning effort. Low keeps answers fast and deterministic-leaning.",
    )
    generation_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16_000,
        description=(
            "Output budget for the grounded answer (thinking counts toward it on Claude 5.x). "
            "Capped at 16K because non-streaming Anthropic requests are limited to ~21K tokens."
        ),
    )
    generation_temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for providers that accept it (OpenAI, OpenRouter). Claude 5.x "
            "models reject non-default sampling parameters, so determinism there comes from "
            "constrained JSON decoding plus low effort. Set to an empty value to omit it."
        ),
    )
    openai_generation_model: str = "gpt-4.1"

    # ------------------------------------------------------------------- openrouter
    openrouter_api_key: SecretStr | None = Field(
        default=None, description="https://openrouter.ai/keys - used by every openrouter provider."
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_models: str = Field(
        default=DEFAULT_OPENROUTER_CHAIN,
        description=(
            "Ordered, comma-separated chain for grounded generation. The first model is the "
            "primary; each later one is tried only when the earlier ones fail."
        ),
    )
    openrouter_contextualizer_models: str = Field(
        default=DEFAULT_OPENROUTER_CHAIN,
        description="Chain used by the contextualizer when CONTEXTUALIZER_PROVIDER=openrouter.",
    )
    openrouter_embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="EMBEDDING_PROVIDER=openrouter model. OpenRouter has no free embedding models.",
    )
    openrouter_strict_json_schema: bool = Field(
        default=True,
        description=(
            "Request strict json_schema output first. A model that rejects it is retried in plain "
            "JSON mode with the schema spelled out in the system prompt."
        ),
    )
    openrouter_free_only: bool = Field(
        default=True,
        description="Refuse any chain entry without the :free suffix so a typo can never spend money.",
    )
    openrouter_timeout_s: float = Field(default=60.0, gt=0, le=600)
    openrouter_requests_per_minute: int = Field(
        default=20,
        ge=0,
        le=600,
        description="Pacing for free models (OpenRouter allows 20/min). 0 disables pacing.",
    )
    openrouter_site_url: str | None = Field(
        default=None, description="Optional HTTP-Referer attribution header."
    )
    openrouter_app_name: str | None = Field(
        default="hybrid-retrieval-engine", description="Optional X-Title attribution header."
    )

    # --------------------------------------------------------------- contextualizer
    contextualizer_enabled: bool = True
    contextualizer_provider: Literal["auto", "anthropic", "openrouter", "heuristic"] = Field(
        default="auto",
        description=(
            "auto = the generation vendor when its key is set, otherwise heuristic. "
            "heuristic = offline preamble (title, section, page), no model calls."
        ),
    )
    contextualizer_model: str = "claude-haiku-5-5"
    contextualizer_max_tokens: int = Field(
        default=2048,
        ge=256,
        le=8192,
        description="Must leave room for Haiku's adaptive thinking plus the ~100-token preamble.",
    )
    contextualizer_concurrency: int = Field(default=8, ge=1, le=64)
    contextualizer_doc_max_chars: int = Field(default=120_000, ge=1_000)
    contextualizer_max_llm_calls_per_document: int = Field(
        default=400,
        ge=1,
        description=(
            "Documents with more chunks than this get heuristic contexts instead of one model "
            "call per chunk. Set it low (e.g. 40) on free OpenRouter models: 50 requests/day."
        ),
    )

    # -------------------------------------------------------------------- reranking
    rerank_provider: Literal["cohere", "fastembed", "none"] = "cohere"
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"
    fastembed_rerank_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        description="Local cross-encoder (80 MB). BAAI/bge-reranker-v2-m3-int8 is stronger (570 MB).",
    )

    # -------------------------------------------------------------------- retrieval
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant k.")
    top_k_retrieval: int = Field(default=50, ge=1, le=1000, description="Fused candidates kept.")
    top_k_rerank: int = Field(
        default=3, ge=1, description="Chunks passed to the LLM after re-ranking."
    )
    hnsw_m: int = Field(default=16, ge=2, le=100)
    hnsw_ef_construction: int = Field(default=64, ge=4, le=1000)
    hnsw_ef_search: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Dynamic candidate list at query time. Must be >= top_k_retrieval.",
    )
    hnsw_iterative_scan: Literal["off", "relaxed_order", "strict_order"] = Field(
        default="relaxed_order",
        description="pgvector >= 0.8 iterative scans keep filtered searches from returning too few rows.",
    )

    # --------------------------------------------------------------------- chunking
    chunk_size_tokens: int = Field(default=400, ge=50, le=4000)
    chunk_overlap_tokens: int = Field(default=60, ge=0, le=1000)
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    max_pdf_pages: int = Field(
        default=500, ge=1, description="Uploads with more pages are rejected before extraction."
    )
    max_chunks_per_document: int = Field(
        default=2000,
        ge=1,
        description="Caps LLM contextualizer calls and embedding requests per ingested document.",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return DEFAULT_DATABASE_URL  # "DATABASE_URL=" in a dotenv file means "not set"
        return normalize_database_url(value) if isinstance(value, str) else value

    @field_validator("database_url_unpooled", mode="before")
    @classmethod
    def _normalize_unpooled_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return normalize_database_url(value) if isinstance(value, str) else value

    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def _clean_openrouter_key(cls, value: object) -> object:
        """Strip what is copied along with a key, and refuse a value that is clearly not one.

        Both secrets of the deployment come from the same file. Pasting the database string
        into this box would send the database password to openrouter.ai as a bearer token on
        every question, so start-up refuses it (without echoing it: hide_input_in_errors).
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.lower().startswith("openrouter_api_key="):
            text = text[len("openrouter_api_key=") :].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
        if text.lower() in {"", "none", "null"}:
            return None
        if "://" in text or any(ch.isspace() for ch in text):
            raise ValueError(
                "OPENROUTER_API_KEY does not look like an OpenRouter key (they start with "
                "sk-or-); was another line pasted?"
            )
        return text

    @field_validator(
        "generation_temperature",
        "api_key",
        "openrouter_site_url",
        "fastembed_threads",
        "client_ip_header",
        mode="before",
    )
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        """``GENERATION_TEMPERATURE=`` / ``API_KEY=`` in .env mean "unset", not an empty value."""
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    @model_validator(mode="after")
    def _validate_relationships(self) -> Settings:
        if self.top_k_rerank > self.top_k_retrieval:
            raise ValueError("TOP_K_RERANK must be <= TOP_K_RETRIEVAL")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.hnsw_ef_construction < 2 * self.hnsw_m:
            raise ValueError("HNSW_EF_CONSTRUCTION must be >= 2 * HNSW_M (pgvector requirement)")
        for name, raw in (
            ("OPENROUTER_MODELS", self.openrouter_models),
            ("OPENROUTER_CONTEXTUALIZER_MODELS", self.openrouter_contextualizer_models),
        ):
            chain = parse_model_chain(raw)
            if not chain:
                raise ValueError(f"{name} must name at least one model id")
            if self.openrouter_free_only:
                paid = [m for m in chain if not is_free_model(m)]
                if paid:
                    raise ValueError(
                        f"{name} contains non-free models {paid}; every id must end in ':free' "
                        "while OPENROUTER_FREE_ONLY=true"
                    )
        return self

    # ------------------------------------------------------------- derived helpers
    @property
    def effective_database_url(self) -> str:
        return self.database_url_unpooled or self.database_url

    @property
    def database_is_pooled(self) -> bool:
        """Neon's PgBouncer endpoints carry ``-pooler`` in the host name."""
        return "-pooler" in (urlsplit(self.effective_database_url).hostname or "")

    @property
    def docs_enabled(self) -> bool:
        """Interactive API docs are hidden on the public showcase."""
        return self.environment != "production"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def uploads_enabled(self) -> bool:
        """In demo mode uploads exist only for the operator, who holds API_KEY."""
        return not self.demo_mode or self.api_key is not None

    @property
    def effective_hnsw_ef_search(self) -> int:
        """HNSW returns at most ``ef_search`` rows, so it can never be below the candidate limit."""
        return max(self.hnsw_ef_search, self.top_k_retrieval)

    @property
    def openrouter_model_chain(self) -> list[str]:
        return parse_model_chain(self.openrouter_models)

    @property
    def openrouter_contextualizer_chain(self) -> list[str]:
        return parse_model_chain(self.openrouter_contextualizer_models)

    @property
    def anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def openrouter_configured(self) -> bool:
        return self.openrouter_api_key is not None

    @property
    def cohere_configured(self) -> bool:
        return bool(self.cohere_api_key)

    @property
    def generator_configured(self) -> bool:
        if self.llm_provider == "anthropic":
            return self.anthropic_configured
        if self.llm_provider == "openrouter":
            return self.openrouter_configured
        return self.openai_configured

    @property
    def embedder_configured(self) -> bool:
        if self.embedding_provider == "openai":
            return self.openai_configured
        if self.embedding_provider == "openrouter":
            return self.openrouter_configured
        return True

    @property
    def reranker_configured(self) -> bool:
        if self.rerank_provider == "cohere":
            return self.cohere_configured
        return True

    @property
    def resolved_contextualizer_provider(self) -> Literal["anthropic", "openrouter", "heuristic"]:
        """Which contextualizer actually runs, given the keys that are present."""
        if not self.contextualizer_enabled:
            return "heuristic"
        choice = self.contextualizer_provider
        if choice == "auto":
            if self.llm_provider == "anthropic" and self.anthropic_configured:
                return "anthropic"
            if self.llm_provider == "openrouter" and self.openrouter_configured:
                return "openrouter"
            return "heuristic"
        if choice == "anthropic" and not self.anthropic_configured:
            return "heuristic"
        if choice == "openrouter" and not self.openrouter_configured:
            return "heuristic"
        return choice


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
