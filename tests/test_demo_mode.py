"""Public-showcase behaviour: closed uploads, budgets, corpus listing, seeding, playground, CORS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.budget import Allowance, BudgetExceeded, QueryBudget
from app.config import Settings, normalize_database_url
from app.demo import load_manifest, sample_questions, seed_demo_corpus
from app.main import Components, create_app

from .conftest import SAMPLE_PAGES, SECOND_DOC_PAGES, InMemoryStore, make_pdf


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://demo")


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    (tmp_path / "sop.pdf").write_bytes(make_pdf(SAMPLE_PAGES))
    (tmp_path / "travel.pdf").write_bytes(make_pdf(SECOND_DOC_PAGES))
    manifest = {
        "documents": [
            {
                "file": "sop.pdf",
                "title": "SOP-QA-017 Cleaning Validation",
                "questions": [
                    "How long must batch records be retained?",
                    "Who approves deviations?",
                ],
            },
            {
                "file": "travel.pdf",
                "title": "Travel Policy",
                "questions": ["What is the claim deadline?"],
            },
            {"file": "missing.pdf", "title": "Ghost", "questions": []},
        ]
    }
    (tmp_path / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.fixture
def demo_settings(settings: Settings, demo_dir: Path) -> Settings:
    settings.demo_mode = True
    settings.demo_seed_dir = str(demo_dir)
    settings.demo_seed_on_startup = False
    settings.demo_warm_models = False
    settings.query_rate_limit_per_minute = 2
    settings.query_global_limit_per_minute = 3
    settings.query_daily_limit = 100
    return settings


# ----------------------------------------------------------------------- uploads closed


async def test_uploads_are_closed_without_api_key_and_gated_with_one(
    demo_settings: Settings, components: Components, sample_pdf_bytes: bytes
) -> None:
    app = create_app(settings=demo_settings, components=components)
    files = {"file": ("a.pdf", sample_pdf_bytes, "application/pdf")}
    async with _client(app) as http:
        closed = await http.post("/api/v1/ingest", files=files)
        assert closed.status_code == 403
        assert "disabled" in closed.json()["detail"]
        health = await http.get("/health")
        assert health.json()["demo_mode"] is True and health.json()["uploads_enabled"] is False

    demo_settings.api_key = SecretStr("operator")
    app = create_app(settings=demo_settings, components=components)
    async with _client(app) as http:
        assert (await http.post("/api/v1/ingest", files=files)).status_code == 401
        ok = await http.post("/api/v1/ingest", files=files, headers={"X-API-Key": "operator"})
        assert ok.status_code == 200
        # The question box stays public in demo mode: API_KEY exists there only to gate uploads.
        asked = await http.post("/api/v1/query", json={"query": "How long are records retained?"})
        assert asked.status_code == 200
        assert (await http.get("/health")).json()["uploads_enabled"] is True


async def test_api_key_still_protects_questions_outside_demo_mode(
    settings: Settings, components: Components
) -> None:
    settings.api_key = SecretStr("operator")
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        assert (
            await http.post("/api/v1/query", json={"query": "anything at all"})
        ).status_code == 401
        ok = await http.post(
            "/api/v1/query", json={"query": "anything at all"}, headers={"X-API-Key": "operator"}
        )
        assert ok.status_code == 200


# ----------------------------------------------------------------------------- budget


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.wall = 1_700_000_000.0

    def mono(self) -> float:
        return self.now

    def time(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.wall += seconds


async def test_budget_limits_per_client_per_minute_and_per_day() -> None:
    settings = Settings(
        _env_file=None,
        _env_prefix="HERMETIC_TEST_",
        query_rate_limit_per_minute=2,
        query_global_limit_per_minute=3,
        query_daily_limit=4,
    )
    clock = FakeClock()
    budget = QueryBudget(settings, allowance_reader=None, clock=clock.mono, wall=clock.time)

    await budget.admit("a")
    await budget.admit("a")
    with pytest.raises(BudgetExceeded) as info:
        await budget.admit("a")
    assert "this address" in info.value.message and 1 <= info.value.retry_after_s <= 60

    await budget.admit("b")  # another client still has room ...
    with pytest.raises(BudgetExceeded) as info:
        await budget.admit("c")  # ... until the global minute is full
    assert "busy" in info.value.message

    clock.advance(61)  # windows expire; the day counter keeps counting
    await budget.admit("a")
    with pytest.raises(BudgetExceeded) as info:
        await budget.admit("b")
    assert "daily" in info.value.message
    assert info.value.retry_after_s <= 86_400

    clock.advance(86_400)  # a new UTC day
    await budget.admit("b")
    snapshot = budget.snapshot()
    assert snapshot["per_client_per_minute"] == 2 and snapshot["used_today"] == 1
    assert snapshot["free_requests_remaining_today"] is None


async def test_budget_respects_openrouter_allowance_and_caches_reads() -> None:
    settings = Settings(
        _env_file=None, _env_prefix="HERMETIC_TEST_", openrouter_allowance_reserve=5
    )
    clock = FakeClock()
    reads: list[int] = []
    remaining = {"value": 7}

    async def reader() -> Allowance | None:
        reads.append(1)
        return Allowance(used=993, limit=1000, remaining=remaining["value"])

    budget = QueryBudget(settings, allowance_reader=reader, clock=clock.mono, wall=clock.time)
    await budget.admit("a")  # 7 - 0 > 5
    await budget.admit("a")  # 7 - 1 > 5
    with pytest.raises(BudgetExceeded) as info:
        await budget.admit("a")  # 7 - 2 == 5 -> refused
    assert "allowance" in info.value.message
    assert len(reads) == 1  # read once, cached for a minute
    assert budget.snapshot()["free_requests_remaining_today"] == 5

    clock.advance(61)
    remaining["value"] = 50
    await budget.admit("a")
    assert len(reads) == 2

    async def failing_reader() -> Allowance | None:
        return None

    unknown = QueryBudget(
        settings, allowance_reader=failing_reader, clock=clock.mono, wall=clock.time
    )
    await unknown.admit("x")  # an unreadable count never blocks on its own
    assert unknown.snapshot()["free_requests_remaining_today"] is None


async def test_query_returns_429_with_retry_after_and_honours_forwarded_ip(
    demo_settings: Settings, components: Components, sample_pdf_bytes: bytes
) -> None:
    components.budget = QueryBudget(demo_settings, allowance_reader=None)
    demo_settings.api_key = None
    demo_settings.trust_proxy_headers = True  # one proxy in front (Render), the default hop
    app = create_app(settings=demo_settings, components=components)
    async with _client(app) as http:
        payload = {"query": "How long must batch records be retained?"}
        first = await http.post(
            "/api/v1/query", json=payload, headers={"X-Forwarded-For": "1.1.1.1"}
        )
        assert first.status_code == 200
        # The visitor writes a fake first hop; the proxy appends the real address after it.
        second = await http.post(
            "/api/v1/query", json=payload, headers={"X-Forwarded-For": "6.6.6.6, 1.1.1.1"}
        )
        assert second.status_code == 200
        third = await http.post(
            "/api/v1/query", json=payload, headers={"X-Forwarded-For": "7.7.7.7, 1.1.1.1"}
        )
        assert third.status_code == 429  # forging the left-most entry does not reset the limit
        assert third.headers["retry-after"].isdigit()
        assert "this address" in third.json()["detail"]
        other = await http.post(
            "/api/v1/query", json=payload, headers={"X-Forwarded-For": "1.1.1.1, 9.9.9.9"}
        )
        assert other.status_code == 200  # a different visitor behind the same proxy
        busy = await http.post(
            "/api/v1/query", json=payload, headers={"X-Forwarded-For": "3.3.3.3"}
        )
        assert busy.status_code == 429 and "busy" in busy.json()["detail"]


# ------------------------------------------------------------------ corpus and seeding


async def test_seeding_is_idempotent_and_reports_missing_files(
    demo_settings: Settings, components: Components, store: InMemoryStore
) -> None:
    report = await seed_demo_corpus(components, demo_settings)
    assert report.ingested == ["sop.pdf", "travel.pdf"]
    assert report.failed == ["missing.pdf"] and report.skipped == []
    titles = {d.title for d in await store.list_documents()}
    assert titles == {"SOP-QA-017 Cleaning Validation", "Travel Policy"}

    again = await seed_demo_corpus(components, demo_settings)
    assert again.ingested == [] and again.skipped == ["sop.pdf", "travel.pdf"]
    assert len(store.documents) == 2


async def test_lifespan_seeds_the_corpus_in_the_background(
    demo_settings: Settings, components: Components, store: InMemoryStore
) -> None:
    demo_settings.demo_seed_on_startup = True
    demo_settings.demo_warm_models = True
    app = create_app(settings=demo_settings, components=components)
    async with app.router.lifespan_context(app):
        await app.state.seed_task
        assert len(store.documents) == 2
        # The models were warmed after seeding: one probe passage went through the re-ranker.
        assert components.reranker.calls == [("warm up", 1, 1)]


async def test_warm_up_failure_never_breaks_startup(
    demo_settings: Settings, components: Components
) -> None:
    class BrokenEmbedder:
        name = "broken"
        dimensions = 256

        async def embed_query(self, text: str):
            raise RuntimeError("model file missing")

        async def embed_documents(self, texts):
            raise RuntimeError("model file missing")

    components.embedder = BrokenEmbedder()
    demo_settings.demo_warm_models = True
    app = create_app(settings=demo_settings, components=components)
    async with app.router.lifespan_context(app):
        await app.state.seed_task  # finishes without raising
        async with _client(app) as http:
            assert (await http.get("/health")).status_code == 200


def test_manifest_helpers(demo_dir: Path, tmp_path: Path) -> None:
    entries = load_manifest(demo_dir)
    assert [e.file for e in entries] == ["sop.pdf", "travel.pdf", "missing.pdf"]
    assert sample_questions(demo_dir) == [
        "How long must batch records be retained?",
        "What is the claim deadline?",
        "Who approves deviations?",
    ]
    assert load_manifest(tmp_path / "nowhere") == []
    (tmp_path / "corpus.json").write_text("{not json", encoding="utf-8")
    assert load_manifest(tmp_path) == []


async def test_documents_endpoint_and_playground_config(
    demo_settings: Settings, components: Components
) -> None:
    await seed_demo_corpus(components, demo_settings)
    components.budget = QueryBudget(demo_settings, allowance_reader=None)
    app = create_app(settings=demo_settings, components=components)
    async with _client(app) as http:
        docs = await http.get("/api/v1/documents")
        assert docs.status_code == 200
        listed = docs.json()["documents"]
        assert {d["filename"] for d in listed} == {"sop.pdf", "travel.pdf"}
        assert all(d["pages_count"] >= 1 and d["chunks_count"] >= 1 for d in listed)

        config = await http.get("/playground/config")
        assert config.status_code == 200
        body = config.json()
        assert body["demo_mode"] is True and body["uploads_enabled"] is False
        assert body["generator"]["provider"] == "anthropic"  # the fixture's llm_provider
        assert body["retrieval"] == {
            "top_k_retrieval": 50,
            "top_k_rerank": 3,
            "rrf_k": 60,
            "max_query_chars": 500,
        }
        assert body["limits"]["per_client_per_minute"] == 2 and body["limits"]["per_day"] == 100
        assert len(body["documents"]) == 2
        assert body["sample_questions"][0] == "How long must batch records be retained?"


# ------------------------------------------------------------------- playground / docs


async def test_playground_is_served_and_can_be_disabled(
    settings: Settings, components: Components
) -> None:
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        page = await http.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "Compliance Retrieval Engine" in page.text
        script = await http.get("/playground/assets/app.js")
        assert script.status_code == 200 and 'fetch("/api/v1/query"' in script.text
        icon = await http.get("/playground/assets/favicon.svg")
        assert icon.status_code == 200 and "svg" in icon.headers["content-type"]

    settings.playground_enabled = False
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        assert (await http.get("/")).status_code == 404
        assert (await http.get("/playground/config")).status_code == 404


async def test_docs_hidden_in_production(settings: Settings, components: Components) -> None:
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        assert (await http.get("/docs")).status_code == 200
    settings.environment = "production"
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        assert (await http.get("/docs")).status_code == 404
        assert (await http.get("/openapi.json")).status_code == 404
        assert (await http.get("/health")).status_code == 200


def test_importing_the_app_does_not_load_the_pdf_library() -> None:
    """A deployed demo with a seeded corpus never parses a PDF; PyMuPDF must stay unloaded."""
    import subprocess
    import sys

    code = "import sys, app.main; print('pymupdf' in sys.modules or 'fitz' in sys.modules)"
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=root, timeout=120
    )
    assert out.stdout.strip() == "False", out.stderr[-400:]


async def test_cors_headers_only_for_configured_origins(
    settings: Settings, components: Components
) -> None:
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        plain = await http.get("/health", headers={"Origin": "https://nick.github.io"})
        assert "access-control-allow-origin" not in plain.headers

    settings.cors_allow_origins = "https://nick.github.io, https://example.org"
    app = create_app(settings=settings, components=components)
    async with _client(app) as http:
        allowed = await http.get("/health", headers={"Origin": "https://nick.github.io"})
        assert allowed.headers["access-control-allow-origin"] == "https://nick.github.io"
        preflight = await http.options(
            "/api/v1/query",
            headers={
                "Origin": "https://example.org",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://example.org"
        denied = await http.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in denied.headers


# ------------------------------------------------------------------ hosted database URLs


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "postgresql://u:p@ep-x.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
            "postgresql+asyncpg://u:p@ep-x.aws.neon.tech/neondb?ssl=require",
        ),
        ("postgres://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
        (
            "postgresql+asyncpg://u:p@host/db?ssl=require&application_name=x",
            "postgresql+asyncpg://u:p@host/db?ssl=require&application_name=x",
        ),
    ],
)
def test_normalize_database_url(raw: str, expected: str) -> None:
    assert normalize_database_url(raw) == expected


def test_database_url_settings_prefer_unpooled_and_detect_pooler() -> None:
    settings = Settings(
        _env_file=None,
        _env_prefix="HERMETIC_TEST_",
        database_url="postgresql://u:p@ep-x-pooler.aws.neon.tech/db?sslmode=require",
        database_url_unpooled="",
    )
    assert settings.database_url_unpooled is None
    assert settings.database_is_pooled is True
    settings = Settings(
        _env_file=None,
        _env_prefix="HERMETIC_TEST_",
        database_url="postgresql://u:p@ep-x-pooler.aws.neon.tech/db?sslmode=require",
        database_url_unpooled="postgresql://u:p@ep-x.aws.neon.tech/db?sslmode=require&channel_binding=require",
    )
    assert (
        settings.effective_database_url
        == "postgresql+asyncpg://u:p@ep-x.aws.neon.tech/db?ssl=require"
    )
    assert settings.database_is_pooled is False
    with pytest.raises(ValueError, match="asyncpg"):
        Settings(
            _env_file=None,
            _env_prefix="HERMETIC_TEST_",
            database_url="postgresql+psycopg2://u:p@h/db",
        )
    # "DATABASE_URL=" left empty in a dotenv file means "not set", not "connect to nothing".
    blank = Settings(_env_file=None, _env_prefix="HERMETIC_TEST_", database_url="  ")
    assert blank.database_url == "postgresql+asyncpg://rag:rag@localhost:5432/rag"


@pytest.mark.parametrize(
    "pasted",
    [
        "psql 'postgresql://u:pw@ep-x.aws.neon.tech/db?sslmode=require&channel_binding=require'",
        '"postgresql://u:pw@ep-x.aws.neon.tech/db?sslmode=require&channel_binding=require"',
        "DATABASE_URL_UNPOOLED=postgresql://u:pw@ep-x.aws.neon.tech/db?sslmode=require",
        "  'postgresql://u:pw@ep-x.aws.neon.tech/db?sslmode=require'  ",
    ],
)
def test_database_url_survives_what_is_copied_along_with_it(pasted: str) -> None:
    # Render's form takes whatever is pasted; a Neon dashboard offers "psql '...'".
    assert (
        normalize_database_url(pasted)
        == "postgresql+asyncpg://u:pw@ep-x.aws.neon.tech/db?ssl=require"
    )


def test_bad_database_url_is_refused_without_echoing_the_password() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as info:
        Settings(
            _env_file=None,
            _env_prefix="HERMETIC_TEST_",
            database_url="ep-x.aws.neon.tech/db?user=u&password=SuperSecret123",
        )
    assert "SuperSecret123" not in str(info.value)
    assert "postgresql://" in str(info.value)  # the message says what is expected


def test_env_example_holds_no_secrets_and_no_trailing_overrides() -> None:
    """The committed template must stay a template: no keys, no connection strings."""
    import re

    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
    assert not re.search(r"sk-or-v1-[0-9a-f]{20,}", text)
    assert "neon.tech/" not in text.replace("...neon.tech/", "")
    active = [line for line in text.splitlines() if line and not line.lstrip().startswith("#")]
    names = [line.split("=", 1)[0] for line in active]
    assert len(names) == len(set(names)), "a name set twice would silently override the first"
    for line in active:
        name, value = line.split("=", 1)
        if name.endswith("API_KEY"):
            assert value == "", f"{name} must be empty in the template"
