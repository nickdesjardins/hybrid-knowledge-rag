"""The published pages are checked against their sources, so they cannot quietly go stale.

Covers the project page (``site/``, served by GitHub Pages), the playground (served by the API),
the recorded runs, the sample corpus, and the deployment files that have to agree with them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.demo import load_manifest
from app.schemas import GroundedAnswer

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
PLAYGROUND = ROOT / "app" / "playground" / "static"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------- recorded runs


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(_read(SITE / "recorded" / "runs.json"))


def test_recorded_runs_are_real_api_responses(record: dict) -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["recorded_at"])
    assert len(record["runs"]) >= 5
    for run in record["runs"]:
        answer = GroundedAnswer.model_validate(run["response"])  # the public schema, unchanged
        assert run["why"] and run["id"]
        if answer.status.value == "answered":
            assert answer.is_grounded and answer.citations
            passages = {p.chunk_id for p in answer.retrieved_passages}
            for citation in answer.citations:
                assert citation.chunk_id in passages  # every quote comes from a retrieved passage
                assert citation.page_number >= 1
        else:
            assert not answer.is_grounded and answer.citations == []
            assert answer.confidence_score == 0.0


def test_recorded_runs_carry_what_the_walkthrough_explains(record: dict) -> None:
    from app.generator.grounded_llm import SYSTEM_PROMPT

    assert record["system_prompt"] == SYSTEM_PROMPT  # re-record after changing the rules
    assert record["documents"], "the corpus listing must not be empty"
    for run in record["runs"]:
        assert run["expect"] in {"answered", "insufficient_context"}
        body = run["response"]
        trace = body["pipeline"]
        assert trace["candidates"] == (
            trace["found_by_both"] + trace["found_by_vector_only"] + trace["found_by_keyword_only"]
        )
        k = trace["rrf_k"]
        for passage in body["retrieved_passages"]:
            ranks = [r for r in (passage["dense_rank"], passage["sparse_rank"]) if r is not None]
            # The walkthrough prints this sum next to the stored score; they must agree.
            assert passage["rrf_score"] == pytest.approx(sum(1 / (k + r) for r in ranks), abs=1e-6)
            assert 1 <= passage["fused_rank"] <= trace["candidates"]
        if body["status"] == "answered":
            assert trace["quotes_verified"] == len(body["citations"]) >= 1


def test_recorded_runs_show_an_answer_and_a_refusal(record: dict) -> None:
    statuses = {run["response"]["status"] for run in record["runs"]}
    assert "answered" in statuses
    assert "insufficient_context" in statuses  # the negative-out case is part of the showcase


def test_recorded_runs_quote_the_sample_documents_verbatim(record: dict) -> None:
    from app.generator.grounded_llm import locate_quote
    from app.ingestion.pdf_parser import parse_pdf_bytes

    texts = {}
    for entry in load_manifest(ROOT / "demo"):
        parsed = parse_pdf_bytes((ROOT / "demo" / entry.file).read_bytes(), entry.file, entry.title)
        texts[entry.title] = (parsed.full_text, {p.page_number: p.text for p in parsed.pages})
    checked = 0
    for run in record["runs"]:
        for citation in run["response"]["citations"]:
            full_text, pages = texts[citation["document_title"]]
            assert locate_quote(citation["verbatim_quote"], full_text), citation["verbatim_quote"]
            # The cited page holds the beginning of the quote.
            opening = " ".join(citation["verbatim_quote"].split()[:6])
            assert locate_quote(opening, pages[citation["page_number"]]), citation
            checked += 1
    assert checked >= 5


def test_recorded_runs_hold_no_secrets(record: dict) -> None:
    blob = json.dumps(record)
    for marker in ("sk-or-", "neon.tech", "npg_", "postgresql://", "Bearer "):
        assert marker not in blob


def test_recorded_stack_matches_the_code(record: dict) -> None:
    settings = Settings(_env_file=None, _env_prefix="HERMETIC_TEST_")
    stack = record["stack"]
    assert stack["generator_chain"] == settings.openrouter_model_chain
    assert stack["embeddings"] == settings.fastembed_model
    assert stack["reranker"] == settings.fastembed_rerank_model
    assert stack["rrf_k"] == settings.rrf_k == 60
    assert stack["top_k_rerank"] == settings.top_k_rerank == 3
    titles = {entry.title for entry in load_manifest(ROOT / "demo")}
    assert {d["title"] for d in record["documents"]} == titles


# ------------------------------------------------------------------------ sample corpus


def test_demo_manifest_lists_existing_pdfs_with_questions() -> None:
    entries = load_manifest(ROOT / "demo")
    assert len(entries) >= 3
    for entry in entries:
        path = ROOT / "demo" / entry.file
        assert path.is_file(), entry.file
        assert path.read_bytes().startswith(b"%PDF")
        assert len(entry.questions) >= 2


# ------------------------------------------------------------------------- static pages


@pytest.mark.parametrize("page", [SITE / "index.html", PLAYGROUND / "index.html"])
def test_pages_use_only_their_own_scripts_and_styles(page: Path) -> None:
    html = _read(page)
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    assert csp, "every page declares a content security policy"
    policy = csp.group(1)
    assert "default-src 'none'" in policy and "script-src 'self'" in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "no inline scripts"
    assert not re.search(r"\son[a-z]+\s*=", html), "no inline event handlers"
    assert "<style" not in html and not re.search(r'\sstyle="', html), "no inline styles"


def test_project_page_assets_exist() -> None:
    html = _read(SITE / "index.html")
    for relative in re.findall(r'(?:href|src)="((?!https?:|#|\./)[^"]+)"', html):
        assert (SITE / relative).is_file(), relative
    assert (SITE / "recorded" / "runs.json").is_file()


def test_playground_assets_exist() -> None:
    html = _read(PLAYGROUND / "index.html")
    for absolute in re.findall(r'(?:href|src)="(/playground/assets/[^"]+)"', html):
        assert (PLAYGROUND / "assets" / absolute.rsplit("/", 1)[1]).is_file(), absolute


def test_live_demo_address_is_written_once_and_is_well_formed() -> None:
    html = _read(SITE / "index.html")
    links = re.findall(r'id="demo-link" href="([^"]+)"', html)
    assert len(links) == 1
    assert re.fullmatch(r"https://[a-z0-9-]+\.(onrender\.com|invalid)/#ask", links[0]), links[0]
    # No other element of the page may hold a Render address.
    assert html.count(".onrender.com/") <= 1


@pytest.mark.parametrize("name", ["explain.js", "explain.css"])
def test_teaching_layer_is_the_same_file_on_both_pages(name: str) -> None:
    assert _read(SITE / "assets" / name) == _read(PLAYGROUND / "assets" / name), (
        f"copy app/playground/static/assets/{name} to site/assets/{name}"
    )


@pytest.mark.parametrize(
    ("page", "own_script"),
    [(SITE / "index.html", "site.js"), (PLAYGROUND / "index.html", "app.js")],
)
def test_pages_load_the_teaching_layer_before_their_own_script(page: Path, own_script: str) -> None:
    html = _read(page)
    assert "explain.css" in html
    scripts = [src.rsplit("/", 1)[-1] for src in re.findall(r'<script src="([^"]+)"', html)]
    assert scripts.index("explain.js") < scripts.index(own_script), scripts


def test_code_links_point_to_files_that_exist() -> None:
    linked = set(re.findall(r'"(app/[a-z_/]+\.py)"', _read(PLAYGROUND / "assets" / "explain.js")))
    linked |= set(re.findall(r"blob/main/(app/[a-z_/]+\.py)", _read(SITE / "index.html")))
    assert len(linked) >= 10
    for path in linked:
        assert (ROOT / path).is_file(), path


def _rrf_lab_order(k: int, vector: bool = True, keyword: bool = True) -> list[str]:
    script = _read(PLAYGROUND / "assets" / "explain.js")
    items = re.findall(
        r'\{ id: "([A-F])", text: "[^"]+", vector: (null|\d+), keyword: (null|\d+)', script
    )
    assert len(items) == 6
    scores = {}
    for item_id, v, s in items:
        ranks = [int(r) for r, on in ((v, vector), (s, keyword)) if on and r != "null"]
        if ranks:
            scores[item_id] = sum(1 / (k + r) for r in ranks)
    return sorted(scores, key=lambda i: (-scores[i], i))


def test_rank_fusion_lab_shows_what_its_text_claims() -> None:
    # The lab tells learners that k = 60 puts the answer (A) first, that a small k lets a single
    # first place win, and that either arm alone puts something else first.
    assert _rrf_lab_order(60)[0] == "A"
    assert _rrf_lab_order(1)[0] == "B"
    assert _rrf_lab_order(60, keyword=False)[0] == "B"
    assert _rrf_lab_order(60, vector=False)[0] == "C"
    # The lab computes the tipping point and says "below k = tip, B would win": that sentence is
    # only true if A stays first for every k from the tipping point up.
    tipping = [k for k in range(1, 101) if _rrf_lab_order(k)[0] == "A"]
    assert tipping and tipping == list(range(tipping[0], 101))
    assert all(_rrf_lab_order(k)[0] == "B" for k in range(1, tipping[0]))


def test_wake_up_icon_path_is_served_by_the_playground() -> None:
    script = _read(SITE / "assets" / "site.js")
    match = re.search(r'new URL\("(/playground/assets/[^"]+)"', script)
    assert match
    assert (PLAYGROUND / "assets" / match.group(1).rsplit("/", 1)[1]).is_file()


# --------------------------------------------------------------------- deployment files


def test_render_blueprint_agrees_with_the_dockerfile_and_settings() -> None:
    blueprint = _read(ROOT / "render.yaml")
    dockerfile = _read(ROOT / "Dockerfile")

    def value(key: str) -> str:
        match = re.search(rf"- key: {key}\n\s+value: \"?([^\"\n#]+?)\"?\s*(?:#.*)?\n", blueprint)
        assert match, key
        return match.group(1).strip()

    assert "plan: free" in blueprint and "runtime: docker" in blueprint
    # Liveness that never queries the database, in both places Render or Docker probe it.
    assert "healthCheckPath: /livez" in blueprint
    assert "/livez" in dockerfile and "/health'" not in dockerfile
    assert value("TRUST_PROXY_HEADERS") == "true" and value("CLIENT_IP_HEADER")
    assert value("DEMO_MODE") == "true"
    assert value("LLM_PROVIDER") == "openrouter"
    assert value("EMBEDDING_PROVIDER") == "fastembed" and value("RERANK_PROVIDER") == "fastembed"
    assert value("FASTEMBED_LOW_MEMORY") == "true"
    # Secrets are asked for by Render, never written in the file.
    for secret in ("OPENROUTER_API_KEY", "DATABASE_URL"):
        assert re.search(rf"- key: {secret}\n\s+sync: false", blueprint), secret
    assert "sk-or-" not in blueprint and "neon.tech" not in blueprint
    # The models baked into the image are the ones the service is told to load.
    assert f"TextEmbedding('{value('FASTEMBED_MODEL')}')" in dockerfile
    assert f"TextCrossEncoder('{value('FASTEMBED_RERANK_MODEL')}')" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile and "USER app" in dockerfile
    # Every key the blueprint sets is a real setting.
    fields = {name.upper() for name in Settings.model_fields}
    for key in re.findall(r"- key: ([A-Z_]+)", blueprint):
        assert key in fields, key
    # The embedding width in the blueprint is the model's.
    assert value("EMBEDDING_DIMENSIONS") == "384"
    assert int(value("TOP_K_RETRIEVAL")) >= Settings(_env_file=None, _env_prefix="X_").top_k_rerank


def test_gitignore_protects_local_secrets() -> None:
    lines = _read(ROOT / ".gitignore").splitlines()
    assert ".env" in lines and ".env.*" in lines and "!.env.example" in lines
    dockerignore = _read(ROOT / ".dockerignore").splitlines()
    assert ".env" in dockerignore and ".env.*" in dockerignore
