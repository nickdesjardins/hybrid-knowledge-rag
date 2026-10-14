# Context-Aware Hybrid Knowledge & Compliance Retrieval Engine

An asynchronous FastAPI service that answers questions over dense regulatory manuals, SOPs, and
technical documentation **without hallucinating**. Every answer is produced from exact passages,
every passage is cited with a verbatim quote and a page number, and the service says "insufficient
context" instead of guessing.

* **The project page:** <https://nickdesjardins.github.io/hybrid-knowledge-rag/> is the address to
  give to people. It is a static page on GitHub Pages: it is there at once, shows seven recorded
  runs with a step-by-step walkthrough of each, and wakes the live demo while it is read.
* **The live demo:** <https://hybrid-retrieval-engine.onrender.com/> runs at zero cost: free
  OpenRouter models for generation, local ONNX models for embeddings and re-ranking, Neon for
  PostgreSQL, Render for hosting. It sleeps between visits and takes about a minute to wake. It
  is closed on purpose: uploads need the operator's key, questions are rate limited and capped
  per day, and the corpus is four fictional quality documents. See
  [Public showcase](#public-showcase).
* **Putting it online:** [docs/DEPLOY.md](docs/DEPLOY.md) walks through it, and says exactly what
  has and has not been verified.

The pipeline is the one described in Anthropic's *Contextual Retrieval* work combined with
PostgreSQL-native hybrid search:

```
PDF ──► PyMuPDF ──► structural chunker ──► Claude Haiku 5.5 context preamble ──► embeddings ──► PostgreSQL
                     (headers / sliding    (Anthropic Contextual Retrieval)                     pgvector HNSW
                      window + overlap)                                                          + tsvector GIN

Query ──► embed ──► ┌ dense:  embedding <=> query          (HNSW, cosine) ┐
                    │                                                     ├─► RRF CTE (k=60) ─► top 50
                    └ sparse: tsv_content @@ websearch_to_tsquery (GIN)  ┘
                                                                              │
                              Cohere Rerank v3.5 (cross-encoder) ◄────────────┘
                                                                              │ top 3
                              Claude Opus 5.5, constrained JSON  ◄────────────┘
                              (answer + verbatim citations)
                                                                              │
                              server-side verbatim verification ◄─────────────┘
                              (quotes re-anchored to stored text; metadata from DB)
                                                                              │
                                                                       GroundedAnswer
```

---

## Why naive RAG fails on compliance documents

**Context collapse.** A chunk such as *"The limit is 0.5 mg/kg and samples are retained for 90
days"* is unintelligible on its own: which product, which site, which section? Naive chunking throws
that away, so the embedding of the chunk is wrong and the model cannot tell two near-identical
clauses from different SOPs apart. This engine prepends a 50-100 token situational context to every
chunk **before** it is embedded and indexed (document title, parent section, core topic). Both the
dense vector and the full-text `tsvector` are built from the contextualized text, so the fix
benefits both retrievers.

**Dense keyword blind spots.** Embedding models are good at concepts and bad at identifiers.
Queries like *"SOP-QA-017"*, *"21 CFR Part 11"*, *"lot 4471-B"* are exactly what compliance users
type, and a bi-encoder will happily return a semantically similar but wrong passage. The sparse arm
(PostgreSQL full-text search with the English dictionary, `websearch_to_tsquery`, `ts_rank_cd`)
catches exact alphanumeric matches. Reciprocal Rank Fusion merges the two candidate lists without
having to calibrate incompatible score scales:

```
RRF(d) = Σ_m 1 / (k + rank_m(d)),   k = 60
```

A document ranked #2 by both retrievers (2/62 ≈ 0.032) beats one ranked #1 by a single retriever
(1/61 ≈ 0.016), which is the behaviour you want: agreement is evidence.

**Lost in the middle.** Stuffing 50 chunks into the prompt does not help. Transformer attention
favours the beginning and the end of the context, and recall over the interior degrades measurably.
A cross-encoder (Cohere Rerank v3.5) reads query and passage *jointly* and reorders the 50 fused
candidates so only the best 3 reach the generator. Fewer, better passages mean higher faithfulness
and lower token cost.

**Ungrounded generation.** Even with good retrieval, a model can paraphrase a number wrong or cite
a page it never saw. The generator is forced into a JSON schema (`answer`, `is_grounded`,
`confidence_score`, `citations[chunk_id, verbatim_quote]`) and the service then **re-anchors every
quote against the stored chunk text**. Quotes that are not an exact substring are dropped, titles
and page numbers come from the database rather than from the model, and an answer whose citations
all fail verification is downgraded to `is_grounded=false` with an explicit `insufficient_context`
status.

---

## Pipeline in detail

### 1. Ingestion (`POST /api/v1/ingest`)

| Stage | Module | What happens |
|---|---|---|
| Parse | `app/ingestion/pdf_parser.py` | PyMuPDF extracts text per page as layout blocks with bounding boxes. Hyphenated line breaks are joined, whitespace normalised. Encrypted or image-only PDFs are rejected with 400. Parsing runs in a worker thread. |
| Chunk | `app/ingestion/chunker.py` | Blocks become units; markdown / numbered / ALL-CAPS headings are detected and recorded as the current section. Units are packed into chunks of `CHUNK_SIZE_TOKENS` with the last `CHUNK_OVERLAP_TOKENS` carried into the next chunk. Oversized paragraphs are recursively exploded into sentences, word windows, and finally token-boundary slices (URLs, dot leaders, CJK prose), always re-joined with the original separators so the stored text stays verbatim. Invariant: `token_count <= CHUNK_SIZE_TOKENS`. Each chunk records the page it starts on. Runs in a worker thread. |
| Contextualize | `app/ingestion/contextualizer.py` | Claude Haiku 5.5 receives the whole document (prompt-cached with `cache_control`) plus the chunk and returns a short situational context. The first chunk is processed alone to create the cache entry, then the rest fan out under a semaphore; any failure, refusal, or truncation falls back to a deterministic heuristic preamble (title, section, page) so ingestion never fails because of the worker. Document text is wrapped in nonce-suffixed tags and declared untrusted in the system prompt. `CONTEXTUALIZER_PROVIDER` switches this stage to the OpenRouter free chain or to the offline heuristic. |
| Embed | `app/retrieval/embedder.py` | Batched `text-embedding-3-small` (1536 dims) by default; `fastembed` for local ONNX models; `hashing` for an offline, non-semantic demo mode. |
| Store | `app/database.py` | One transaction inserts the `documents` row and all `document_chunks` rows. `tsv_content` is a `GENERATED ALWAYS AS (to_tsvector('english', contextualized_content)) STORED` column, so it can never drift from the text. |

Schema (created on startup, idempotent):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (id UUID PRIMARY KEY, title TEXT, filename TEXT, page_count INTEGER,
                        created_at TIMESTAMPTZ);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER, page_number INTEGER,       -- the page the chunk starts on
    raw_content TEXT, contextualized_content TEXT,
    embedding vector(1536),
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', contextualized_content)) STORED,
    page_map JSONB,                                 -- [[offset, page], ...] inside raw_content
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX idx_chunks_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_chunks_tsv  ON document_chunks USING gin (tsv_content);
```

### 2. Hybrid retrieval (`app/retrieval/hybrid_search.py`)

One round-trip, one snapshot, one CTE:

```sql
WITH dense_search AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY distance, id) AS rank
    FROM (SELECT id, embedding <=> :query_vector AS distance
          FROM document_chunks
          ORDER BY embedding <=> :query_vector LIMIT 50) sub
),
sparse_search AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC, id) AS rank
    FROM (SELECT id,
                 ts_rank_cd(tsv_content, strict_query) + ts_rank_cd(tsv_content, relaxed_query) AS score
          FROM document_chunks
          WHERE tsv_content @@ relaxed_query          -- strict_query = websearch_to_tsquery('english', :q)
          ORDER BY score DESC LIMIT 50) sub           -- relaxed_query = strict_query with '&' -> '|'
),
rrf_fusion AS (
    SELECT COALESCE(d.id, s.id) AS id,
           COALESCE(1.0 / (60 + d.rank), 0.0) + COALESCE(1.0 / (60 + s.rank), 0.0) AS rrf_score,
           d.rank AS dense_rank, s.rank AS sparse_rank
    FROM dense_search d FULL OUTER JOIN sparse_search s ON d.id = s.id
)
SELECT c.*, doc.title, f.rrf_score, f.dense_rank, f.sparse_rank
FROM rrf_fusion f JOIN document_chunks c ON c.id = f.id JOIN documents doc ON doc.id = c.document_id
ORDER BY f.rrf_score DESC, f.dense_rank NULLS LAST, f.sparse_rank NULLS LAST, c.id
LIMIT 50;
```

Implementation notes:

- The `ORDER BY ... LIMIT` lives inside the sub-select so PostgreSQL can satisfy it with an HNSW
  index scan; the window function only numbers the 50 survivors. The live test suite checks with
  `EXPLAIN` that `idx_chunks_hnsw` is used.
- `websearch_to_tsquery` ANDs every term, so the question *"How long must batch records be
  retained?"* would match nothing (no clause contains "long"). The sparse arm therefore matches on
  an OR-relaxed copy of the query and ranks by `ts_rank_cd(strict) + ts_rank_cd(relaxed)`: chunks
  containing every term still rank first, partial matches still surface, and exact identifiers such
  as `21 CFR Part 11` behave as before.
- `SET LOCAL hnsw.ef_search` is issued in the same transaction (HNSW returns at most `ef_search`
  rows, so it is always raised to at least `TOP_K_RETRIEVAL`). With pgvector >= 0.8,
  `hnsw.iterative_scan = relaxed_order` keeps filtered searches (`filter_document_id`) from
  returning too few rows.
- The query vector is sent as a pgvector text literal and cast server-side, so no custom asyncpg
  codec is needed. Every bind parameter is wrapped in a `CAST` so asyncpg's type inference is stable
  even though names repeat.
- `reciprocal_rank_fusion()` is a pure-Python twin of the CTE used by the unit tests and the
  in-memory test double; the live-database test asserts the two agree to 1e-9.

### 3. Cross-encoder re-ranking (`app/retrieval/reranker.py`)

`cohere.AsyncClientV2.rerank(model="rerank-v3.5", query, documents=[contextualized_content ...],
top_n=3)`. Returns the top 3 with relevance scores. `RERANK_PROVIDER=none` keeps RRF order
(degraded mode, reported by `/health`).

### 4. Grounded generation (`app/generator/grounded_llm.py`)

- The three passages are rendered as
  `<passage-ID chunk_id=... page_number=...><context-ID>...</context-ID><text-ID>...</text-ID></passage-ID>`
  followed by `<question-ID>`. `ID` is a random per-request suffix, so text inside a document can
  never close the passage block or inject a question of its own; attribute values are escaped and
  the system prompt declares passage contents untrusted data.
- The system prompt is the strict one from the spec: *"Answer the question using strictly the
  provided facts. For every factual claim, extract an exact, verbatim quotation from the text. If
  the text does not contain sufficient facts to answer with certainty, set is_grounded to False,
  state what is missing in the answer, and provide an empty citations list."*
- Claude Opus 5.5 via `messages.create(output_config={"effort": "low", "format": {json_schema}})`.
  The stop reason is inspected **before** the JSON is validated, so safety refusals
  (`stop_reason == "refusal"`) surface as `status: "refused"` and `max_tokens` truncation becomes
  an actionable error instead of an opaque validation failure.
- **Verification layer:** quotes are matched against the stored chunk text only (never the
  synthesised preamble) modulo whitespace, case, typographic quotes/dashes and Unicode
  normalisation (ligatures, combining accents, zero-width characters), then replaced by the exact
  source span. Unknown `chunk_id`s, overlong quotes and non-matching quotes are dropped;
  duplicates are collapsed; at most 10 citations.
- **Exact page attribution:** a chunk can start on one page and continue on the next, or begin
  with overlap carried from the previous page. Each chunk therefore stores a small page map
  (`page_map`: the character offsets at which the page changes), and a citation reports the page
  on which its quote *begins*, not the page its chunk starts on. The model never supplies a page
  number at all.
- **Retrieval trace:** every answer also returns `retrieved_passages`: for each of the passages
  given to the model, its rank in the vector arm, its rank in the keyword arm (or none), the RRF
  score, its position in the fused list, the re-rank score, and whether a verified citation
  quotes it. A `pipeline` object adds what each stage did: how many candidates each search arm
  contributed, how many quotes the model proposed and how many survived verification, and the
  milliseconds spent embedding, searching, re-ranking and generating. The playground turns both
  into a step-by-step walkthrough of that answer.

#### A note on determinism and `temperature=0`

Claude 5.x models (Opus 5.5, Sonnet 5.5, Haiku 5.5) reject `temperature`, `top_p` and `top_k`
with HTTP 400, and the `anthropic` 1.x SDK no longer exposes the parameter. Determinism is enforced
instead by constrained JSON decoding, `effort="low"`, a prompt that forbids inference, and the
server-side quote verification above. The OpenAI path (`LLM_PROVIDER=openai`) still sends
`GENERATION_TEMPERATURE=0.0`.

---

## HNSW tuning: `m = 16`, `ef_construction = 64`

pgvector's HNSW index is a multi-layer proximity graph. Three parameters matter:

| Parameter | Default here | Effect |
|---|---|---|
| `m` | 16 | Maximum number of bidirectional links per node per layer. Higher `m` = denser graph = better recall and more robust navigation on high-dimensional data, at the cost of memory (roughly `m * 2 * 4 bytes` per node per layer on top of the vector itself) and slower inserts. 16 is the sweet spot for 768-1536-dim embeddings; go to 24-32 only if recall at your target `ef_search` is below ~95%. |
| `ef_construction` | 64 | Size of the dynamic candidate list while **building** the graph. Larger values explore more neighbours per insert, producing a higher-quality graph (better recall for the same `ef_search`) but building proportionally slower. 64 builds fast enough for incremental ingestion through the API; for a one-shot bulk load of millions of chunks, build with 128-200 (quality is baked into the graph, no `REINDEX` needed later). pgvector requires `ef_construction >= 2 * m`; the settings validator enforces it so a bad pair fails at startup with a clear message instead of inside `CREATE INDEX`. |
| `hnsw.ef_search` | 100 (runtime, `SET LOCAL`) | Candidate list at **query** time. It bounds both recall and the number of rows a scan can return, so it must be >= the `LIMIT` (the service enforces `max(HNSW_EF_SEARCH, TOP_K_RETRIEVAL)`). Raising it is the cheapest way to buy recall after the index exists; 100-200 is typical for top-50 candidate sets. |

Practical guidance:

- **Memory.** The whole graph should fit in `shared_buffers` / OS cache for predictable latency.
  Budget `dims * 4 bytes + ~m * 2 * 8 bytes` per chunk (1536 dims ≈ 6.4 KB per chunk, ~6.5 GB per
  million chunks). Matryoshka-style truncation (`EMBEDDING_DIMENSIONS=768` with
  `text-embedding-3-*`) halves that for a small recall cost.
- **Build speed.** HNSW builds are bounded by `maintenance_work_mem`; the compose file sets 256 MB
  and `max_parallel_maintenance_workers=2`. If the index does not fit in `maintenance_work_mem`
  pgvector falls back to a much slower on-disk build and logs a notice.
- **Filtered search.** `WHERE document_id = ...` plus `ORDER BY embedding <=> q LIMIT 50` can return
  fewer than 50 rows on pgvector < 0.8 because the index scan stops after `ef_search` candidates.
  pgvector 0.8 iterative scans fix this; the service enables `relaxed_order` when the server
  supports it.
- **When to use IVFFlat instead.** IVFFlat builds faster and uses less memory but needs training
  data (cannot be created on an empty table), and recall drops as the table grows past the trained
  centroids. For an API that ingests continuously, HNSW is the right default.

---

## Quick start

### Prerequisites

- Docker + Docker Compose (for PostgreSQL 16 with pgvector), or `pip install pgserver` for an
  embedded server without Docker
- Python 3.11+ (3.12 recommended)
- Provider keys, depending on the stack you choose (next section). The paid defaults expect
  `ANTHROPIC_API_KEY` (generation + contextualizer), `OPENAI_API_KEY` (embeddings) and
  `COHERE_API_KEY` (re-ranking); the zero-cost stack needs only an OpenRouter key.

> `API_KEY` is **not** a provider key. It is the optional inbound secret that protects this
> service's own `/api/v1/*` endpoints (`X-API-Key` header). Leave it empty for local work.

### Zero-cost stack: OpenRouter free models + local embeddings and re-ranking

Every stage has a free option. Pick them with six settings (a ready-made block sits at the **end**
of `.env.example`: in a dotenv file the last value of a name wins, so the block overrides the
section defaults above it). The app reads `.env` and then `.env.local`; both are git-ignored.

| Stage | Paid default | Free option | Setting |
|---|---|---|---|
| Generation | Claude Opus 5.5 | Ordered chain of OpenRouter `:free` models | `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY` |
| Context preambles | Claude Haiku 5.5 | Offline heuristic (title, section, page), or the same free chain | `CONTEXTUALIZER_PROVIDER=heuristic` or `openrouter` |
| Embeddings | OpenAI `text-embedding-3-small` | Local ONNX `BAAI/bge-small-en-v1.5` (384 dims, 67 MB) | `EMBEDDING_PROVIDER=fastembed`, `EMBEDDING_DIMENSIONS=384` |
| Re-ranking | Cohere Rerank v3.5 | Local cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` (80 MB) | `RERANK_PROVIDER=fastembed` |
| Database | - | PostgreSQL in Docker, or `pgserver` | - |

```bash
pip install -r requirements-local-models.txt       # fastembed (onnxruntime + model weights)
python scripts/openrouter_smoke.py                  # 1-2 free requests: catalogue, allowance, one grounded answer
```

How the OpenRouter path works (the same design as the Phase 1 extraction service):

- `OPENROUTER_MODELS` is an ordered chain. The first model is the primary; each later one is
  tried only when the earlier ones fail (429, in-band upstream error, withdrawal, truncation,
  invalid JSON). The default, `nvidia/nemotron-3-super-120b-a12b:free` then
  `liquid/lfm-2.5-2.6b:free`, is the chain Phase 1 measured on 7 October 2026; both support
  strict JSON-schema output and sit on different upstream providers.
- Strict `json_schema` output is requested first. A model that rejects it is retried in plain
  JSON mode with the schema spelled out in the system prompt; code fences and surrounding prose
  are stripped before validation. `temperature=0.0` is sent.
- `OPENROUTER_FREE_ONLY=true` refuses any chain entry without the `:free` suffix, so a typo can
  never spend money.
- Request starts are paced to `OPENROUTER_REQUESTS_PER_MINUTE` (20, OpenRouter's free-model
  limit). The daily allowance is 50 requests without credits and 1000 after a one-time $10
  purchase. The contextualizer costs **one request per chunk**, which is why the preset uses
  the heuristic preamble; `CONTEXTUALIZER_MAX_LLM_CALLS_PER_DOCUMENT` caps the spend when you
  do switch it to `openrouter`.
- Free models are withdrawn often. The smoke script reports which chain ids the public catalogue
  still lists and how much of today's allowance remains.
- **Data handling:** free endpoints may keep and train on what you send them. Use sample or
  redacted documents, never real compliance records, while running on free models.

What you give up: the local embedding model is smaller than `text-embedding-3-small` (recall
drops a little on nuanced queries), the MiniLM cross-encoder is weaker than Cohere's
(`BAAI/bge-reranker-v2-m3-int8` narrows the gap at 570 MB), and free generation models are
slower and less consistent than Claude. The grounding layer does not change: every citation is
still verified against the stored text, whichever model wrote it.

### 1. Start PostgreSQL

```bash
docker compose up -d postgres
```

The container is `pgvector/pgvector:pg16` with a persistent `pgdata` volume and a `pg_isready`
healthcheck. Credentials default to `rag / rag / rag` on port 5432 and can be overridden through
`POSTGRES_*` variables in `.env`.

### 2. Install and configure

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in the API keys
```

### 3. Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On startup the service creates the extension, tables and indexes (`AUTO_CREATE_SCHEMA=true`) and
verifies that an existing `embedding` column has the configured width (changing
`EMBEDDING_DIMENSIONS` after ingestion fails fast instead of at the first insert). The module-level
`app` is built lazily, so `uvicorn --factory app.main:create_app` works too. Interactive docs:
<http://localhost:8000/docs>.

To run the API in Docker as well (image runs as a non-root user with a `HEALTHCHECK`):

```bash
docker compose --profile full up --build
```

### Security and limits

- **Auth.** Set `API_KEY` and every `/api/v1/*` request must carry it in `X-API-Key`
  (constant-time comparison; `/health` and `/livez` stay open for probes). Leave it empty for
  local work.
- **Request bodies.** A pure-ASGI middleware runs before FastAPI reads any body. For uploads it
  checks the key first (so an anonymous upload is refused before a byte is read), then
  `MAX_UPLOAD_BYTES`; every other POST is limited to `MAX_JSON_BODY_BYTES` (32 KB). Both limits
  apply to `Content-Length` and to streamed bodies. `MAX_PDF_PAGES` is checked before any text
  extraction and `MAX_CHUNKS_PER_DOCUMENT` bounds the number of LLM and embedding calls one
  upload can trigger (both return 413).
- **Questions** are 3 to 2000 characters, refused when they contain control characters, and
  folded like the documents (ligatures to letters; superscripts, subscripts and fractions kept,
  so `10³` never becomes `103`).
- **Error responses** never echo upstream provider bodies (which can contain request ids, quota
  details or key fragments); clients get a provider name, a retryable/permanent hint and an
  incident id that matches the server log line. Database failures are a 503 with an incident id
  and no SQL, and validation errors never repeat the rejected input.
- **Liveness vs health.** `/livez` answers without touching anything; `/health` checks the
  database and pgvector. Point host probes and uptime monitors at `/livez`, so they never keep a
  serverless database awake.
- **Logs** carry identifiers and sizes, not document text or model drafts.
- **Compose** binds PostgreSQL and the API to `127.0.0.1` by default; override `POSTGRES_BIND` /
  `API_BIND` deliberately, and change the default database password before doing so.

---

## Public showcase

`DEMO_MODE=true` turns the service into something that can be left on the open internet on a
free account. `render.yaml` sets it; `python scripts/run_local_demo.py` runs the app in demo mode
on your machine, with the settings in `.env.local` (it does not read `render.yaml`).

| | Normal mode | Demo mode |
|---|---|---|
| `POST /api/v1/ingest` | Open, or `X-API-Key` when `API_KEY` is set | Only with `API_KEY`; without one it answers 403 |
| `POST /api/v1/query` | Open, or `X-API-Key` when `API_KEY` is set | Open to everyone, but budgeted (below) |
| Corpus | Whatever you ingest | `demo/` is ingested once into an empty database |
| `/docs` | Served | Hidden when `ENVIRONMENT=production` |

**The budget** (`app/budget.py`) refuses a question with `429` and a `Retry-After` header when
one visitor address passes 10 a minute or 60 a day, everyone together passes 20 a minute or 800
a day, or OpenRouter's own count of the day's free requests (read once a minute) falls to a
reserve (50, or a fifth of the account's daily allowance when that is smaller). That last rule is
the one a restart cannot erase: a free host forgets everything in memory. Before the budget, two
cheap checks spend nothing: questions over 500 characters get `422`, and a fifth question while
four are being answered gets `503` with `Retry-After`.

**Who is a visitor.** Behind Render the address comes from `CF-Connecting-IP`, which Cloudflare
overwrites (`CLIENT_IP_HEADER`). The left-most `X-Forwarded-For` entry is never used, because
proxies append to that header and the client writes its start; the fallback is the entry
`TRUSTED_PROXY_HOPS` places from the right. IPv6 addresses count per /64 network. With
`TRUST_PROXY_HEADERS=false` (the default) only the socket address is used.

**The playground** at `/` is a handful of static files with no build step. It asks a question,
shows the answer with its verbatim citations, and then walks through how that answer was built,
one stage at a time, with the numbers from that very request: the vector and keyword ranks, the
Reciprocal Rank Fusion arithmetic, how the cross-encoder reordered the candidates, how many
quotes survived verification, and where the time went. Each step links to the file that
implements it. Below the question box, a rank-fusion lab lets you change `k` or switch a search
arm off and watch the fused order change, and a glossary explains each concept in a few
sentences. The teaching layer (`assets/explain.js` and `explain.css`) is shared verbatim with the
project page; `tests/test_site.py` keeps the two copies identical and checks that every code link
points to a file that exists.

**The project page** (`site/`) is published by GitHub Pages. It shows `site/recorded/runs.json`,
which `scripts/record_runs.py` writes by driving the real application; nothing in it is edited
by hand, and `tests/test_site.py` re-checks every recorded quote against the sample PDFs. Each
recorded run opens into the same walkthrough as the playground, so the page teaches even while
the live demo sleeps.

**Fitting in 512 MB.** The two local models first took 643 MB and grew to 856 MB within a few
questions, which a free Render instance would not survive. Three changes brought that to 390 MB
steady and 441 MB at peak (measured on Windows, both models loaded, after a real generation):

| Change | Effect |
|---|---|
| `FASTEMBED_LOW_MEMORY=true`: ONNX Runtime's CPU memory arena off | 843 MB to 385 MB steady in isolation, about 15% slower inference |
| PyMuPDF imported only when a PDF is parsed | 26 MB less in a service that only answers questions |
| Models loaded on first use, then warmed in the background in demo mode | Start-up stays fast; the first visitor does not pay for loading |

The Docker image downloads both models at build time and runs with `HF_HUB_OFFLINE=1`, because a
free instance starts with an empty disk every time it wakes.

---

## curl examples

### Health

```bash
curl -s http://localhost:8000/health | jq
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "database_connected": true,
  "pgvector_installed": true,
  "pgvector_version": "0.8.1",
  "providers": {
    "embeddings":     {"provider": "openai",    "configured": true, "model": "text-embedding-3-small"},
    "reranker":       {"provider": "cohere",    "configured": true, "model": "rerank-v3.5"},
    "contextualizer": {"provider": "anthropic", "configured": true, "model": "claude-haiku-5-5"},
    "generator":      {"provider": "anthropic", "configured": true, "model": "claude-opus-5-5"}
  }
}
```

`/health` returns 503 when the database is unreachable and `"status": "degraded"` when a provider is
missing.

### Ingest a PDF

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@docs/SOP-QA-017-cleaning-validation.pdf" \
  -F "title=SOP-QA-017 Cleaning Validation" | jq
```

```json
{
  "document_id": "6b0f1c2e-3f7d-4a1e-9c1b-0c1f3a2d9e55",
  "filename": "SOP-QA-017-cleaning-validation.pdf",
  "title": "SOP-QA-017 Cleaning Validation",
  "chunks_count": 142,
  "pages_count": 38,
  "processing_time_ms": 24180.5
}
```

### Ask a question

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How long must batch records be retained after product expiry?"}' | jq
```

```json
{
  "query": "How long must batch records be retained after product expiry?",
  "answer": "Batch records must be kept for at least seven years after the product expiry date (SOP-QA-017, page 3).",
  "is_grounded": true,
  "confidence_score": 0.96,
  "citations": [
    {
      "document_title": "SOP-QA-017 Cleaning Validation",
      "page_number": 3,
      "chunk_id": "0d2a7c7e-8f0b-4c7a-9b8e-2f1c6a4d1b90",
      "verbatim_quote": "Batch records must be retained for a minimum of seven years after the product expiry date."
    }
  ],
  "retrieval_latency_ms": 412.7,
  "total_latency_ms": 2890.3,
  "status": "answered",
  "generation_model": "claude-opus-5-5"
}
```

### Restrict to one document

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who approves deviation reports?", "filter_document_id": "6b0f1c2e-3f7d-4a1e-9c1b-0c1f3a2d9e55"}' | jq
```

### Negative-out example

```json
{
  "query": "What is the CEO's salary?",
  "answer": "The retrieved passages describe cleaning validation, deviation approvals and record retention; they contain no information about executive compensation.",
  "is_grounded": false,
  "confidence_score": 0.0,
  "citations": [],
  "status": "insufficient_context",
  ...
}
```

---

## Configuration

All settings live in `app/config.py` and are read from the environment or `.env`
(see `.env.example`). The ones you will actually tune:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://rag:rag@localhost:5432/rag` | asyncpg driver required |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `openai` / `text-embedding-3-small` / `1536` | Dimensions are baked into the `vector(N)` column; change them before first ingestion |
| `LLM_PROVIDER` / `GENERATION_MODEL` / `GENERATION_EFFORT` | `anthropic` / `claude-opus-5-5` / `low` | `openai` and `openrouter` are the alternatives; effort is the Claude 5.x cost/latency lever |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODELS` / `OPENROUTER_FREE_ONLY` | unset / Phase 1 chain / `true` | Ordered `:free` chain with fallback; see the zero-cost section |
| `CONTEXTUALIZER_PROVIDER` / `CONTEXTUALIZER_MODEL` | `auto` / `claude-haiku-5-5` | `heuristic` makes no model calls; `openrouter` costs one free request per chunk |
| `RERANK_PROVIDER` / `COHERE_RERANK_MODEL` / `FASTEMBED_RERANK_MODEL` | `cohere` / `rerank-v3.5` / `Xenova/ms-marco-MiniLM-L-6-v2` | `fastembed` runs a local cross-encoder; `none` keeps RRF order |
| `RRF_K` / `TOP_K_RETRIEVAL` / `TOP_K_RERANK` | `60` / `50` / `3` | The spec's fusion and narrowing parameters |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` / `HNSW_EF_SEARCH` | `16` / `64` / `100` | Build-time values only apply when the index is first created |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `400` / `60` | Regex token counter; plug in tiktoken via `counter=` if you need BPE-exact sizes |
| `API_KEY` | unset | Enables `X-API-Key` auth on `/api/v1/*` |
| `MAX_UPLOAD_BYTES` / `MAX_PDF_PAGES` / `MAX_CHUNKS_PER_DOCUMENT` | `50 MiB` / `500` / `2000` | Ingestion ceilings, all enforced before expensive work |
| `GENERATION_MAX_TOKENS` / `CONTEXTUALIZER_MAX_TOKENS` | `4096` / `2048` | Thinking counts toward these budgets on Claude 5.x; the generator cap is 16K (non-streaming SDK limit) |
| `DB_CONNECT_TIMEOUT_S` / `HEALTH_TIMEOUT_S` | `10` / `5` | Fail fast on unreachable hosts; `/health` never hangs on a black-holed database |
| `DB_STARTUP_RETRIES` | `3` | Start-up retries while a serverless database resumes (a wrong setting is never retried) |
| `MAX_JSON_BODY_BYTES` | `32768` | Every POST except uploads, checked before the body is parsed |
| `DEMO_MODE` | `false` | Public showcase: closed uploads, budgeted questions, seeded corpus |
| `QUERY_RATE_LIMIT_PER_MINUTE` / `QUERY_DAILY_LIMIT_PER_CLIENT` | `10` / `60` | Per visitor address (demo mode) |
| `QUERY_GLOBAL_LIMIT_PER_MINUTE` / `QUERY_DAILY_LIMIT` | `20` / `800` | For everyone together (demo mode) |
| `QUERY_MAX_INFLIGHT` / `DEMO_MAX_QUERY_CHARS` | `4` / `500` | Load shedding and question length (demo mode) |
| `TRUST_PROXY_HEADERS` / `CLIENT_IP_HEADER` / `TRUSTED_PROXY_HOPS` | `false` / unset / `1` | Where the visitor address comes from behind a proxy (see Public showcase) |
| `LOCAL_INFERENCE_CONCURRENCY` | `1` | Local model calls at once (embedding and re-ranking share it) |
| `DOCUMENTS_CACHE_TTL_S` | `300` | Corpus listing cache, so page views do not wake the database |

---

## Tests

The default suite is fully hermetic: no network, no PostgreSQL, no API keys. External calls are
replaced by deterministic doubles (feature-hashed embeddings, a lexical re-ranker, an in-memory
store that runs the same RRF math as the SQL CTE, and a fake generator that can be switched to
hallucinate, refuse, or fail).

```bash
pytest
```

What is covered (the suite scrubs every `Settings`-related environment variable at start, so a
developer shell full of real keys or odd `LOG_LEVEL` values cannot change the outcome):

- `tests/test_rrf.py` - RRF scoring, tie-breaking identical to the SQL `ORDER BY`, limits,
  duplicate handling, SQL CTE structure, vector literal serialisation.
- `tests/test_chunker.py` - token budget invariant (including a BPE-like non-additive counter),
  overlap carry, headings/sections, sentence/word/token-slice splitting, termination on
  whitespace-free runs, page attribution, hyphenation handling, PDF round trip, page ceilings.
- `tests/test_api.py` - `/health`, `/api/v1/ingest` (valid, non-PDF, empty, corrupt, oversized via
  header and via streamed body, too many pages/chunks), `/api/v1/query` (grounded answer, lexical
  identifier retrieval, document filter, empty corpus short-circuit, hallucinated-quote
  rejection, insufficient context, refusal, provider outage 502/503 with sanitised details,
  validation 422, API-key auth), plus unit tests for quote anchoring (Unicode folds, ligatures,
  combining accents), preamble exclusion and prompt-injection resistance.
- `tests/test_adapters.py` - the real `AnthropicGroundedGenerator`, `OpenAIGroundedGenerator`,
  `AnthropicContextualizer`, `CohereReranker` and `OpenAIEmbedder` driven by stub SDK clients:
  request shapes (no sampling parameters on Claude, structured-output schema, cache breakpoints,
  nonce tags), refusal/truncation handling, exception-to-`ProviderError` mapping, batching, and
  the contextualizer's cache warm-up ordering.
- `tests/test_config.py` - settings parsing quirks (empty values mean unset, case-insensitive log
  level) and cross-field rules.
- `tests/test_openrouter.py` - the zero-cost stack: chain fallback on 429 / in-band errors /
  withdrawal / truncation / invalid JSON, strict-schema-to-JSON-mode retry, key rejection,
  content-filter refusals, request pacing, the free-only guard, the OpenRouter contextualizer
  with its call cap, the local cross-encoder re-ranker and the OpenRouter/fastembed embedders.
- `tests/test_demo_mode.py` - the public showcase: closed uploads, the question budget (per
  visitor, per minute, per day, OpenRouter's own count), corpus seeding, model warm-up, the
  playground, hidden docs in production, CORS, hosted-database URLs, and that the committed
  `.env.example` holds no secret.
- `tests/test_page_attribution.py` - page maps across page breaks and overlap, and that a
  citation carries the page its quote is on.
- `tests/test_site.py` - holds the published pages to their sources: every recorded quote is
  found verbatim in the sample PDF on the page it cites, the recorded stack matches the code,
  the pages load only their own scripts, and `render.yaml`, the `Dockerfile` and the settings
  agree with each other.

Live PostgreSQL integration tests run only when `TEST_DATABASE_URL` is set. They create and drop
their own schema (`rag_test`), verify the DDL and index definitions, assert that the SQL RRF scores
*and order* equal the Python reference, exercise the document filter, check the dimension-drift
guard, and confirm with `EXPLAIN` that the dense arm uses `idx_chunks_hnsw`:

```bash
TEST_DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag pytest tests/test_integration_db.py -v
```

No Docker available? `pip install pgserver` ships PostgreSQL with pgvector inside the virtualenv:

```bash
python scripts/run_live_db_tests.py
```

---

## Benchmark notes

`scripts/benchmark.py` fires concurrent queries at a running instance and reports p50/p95/p99 for
wall-clock, server total, and server retrieval time (the `retrieval_latency_ms` field covers query
embedding + SQL + re-ranking; `total_latency_ms` adds generation and verification):

```bash
python scripts/benchmark.py --url http://localhost:8000 --runs 40 --concurrency 4 \
  "How long must batch records be retained?" \
  "Who approves deviation reports?" \
  "What is the residue acceptance limit for product-contact equipment?"
```

Latency budget to expect per query with the default providers (indicative, measure your own):

| Stage | Typical | Dominant cost |
|---|---|---|
| Query embedding (OpenAI) | 40-120 ms | network round-trip |
| Hybrid CTE (100k chunks, warm cache) | 5-30 ms | HNSW scan + GIN scan |
| Cohere Rerank, 50 x ~450 tokens | 150-400 ms | cross-encoder inference |
| **Retrieval subtotal (p95 target)** | **< 800 ms** | |
| Claude Opus 5.5, effort=low, 3 passages | 1.5-4 s | generation |

How to stay inside the retrieval budget as the corpus grows:

1. Keep the HNSW graph in memory (`shared_buffers`), watch `pg_stat_user_indexes` for
   `idx_chunks_hnsw` reads.
2. Lower `HNSW_EF_SEARCH` toward `TOP_K_RETRIEVAL` once recall is validated; latency is roughly
   linear in `ef_search`.
3. Reduce `TOP_K_RETRIEVAL` (e.g. 30) if the re-ranker dominates; RRF quality is stable well below
   50 candidates for most corpora.
4. Use `EMBEDDING_DIMENSIONS=768` (Matryoshka truncation) to halve index size and distance cost.
5. Cache query embeddings for repeated questions (Phase 4 of the roadmap adds a Redis semantic
   cache in front of the whole pipeline).

Quality proof points to track alongside latency: citation verification pass rate (log line
`Dropped N unverifiable citation(s)`), share of `insufficient_context` answers on a golden set, and
RAG-triad faithfulness via Ragas/DeepEval.

---

## Project layout

```
hybrid-retrieval-engine/
├── docker-compose.yml           # pgvector/pgvector:pg16 (+ optional api profile)
├── Dockerfile                   # production image: non-root, local models baked in, offline
├── render.yaml                  # Render Blueprint: one free web service in demo mode
├── .dockerignore
├── requirements.txt
├── .env.example
├── pyproject.toml               # pytest (asyncio auto mode) + ruff config
├── app/
│   ├── config.py                # pydantic-settings (paid defaults + zero-cost switches)
│   ├── openrouter.py            # free-model chain: fallback, strict schema -> JSON mode, pacing
│   ├── budget.py                # demo mode: who may ask, how often, and the day's allowance
│   ├── demo.py                  # the sample corpus manifest and its idempotent seeding
│   ├── database.py              # async engine, DDL bootstrap, transactional writes
│   ├── schemas.py               # API + internal Pydantic models
│   ├── errors.py                # ProviderError / PDFParseError / ConfigurationError
│   ├── textnorm.py              # Unicode folding shared by PDF text and questions
│   ├── playground/static/       # the browser playground served at / (no build step);
│   │                            # assets/explain.js is the walkthrough, lab and glossary
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PyMuPDF blocks with page numbers and bboxes
│   │   ├── chunker.py           # structural recursive chunker with overlap and page maps
│   │   ├── pipeline.py          # parse -> chunk -> contextualize -> embed -> store
│   │   └── contextualizer.py    # Contextual Retrieval worker: Claude / OpenRouter / heuristic
│   ├── retrieval/
│   │   ├── embedder.py          # OpenAI / OpenRouter / fastembed (local) / hashing embedders
│   │   ├── hybrid_search.py     # RRF CTE + pure-Python reference
│   │   └── reranker.py          # Cohere Rerank v3.5 / fastembed cross-encoder / passthrough
│   ├── generator/
│   │   └── grounded_llm.py      # constrained output + verbatim citation verification
│   └── main.py                  # FastAPI routes, request guard middleware, auth, playground, CORS
├── requirements-local-models.txt # fastembed for the zero-cost stack
├── demo/                        # four fictional quality documents (PDF) + corpus.json
├── site/                        # the project page, published on GitHub Pages
│   ├── index.html               # the one place that holds the live demo's address
│   ├── assets/                  # site.css, site.js, theme.js, favicon.svg, and a copy of
│   │                            # explain.js/.css (the teaching layer): no build step
│   └── recorded/runs.json       # seven real runs, saved as the service returned them
├── docs/DEPLOY.md               # putting it online, step by step
├── .github/workflows/           # tests.yml (suite + lint), pages.yml (publishes site/)
├── scripts/
│   ├── benchmark.py             # p50/p95/p99 latency against a running instance
│   ├── make_demo_corpus.py      # writes the sample PDFs and their manifest
│   ├── record_runs.py           # seeds the corpus and records the runs shown on the page
│   ├── run_local_demo.py        # the showcase on your machine, in demo mode
│   ├── openrouter_smoke.py      # live check of the free-model chain (1-2 free requests)
│   └── run_live_db_tests.py     # live-DB tests on embedded PostgreSQL (pgserver)
└── tests/                       # hermetic suite + opt-in live PostgreSQL tests
```

---

## Production hardening checklist

This repository is the Phase 2 milestone of a four-phase Applied AI curriculum; the items below are
intentionally out of scope here and land in later phases:

- Per-user authentication and quotas (today: one shared `API_KEY`, and in demo mode an
  in-process budget per visitor address, which a restart forgets and a second worker would not
  share).
- Idempotent re-ingestion (content hash on `documents`) and a `DELETE /api/v1/documents/{id}`.
- OCR for scanned PDFs, and tables: text is extracted as layout blocks only.
- Observability: Langfuse tracing of every stage, token and cost telemetry (Phase 4).
- Evaluation CI: golden query set, RAG-triad metrics, PR gating on faithfulness (Phase 4).
- Redis semantic cache for repeated queries (Phase 4).
