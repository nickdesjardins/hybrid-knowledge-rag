"""Seed the demo corpus into the configured database, then record real runs for the project page.

It drives the real application in-process (same code path as the deployed service): local
embeddings and re-ranking, the PostgreSQL hybrid-search CTE, and the OpenRouter free-model
chain. Every recorded answer is what the service actually returned; nothing is edited by hand.

    python scripts/record_runs.py                 # seeds if needed, records site/recorded/runs.json
    python scripts/record_runs.py --seed-only     # only make sure the corpus is in the database
    python scripts/record_runs.py --reseed        # replace the sample documents, then record
    python scripts/record_runs.py --maps-only     # add a vector map to every recorded run

Spends one free OpenRouter request per recorded question (seven by default).

``--maps-only`` spends none and writes nothing to the database: for each recorded question it
replays retrieval (embedding, hybrid search, re-ranking; no language model), checks that the
replay finds exactly the passages that were recorded, and stores the run's ``vector_map`` next
to its response, which stays as the service returned it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
from datetime import UTC, datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.demo import load_manifest, seed_demo_corpus  # noqa: E402
from app.generator.grounded_llm import SYSTEM_PROMPT  # noqa: E402
from app.main import build_components, create_app  # noqa: E402
from app.retrieval.vector_map import VectorMapper  # noqa: E402

OUT = ROOT / "site" / "recorded" / "runs.json"

#: The outcome each question should have. The page counts results against this, so a wrongful
#: "not in the documents" on an answerable question shows as a miss, not as a success.
NOT_IN_CORPUS = {"not-in-corpus"}

#: What a replayed passage must share with the recorded one, field by field.
PASSAGE_KEYS = ("chunk_id", "dense_rank", "sparse_rank", "fused_rank", "rerank_position")

#: What a map point must show exactly as its recorded passage does.
PASSAGE_LABELS = ("document_title", "page_number", "preview")

#: (id, question, why it is on the page)
QUESTIONS: list[tuple[str, str, str]] = [
    (
        "retention",
        "How long must batch records be retained after product expiry?",
        "A plain factual lookup: one clause in one policy answers it.",
    ),
    (
        "regulation",
        "Which regulation governs electronic records and audit trails?",
        "An exact identifier (21 CFR Part 11): the kind of query keyword search exists for.",
    ),
    (
        "sampling",
        "What sampling plan applies to incoming excipients?",
        "A standard number and a quality limit that must be quoted exactly, not paraphrased.",
    ),
    (
        "closure",
        "Who approves the closure of a critical deviation?",
        "Two similar clauses exist (minor/major vs critical); the right one must be cited.",
    ),
    (
        "cross-document",
        "How long are cleaning validation reports kept?",
        "The cleaning SOP only points to the retention policy; the answer lives in the other document.",
    ),
    (
        "hold-time",
        "What is the maximum dirty hold time before cleaning must start?",
        "A number with a unit, next to a different hold time that must not be confused with it.",
    ),
    (
        "not-in-corpus",
        "What is the company's annual revenue?",
        "Nothing in the corpus says this. The engine has to say so instead of guessing.",
    ),
]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def dump_record(value: object, level: int = 0) -> str:
    """``json.dumps(indent=2)``, except that a list of numbers stays on one line.

    A map carries a 384-number vector per question and per spotlight chunk; one number per line
    would make the file tens of thousands of lines long and its diffs unreadable.
    """
    pad = "  " * level
    if isinstance(value, dict) and value:
        items = [
            f"{pad}  {json.dumps(key, ensure_ascii=False)}: {dump_record(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and value:
        if all(_is_number(item) for item in value):
            return "[" + ", ".join(json.dumps(item) for item in value) + "]"
        items = [f"{pad}  {dump_record(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(value, ensure_ascii=False)


def write_record(record: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dump_record(record) + "\n", encoding="utf-8")


async def record_maps() -> int:
    """Add a vector map to every recorded run: no seeding, no language model, no database write."""
    record = json.loads(OUT.read_text(encoding="utf-8"))
    stack = record["stack"]
    settings = Settings(
        log_level="WARNING",
        auto_create_schema=False,  # read-only: not even the idempotent DDL
        demo_mode=False,  # no question budget: nothing here asks OpenRouter anything
        top_k_retrieval=stack["top_k_retrieval"],
        top_k_rerank=stack["top_k_rerank"],
        rrf_k=stack["rrf_k"],
    )
    components = await build_components(settings)
    components.generator = None  # never needed here, so never there to be called
    try:
        if components.db is None or components.db.pgvector_version is None:
            print("The database is unreachable (or has no pgvector); runs.json is unchanged.")
            return 1
        in_use = {
            "embeddings": getattr(components.embedder, "model_name", components.embedder.name),
            "reranker": getattr(components.reranker, "model", None),
        }
        for key, model in in_use.items():
            if model != stack[key]:
                print(f"The runs were recorded with {key} {stack[key]}, this setup uses {model}.")
                return 1
        mapper = components.vector_map or VectorMapper(components.db, settings)  # type: ignore[arg-type]

        maps: dict[str, dict] = {}
        for run in record["runs"]:
            body = run["response"]
            question = body["query"]  # as the service stored it, after folding
            query_vector = await components.embedder.embed_query(question)
            hybrid = await components.searcher.search_with_arms(
                query_text=question,
                query_vector=query_vector,
                document_id=None,
                limit=settings.top_k_retrieval,
            )
            candidates = hybrid.candidates
            reranked = (
                await components.reranker.rerank(question, candidates, top_n=settings.top_k_rerank)
                if candidates
                else []
            )
            fused = {chunk.id: position for position, chunk in enumerate(candidates, start=1)}
            replayed = [
                (c.id, c.dense_rank, c.sparse_rank, fused[c.id], c.rerank_position)
                for c in reranked
            ]
            recorded = [
                tuple(passage[key] for key in PASSAGE_KEYS)
                for passage in body["retrieved_passages"]
            ]
            expected = body["pipeline"]["candidates"]
            if replayed != recorded or len(candidates) != expected:
                print(f"[{run['id']}] the replay differs from the recording; nothing written")
                print(f"    candidates: recorded {expected}, now {len(candidates)}")
                print(f"    recorded {PASSAGE_KEYS}: {recorded}")
                print(f"    replayed {PASSAGE_KEYS}: {replayed}")
                return 1
            vector_map = await mapper.build(
                query_vector=query_vector,
                candidates=candidates,
                cut=hybrid.cut,
                reranked=reranked,
            )
            if vector_map is None:
                print(
                    f"[{run['id']}] no map (under 3 chunks, over VECTOR_MAP_MAX_CHUNKS, or "
                    "chunks that do not spread in two directions)"
                )
                return 1
            # The map's labels come from another statement (Database.chunk_vectors) than the
            # passages' did: the page shows both for the same chunk, so they must agree.
            points = {point.chunk_id: point for point in vector_map.points}
            for passage in body["retrieved_passages"]:
                point = points[passage["chunk_id"]]
                shown = (point.document_title, point.page_number, point.preview)
                if shown != tuple(passage[key] for key in PASSAGE_LABELS):
                    print(f"[{run['id']}] the map labels a passage differently; nothing written")
                    print(f"    passage {PASSAGE_LABELS}: {[passage[k] for k in PASSAGE_LABELS]}")
                    print(f"    map point: {list(shown)}")
                    return 1
            maps[run["id"]] = vector_map.model_dump(mode="json")
            nearest = max(vector_map.points, key=lambda p: p.similarity)
            print(
                f"[{run['id']}] replay matches ({len(candidates)} candidates); question at "
                f"({vector_map.query.x:+.4f}, {vector_map.query.y:+.4f}), nearest chunk "
                f"{nearest.document_title[:10]} #{nearest.chunk_index} "
                f"(similarity {nearest.similarity:.4f}), {len(vector_map.vectors)} spotlight vectors"
            )

        for run in record["runs"]:
            run["vector_map"] = maps[run["id"]]
        record["vector_maps_recorded_at"] = _now()
        write_record(record)
        first = next(iter(maps.values()))
        print(
            f"wrote {OUT.relative_to(ROOT)}: {len(maps)} maps of {first['corpus_chunks']} chunks, "
            f"{first['dimensions']} dimensions, explained variance {first['explained_variance']}"
        )
        return 0
    finally:
        await components.close()


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="delete the documents named in demo/corpus.json first, then ingest them again "
        "(needed after the sample PDFs or the chunking change)",
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="replay retrieval for the recorded runs and add their vector maps (no seeding, "
        "no language model, read-only against the database)",
    )
    args = parser.parse_args()
    if args.maps_only:
        return await record_maps()

    settings = Settings(demo_mode=True, demo_seed_on_startup=False, log_level="WARNING")
    components = await build_components(settings)
    try:
        if args.reseed:
            names = [entry.file for entry in load_manifest(settings.demo_seed_dir)]
            deleted = await components.db.delete_documents_by_filename(names)
            # The cached listing and the map's projection still hold the deleted rows.
            components.catalogue.invalidate()
            if components.vector_map is not None:
                components.vector_map.invalidate()
            print(f"reseed: deleted {deleted} of {len(names)} sample documents")
        report = await seed_demo_corpus(components, settings)
        documents = await components.catalogue.list_documents()
        print(
            f"corpus: {len(report.ingested)} ingested, {len(report.skipped)} already present, "
            f"{len(report.failed)} failed -> {len(documents)} documents, "
            f"{sum(d.chunks_count for d in documents)} chunks"
        )
        if report.failed:
            print("failed:", report.failed)
            return 1
        if args.seed_only:
            return 0
        if components.generator is None:
            print("No generator configured (OPENROUTER_API_KEY missing?). Nothing recorded.")
            return 2

        app = create_app(settings=settings, components=components)
        runs = []
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://record"
        ) as client:
            health = (await client.get("/health")).json()
            for run_id, question, why in QUESTIONS:
                started = time.perf_counter()
                response = await client.post("/api/v1/query", json={"query": question})
                wall_ms = round((time.perf_counter() - started) * 1000.0, 1)
                body = response.json()
                if response.status_code != 200:
                    print(f"[{run_id}] HTTP {response.status_code}: {body}")
                    return 1
                expect = "insufficient_context" if run_id in NOT_IN_CORPUS else "answered"
                runs.append(
                    {
                        "id": run_id,
                        "why": why,
                        "expect": expect,
                        "wall_ms": wall_ms,
                        "response": body,
                    }
                )
                if body["status"] != expect:
                    print(f"    NOTE: expected {expect}, got {body['status']}")
                first = body["citations"][0] if body["citations"] else None
                print(
                    f"[{run_id}] {body['status']} conf={body['confidence_score']} "
                    f"retrieval={body['retrieval_latency_ms']:.0f}ms total={body['total_latency_ms']:.0f}ms "
                    f"model={body['generation_model']}"
                )
                print(f"    answer: {body['answer'][:160]}")
                if first:
                    print(
                        f"    cite  : {first['document_title'][:40]} p.{first['page_number']}: "
                        f"{first['verbatim_quote'][:110]}"
                    )

        record = {
            "recorded_at": _now(),
            "stack": {
                "database": f"PostgreSQL + pgvector {health.get('pgvector_version')}",
                "embeddings": health["providers"]["embeddings"]["model"],
                "reranker": health["providers"]["reranker"]["model"],
                "generator_chain": settings.openrouter_model_chain,
                "top_k_retrieval": settings.top_k_retrieval,
                "top_k_rerank": settings.top_k_rerank,
                "rrf_k": settings.rrf_k,
            },
            "documents": [
                {"title": d.title, "pages": d.pages_count, "chunks": d.chunks_count}
                for d in documents
            ],
            "system_prompt": SYSTEM_PROMPT,
            "runs": runs,
        }
        write_record(record)
        print(f"wrote {OUT.relative_to(ROOT)} ({len(runs)} runs)")
        return 0
    finally:
        await components.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
