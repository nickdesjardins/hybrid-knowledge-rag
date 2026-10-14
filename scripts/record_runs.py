"""Seed the demo corpus into the configured database, then record real runs for the project page.

It drives the real application in-process (same code path as the deployed service): local
embeddings and re-ranking, the PostgreSQL hybrid-search CTE, and the OpenRouter free-model
chain. Every recorded answer is what the service actually returned; nothing is edited by hand.

    python scripts/record_runs.py                 # seeds if needed, records site/recorded/runs.json
    python scripts/record_runs.py --seed-only     # only make sure the corpus is in the database
    python scripts/record_runs.py --reseed        # replace the sample documents, then record

Spends one free OpenRouter request per recorded question (seven by default).
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

OUT = ROOT / "site" / "recorded" / "runs.json"

#: The outcome each question should have. The page counts results against this, so a wrongful
#: "not in the documents" on an answerable question shows as a miss, not as a success.
NOT_IN_CORPUS = {"not-in-corpus"}

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
    args = parser.parse_args()

    settings = Settings(demo_mode=True, demo_seed_on_startup=False, log_level="WARNING")
    components = await build_components(settings)
    try:
        if args.reseed:
            names = [entry.file for entry in load_manifest(settings.demo_seed_dir)]
            deleted = await components.db.delete_documents_by_filename(names)
            components.catalogue.invalidate()  # the cached listing still holds the deleted rows
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
            "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)} ({len(runs)} runs)")
        return 0
    finally:
        await components.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
