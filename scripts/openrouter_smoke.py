"""Live check of the zero-cost generation path: OpenRouter free models + the grounded prompt.

Spends 1-2 free requests from the account's daily allowance (50/day without credits).

    python scripts/openrouter_smoke.py            # reads OPENROUTER_API_KEY from .env / env

What it prints:
1. which models of OPENROUTER_MODELS the public catalogue still lists (no key needed);
2. the account's free-request allowance for today (GET /key, sends the key);
3. one grounded answer over a built-in passage, with the model that answered, the status,
   and the verified citation.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.generator.grounded_llm import OpenRouterGroundedGenerator, ground_answer  # noqa: E402
from app.schemas import RerankedChunk  # noqa: E402

PASSAGE = (
    "Batch records must be retained for a minimum of seven years after the product expiry "
    "date. Electronic records require an audit trail compliant with 21 CFR Part 11 and must be "
    "backed up weekly."
)
QUESTION = "How long must batch records be retained after the product expiry date?"


def listed_models(base_url: str) -> set[str]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models", headers={"User-Agent": "hybrid-retrieval-engine smoke"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return {m["id"] for m in json.load(response)["data"]}


def allowance(base_url: str, key: str) -> dict | None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/key",
        headers={"Authorization": f"Bearer {key}", "User-Agent": "hybrid-retrieval-engine smoke"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response).get("data")
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"could not read the key allowance: {type(exc).__name__}")
        return None


async def main() -> int:
    settings = Settings(llm_provider="openrouter")
    if settings.openrouter_api_key is None:
        print("OPENROUTER_API_KEY is not set (put it in .env). Nothing was sent.")
        return 2

    chain = settings.openrouter_model_chain
    try:
        listed = listed_models(settings.openrouter_base_url)
        for model in chain:
            print(
                f"catalogue: {model}: {'listed' if model in listed else 'NOT LISTED (withdrawn?)'}"
            )
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"could not read the public catalogue: {type(exc).__name__}")

    data = allowance(settings.openrouter_base_url, settings.openrouter_api_key.get_secret_value())
    if data:
        daily = data.get("free_model_daily_requests") or {}
        print(
            f"free requests today: used={daily.get('used')} remaining={daily.get('remaining')} "
            f"limit={daily.get('limit')} (free tier: {data.get('is_free_tier')})"
        )

    chunk = RerankedChunk(
        id="smoke-chunk-1",
        document_id="smoke-doc",
        document_title="SOP-QA-017 Records Retention",
        chunk_index=0,
        page_number=3,
        raw_content=PASSAGE,
        contextualized_content=f"Document: SOP-QA-017. Section: 3. Records.\n\n{PASSAGE}",
        rrf_score=0.03,
        relevance_score=0.9,
        rerank_position=1,
    )
    generator = OpenRouterGroundedGenerator(settings)
    print(f"asking {chain[0]} (fallbacks: {chain[1:] or 'none'}) ...")
    result = await generator.generate(QUESTION, [chunk])
    answer = ground_answer(
        query=QUESTION,
        result=result,
        chunks=[chunk],
        retrieval_latency_ms=0.0,
        total_latency_ms=0.0,
    )
    print(f"model used : {result.model}")
    print(
        f"status     : {answer.status.value}  grounded={answer.is_grounded}  "
        f"confidence={answer.confidence_score}"
    )
    print(f"answer     : {answer.answer}")
    for citation in answer.citations:
        print(f'citation   : p.{citation.page_number} "{citation.verbatim_quote}"')
    return 0 if answer.is_grounded else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
