"""Latency benchmark for POST /api/v1/query.

Example::

    python scripts/benchmark.py --url http://localhost:8000 --runs 30 --concurrency 4 \
        "How long must batch records be retained?" "Who approves deviation reports?"

Reports wall-clock, server-side total, and server-side retrieval latency percentiles. Use it to
check the sub-800 ms p95 retrieval target after tuning HNSW_EF_SEARCH / TOP_K_RETRIEVAL.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


async def run(
    url: str, queries: list[str], runs: int, concurrency: int, document_id: str | None
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    wall: list[float] = []
    total: list[float] = []
    retrieval: list[float] = []
    statuses: dict[str, int] = {}

    async with httpx.AsyncClient(base_url=url, timeout=120.0) as client:

        async def one(query: str) -> None:
            payload = {"query": query}
            if document_id:
                payload["filter_document_id"] = document_id
            async with semaphore:
                started = time.perf_counter()
                response = await client.post("/api/v1/query", json=payload)
                elapsed = (time.perf_counter() - started) * 1000.0
            wall.append(elapsed)
            if response.status_code != 200:
                statuses[f"http_{response.status_code}"] = (
                    statuses.get(f"http_{response.status_code}", 0) + 1
                )
                return
            body = response.json()
            total.append(body["total_latency_ms"])
            retrieval.append(body["retrieval_latency_ms"])
            statuses[body["status"]] = statuses.get(body["status"], 0) + 1

        tasks = [one(queries[i % len(queries)]) for i in range(runs)]
        started = time.perf_counter()
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started

    def row(name: str, values: list[float]) -> str:
        if not values:
            return f"{name:<22} n/a"
        return (
            f"{name:<22} p50={percentile(values, 50):8.1f}  p95={percentile(values, 95):8.1f}  "
            f"p99={percentile(values, 99):8.1f}  mean={statistics.fmean(values):8.1f}  max={max(values):8.1f}"
        )

    print(
        f"\n{runs} requests, concurrency={concurrency}, {elapsed:.2f}s wall, "
        f"{runs / elapsed if elapsed else 0:.2f} req/s"
    )
    print(f"statuses: {statuses}")
    print(row("wall-clock ms", wall))
    print(row("server total ms", total))
    print(row("server retrieval ms", retrieval))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("queries", nargs="+", help="One or more questions to cycle through")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--document-id", default=None)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.queries, args.runs, args.concurrency, args.document_id))


if __name__ == "__main__":
    main()
