"""A citation's page must be the page the quote is on, not the page its chunk starts on."""

from __future__ import annotations

from httpx import AsyncClient

from app.generator.grounded_llm import LLMCitation, locate_quote_span, verify_citations
from app.ingestion.chunker import (
    TextUnit,
    chunk_document,
    pack_units,
    page_for_offset,
)
from app.ingestion.pdf_parser import parse_pdf_bytes
from app.retrieval.hybrid_search import _decode_page_map
from app.schemas import RerankedChunk

from .conftest import InMemoryStore, make_pdf

TWO_PAGES = [
    [
        "WIDGET POLICY",
        "This policy introduces the general handling rules for widgets in the warehouse.",
    ],
    ["The widget warranty period is exactly five years from the delivery date."],
]


def test_page_for_offset() -> None:
    page_map = [(0, 3), (120, 4), (300, 5)]
    assert page_for_offset(page_map, 0, default=9) == 3
    assert page_for_offset(page_map, 119, default=9) == 3
    assert page_for_offset(page_map, 120, default=9) == 4
    assert page_for_offset(page_map, 10_000, default=9) == 5
    assert page_for_offset(None, 50, default=9) == 9
    assert page_for_offset([], 50, default=9) == 9


def test_chunk_spanning_two_pages_records_where_the_page_changes() -> None:
    units = [
        TextUnit(1, "First paragraph on page one."),
        TextUnit(1, "Second paragraph on page one."),
        TextUnit(2, "Third paragraph is on page two."),
    ]
    (chunk,) = pack_units(units, chunk_size=200, overlap=0)
    assert chunk.page_number == 1  # the chunk starts on page 1 ...
    offset = chunk.text.index("Third paragraph")
    assert chunk.page_map == [(0, 1), (offset, 2)]  # ... and continues on page 2
    assert page_for_offset(chunk.page_map, chunk.text.index("Second"), chunk.page_number) == 1
    assert page_for_offset(chunk.page_map, offset + 5, chunk.page_number) == 2


def test_overlap_carried_from_the_previous_page_keeps_its_page() -> None:
    units = [
        TextUnit(1, "Alpha beta gamma delta epsilon zeta eta theta."),
        TextUnit(2, "Iota kappa lambda mu nu xi omicron pi rho."),
    ]
    first, second = pack_units(units, chunk_size=16, overlap=4)
    assert first.page_map == [(0, 1)]
    assert second.page_number == 2
    assert second.overlap_text and second.text.startswith(second.overlap_text)
    body_offset = second.text.index("Iota")
    # The carried words still belong to page 1; the body starts on page 2.
    assert second.page_map == [(0, 1), (body_offset, 2)]
    assert page_for_offset(second.page_map, 0, second.page_number) == 1
    assert page_for_offset(second.page_map, body_offset, second.page_number) == 2


def test_single_page_chunks_have_a_one_entry_map() -> None:
    chunks = pack_units([TextUnit(4, "Only page four here.")], chunk_size=50, overlap=0)
    assert chunks[0].page_map == [(0, 4)]


def test_pdf_chunk_across_a_page_break_maps_both_pages() -> None:
    document = parse_pdf_bytes(make_pdf(TWO_PAGES), "widgets.pdf")
    (chunk,) = chunk_document(document, chunk_size=200, overlap=0)
    assert chunk.page_number == 1
    assert [page for _, page in chunk.page_map] == [1, 2]
    start, _ = locate_quote_span("warranty period is exactly five years", chunk.text)
    assert page_for_offset(chunk.page_map, start, chunk.page_number) == 2


def _chunk(page_map) -> RerankedChunk:
    text = "Rules for page one apply here.\n\nThe retention period on page two is seven years."
    return RerankedChunk(
        id="c1",
        document_id="d1",
        document_title="Doc",
        chunk_index=0,
        page_number=1,
        raw_content=text,
        contextualized_content=f"Ctx.\n\n{text}",
        rrf_score=0.03,
        relevance_score=0.9,
        rerank_position=1,
        page_map=page_map,
    )


def test_verify_citations_uses_the_page_of_the_quote() -> None:
    chunk = _chunk([(0, 1), (32, 2)])
    assert chunk.raw_content[32:].startswith("The retention period")
    verified, dropped = verify_citations(
        [
            LLMCitation(chunk_id="c1", verbatim_quote="Rules for page one apply here."),
            LLMCitation(
                chunk_id="c1", verbatim_quote="retention period on page two is seven years"
            ),
        ],
        [chunk],
    )
    assert dropped == 0
    assert [c.page_number for c in verified] == [1, 2]


def test_verify_citations_falls_back_to_the_chunk_page_without_a_map() -> None:
    verified, _ = verify_citations(
        [LLMCitation(chunk_id="c1", verbatim_quote="retention period on page two")], [_chunk(None)]
    )
    assert verified[0].page_number == 1  # rows ingested before page maps existed


def test_decode_page_map_accepts_text_lists_and_garbage() -> None:
    assert _decode_page_map("[[0, 1], [40, 2]]") == [(0, 1), (40, 2)]
    assert _decode_page_map([[0, 1], [40, 2]]) == [(0, 1), (40, 2)]
    assert _decode_page_map(None) is None
    assert _decode_page_map("not json") is None
    assert _decode_page_map("[1, 2, 3]") is None


async def test_api_cites_the_page_the_quote_is_on(
    client: AsyncClient, store: InMemoryStore
) -> None:
    response = await client.post(
        "/api/v1/ingest", files={"file": ("widgets.pdf", make_pdf(TWO_PAGES), "application/pdf")}
    )
    assert response.status_code == 200 and response.json()["chunks_count"] == 1
    assert response.json()["pages_count"] == 2
    assert store.chunks[0]["page_number"] == 1 and store.chunks[0]["page_map"][-1][1] == 2

    answer = await client.post(
        "/api/v1/query", json={"query": "What is the widget warranty period?"}
    )
    body = answer.json()
    assert body["status"] == "answered"
    citation = body["citations"][0]
    assert "five years" in citation["verbatim_quote"]
    assert citation["page_number"] == 2  # the chunk starts on page 1; the quote is on page 2
    assert body["retrieved_passages"][0]["page_number"] == 1
    listed = (await client.get("/api/v1/documents")).json()["documents"]
    assert listed[0]["pages_count"] == 2
