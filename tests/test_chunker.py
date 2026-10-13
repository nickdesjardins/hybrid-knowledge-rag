"""Unit tests for structural chunking, overlap, page attribution, and context prepending."""

from __future__ import annotations

import time

import pytest

from app.ingestion.chunker import (
    Chunk,
    TextUnit,
    chunk_document,
    chunk_text,
    count_tokens,
    explode_unit,
    is_heading,
    pack_units,
    slice_word,
    tail_words,
)
from app.ingestion.contextualizer import (
    HeuristicContextualizer,
    apply_context,
    document_as_prompt_text,
)
from app.ingestion.pdf_parser import (
    PageText,
    ParsedDocument,
    TextBlock,
    clean_block_text,
    parse_pdf_bytes,
)

LONG_MARKDOWN = "\n\n".join(
    [
        "# Cleaning Validation Procedure",
        "## 1. Purpose",
        " ".join(
            f"Purpose sentence number {i} describes why cleaning validation matters."
            for i in range(1, 9)
        ),
        "## 2. Scope",
        " ".join(
            f"Scope sentence number {i} covers equipment on production line {i}."
            for i in range(1, 9)
        ),
        "## 3. Acceptance Criteria",
        " ".join(
            f"Criterion {i} requires residue below {i} ppm after rinsing." for i in range(1, 9)
        ),
        "## 4. Records",
        " ".join(
            f"Record {i} must be archived for {i + 4} years in the QA vault." for i in range(1, 9)
        ),
    ]
)


# ------------------------------------------------------------------------ tokens/headings


def test_count_tokens_counts_words_and_punctuation() -> None:
    assert count_tokens("Hello, world!") == 4
    assert count_tokens("") == 0
    assert count_tokens("SOP-QA-017 applies.") == 7  # SOP - QA - 017 applies .


@pytest.mark.parametrize(
    "text, expected",
    [
        ("# Title", True),
        ("## 2.1 Scope", True),
        ("3. Records", True),
        ("4.2.1 Residue Limits", True),
        ("QUALITY MANUAL", True),
        ("This is a normal sentence that ends with a period.", False),
        ("records are kept, usually", False),
        ("", False),
        ("x" * 200, False),
    ],
)
def test_is_heading(text: str, expected: bool) -> None:
    assert is_heading(text) is expected


# ------------------------------------------------------------------------------ packing


def test_chunks_respect_token_budget() -> None:
    chunks = chunk_text(LONG_MARKDOWN, chunk_size=60, overlap=10)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.token_count <= 60, chunk.text
        assert chunk.token_count == count_tokens(chunk.text)


def test_chunk_indices_are_sequential_and_text_non_empty() -> None:
    chunks = chunk_text(LONG_MARKDOWN, chunk_size=60, overlap=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.text.strip() for c in chunks)
    assert all(c.body.strip() for c in chunks)


def test_sliding_window_overlap_carries_previous_tail() -> None:
    chunks = chunk_text(LONG_MARKDOWN, chunk_size=60, overlap=10)
    assert chunks[0].overlap_text == ""
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.overlap_text == tail_words(previous.body, 10)
        assert current.overlap_text
        assert count_tokens(current.overlap_text) <= 10
        assert current.text.startswith(current.overlap_text)


def test_zero_overlap_produces_disjoint_chunks() -> None:
    chunks = chunk_text(LONG_MARKDOWN, chunk_size=60, overlap=0)
    assert all(c.overlap_text == "" for c in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert not current.text.startswith(" ".join(previous.text.split()[-3:]))


def test_exploded_sentences_are_rejoined_with_spaces_verbatim() -> None:
    paragraph = " ".join(f"Sentence number {i} is short." for i in range(1, 13))
    chunks = chunk_text(paragraph, chunk_size=30, overlap=0)
    assert len(chunks) > 1
    # Every chunk body is a verbatim substring of the paragraph (no injected newlines).
    assert all(c.body in paragraph for c in chunks)
    assert "".join(c.body + " " for c in chunks).strip() == paragraph


def test_headings_become_sections_and_start_chunks() -> None:
    chunks = chunk_text(LONG_MARKDOWN, chunk_size=80, overlap=0)
    sections = {c.section for c in chunks}
    assert "1. Purpose" in sections
    assert "4. Records" in sections
    assert any("## 4. Records" in c.body for c in chunks)
    records_chunks = [c for c in chunks if c.section == "4. Records"]
    assert records_chunks and all("Record" in c.text for c in records_chunks)
    # The very first chunk starts at the document title.
    assert chunks[0].text.startswith("# Cleaning Validation Procedure")


def test_heading_merged_into_short_buffer_relabels_the_chunk() -> None:
    units = [
        TextUnit(1, "Tiny tail."),
        TextUnit(1, "## Next Section", is_heading=True),
        TextUnit(1, "Body of the next section with several words in it."),
    ]
    chunks = pack_units(units, chunk_size=60, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].section == "Next Section"
    assert chunks[0].text.startswith("Tiny tail.\n\n## Next Section")


def test_slice_word_splits_whitespace_free_runs_at_token_boundaries() -> None:
    run = "a.b.c.d.e.f.g"  # 13 regex tokens
    pieces = slice_word(run, max_tokens=4)
    assert "".join(pieces) == run
    assert all(count_tokens(p) <= 4 for p in pieces)
    assert len(pieces) == 4
    assert slice_word("indivisible", max_tokens=1) == ["indivisible"]


def test_explode_unit_marks_joiners_for_sentences_and_slices() -> None:
    unit = TextUnit(2, "First one. Second one. Third one.")
    pieces = explode_unit(unit, max_tokens=5)
    assert [p.text for p in pieces] == ["First one.", "Second one.", "Third one."]
    assert [p.joiner for p in pieces] == ["\n\n", " ", " "]
    assert all(p.page_number == 2 and not p.is_heading for p in pieces)

    url_unit = TextUnit(1, "See https://example.com/a/b/c/d/e/f/g/h now.", joiner=" ")
    pieces = explode_unit(url_unit, max_tokens=6)
    texts = [p.text for p in pieces]
    assert texts[0] == "See"
    assert pieces[0].joiner == " "  # keeps the original unit's joiner
    url_pieces = [p for p in pieces if p.joiner == ""]
    assert url_pieces, "slices of one run must attach without a separator"
    rebuilt = pieces[0].text
    for piece in pieces[1:]:
        rebuilt += piece.joiner + piece.text
    assert rebuilt == url_unit.text


def test_whitespace_free_runs_terminate_and_stay_verbatim() -> None:
    leader = "." * 3000
    url = "https://example.com/" + "/".join(f"segment-{i}" for i in range(200))
    text = f"Table of contents {leader} 42\n\nSee {url} for details."
    started = time.perf_counter()
    chunks = chunk_text(text, chunk_size=50, overlap=0)
    assert time.perf_counter() - started < 5.0
    assert chunks
    assert all(c.token_count <= 50 for c in chunks)
    rebuilt = "".join(c.body for c in chunks)
    assert leader in rebuilt
    assert url.replace(" ", "") in rebuilt.replace(" ", "").replace("\n", "")


def test_indivisible_token_wider_than_budget_gets_its_own_chunk() -> None:
    char_counter = len  # a counter that makes one long word exceed any budget
    units = [TextUnit(1, "short words here"), TextUnit(1, "x" * 200), TextUnit(1, "and after")]
    chunks = pack_units(units, chunk_size=50, overlap=5, counter=char_counter)
    # Accepted as its own chunk (after the usual overlap carry) instead of looping forever.
    assert chunks[1].body == "x" * 200
    assert chunks[1].overlap_text == "here"
    assert chunks[1].token_count > 50  # the documented exception for indivisible tokens
    assert chunks[0].token_count <= 50 and chunks[2].token_count <= 50


def test_non_additive_counter_still_respects_the_budget() -> None:
    def bpe_like(text: str) -> int:  # every paragraph joiner costs one extra token
        return count_tokens(text) + text.count("\n\n")

    chunks = chunk_text(LONG_MARKDOWN, chunk_size=60, overlap=10, counter=bpe_like)
    assert len(chunks) > 3
    for chunk in chunks:
        assert bpe_like(chunk.text) <= 60, chunk.text
        assert chunk.token_count == bpe_like(chunk.text)


def test_oversized_unit_is_split_inside_pack_units() -> None:
    units = [TextUnit(1, " ".join(f"Clause {i} applies." for i in range(60)))]
    chunks = pack_units(units, chunk_size=50, overlap=5)
    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    assert all(c.page_number == 1 for c in chunks)
    assert chunks[1].overlap_text and count_tokens(chunks[1].overlap_text) <= 5


def test_giant_single_sentence_is_windowed() -> None:
    units = [TextUnit(1, "x " * 300)]
    chunks = pack_units(units, chunk_size=40, overlap=8)
    assert all(c.token_count <= 40 for c in chunks)
    assert sum(len(c.body.split()) for c in chunks) == 300


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", chunk_size=50, overlap=5) == []
    assert chunk_text("\n\n   \n", chunk_size=50, overlap=5) == []


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        pack_units([], chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        pack_units([], chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        slice_word("abc", max_tokens=0)


# ---------------------------------------------------------------------- page attribution


def _doc(pages: list[list[str]]) -> ParsedDocument:
    page_objects = []
    for number, paragraphs in enumerate(pages, start=1):
        blocks = [
            TextBlock(page_number=number, text=p, bbox=(0.0, float(i), 1.0, float(i + 1)))
            for i, p in enumerate(paragraphs)
        ]
        page_objects.append(
            PageText(page_number=number, text="\n\n".join(paragraphs), blocks=blocks)
        )
    return ParsedDocument(title="Doc", filename="doc.pdf", pages=page_objects)


def test_page_numbers_follow_first_unit_in_chunk() -> None:
    document = _doc(
        [
            ["Page one paragraph alpha.", "Page one paragraph beta."],
            ["Page two paragraph gamma.", "Page two paragraph delta."],
            ["Page three paragraph epsilon."],
        ]
    )
    chunks = chunk_document(document, chunk_size=8, overlap=0)
    assert [c.page_number for c in chunks] == [1, 1, 2, 2, 3]
    assert chunks[2].text == "Page two paragraph gamma."

    merged = chunk_document(document, chunk_size=12, overlap=0)
    assert [c.page_number for c in merged] == [1, 2, 3]
    assert merged[1].text == "Page two paragraph gamma.\n\nPage two paragraph delta."


def test_chunks_merge_small_paragraphs_across_pages_but_keep_start_page() -> None:
    document = _doc([["Alpha."], ["Beta."], ["Gamma."]])
    chunks = chunk_document(document, chunk_size=50, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert "Gamma." in chunks[0].text


# ------------------------------------------------------------------------------ contexts


def test_apply_context_prepends_preamble() -> None:
    assert apply_context("Doc X, section 2.", "The body.") == "Doc X, section 2.\n\nThe body."
    assert apply_context("", "The body.") == "The body."
    assert apply_context("  ctx  ", "body") == "ctx\n\nbody"


async def test_heuristic_contextualizer_mentions_title_section_and_page() -> None:
    document = _doc([["3. Records", "Batch records are retained."]])
    chunks = chunk_document(document, chunk_size=50, overlap=0)
    contexts = await HeuristicContextualizer().contextualize(document, chunks)
    assert len(contexts) == len(chunks)
    assert "Document: Doc." in contexts[0]
    assert "Section: 3. Records." in contexts[0]
    assert "Page 1 of 1." in contexts[0]


def test_document_prompt_text_is_truncated_with_marker() -> None:
    document = _doc([["word " * 500]])
    rendered = document_as_prompt_text(document, max_chars=200)
    assert rendered.startswith("Title: Doc")
    assert rendered.endswith("[... document truncated for context generation ...]")
    assert len(rendered) < 300


# ------------------------------------------------------------------------- PDF parsing


def test_clean_block_text_undoes_hyphenation_but_keeps_hard_hyphens() -> None:
    assert clean_block_text("cleaning valida-\ntion matters") == "cleaning validation matters"
    assert clean_block_text("procedure SOP-QA-\n017 applies") == "procedure SOP-QA-017 applies"
    assert clean_block_text("during 2023-\n2024 and COVID-\n19") == "during 2023-2024 and COVID-19"
    assert clean_block_text("re­\nquired  twice") == "required twice"
    assert clean_block_text("  multiple   spaces\tand\nlines ") == "multiple spaces and lines"


def test_pdf_parse_and_chunk_roundtrip(sample_pdf_bytes: bytes) -> None:
    document = parse_pdf_bytes(sample_pdf_bytes, "sop-qa-017.pdf")
    assert document.page_count == 3
    assert document.title == "sop qa 017"
    assert "SOP-QA-017" in document.pages[0].text
    assert document.pages[2].blocks and all(b.bbox[2] > b.bbox[0] for b in document.pages[2].blocks)

    chunks = chunk_document(document, chunk_size=120, overlap=20)
    assert chunks
    assert {c.page_number for c in chunks} == {1, 2, 3}
    retained = [c for c in chunks if "seven years" in c.text]
    assert retained and retained[0].page_number == 3


def test_pdf_parse_rejects_garbage_empty_and_too_many_pages(sample_pdf_bytes: bytes) -> None:
    from app.errors import PDFParseError, PDFTooLargeError

    with pytest.raises(PDFParseError):
        parse_pdf_bytes(b"", "empty.pdf")
    with pytest.raises(PDFParseError):
        parse_pdf_bytes(b"this is not a pdf", "junk.pdf")
    with pytest.raises(PDFTooLargeError):
        parse_pdf_bytes(sample_pdf_bytes, "sop.pdf", max_pages=2)
    assert parse_pdf_bytes(sample_pdf_bytes, "sop.pdf", max_pages=3).page_count == 3


def test_chunk_dataclass_defaults() -> None:
    chunk = Chunk(chunk_index=0, page_number=1, text="x", token_count=1)
    assert chunk.section is None
    assert chunk.overlap_text == ""
    assert chunk.body == "x"
