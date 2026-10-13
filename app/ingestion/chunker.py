"""Structure-aware recursive chunker with token overlap.

Strategy (coarse -> fine):

1. Pages are split into *units*: PDF layout blocks (paragraphs) or, for plain text, blank-line
   separated paragraphs. Markdown/numbered headings are detected and recorded as the current section.
2. Units are packed greedily into chunks of at most ``chunk_size`` tokens. A heading starts a new
   chunk once the current one is reasonably full, so chunks align with structural boundaries.
3. A unit that cannot fit in a chunk is recursively exploded: into sentences, then into word
   windows, and finally (for whitespace-free runs such as URLs, dot leaders or CJK prose) into
   token-boundary slices. Exploded pieces are re-joined with the original separators, so the
   stored chunk text stays verbatim with the source paragraph.
4. The last ``overlap`` tokens of each chunk are carried into the next chunk (sliding window), so a
   fact straddling a boundary is retrievable from either side.

Token counting uses a fast, dependency-free regex tokenizer (one token per word or punctuation
mark). It tracks BPE token counts closely enough for sizing and keeps the test suite hermetic; pass
``counter=`` to plug in ``tiktoken`` or a provider tokenizer. The ``token_count <= chunk_size``
invariant is exact for additive counters such as the default one; for BPE counters the packer
budgets the joiner tokens and re-checks the joined text, so it holds up to tokenizer merge
effects at the seams.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from app.ingestion.pdf_parser import ParsedDocument

TokenCounter = Callable[[str], int]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)]|[IVXLC]+[.)])\s+[A-Za-z]")
_SENTENCE_END_CHARS = (".", ";", ",", ":")

PARAGRAPH_JOINER = "\n\n"
SENTENCE_JOINER = " "
SLICE_JOINER = ""  # pieces of one whitespace-free run are re-joined without a separator


def count_tokens(text: str) -> int:
    """Approximate token count: words and punctuation marks."""
    return len(_TOKEN_RE.findall(text))


def is_heading(text: str) -> bool:
    """Heuristic heading detector for markdown-style, numbered, or ALL-CAPS titles."""
    stripped = text.strip()
    if not stripped or len(stripped) > 120 or count_tokens(stripped) > 16:
        return False
    if _MARKDOWN_HEADING_RE.match(stripped):
        return True
    if stripped.endswith(_SENTENCE_END_CHARS):
        return False
    if _NUMBERED_HEADING_RE.match(stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters):
        return True
    return False


@dataclass(slots=True)
class TextUnit:
    page_number: int
    text: str
    is_heading: bool = False
    joiner: str = PARAGRAPH_JOINER  # how this unit attaches to the previous unit inside a chunk


@dataclass(slots=True)
class Chunk:
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    section: str | None = None
    overlap_text: str = ""  # prefix carried over from the previous chunk ("" for the first)

    @property
    def body(self) -> str:
        """Chunk text without the carried-over overlap prefix."""
        if self.overlap_text and self.text.startswith(self.overlap_text):
            return self.text[len(self.overlap_text) :].lstrip()
        return self.text


# ----------------------------------------------------------------------- unit building


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def build_units_from_document(document: ParsedDocument) -> list[TextUnit]:
    units: list[TextUnit] = []
    for page in document.pages:
        if page.blocks:
            for block in page.blocks:
                units.append(
                    TextUnit(page.page_number, block.text, is_heading=is_heading(block.text))
                )
        else:
            for paragraph in split_paragraphs(page.text):
                units.append(
                    TextUnit(page.page_number, paragraph, is_heading=is_heading(paragraph))
                )
    return [u for u in units if u.text]


def build_units_from_text(text: str, page_number: int = 1) -> list[TextUnit]:
    return [
        TextUnit(page_number, paragraph, is_heading=is_heading(paragraph))
        for paragraph in split_paragraphs(text)
    ]


# ------------------------------------------------------------------ recursive splitting


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def slice_word(word: str, max_tokens: int, counter: TokenCounter = count_tokens) -> list[str]:
    """Split one whitespace-free run (URL, dot leader, CJK prose) at token boundaries.

    Returns ``[word]`` unchanged when the run is a single indivisible token.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    spans = [match.span() for match in _TOKEN_RE.finditer(word)]
    if len(spans) <= 1:
        return [word]
    pieces: list[str] = []
    start = 0
    while start < len(spans):
        end = start
        while (
            end + 1 < len(spans)
            and counter(word[spans[start][0] : spans[end + 1][1]]) <= max_tokens
        ):
            end += 1
        pieces.append(word[spans[start][0] : spans[end][1]])
        start = end + 1
    return pieces


def tail_words(text: str, max_tokens: int, counter: TokenCounter = count_tokens) -> str:
    """Return the trailing words of ``text`` that fit in ``max_tokens`` (overlap carry-over)."""
    if max_tokens <= 0 or not text:
        return ""
    words = text.split()
    tail: list[str] = []
    tokens = 0
    for word in reversed(words):
        word_tokens = counter(word)
        if tokens + word_tokens > max_tokens:
            break
        tail.append(word)
        tokens += word_tokens
    return " ".join(reversed(tail))


def explode_unit(
    unit: TextUnit, max_tokens: int, counter: TokenCounter = count_tokens
) -> list[TextUnit]:
    """Split an oversized unit into sentence units, word windows, and token slices.

    The first piece keeps the original unit's joiner; later pieces attach with a space, except
    slices of one whitespace-free run, which attach with nothing so the run stays verbatim.
    """
    pieces: list[TextUnit] = []

    def add(text: str, joiner: str) -> None:
        pieces.append(TextUnit(unit.page_number, text, is_heading=False, joiner=joiner))

    for sentence in split_sentences(unit.text):
        if counter(sentence) <= max_tokens:
            add(sentence, SENTENCE_JOINER)
            continue
        window: list[str] = []
        window_tokens = 0

        def flush_window() -> None:
            nonlocal window, window_tokens
            if window:
                add(" ".join(window), SENTENCE_JOINER)
                window, window_tokens = [], 0

        for word in sentence.split():
            word_tokens = counter(word)
            if word_tokens > max_tokens:
                flush_window()
                for index, piece in enumerate(slice_word(word, max_tokens, counter)):
                    add(piece, SENTENCE_JOINER if index == 0 else SLICE_JOINER)
                continue
            if window and window_tokens + word_tokens > max_tokens:
                flush_window()
            window.append(word)
            window_tokens += word_tokens
        flush_window()

    if pieces:
        pieces[0].joiner = unit.joiner
    return pieces


# --------------------------------------------------------------------------- packing


def _join_units(units: Sequence[TextUnit]) -> str:
    if not units:
        return ""
    body = units[0].text
    for unit in units[1:]:
        body += unit.joiner + unit.text
    return body


def pack_units(
    units: Iterable[TextUnit],
    *,
    chunk_size: int,
    overlap: int,
    min_chunk_tokens: int | None = None,
    counter: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Pack units into chunks of at most ``chunk_size`` tokens with ``overlap`` tokens carried over.

    A chunk is ``carry + body`` with ``tokens(carry) <= overlap`` and the body budgeted against
    ``chunk_size - tokens(carry)``. Joiner tokens are charged for non-additive counters, and the
    joined text is re-counted at flush time; if it still overshoots, the last unit is pushed back.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    min_chunk_tokens = chunk_size // 4 if min_chunk_tokens is None else min_chunk_tokens
    # 0 for the default regex counter; 1 for BPE tokenizers that encode "\n\n" as a token.
    joiner_cost = max(0, counter(f"a{PARAGRAPH_JOINER}a") - 2 * counter("a"))
    max_unit_tokens = max(1, chunk_size - overlap - joiner_cost)

    chunks: list[Chunk] = []
    buffer: list[TextUnit] = []
    buffer_tokens = 0
    buffer_page: int | None = None
    buffer_section: str | None = None
    current_section: str | None = None
    carry = ""  # overlap text carried from the previous chunk
    carry_tokens = 0
    pending: deque[TextUnit] = deque(units)

    def flush() -> None:
        nonlocal buffer, buffer_tokens, buffer_page, buffer_section, carry, carry_tokens
        while buffer:
            body = _join_units(buffer)
            text = f"{carry}{PARAGRAPH_JOINER}{body}" if carry else body
            token_count = counter(text)
            if token_count > chunk_size and len(buffer) > 1:
                # A non-additive counter overshot at a seam: retry without the last unit.
                pending.appendleft(buffer.pop())
                continue
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    page_number=buffer_page or 1,
                    text=text,
                    token_count=token_count,
                    section=buffer_section,
                    overlap_text=carry,
                )
            )
            carry = tail_words(body, overlap, counter)
            carry_tokens = counter(carry) if carry else 0
            buffer, buffer_tokens, buffer_page, buffer_section = [], 0, None, None

    while pending:
        unit = pending.popleft()
        if not unit.text.strip():
            continue
        if unit.is_heading:
            if buffer and buffer_tokens >= min_chunk_tokens:
                flush()
            current_section = unit.text.lstrip("#").strip()
            if buffer:
                # The heading joins a short buffer: the chunk is mostly the new section.
                buffer_section = current_section

        unit_tokens = counter(unit.text)
        if unit_tokens > max_unit_tokens:
            pieces = explode_unit(unit, max_unit_tokens, counter)
            if len(pieces) == 1 and pieces[0].text == unit.text:
                # Indivisible (a single token wider than the budget): give it its own chunk
                # rather than looping forever trying to split it.
                flush()
                buffer_page, buffer_section = unit.page_number, current_section
                buffer, buffer_tokens = [unit], unit_tokens
                flush()
                continue
            pending.extendleft(reversed(pieces))
            continue

        capacity = chunk_size - carry_tokens - (joiner_cost if carry else 0)
        cost = unit_tokens + (joiner_cost if buffer else 0)
        if buffer and buffer_tokens + cost > capacity:
            flush()
            cost = unit_tokens
        if not buffer:
            buffer_page = unit.page_number
            buffer_section = current_section
        buffer.append(unit)
        buffer_tokens += cost

    flush()
    return chunks


def chunk_document(
    document: ParsedDocument,
    *,
    chunk_size: int = 400,
    overlap: int = 60,
    min_chunk_tokens: int | None = None,
    counter: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Chunk a parsed PDF along structural boundaries with sliding-window overlap."""
    units = build_units_from_document(document)
    return pack_units(
        units,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_tokens=min_chunk_tokens,
        counter=counter,
    )


def chunk_text(
    text: str,
    *,
    chunk_size: int = 400,
    overlap: int = 60,
    page_number: int = 1,
    min_chunk_tokens: int | None = None,
    counter: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Chunk a plain (optionally markdown) string. Convenience wrapper used by tests and scripts."""
    units = build_units_from_text(text, page_number=page_number)
    return pack_units(
        units,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_tokens=min_chunk_tokens,
        counter=counter,
    )
