"""PDF text extraction that preserves page boundaries and block coordinates (PyMuPDF).

Each page is decomposed into layout blocks. A block carries its bounding box (PDF points) so the
chunker can treat it as a paragraph-level structural unit instead of splitting on raw characters.

PyMuPDF does not support multithreaded use of its shared context, so all parsing runs on one
dedicated worker thread (the event loop stays free; concurrent uploads queue behind each other).
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import PurePath

import pymupdf

from app.errors import PDFParseError, PDFTooLargeError

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
# Only a lowercase-to-lowercase break is a hyphenation artefact ("valida-\ntion"). Hard hyphens
# in identifiers and compounds ("SOP-QA-\n017", "COVID-\n19", "2023-\n2024") must survive.
_SOFT_HYPHEN_BREAK_RE = re.compile(r"([a-z])-\n([a-z])")
_HARD_HYPHEN_BREAK_RE = re.compile(r"-\n(?=\S)")
_SOFT_HYPHEN_RE = re.compile(r"­\n?")

_PDF_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pymupdf")


@dataclass(slots=True)
class TextBlock:
    page_number: int  # 1-indexed
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points


@dataclass(slots=True)
class PageText:
    page_number: int  # 1-indexed
    text: str
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    filename: str
    pages: list[PageText]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


def clean_block_text(raw: str) -> str:
    """Normalise a block: undo hyphenation, keep hard hyphens, collapse whitespace."""
    text = _SOFT_HYPHEN_RE.sub("", raw)
    text = _SOFT_HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _HARD_HYPHEN_BREAK_RE.sub("-", text)
    text = text.replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _title_from(filename: str, metadata: dict[str, str], explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    meta_title = (metadata.get("title") or "").strip()
    if meta_title:
        return meta_title
    stem = PurePath(filename).stem.replace("_", " ").replace("-", " ").strip()
    return stem or "Untitled document"


def parse_pdf_bytes(
    data: bytes,
    filename: str,
    title: str | None = None,
    *,
    max_pages: int | None = None,
) -> ParsedDocument:
    """Extract text from PDF bytes, page by page, block by block.

    Raises ``PDFParseError`` when the file is not a readable PDF or contains no extractable text
    (image-only scans need OCR, which is out of scope for this engine), and ``PDFTooLargeError``
    when the page count exceeds ``max_pages`` (checked before any text is extracted).
    """
    if not data:
        raise PDFParseError("Empty upload")
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises a few different exception types
        raise PDFParseError(f"Could not open PDF: {exc}") from exc

    try:
        if document.is_encrypted and not document.authenticate(""):
            raise PDFParseError("PDF is password protected")
        if max_pages is not None and document.page_count > max_pages:
            raise PDFTooLargeError(
                f"PDF has {document.page_count} pages; limit is MAX_PDF_PAGES={max_pages}"
            )

        metadata = {k: str(v) for k, v in (document.metadata or {}).items() if v}
        pages: list[PageText] = []
        for index, page in enumerate(document):
            page_number = index + 1
            blocks: list[TextBlock] = []
            for block in page.get_text("blocks"):
                # (x0, y0, x1, y1, text, block_no, block_type) - block_type 0 == text
                if len(block) >= 7 and block[6] != 0:
                    continue
                cleaned = clean_block_text(block[4])
                if not cleaned:
                    continue
                bbox = (float(block[0]), float(block[1]), float(block[2]), float(block[3]))
                blocks.append(TextBlock(page_number=page_number, text=cleaned, bbox=bbox))
            page_text = "\n\n".join(b.text for b in blocks)
            pages.append(PageText(page_number=page_number, text=page_text, blocks=blocks))
    finally:
        document.close()

    if not any(page.text for page in pages):
        raise PDFParseError("PDF contains no extractable text (scanned image without OCR?)")

    return ParsedDocument(
        title=_title_from(filename, metadata, title),
        filename=filename,
        pages=pages,
        metadata=metadata,
    )


async def parse_pdf_bytes_async(
    data: bytes,
    filename: str,
    title: str | None = None,
    *,
    max_pages: int | None = None,
) -> ParsedDocument:
    """Run the blocking PyMuPDF parse on the dedicated single worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _PDF_EXECUTOR, lambda: parse_pdf_bytes(data, filename, title, max_pages=max_pages)
    )
