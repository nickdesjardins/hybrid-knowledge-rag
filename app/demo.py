"""The showcase corpus: a manifest of sample PDFs, seeded once into an empty database.

``<DEMO_SEED_DIR>/corpus.json``::

    {"documents": [{"file": "SOP-QA-017.pdf", "title": "...", "questions": ["..."]}, ...]}

Seeding is idempotent: a document whose filename is already in the database is skipped, so a
restart on a free host (empty disk, same database) ingests nothing twice.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.errors import RetrievalEngineError
from app.ingestion.pipeline import IngestComponents, ingest_pdf
from app.schemas import DocumentSummary

logger = logging.getLogger(__name__)

MANIFEST_NAME = "corpus.json"


class DocumentCatalogue(Protocol):
    async def list_documents(self) -> list[DocumentSummary]: ...


class SeedComponents(IngestComponents, Protocol):
    catalogue: DocumentCatalogue


@dataclass(slots=True)
class ManifestEntry:
    file: str
    title: str
    questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SeedReport:
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def load_manifest(directory: str | Path) -> list[ManifestEntry]:
    """Read ``corpus.json``; an absent folder or manifest means an empty corpus."""
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            ManifestEntry(
                file=str(item["file"]),
                title=str(item.get("title") or Path(str(item["file"])).stem),
                questions=[str(q) for q in item.get("questions", [])],
            )
            for item in raw.get("documents", [])
        ]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("Demo manifest %s is unreadable: %s", path, exc)
        return []
    return entries


def sample_questions(directory: str | Path, limit: int = 12) -> list[str]:
    """Questions from the manifest, interleaved across documents, for the playground chips."""
    entries = load_manifest(directory)
    questions: list[str] = []
    round_index = 0
    while len(questions) < limit:
        added = False
        for entry in entries:
            if round_index < len(entry.questions):
                questions.append(entry.questions[round_index])
                added = True
                if len(questions) >= limit:
                    break
        if not added:
            break
        round_index += 1
    return questions


async def seed_demo_corpus(components: SeedComponents, settings: Settings) -> SeedReport:
    """Ingest every manifest PDF that is not yet in the database. Never raises."""
    report = SeedReport()
    directory = Path(settings.demo_seed_dir)
    entries = load_manifest(directory)
    if not entries:
        logger.info("Demo corpus: nothing to seed (%s has no %s)", directory, MANIFEST_NAME)
        return report
    try:
        existing = {doc.filename for doc in await components.catalogue.list_documents()}
    except Exception as exc:  # noqa: BLE001 - seeding must never take the service down
        logger.warning("Demo corpus: could not list documents (%s); seeding skipped", exc)
        return report
    for entry in entries:
        if entry.file in existing:
            report.skipped.append(entry.file)
            continue
        path = directory / entry.file
        if not path.is_file():
            logger.warning("Demo corpus: %s is listed in the manifest but missing on disk", path)
            report.failed.append(entry.file)
            continue
        try:
            result = await ingest_pdf(components, path.read_bytes(), entry.file, entry.title)
        except (RetrievalEngineError, OSError) as exc:
            logger.warning("Demo corpus: %s failed to ingest: %s", entry.file, exc)
            report.failed.append(entry.file)
            continue
        report.ingested.append(entry.file)
        logger.info(
            "Demo corpus: seeded %s (%d pages, %d chunks)",
            entry.file,
            result.pages_count,
            result.chunks_count,
        )
    return report


def questions_for(entries: Sequence[ManifestEntry]) -> list[str]:
    return [question for entry in entries for question in entry.questions]
