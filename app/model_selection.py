"""Validated, last-known-good model lists; runtime reads never call a model.

The daily job owns qualification. Runtime processes adopt its file on their first request,
hourly thereafter, and after failures (at most once a minute). Explicit model lists stay pinned.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import tempfile
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.config import DEFAULT_OPENROUTER_CHAIN, Settings, parse_model_chain

logger = logging.getLogger(__name__)
STAMP = "%Y-%m-%dT%H:%M:%SZ"
MAX_BYTES = 64_000
MAX_AGE = timedelta(days=14)
ROLES = ("generation", "contextualizer")
FREE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,95}:free")


def moment(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    return datetime.strptime(value, STAMP).replace(tzinfo=UTC)


def fingerprint(settings: Settings | None = None) -> str:
    from app.generator.grounded_llm import SYSTEM_PROMPT, LLMGroundedOutput
    from app.ingestion.contextualizer import CONTEXT_INSTRUCTION, CONTEXT_SYSTEM_PROMPT

    settings = settings or Settings(_env_file=None)
    material = [
        "hybrid-model-check-v1",
        SYSTEM_PROMPT,
        LLMGroundedOutput.model_json_schema(),
        CONTEXT_SYSTEM_PROMPT,
        CONTEXT_INSTRUCTION,
        settings.generation_max_tokens,
        settings.generation_temperature,
        settings.contextualizer_max_tokens,
        settings.openrouter_strict_json_schema,
        settings.openrouter_base_url.rstrip("/"),
        DEFAULT_OPENROUTER_CHAIN,
        Path(__file__).with_name("model_check.py").read_text(encoding="utf-8"),
    ]
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def read_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    if urlsplit(url).scheme != "https":
        raise ValueError("model metadata requires HTTPS")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hybrid-model-check",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        # Catalogue may be much larger than the published file; still bound input size.
        body = response.read(5_000_001)
    if len(body) > 5_000_000:
        raise ValueError("model metadata exceeds size limit")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("model metadata is not an object")
    return value


def eligible(data: dict[str, Any]) -> list[str]:
    """Only explicitly zero-priced text models supporting structured output qualify."""
    models = []
    for row in data.get("data", []):
        if not isinstance(row, dict) or not FREE_ID.fullmatch(str(row.get("id", ""))):
            continue
        prices = row.get("pricing")
        architecture = row.get("architecture")
        supported = row.get("supported_parameters")
        if not isinstance(prices, dict) or not isinstance(architecture, dict):
            continue
        outputs = architecture.get("output_modalities")
        if not isinstance(outputs, list) or not isinstance(supported, list):
            continue
        try:
            if any(isinstance(prices[key], bool) for key in ("prompt", "completion")):
                continue
            free = float(prices["prompt"]) == 0 and float(prices["completion"]) == 0
        except (KeyError, TypeError, ValueError):
            continue
        if (
            free
            and "text" in outputs
            and any(p in supported for p in ("structured_outputs", "response_format"))
        ):
            models.append(row["id"])
    return sorted(set(models))


def parse(text: str, *, expected: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    if len(text.encode()) > MAX_BYTES:
        raise ValueError("published model list is too large")
    record = json.loads(text)
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError("unsupported published model list")
    if record.get("fingerprint") != expected:
        raise ValueError("published model tests use different settings or prompts")
    tested = moment(record.get("tested"))
    if tested > now + timedelta(minutes=5) or now - tested > MAX_AGE:
        raise ValueError("published model evidence is future-dated or older than 14 days")
    chains = record.get("chains")
    if not isinstance(chains, dict):
        raise ValueError("missing model chains")
    for role in ROLES:
        chain = chains.get(role)
        if (
            not isinstance(chain, list)
            or not 1 <= len(chain) <= 2
            or not all(isinstance(m, str) and FREE_ID.fullmatch(m) for m in chain)
            or len(set(chain)) != len(chain)
        ):
            raise ValueError(f"invalid free model chain for {role}")
    cursor = record.get("cursor", 0)
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
        raise ValueError("invalid candidate cursor")
    attempted = moment(record.get("attempted", record["tested"]))
    if attempted > now + timedelta(minutes=5) or attempted < tested:
        raise ValueError("invalid last-attempt timestamp")
    catalogue = record.get("catalogue", [])
    if not isinstance(catalogue, list) or not all(
        isinstance(model, str) and FREE_ID.fullmatch(model) for model in catalogue
    ):
        raise ValueError("invalid published catalogue")
    failed = record.get("failed_models", {})
    if not isinstance(failed, dict) or any(role not in ROLES for role in failed):
        raise ValueError("invalid failure state")
    for role, models in failed.items():
        if not isinstance(models, list) or any(model not in chains[role] for model in models):
            raise ValueError("failures must refer to retained models in their tested role")
    unavailable = record.get("unavailable_roles", [])
    if not isinstance(unavailable, list) or any(role not in ROLES for role in unavailable):
        raise ValueError("invalid unavailable role state")
    return record


def atomic_write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(record, handle, indent=2)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PublishedModels:
    def __init__(self, settings: Settings, role: str) -> None:
        if role not in ROLES:
            raise ValueError("unknown model role")
        self.settings = settings
        self.role = role
        self.models = parse_model_chain(DEFAULT_OPENROUTER_CHAIN)
        self.tested: datetime | None = None
        self._checked = float("-inf")
        self._lock = asyncio.Lock()
        self.expected = fingerprint(settings)
        self.cache = Path(settings.model_cache_path)
        try:
            self._adopt(parse(self.cache.read_text(encoding="utf-8"), expected=self.expected))
        except (OSError, ValueError, TypeError):
            pass

    def _adopt(self, record: dict[str, Any]) -> bool:
        tested = moment(record["tested"])
        if self.tested is not None and tested <= self.tested:
            return False
        self.models[:] = record["chains"][self.role]
        self.tested = tested
        return True

    async def refresh(self, *, after_failure: bool = False) -> list[str]:
        interval = 60 if after_failure else self.settings.model_refresh_seconds
        if not self.settings.model_published_url or time.monotonic() - self._checked < interval:
            return list(self.models)
        async with self._lock:
            if time.monotonic() - self._checked < interval:
                return list(self.models)
            self._checked = time.monotonic()
            try:
                raw = await asyncio.to_thread(read_json, self.settings.model_published_url)
                record = parse(json.dumps(raw), expected=self.expected)
                if self._adopt(record):
                    try:
                        await asyncio.to_thread(atomic_write, self.cache, record)
                    except OSError:
                        logger.warning("Could not persist the published model list")
                    logger.info("Adopted tested %s models dated %s", self.role, record["tested"])
            except (OSError, HTTPException, ValueError, TypeError) as exc:
                logger.warning(
                    "Keeping current %s chain: publication unavailable (%s)",
                    self.role,
                    type(exc).__name__,
                )
            return list(self.models)
