"""OpenRouter access: one vendor, an ordered chain of free models, resilience by fallback.

This is the pattern the Phase 1 extraction service used, ported to this engine:

* The service talks to OpenRouter over its OpenAI-compatible Chat Completions API and gets its
  resilience from routing across *several models* rather than several vendors. The dominant
  failure mode of a free model is not a bad answer but a transient 429, an in-band "upstream
  provider error", or a withdrawal (404), each on one specific model.
* Structured output is requested through the strict ``json_schema`` response format first; a
  model that rejects it is retried in plain JSON mode with the schema spelled out in the
  system prompt. Either way the caller validates the JSON before it can reach a user.
* Request starts are paced to OpenRouter's free-model allowance (20 per minute) so a burst of
  chunks never trips the limit.
* Error messages returned to callers are short and sanitised; upstream bodies go to logs only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.errors import ConfigurationError, ProviderError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- schemas


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A strict-mode JSON Schema for OpenAI-compatible structured outputs.

    Strict mode requires every property to be listed in ``required`` and
    ``additionalProperties: false`` on every object, including those under ``$defs``.
    """

    def harden(node: Any) -> Any:
        if isinstance(node, dict):
            hardened = {key: harden(value) for key, value in node.items()}
            if hardened.get("type") == "object" and "properties" in hardened:
                hardened["required"] = list(hardened["properties"].keys())
                hardened["additionalProperties"] = False
            return hardened
        if isinstance(node, list):
            return [harden(item) for item in node]
        return node

    return harden(model.model_json_schema())


# --------------------------------------------------------------------------- parsing

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
#: Some models emit prose or a chain of thought before the JSON despite the instructions;
#: recover the last balanced-looking object rather than failing the call.
_TRAILING_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str | None) -> dict[str, Any]:
    """Turn a model's raw text into a JSON object (``dict``) or raise ``ValueError``."""
    if raw is None or not raw.strip():
        raise ValueError("model returned an empty response")
    match = _CODE_FENCE_RE.match(raw)
    cleaned = match.group(1) if match else raw.strip()
    try:
        payload: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        found = _TRAILING_OBJECT_RE.search(cleaned)
        if not found:
            raise ValueError(f"invalid JSON at position {exc.pos}: {exc.msg}") from exc
        try:
            payload = json.loads(found.group(0))
        except json.JSONDecodeError as inner:
            raise ValueError(f"invalid JSON at position {inner.pos}: {inner.msg}") from inner
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


# ------------------------------------------------------------------------ rate limit


class RateLimiter:
    """Spaces request starts so a burst can never exceed a per-minute allowance."""

    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next_start - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_start = max(now, self._next_start) + self.interval


# ----------------------------------------------------------------------------- chat


@dataclass(slots=True)
class ChatOutcome:
    model: str  # the model that actually answered (fallbacks may have been used)
    content: str
    finish_reason: str | None
    refused: bool = False  # the model's content filter declined
    json_mode: bool = False  # strict schema was rejected; plain JSON mode was used instead
    parsed: Any = None  # result of the caller's ``validate`` callback, when given


class _StrictSchemaUnsupported(Exception):
    """The model rejected the strict ``json_schema`` response format."""


class OpenRouterChat:
    """Walks an ordered model chain over the OpenAI-compatible API."""

    def __init__(
        self,
        settings: Settings,
        *,
        models: Sequence[str],
        client: object | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.models = list(models)
        if not self.models:
            raise ConfigurationError("the OpenRouter model chain is empty")
        if client is None and settings.openrouter_api_key is None:
            raise ConfigurationError("OPENROUTER_API_KEY is required for the openrouter providers")
        headers: dict[str, str] = {}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name
        self._client = client or AsyncOpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),  # type: ignore[union-attr]
            base_url=settings.openrouter_base_url,
            timeout=settings.openrouter_timeout_s,
            max_retries=0,  # the chain owns retry semantics (fallback), not the SDK
            default_headers=headers or None,
        )
        self.limiter = limiter or RateLimiter(settings.openrouter_requests_per_minute)
        self.strict_schema = settings.openrouter_strict_json_schema

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float | None = 0.0,
        schema: dict[str, Any] | None = None,
        schema_name: str = "output",
        json_mode_system_suffix: str | None = None,
        validate: Callable[[str], Any] | None = None,
    ) -> ChatOutcome:
        """Try each model in order; return the first usable answer.

        A model is skipped (and the next one tried) on rate limits, timeouts, upstream errors,
        withdrawal, truncation, empty output, and when ``validate`` raises on its text. When the
        strict schema is rejected, the same model is retried once in plain JSON mode.
        """
        failures: list[str] = []
        any_retryable = False
        for model in self.models:
            modes = ["strict", "json"] if (schema and self.strict_schema) else ["json"]
            if schema is None:
                modes = ["text"]
            for mode in modes:
                try:
                    outcome = await self._call(
                        model,
                        messages,
                        max_tokens,
                        temperature,
                        schema,
                        schema_name,
                        mode,
                        json_mode_system_suffix,
                    )
                except _StrictSchemaUnsupported:
                    logger.info("%s rejected strict json_schema; retrying in JSON mode", model)
                    continue
                except ProviderError as exc:
                    failures.append(f"{model}: {exc.message}")
                    any_retryable = any_retryable or exc.retryable
                    if exc.provider == "openrouter-auth":
                        raise  # no model can fix a rejected key
                    logger.warning(
                        "OpenRouter model %s failed (%s); trying next", model, exc.message
                    )
                    break
                if outcome.refused or validate is None:
                    return outcome
                try:
                    outcome.parsed = validate(outcome.content)
                except Exception as exc:  # noqa: BLE001 - any validation failure cascades
                    failures.append(f"{model}: invalid output ({type(exc).__name__})")
                    logger.warning("OpenRouter model %s returned invalid output: %s", model, exc)
                    break
                return outcome
        raise ProviderError(
            "openrouter",
            f"all models failed ({'; '.join(failures)})",
            retryable=any_retryable,
        )

    async def _call(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float | None,
        schema: dict[str, Any] | None,
        schema_name: str,
        mode: str,
        json_mode_system_suffix: str | None,
    ) -> ChatOutcome:
        import openai

        request_messages = [dict(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if mode == "strict":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        elif mode == "json":
            kwargs["response_format"] = {"type": "json_object"}
            if (
                json_mode_system_suffix
                and request_messages
                and request_messages[0].get("role") == "system"
            ):
                request_messages[0]["content"] += json_mode_system_suffix

        await self.limiter.wait()
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise ProviderError("openrouter-auth", "OpenRouter rejected the API key (401)") from exc
        except openai.RateLimitError as exc:
            raise ProviderError(model, "rate limited (429)", retryable=True) from exc
        except openai.NotFoundError as exc:
            raise ProviderError(model, "model not found (404); withdrawn?") from exc
        except openai.BadRequestError as exc:
            body = json.dumps(getattr(exc, "body", None), default=str).lower()
            if mode == "strict" and any(
                marker in body for marker in ("response_format", "json_schema", "structured")
            ):
                raise _StrictSchemaUnsupported() from exc
            raise ProviderError(model, "request rejected (400)") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(
                model, f"HTTP {exc.status_code}", retryable=exc.status_code >= 500
            ) from exc
        except openai.APITimeoutError as exc:
            raise ProviderError(model, "request timed out", retryable=True) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(model, "connection error", retryable=True) from exc

        # OpenRouter reports upstream provider failures in-band: HTTP 200 with an "error"
        # member instead of choices. This is the most common free-model failure.
        error = getattr(completion, "error", None)
        if error:
            code = error.get("code") if isinstance(error, dict) else None
            logger.warning("OpenRouter in-band error from %s: %s", model, error)
            raise ProviderError(model, f"upstream provider error (code {code})", retryable=True)

        choices = getattr(completion, "choices", None)
        if not choices:
            raise ProviderError(model, "no choices in response", retryable=True)
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise ProviderError(
                model, "response truncated at max_tokens; raise GENERATION_MAX_TOKENS"
            )
        if finish_reason == "content_filter":
            return ChatOutcome(model, "", finish_reason, refused=True, json_mode=(mode == "json"))
        content = getattr(getattr(choice, "message", None), "content", None) or ""
        if not content.strip():
            raise ProviderError(model, "empty response", retryable=True)
        return ChatOutcome(model, content, finish_reason, json_mode=(mode == "json"))


ValidateCallback = Callable[[str], Any] | Callable[[str], Awaitable[Any]]
