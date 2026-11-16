"""The discovery job and runtime selection agree on evidence, budgets and manual pins."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from http.client import IncompleteRead
from types import SimpleNamespace

import pytest

from app import model_check, model_selection
from app.config import DEFAULT_OPENROUTER_CHAIN, Settings, parse_model_chain
from app.errors import ProviderError
from app.model_selection import PublishedModels, eligible, fingerprint, parse
from app.openrouter import ChatOutcome, OpenRouterChat, RateLimiter


def settings(tmp_path, **kwargs):
    return Settings(
        _env_file=None,
        _env_prefix="HERMETIC_",
        model_cache_path=str(tmp_path / "chain.json"),
        **kwargs,
    )


def publication(config, **kwargs):
    now = datetime.now(UTC)
    return {
        "version": 1,
        "fingerprint": fingerprint(config),
        "tested": now.strftime(model_selection.STAMP),
        "attempted": now.strftime(model_selection.STAMP),
        "chains": {role: ["vendor/current:free"] for role in model_selection.ROLES},
        "catalogue": ["vendor/current:free"],
        "cursor": 0,
        **kwargs,
    }


@pytest.mark.parametrize(
    "price",
    [
        {},
        {"prompt": "0"},
        {"prompt": None, "completion": 0},
        {"prompt": False, "completion": 0},
        {"prompt": "nan", "completion": 0},
        {"prompt": -1, "completion": 0},
        {"prompt": 0, "completion": "0.1"},
    ],
)
def test_missing_invalid_or_nonzero_prices_are_not_free(price):
    row = {
        "id": "vendor/model:free",
        "pricing": price,
        "architecture": {"output_modalities": ["text"]},
        "supported_parameters": ["response_format"],
    }
    assert eligible({"data": [row]}) == []


def test_catalogue_checks_exact_price_capability_and_id():
    good = {
        "id": "vendor/model:free",
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {"output_modalities": ["text"]},
        "supported_parameters": ["structured_outputs"],
    }
    assert eligible(
        {"data": [good, {**good, "id": "vendor/paid"}, {**good, "supported_parameters": []}]}
    ) == [good["id"]]


def test_candidates_rotate_past_a_full_chain():
    current = ["v/a:free", "v/b:free"]
    listing = current + ["v/c:free", "v/d:free", "v/e:free"]
    first, cursor = model_check.candidates(current, listing, 0, 3)
    second, _ = model_check.candidates(current, listing, cursor, 3)
    assert first == current + ["v/c:free"]
    assert second == current + ["v/d:free"]


def test_discovery_does_not_retest_unrelated_removals_or_catalogue_reordering(tmp_path):
    config = settings(tmp_path)
    previous = publication(config, catalogue=["vendor/current:free", "vendor/other:free"])
    now = datetime.now(UTC)
    assert model_check.due(previous, list(reversed(previous["catalogue"])), now, False) is None
    assert model_check.due(previous, ["vendor/current:free"], now, False) is None
    assert model_check.due(previous, [], now, False)
    assert model_check.due(previous, previous["catalogue"], now, True)
    previous["attempted"] = (now - timedelta(days=1)).strftime(model_selection.STAMP)
    assert model_check.due(previous, [*previous["catalogue"], "new/model:free"], now, False)
    assert model_check.due(previous, previous["catalogue"], now + timedelta(days=7), False)


@pytest.mark.parametrize(
    "change",
    [
        {"fingerprint": "other"},
        {"tested": "2000-01-01T00:00:00Z"},
        {"chains": {"generation": ["v/paid"], "contextualizer": ["v/x:free"]}},
    ],
)
def test_runtime_rejects_untrusted_or_stale_publications(tmp_path, change):
    config = settings(tmp_path)
    with pytest.raises(ValueError):
        parse(json.dumps(publication(config, **change)), expected=fingerprint(config))


@pytest.mark.parametrize("failure", [TimeoutError(), IncompleteRead(b"truncated", 20)])
async def test_runtime_adopts_persists_and_keeps_last_good_during_outage(
    tmp_path, monkeypatch, failure
):
    config = settings(tmp_path)
    record = publication(config)
    monkeypatch.setattr(model_selection, "read_json", lambda url: record)
    reader = PublishedModels(config, "generation")
    assert await reader.refresh() == ["vendor/current:free"]
    assert PublishedModels(config, "generation").models == ["vendor/current:free"]

    def unavailable(url):
        raise failure

    monkeypatch.setattr(model_selection, "read_json", unavailable)
    reader._checked = float("-inf")
    assert await reader.refresh() == ["vendor/current:free"]
    assert reader.tested == model_selection.moment(record["tested"])


async def test_running_instance_adopts_newer_publication_after_successful_requests(
    tmp_path, monkeypatch
):
    config = settings(tmp_path)
    old = publication(
        config, tested=(datetime.now(UTC) - timedelta(days=1)).strftime(model_selection.STAMP)
    )
    new = publication(config, chains={role: ["vendor/new:free"] for role in model_selection.ROLES})
    records = iter([old, new])
    monkeypatch.setattr(model_selection, "read_json", lambda url: next(records))
    reader = PublishedModels(config, "generation")
    assert await reader.refresh() == ["vendor/current:free"]
    reader._checked -= config.model_refresh_seconds
    assert await reader.refresh() == ["vendor/new:free"]


def test_blank_is_automatic_but_explicit_models_stay_pinned(tmp_path):
    config = settings(tmp_path, openrouter_models="")
    assert config.openrouter_model_chain == parse_model_chain(DEFAULT_OPENROUTER_CHAIN)
    pinned = settings(tmp_path, openrouter_models="mine/pinned:free")
    assert pinned.openrouter_model_chain == ["mine/pinned:free"]


@pytest.mark.parametrize("outcome,exit_code", [("inconclusive", 0), ("failed", 1)])
async def test_failed_check_preserves_evidence_and_distinguishes_transient_errors(
    tmp_path, monkeypatch, outcome, exit_code
):
    config = settings(tmp_path)
    old = publication(
        config,
        tested=(datetime.now(UTC) - timedelta(days=7)).strftime(model_selection.STAMP),
        attempted=(datetime.now(UTC) - timedelta(days=7)).strftime(model_selection.STAMP),
    )
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-only")
    catalogue = {
        "data": [
            {
                "id": "vendor/current:free",
                "pricing": {"prompt": 0, "completion": 0},
                "architecture": {"output_modalities": ["text"]},
                "supported_parameters": ["response_format"],
            }
        ]
    }
    monkeypatch.setattr(
        model_check,
        "read_json",
        lambda url, *args: (
            catalogue
            if url.endswith("/models")
            else {"data": {"free_model_daily_requests": {"remaining": 100}}}
        ),
    )

    async def unanswered(config, model, limiter):
        return {
            "model": model,
            "roles": [],
            "results": [
                {"case": case, "outcome": outcome, "reason": "controlled failure"}
                for case in model_check.CASES
            ],
        }

    monkeypatch.setattr(model_check, "qualify", unanswered)
    out = tmp_path / "out.json"
    assert (
        await model_check.run(argparse.Namespace(previous=previous, out=out, force=False))
        == exit_code
    )
    kept = json.loads(out.read_text())
    assert kept["chains"] == old["chains"] and kept["tested"] == old["tested"]
    assert kept["retry"] and kept["attempted"] != old["tested"]
    assert bool(kept["failed_models"]) == (outcome == "failed")


async def test_insufficient_budget_sends_no_model_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-only")
    monkeypatch.setattr(
        model_check,
        "read_json",
        lambda url, *args: (
            {"data": []}
            if url.endswith("/models")
            else {"data": {"free_model_daily_requests": {"remaining": 10}}}
        ),
    )

    async def forbidden(*args):
        pytest.fail("budget guard must run before model calls")

    monkeypatch.setattr(model_check, "qualify", forbidden)
    assert (
        await model_check.run(argparse.Namespace(previous=None, out=tmp_path / "out", force=True))
        == 1
    )


@pytest.mark.parametrize("pinned", [True, False])
async def test_chat_refreshes_before_successful_calls_but_respects_manual_pins(
    tmp_path, monkeypatch, pinned
):
    config = settings(
        tmp_path,
        openrouter_api_key="sk-or-test-only",
        openrouter_models="vendor/pinned:free" if pinned else "",
    )
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: SimpleNamespace())
    reads = []

    def published(url):
        reads.append(url)
        return publication(config)

    async def call(self, model, *args):
        return ChatOutcome(model=model, content="ok", finish_reason="stop")

    monkeypatch.setattr(model_selection, "read_json", published)
    monkeypatch.setattr(OpenRouterChat, "_call", call)
    chat = OpenRouterChat(config, models=config.openrouter_model_chain, role="generation")
    result = await chat.complete(messages=[], max_tokens=1)
    assert result.model == ("vendor/pinned:free" if pinned else "vendor/current:free")
    assert len(reads) == (0 if pinned else 1)


@pytest.mark.parametrize("bad_quote", [True, False])
async def test_qualification_uses_actual_grounding_and_keeps_roles_separate(
    tmp_path, monkeypatch, bad_quote
):
    config = settings(tmp_path, openrouter_api_key="sk-or-test-only")
    calls = []
    closed = []

    async def close():
        closed.append(True)

    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: SimpleNamespace(close=close))

    async def call(self, model, messages, max_tokens, *args):
        calls.append((messages, max_tokens))
        case = model_check.CASES[len(calls) - 1]
        if case == "contextualizer":
            content = "This records policy covers record retention and electronic audit trails."
        else:
            content = json.dumps(
                {
                    "answer": "Seven years."
                    if case != "unanswerable"
                    else "No stock price provided.",
                    "is_grounded": case != "unanswerable",
                    "confidence_score": 0 if case == "unanswerable" else 1,
                    "citations": []
                    if case == "unanswerable"
                    else [
                        {
                            "chunk_id": "records-1",
                            "verbatim_quote": "Invented seven years."
                            if bad_quote
                            else model_check.PASSAGE,
                        }
                    ],
                }
            )
        return ChatOutcome(model=model, content=content, finish_reason="stop")

    monkeypatch.setattr(OpenRouterChat, "_call", call)
    report = await model_check.qualify(config, "vendor/test:free", RateLimiter(0))
    assert report["roles"] == (["contextualizer"] if bad_quote else list(model_selection.ROLES))
    assert [tokens for _, tokens in calls] == [config.generation_max_tokens] * 3 + [
        config.contextualizer_max_tokens
    ]
    assert "PASSWORD_ACCEPTED" in calls[2][0][1]["content"]
    assert closed == [True]


@pytest.mark.parametrize(
    "next_run",
    ["cooldown", "catalogue_outage", "key_outage", "quota", "unknown_quota", "inconclusive"],
)
async def test_known_quality_failure_stays_red_across_deferred_or_inconclusive_runs(
    tmp_path, monkeypatch, next_run
):
    config = settings(tmp_path)
    now = datetime.now(UTC)
    record = publication(
        config,
        tested=(now - timedelta(days=7)).strftime(model_selection.STAMP),
        attempted=(now - timedelta(days=1)).strftime(model_selection.STAMP),
        retry=True,
        failed_models={"generation": ["vendor/current:free"]},
    )
    if next_run == "cooldown":
        record["attempted"] = now.strftime(model_selection.STAMP)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-only")

    def metadata(url, *args):
        if url.endswith("/models"):
            if next_run == "catalogue_outage":
                raise TimeoutError()
            return {
                "data": [
                    {
                        "id": "vendor/current:free",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "architecture": {"output_modalities": ["text"]},
                        "supported_parameters": ["response_format"],
                    }
                ]
            }
        if next_run == "key_outage":
            raise TimeoutError()
        remaining = None if next_run == "unknown_quota" else 10 if next_run == "quota" else 100
        return {"data": {"free_model_daily_requests": {"remaining": remaining}}}

    async def probe(config, model, limiter):
        assert next_run == "inconclusive", "deferrals must not spend model requests"
        return {
            "model": model,
            "roles": ["contextualizer"],
            "results": [
                {
                    "case": case,
                    "outcome": "passed" if case == "contextualizer" else "inconclusive",
                    "reason": "",
                }
                for case in model_check.CASES
            ],
        }

    monkeypatch.setattr(model_check, "read_json", metadata)
    monkeypatch.setattr(model_check, "qualify", probe)
    out = tmp_path / "out.json"
    assert await model_check.run(argparse.Namespace(previous=previous, out=out, force=False)) == 1
    retained = json.loads((out if out.exists() else previous).read_text())
    assert retained["tested"] == record["tested"]
    assert retained["chains"] == record["chains"]
    assert retained["failed_models"] == {"generation": ["vendor/current:free"]}


def test_role_success_clears_only_that_roles_known_failure(tmp_path):
    config = settings(tmp_path)
    record = publication(
        config,
        failed_models={role: ["vendor/current:free"] for role in model_selection.ROLES},
    )
    report = {
        "model": "vendor/current:free",
        "roles": ["generation"],
        "results": [
            {"case": "grounded", "outcome": "passed"},
            {"case": "unanswerable", "outcome": "passed"},
            {"case": "injection", "outcome": "passed"},
            {"case": "contextualizer", "outcome": "inconclusive"},
        ],
    }
    retained = model_check.retain(
        record, record["catalogue"], tmp_path / "out", now=datetime.now(UTC), reports=[report]
    )
    assert retained["failed_models"] == {"contextualizer": ["vendor/current:free"]}
    assert retained["tested"] == record["tested"]
    assert model_check.unresolved(retained)


def test_withdrawal_remains_a_problem_after_a_metadata_outage(tmp_path):
    config = settings(tmp_path)
    record = publication(config)
    retained = model_check.retain(record, [], tmp_path / "out", now=datetime.now(UTC))
    reloaded = parse(json.dumps(retained), expected=fingerprint(config))
    assert model_check.unresolved(reloaded)
    assert reloaded["unavailable_roles"] == list(model_selection.ROLES)
    restored = model_check.retain(
        reloaded, record["catalogue"], tmp_path / "out", now=datetime.now(UTC)
    )
    assert not model_check.unresolved(restored)


async def test_concurrent_refresh_does_not_change_an_inflight_fallback_chain(tmp_path, monkeypatch):
    config = settings(tmp_path)
    chat = OpenRouterChat(
        config, models=["vendor/first:free", "vendor/fallback:free"], client=SimpleNamespace()
    )
    calls = []

    async def call(self, model, *args):
        calls.append(model)
        if model == "vendor/first:free":
            self.models[:] = ["vendor/new:free"]
            raise ProviderError("openrouter", "rate limited", retryable=True)
        return ChatOutcome(model=model, content="ok", finish_reason="stop")

    monkeypatch.setattr(OpenRouterChat, "_call", call)
    result = await chat.complete(messages=[], max_tokens=1)
    assert result.model == "vendor/fallback:free"
    assert calls == ["vendor/first:free", "vendor/fallback:free"]


async def test_only_published_passing_replacement_clears_a_failed_incumbent(tmp_path, monkeypatch):
    config = settings(tmp_path)
    now = datetime.now(UTC)
    record = publication(
        config,
        tested=(now - timedelta(days=7)).strftime(model_selection.STAMP),
        attempted=(now - timedelta(days=1)).strftime(model_selection.STAMP),
        failed_models={"generation": ["vendor/current:free"]},
        retry=True,
    )
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-only")
    models = ["vendor/current:free", "vendor/replacement:free"]

    def metadata(url, *args):
        if url.endswith("/models"):
            return {
                "data": [
                    {
                        "id": model,
                        "pricing": {"prompt": "0", "completion": "0"},
                        "architecture": {"output_modalities": ["text"]},
                        "supported_parameters": ["response_format"],
                    }
                    for model in models
                ]
            }
        return {"data": {"free_model_daily_requests": {"remaining": 100}}}

    async def probe(config, model, limiter):
        replacement = model == models[1]
        return {
            "model": model,
            "roles": list(model_selection.ROLES) if replacement else ["contextualizer"],
            "results": [
                {
                    "case": case,
                    "outcome": "passed"
                    if replacement or case == "contextualizer"
                    else "inconclusive",
                    "reason": "",
                }
                for case in model_check.CASES
            ],
        }

    monkeypatch.setattr(model_check, "read_json", metadata)
    monkeypatch.setattr(model_check, "qualify", probe)
    tentative = await probe(config, models[1], None)
    retained = model_check.retain(
        record, models, tmp_path / "retained.json", now=now, reports=[tentative]
    )
    assert retained["failed_models"] == record["failed_models"]
    out = tmp_path / "out.json"
    assert await model_check.run(argparse.Namespace(previous=previous, out=out, force=False)) == 0
    published = json.loads(out.read_text())
    assert published["chains"]["generation"] == ["vendor/replacement:free"]
    assert not model_check.unresolved(published)
    assert published["tested"] != record["tested"]
