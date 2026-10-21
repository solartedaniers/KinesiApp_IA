"""Pruebo GeminiLlmClient sin red: reemplazo la llamada de streaming del SDK."""

import asyncio

import httpx
import pytest
from google.genai import errors, types

from kinesiapp_ai.chat.gemini_client import GeminiLlmClient
from kinesiapp_ai.chat.llm import (
    LlmDailyQuotaError,
    LlmOverloadedError,
    LlmQuotaError,
    LlmRequest,
    LlmTurn,
    LlmUnavailableError,
)

REQUEST = LlmRequest(system_instruction="instrucciones", turns=[LlmTurn(role="user", text="hola")])


class _Chunk:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.usage_metadata = None


def _client(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list,
    error: Exception | None = None,
    fail_at_start: bool = False,
    thinking_level: str | None = "LOW",
    first_text_timeout: float = 5,
    pause: float = 0,
):
    client = GeminiLlmClient(
        api_key="test-key",
        model_name="test-model",
        timeout_seconds=5,
        first_text_timeout_seconds=first_text_timeout,
        max_output_tokens=100,
        thinking_level=thinking_level,
    )
    calls = []

    async def stream():
        for chunk in chunks:
            await asyncio.sleep(pause)
            yield _Chunk(chunk)
        if error and not fail_at_start:
            raise error

    async def generate_content_stream(**kwargs):
        calls.append(kwargs)
        if error and fail_at_start:
            raise error
        return stream()

    monkeypatch.setattr(client._client.aio.models, "generate_content_stream", generate_content_stream)
    return client, calls


def _collect(client: GeminiLlmClient, received: list[str]) -> None:
    async def main():
        async for delta in client.stream_reply(REQUEST):
            received.append(delta)

    asyncio.run(main())


def test_streams_each_text_chunk_with_instructions_turns_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client(monkeypatch, ["Hola, ", None, "atleta."])
    received: list[str] = []

    _collect(client, received)

    assert received == ["Hola, ", "atleta."]
    call = calls[0]
    assert call["model"] == "test-model"
    assert (call["config"].system_instruction, call["config"].max_output_tokens) == ("instrucciones", 100)
    assert [(content.role, content.parts[0].text) for content in call["contents"]] == [("user", "hola")]


def test_the_configured_thinking_level_is_sent_so_reasoning_does_not_eat_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client(monkeypatch, ["ok"])

    _collect(client, [])

    assert calls[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_without_thinking_level_the_model_default_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client(monkeypatch, ["ok"], thinking_level=None)

    _collect(client, [])

    assert calls[0]["config"].thinking_config is None


@pytest.mark.parametrize(
    "error",
    [
        errors.ClientError(429, {"error": {"message": "quota"}}),
        errors.ServerError(503, {"error": {"message": "overloaded"}}),
        httpx.TimeoutException("timeout"),
        TimeoutError(),
    ],
    ids=["quota", "server", "httpx-timeout", "timeout"],
)
def test_provider_failures_before_any_text_become_llm_unavailable(monkeypatch: pytest.MonkeyPatch, error) -> None:
    client, _ = _client(monkeypatch, [], error, fail_at_start=True)

    with pytest.raises(LlmUnavailableError):
        _collect(client, [])


def test_a_failure_mid_stream_keeps_what_was_sent_and_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, ["Primera parte"], errors.ServerError(500, {"error": {"message": "cut"}}))
    received: list[str] = []

    with pytest.raises(LlmUnavailableError):
        _collect(client, received)

    assert received == ["Primera parte"]


@pytest.mark.parametrize("chunks", [[], [None], [""]], ids=["nothing", "blocked", "empty"])
def test_a_reply_without_text_is_llm_unavailable(monkeypatch: pytest.MonkeyPatch, chunks: list) -> None:
    client, _ = _client(monkeypatch, chunks)

    with pytest.raises(LlmUnavailableError):
        _collect(client, [])


def test_the_failure_names_the_provider_status_code(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, [], errors.ClientError(429, {"error": {"message": "quota"}}), fail_at_start=True)

    with pytest.raises(LlmUnavailableError, match="ClientError 429"):
        _collect(client, [])


@pytest.mark.parametrize(
    "error",
    [
        errors.ServerError(503, {"error": {"message": "high demand"}}),
        errors.ServerError(504, {"error": {"message": "deadline"}}),
        httpx.ReadTimeout("timeout"),
        TimeoutError(),
    ],
    ids=["503", "504", "httpx-timeout", "timeout"],
)
def test_saturation_and_timeouts_before_any_text_are_overloaded_so_the_fallback_answers(
    monkeypatch: pytest.MonkeyPatch, error
) -> None:
    client, _ = _client(monkeypatch, [], error, fail_at_start=True)

    with pytest.raises(LlmOverloadedError):
        _collect(client, [])


def _quota_error(quota_id: str) -> errors.ClientError:
    # Copio la forma del 429 que devolvió Google el 21-oct
    violation = {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests", "quotaId": quota_id}
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [violation]}]
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}})


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), LlmDailyQuotaError),
        (_quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), LlmQuotaError),
        (errors.ClientError(429, {"error": {"message": "quota"}}), LlmQuotaError),
    ],
    ids=["daily", "per-minute", "without-details"],
)
def test_quota_tells_daily_from_per_minute_and_lets_the_fallback_answer(
    monkeypatch: pytest.MonkeyPatch, error, expected
) -> None:
    client, _ = _client(monkeypatch, [], error, fail_at_start=True)

    with pytest.raises(LlmOverloadedError) as raised:
        _collect(client, [])

    assert type(raised.value) is expected


@pytest.mark.parametrize(
    "error",
    [errors.ClientError(400, {"error": {"message": "bad"}}), errors.ClientError(404, {"error": {"message": "gone"}})],
    ids=["400", "404"],
)
def test_request_errors_are_not_overloaded_because_the_fallback_would_fail_the_same(
    monkeypatch: pytest.MonkeyPatch, error
) -> None:
    client, _ = _client(monkeypatch, [], error, fail_at_start=True)

    with pytest.raises(LlmUnavailableError) as raised:
        _collect(client, [])

    assert not isinstance(raised.value, LlmOverloadedError)


def test_a_model_that_sends_no_text_in_time_is_cut_as_overloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reproduzco el 21-oct: el principal se quedó callado y el chat esperó 30 s para nada
    client, _ = _client(monkeypatch, ["tarde"], first_text_timeout=0.05, pause=1)

    with pytest.raises(LlmOverloadedError):
        _collect(client, [])


def test_after_the_first_text_the_first_text_deadline_no_longer_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, ["Hola, ", "atleta."], first_text_timeout=0.15, pause=0.1)
    received: list[str] = []

    _collect(client, received)

    assert received == ["Hola, ", "atleta."]
