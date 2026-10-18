"""Pruebo GeminiLlmClient sin red: reemplazo la llamada de streaming del SDK."""

import asyncio

import httpx
import pytest
from google.genai import errors, types

from kinesiapp_ai.chat.gemini_client import GeminiLlmClient
from kinesiapp_ai.chat.llm import LlmOverloadedError, LlmRequest, LlmTurn, LlmUnavailableError

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
):
    client = GeminiLlmClient(
        api_key="test-key", model_name="test-model", timeout_seconds=5, max_output_tokens=100, thinking_level=thinking_level
    )
    calls = []

    async def stream():
        for chunk in chunks:
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


def test_a_503_is_reported_as_overloaded_and_other_failures_are_not(monkeypatch: pytest.MonkeyPatch) -> None:
    overloaded, _ = _client(monkeypatch, [], errors.ServerError(503, {"error": {"message": "high demand"}}), fail_at_start=True)
    with pytest.raises(LlmOverloadedError):
        _collect(overloaded, [])

    quota, _ = _client(monkeypatch, [], errors.ClientError(429, {"error": {"message": "quota"}}), fail_at_start=True)
    with pytest.raises(LlmUnavailableError) as raised:
        _collect(quota, [])
    assert not isinstance(raised.value, LlmOverloadedError)
