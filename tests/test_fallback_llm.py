"""Pruebo cuándo el asistente recurre al modelo de respaldo y cuándo no."""

import asyncio

import pytest

from kinesiapp_ai.chat.fallback_llm import FallbackLlmClient
from kinesiapp_ai.chat.llm import LlmOverloadedError, LlmRequest, LlmTurn, LlmUnavailableError

REQUEST = LlmRequest(system_instruction="instrucciones", turns=[LlmTurn(role="user", text="hola")])


class _Model:
    def __init__(self, deltas: list[str], error: Exception | None = None) -> None:
        self.deltas, self.error = deltas, error
        self.calls = 0

    async def stream_reply(self, request: LlmRequest):
        self.calls += 1
        for delta in self.deltas:
            yield delta
        if self.error:
            raise self.error


def _collect(client: FallbackLlmClient, received: list[str]) -> None:
    async def main():
        async for delta in client.stream_reply(REQUEST):
            received.append(delta)

    asyncio.run(main())


def test_a_healthy_primary_never_touches_the_fallback() -> None:
    primary, fallback = _Model(["Hola"]), _Model(["respaldo"])
    received: list[str] = []

    _collect(FallbackLlmClient(primary, fallback), received)

    assert received == ["Hola"] and fallback.calls == 0


def test_an_overloaded_primary_before_any_text_retries_once_with_the_fallback() -> None:
    primary, fallback = _Model([], LlmOverloadedError("503")), _Model(["Desde ", "el respaldo"])
    received: list[str] = []

    _collect(FallbackLlmClient(primary, fallback), received)

    assert received == ["Desde ", "el respaldo"]
    assert (primary.calls, fallback.calls) == (1, 1)


def test_an_overload_mid_stream_does_not_retry() -> None:
    primary, fallback = _Model(["Parte"], LlmOverloadedError("503")), _Model(["respaldo"])
    received: list[str] = []

    with pytest.raises(LlmOverloadedError):
        _collect(FallbackLlmClient(primary, fallback), received)

    assert received == ["Parte"] and fallback.calls == 0


def test_other_failures_of_the_primary_do_not_retry() -> None:
    primary, fallback = _Model([], LlmUnavailableError("429 quota")), _Model(["respaldo"])

    with pytest.raises(LlmUnavailableError):
        _collect(FallbackLlmClient(primary, fallback), [])

    assert fallback.calls == 0


def test_when_the_fallback_also_fails_the_assistant_is_unavailable() -> None:
    primary, fallback = _Model([], LlmOverloadedError("503")), _Model([], LlmOverloadedError("503"))

    with pytest.raises(LlmUnavailableError):
        _collect(FallbackLlmClient(primary, fallback), [])

    assert (primary.calls, fallback.calls) == (1, 1)
