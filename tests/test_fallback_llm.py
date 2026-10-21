"""Pruebo cuándo el asistente recurre al modelo de respaldo y cuándo no."""

import asyncio
import logging

import pytest

from kinesiapp_ai.chat.fallback_llm import FallbackLlmClient
from kinesiapp_ai.chat.llm import (
    LlmDailyQuotaError,
    LlmOverloadedError,
    LlmQuotaError,
    LlmRequest,
    LlmTurn,
    LlmUnavailableError,
)

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
    primary, fallback = _Model([], LlmUnavailableError("400 bad request")), _Model(["respaldo"])

    with pytest.raises(LlmUnavailableError):
        _collect(FallbackLlmClient(primary, fallback), [])

    assert fallback.calls == 0


def test_when_the_fallback_also_fails_the_assistant_is_unavailable() -> None:
    primary, fallback = _Model([], LlmOverloadedError("503")), _Model([], LlmOverloadedError("503"))

    with pytest.raises(LlmUnavailableError):
        _collect(FallbackLlmClient(primary, fallback), [])

    assert (primary.calls, fallback.calls) == (1, 1)


def test_a_primary_out_of_quota_lets_the_fallback_answer_and_logs_why(caplog: pytest.LogCaptureFixture) -> None:
    # La cuota es por modelo: el 21-oct 3.8-flash agotó la diaria y el respaldo todavía tenía la suya
    primary = _Model([], LlmDailyQuotaError("Gemini gemini-3.8-flash failed: ClientError 429"))
    fallback = _Model(["respaldo"])
    received: list[str] = []

    with caplog.at_level(logging.WARNING):
        _collect(FallbackLlmClient(primary, fallback), received)

    assert received == ["respaldo"]
    assert "gemini-3.8-flash failed: ClientError 429" in caplog.text


@pytest.mark.parametrize(
    ("primary_error", "fallback_error", "expected"),
    [
        (LlmDailyQuotaError("429"), LlmDailyQuotaError("429"), LlmDailyQuotaError),
        (LlmDailyQuotaError("429"), LlmQuotaError("429"), LlmQuotaError),
        (LlmOverloadedError("timeout"), LlmDailyQuotaError("429"), LlmQuotaError),
        (LlmQuotaError("429"), LlmOverloadedError("503"), LlmOverloadedError),
    ],
    ids=["both-daily", "fallback-per-minute", "primary-timed-out", "fallback-saturated"],
)
def test_only_when_both_models_spent_their_daily_quota_the_limit_is_daily(primary_error, fallback_error, expected) -> None:
    primary, fallback = _Model([], primary_error), _Model([], fallback_error)

    with pytest.raises(LlmUnavailableError) as raised:
        _collect(FallbackLlmClient(primary, fallback), [])

    assert type(raised.value) is expected
