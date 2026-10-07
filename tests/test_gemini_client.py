"""GeminiLlmClient sin red: se reemplaza la llamada del SDK."""
import httpx
import pytest
from google.genai import errors

from kinesiapp_ai.chat.gemini_client import GeminiLlmClient
from kinesiapp_ai.chat.llm import LlmRequest, LlmTurn, LlmUnavailableError

REQUEST = LlmRequest(system_instruction="instrucciones", turns=[LlmTurn(role="user", text="hola")])


class _Response:
    def __init__(self, text):
        self.text = text


def _client(monkeypatch, outcome) -> GeminiLlmClient:
    client = GeminiLlmClient(api_key="test-key", model_name="test-model", timeout_seconds=5, max_output_tokens=100)
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(client._client.models, "generate_content", generate_content)
    client.calls = calls
    return client


def test_sends_system_instruction_turns_and_model(monkeypatch):
    client = _client(monkeypatch, "respuesta")
    assert client.generate_reply(REQUEST) == "respuesta"
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["config"].system_instruction == "instrucciones"
    assert call["config"].max_output_tokens == 100
    assert [(c.role, c.parts[0].text) for c in call["contents"]] == [("user", "hola")]


@pytest.mark.parametrize(
    "outcome",
    [
        errors.ClientError(429, {"error": {"message": "quota"}}),
        errors.ServerError(503, {"error": {"message": "overloaded"}}),
        httpx.TimeoutException("timeout"),
        None,
        "",
    ],
    ids=["quota", "server", "timeout", "blocked", "empty"],
)
def test_provider_failures_become_llm_unavailable(monkeypatch, outcome):
    with pytest.raises(LlmUnavailableError):
        _client(monkeypatch, outcome).generate_reply(REQUEST)
