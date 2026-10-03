from dataclasses import dataclass
from typing import Literal, Protocol


class LlmUnavailableError(Exception):
    """El proveedor no pudo responder: timeout, cuota agotada, red o respuesta bloqueada."""


@dataclass(frozen=True)
class LlmTurn:
    role: Literal["user", "model"]
    text: str


@dataclass(frozen=True)
class LlmRequest:
    system_instruction: str
    turns: list[LlmTurn]


class LlmClient(Protocol):
    """Frontera hacia el proveedor de LLM: el resto del dominio no sabe que es Gemini."""

    def generate_reply(self, request: LlmRequest) -> str: ...


class UnconfiguredLlmClient:
    """Se usa cuando no hay API key: el chat degrada a 503 en vez de impedir que arranque la app."""

    def generate_reply(self, request: LlmRequest) -> str:
        raise LlmUnavailableError("No LLM API key configured")
