from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol


class LlmUnavailableError(Exception):
    """El proveedor no pudo responder: timeout, cuota agotada, red o respuesta bloqueada."""


class LlmOverloadedError(LlmUnavailableError):
    """El modelo está saturado (503): es temporal y otro modelo sí podría responder."""


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

    def stream_reply(self, request: LlmRequest) -> AsyncIterator[str]:
        """Entrego la respuesta por partes, en cuanto el proveedor las genera."""
        ...
