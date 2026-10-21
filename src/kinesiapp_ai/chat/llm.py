from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol


class LlmUnavailableError(Exception):
    """El proveedor no pudo responder: timeout, cuota agotada, red o respuesta bloqueada."""


class LlmOverloadedError(LlmUnavailableError):
    """El modelo está saturado (503 o 504), agotó su cuota o no dio texto a tiempo: otro modelo sí podría responder."""


class LlmQuotaError(LlmOverloadedError):
    """El modelo agotó su cuota por minuto (429). La cuota es por modelo: el de respaldo tiene la suya."""


class LlmDailyQuotaError(LlmQuotaError):
    """El modelo agotó su cuota diaria (429): no vuelve hasta que el proveedor la reinicie."""


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
