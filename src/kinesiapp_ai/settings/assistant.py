from typing import Literal

from pydantic import Field, SecretStr, model_validator

from kinesiapp_ai.settings.base import EnvSettings


class AssistantSettings(EnvSettings):
    """Configuro el asistente con Gemini y los límites de lo que recibo del backend."""

    # Sin valor por defecto, como todo secreto: si falta, la IA no arranca
    GEMINI_API_KEY: SecretStr = Field(min_length=1)
    # El 21-oct medí el primer texto: 3.5-flash-lite 0.5-0.9 s; 3.8-flash agotó sus 20 pedidos diarios del plan gratis
    GEMINI_MODEL: str = Field(default="gemini-3.5-flash-lite", min_length=1)
    # Opcional: si falta, cuando el principal se satura respondo ASSISTANT_UNAVAILABLE sin reintentar
    GEMINI_FALLBACK_MODEL: str | None = Field(default=None, min_length=1)
    GEMINI_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)
    # Plazo para el primer texto de cada modelo. Un modelo sano empieza en ~1 s: con 6 s, si el principal se queda
    # callado, el respaldo responde y el peor caso queda en ~12 s en vez de los 24 s que esperábamos con 12 s
    GEMINI_FIRST_TEXT_TIMEOUT_SECONDS: float = Field(default=6.0, gt=0)
    GEMINI_MAX_OUTPUT_TOKENS: int = Field(default=1024, gt=0)
    # Fijo cuánto razona el modelo antes de responder, porque ese razonamiento también cuenta contra
    # GEMINI_MAX_OUTPUT_TOKENS; si lo dejo vacío, uso el del modelo, para los que no aceptan el ajuste
    # MINIMAL porque con LOW 3.5-flash-lite tardó una vez 43 s en dar el primer texto
    GEMINI_THINKING_LEVEL: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"] | None = "MINIMAL"
    CHAT_QUESTION_MAX_CHARS: int = Field(default=1000, gt=0)
    # Recorto aquí el historial que me manda el backend: solo los últimos mensajes van al prompt
    CHAT_HISTORY_MAX_MESSAGES: int = Field(default=10, ge=0)
    CHAT_PROMPT_MAX_REPETITIONS: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _first_text_fits_in_the_request(self) -> "AssistantSettings":
        if self.GEMINI_FIRST_TEXT_TIMEOUT_SECONDS > self.GEMINI_TIMEOUT_SECONDS:
            raise ValueError("GEMINI_FIRST_TEXT_TIMEOUT_SECONDS cannot exceed GEMINI_TIMEOUT_SECONDS")
        return self
