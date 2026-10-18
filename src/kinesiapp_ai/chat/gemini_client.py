import logging
import time
from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from kinesiapp_ai.chat.llm import LlmOverloadedError, LlmRequest, LlmUnavailableError

logger = logging.getLogger(__name__)

_OVERLOADED_STATUS = 503


class GeminiLlmClient:
    """Adapter sobre el SDK oficial google-genai, en streaming. Cualquier falla del proveedor sale como LlmUnavailableError."""

    def __init__(
        self, api_key: str, model_name: str, timeout_seconds: float, max_output_tokens: int, thinking_level: str | None
    ) -> None:
        # Hago un solo intento: ante una falla respondo no disponible, sin reintentos que gasten cuota
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=round(timeout_seconds * 1000), retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        # Con el razonamiento por defecto, el modelo gastaba casi todo max_output_tokens pensando: la respuesta
        # salía truncada y el primer texto tardaba varias veces más
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None

    async def stream_reply(self, request: LlmRequest) -> AsyncIterator[str]:
        produced = False
        started = time.perf_counter()
        first_text_ms = None
        usage = None
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=[
                    types.Content(role=turn.role, parts=[types.Part(text=turn.text)]) for turn in request.turns
                ],
                config=types.GenerateContentConfig(
                    system_instruction=request.system_instruction,
                    max_output_tokens=self._max_output_tokens,
                    thinking_config=self._thinking,
                ),
            )
            async for chunk in stream:
                usage = chunk.usage_metadata or usage
                if chunk.text:
                    produced = True
                    if first_text_ms is None:
                        first_text_ms = (time.perf_counter() - started) * 1000
                    yield chunk.text
        except (errors.APIError, httpx.HTTPError, TimeoutError) as error:
            # Registro el código HTTP para distinguir cuota (429), modelo inexistente (404) o saturación (503)
            code = getattr(error, "code", None)
            failure = LlmOverloadedError if code == _OVERLOADED_STATUS else LlmUnavailableError
            raise failure(f"Gemini failed: {type(error).__name__} {code or ''}".strip()) from error
        # Lo registro porque sin esto los logs no decían cuánto del chat era espera de Gemini ni cuánto razonó
        logger.info(
            "Gemini %s: first text %s ms, total %.0f ms, thoughts %s tokens, output %s tokens",
            self._model_name,
            f"{first_text_ms:.0f}" if first_text_ms is not None else "-",
            (time.perf_counter() - started) * 1000,
            getattr(usage, "thoughts_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )
        # Sin texto la respuesta vino bloqueada por seguridad o cortada antes de generar nada
        if not produced:
            raise LlmUnavailableError("Empty response from Gemini")
