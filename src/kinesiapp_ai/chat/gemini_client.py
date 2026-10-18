from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from kinesiapp_ai.chat.llm import LlmRequest, LlmUnavailableError


class GeminiLlmClient:
    """Adapter sobre el SDK oficial google-genai, en streaming. Cualquier falla del proveedor sale como LlmUnavailableError."""

    def __init__(self, api_key: str, model_name: str, timeout_seconds: float, max_output_tokens: int) -> None:
        # Hago un solo intento: ante una falla respondo no disponible, sin reintentos que gasten cuota
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=round(timeout_seconds * 1000), retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens

    async def stream_reply(self, request: LlmRequest) -> AsyncIterator[str]:
        produced = False
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=[
                    types.Content(role=turn.role, parts=[types.Part(text=turn.text)]) for turn in request.turns
                ],
                config=types.GenerateContentConfig(
                    system_instruction=request.system_instruction,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
            async for chunk in stream:
                if chunk.text:
                    produced = True
                    yield chunk.text
        except (errors.APIError, httpx.HTTPError, TimeoutError) as error:
            raise LlmUnavailableError(f"Gemini failed: {type(error).__name__}") from error
        # Sin texto la respuesta vino bloqueada por seguridad o cortada antes de generar nada
        if not produced:
            raise LlmUnavailableError("Empty response from Gemini")
