import httpx
from google import genai
from google.genai import errors, types

from app.chat.llm import LlmRequest, LlmUnavailableError


class GeminiLlmClient:
    """SDK oficial google-genai. Cualquier falla del proveedor sale como LlmUnavailableError."""

    def __init__(self, api_key: str, model_name: str, timeout_seconds: float, max_output_tokens: int) -> None:
        # Un solo intento: ante una falla se degrada a 503 (§6.8), sin reintentos que gasten cuota
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=round(timeout_seconds * 1000), retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens

    def generate_reply(self, request: LlmRequest) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    types.Content(role=turn.role, parts=[types.Part(text=turn.text)]) for turn in request.turns
                ],
                config=types.GenerateContentConfig(
                    system_instruction=request.system_instruction,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
        except (errors.APIError, httpx.HTTPError) as error:
            raise LlmUnavailableError(str(error)) from error
        # Sin texto: respuesta bloqueada por seguridad o cortada antes de generar nada
        if not response.text:
            raise LlmUnavailableError("Empty response from Gemini")
        return response.text
