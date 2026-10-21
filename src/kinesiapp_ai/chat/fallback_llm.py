import logging
from collections.abc import AsyncIterator

from kinesiapp_ai.chat.llm import LlmClient, LlmDailyQuotaError, LlmOverloadedError, LlmQuotaError, LlmRequest

logger = logging.getLogger(__name__)


class FallbackLlmClient:
    """Decorator: si el modelo principal falla de forma pasajera antes de mandar texto, intento una vez con el de respaldo.

    A mitad del stream no reintento: el cliente ya recibió parte de la respuesta y otra la duplicaría.
    """

    def __init__(self, primary: LlmClient, fallback: LlmClient) -> None:
        self._primary = primary
        self._fallback = fallback

    async def stream_reply(self, request: LlmRequest) -> AsyncIterator[str]:
        sent = False
        try:
            async for delta in self._primary.stream_reply(request):
                sent = True
                yield delta
            return
        except LlmOverloadedError as error:
            if sent:
                raise
            primary_error = error
            # El mensaje trae el modelo y el código (429, 503 o timeout), nunca la clave
            logger.warning("Primary LLM failed before any text (%s), retrying once with the fallback model", error)
        try:
            async for delta in self._fallback.stream_reply(request):
                yield delta
        except LlmQuotaError as error:
            # Solo digo "vuelve mañana" si los dos agotaron la diaria; si el principal falló por otra cosa, quizá vuelva pronto
            if isinstance(error, LlmDailyQuotaError) and isinstance(primary_error, LlmDailyQuotaError):
                raise
            raise LlmQuotaError(str(error)) from error
