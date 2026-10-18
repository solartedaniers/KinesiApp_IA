import logging
from collections.abc import AsyncIterator

from kinesiapp_ai.chat.llm import LlmClient, LlmOverloadedError, LlmRequest

logger = logging.getLogger(__name__)


class FallbackLlmClient:
    """Decorator: si el modelo principal está saturado antes de mandar texto, intento una sola vez con el de respaldo.

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
        except LlmOverloadedError:
            if sent:
                raise
            logger.warning("Primary LLM overloaded, retrying once with the fallback model")
        async for delta in self._fallback.stream_reply(request):
            yield delta
