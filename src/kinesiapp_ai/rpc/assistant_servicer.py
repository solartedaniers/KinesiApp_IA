from collections.abc import AsyncIterator
from typing import TypeVar

import grpc
from kinesiapp.ai.v1 import assistant_pb2, assistant_pb2_grpc

from kinesiapp_ai.chat.analysis_assistant import AnalysisAssistant
from kinesiapp_ai.rpc.assistant_messages import AssistantMessages

Response = TypeVar("Response", assistant_pb2.ExplainAnalysisResponse, assistant_pb2.AnswerQuestionResponse)


class AssistantServicer(assistant_pb2_grpc.AssistantServiceServicer):
    """Implemento kinesiapp.ai.v1.AssistantService: reenvío cada parte del LLM y cierro con la respuesta completa."""

    def __init__(self, assistant: AnalysisAssistant, messages: AssistantMessages) -> None:
        self._assistant = assistant
        self._messages = messages

    async def ExplainAnalysis(
        self, request: assistant_pb2.ExplainAnalysisRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[assistant_pb2.ExplainAnalysisResponse]:
        deltas = self._assistant.explain(self._messages.analysis(request.facts), self._messages.audience(request.audience))
        async for response in self._relay(deltas, assistant_pb2.ExplainAnalysisResponse):
            yield response

    async def AnswerQuestion(
        self, request: assistant_pb2.AnswerQuestionRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[assistant_pb2.AnswerQuestionResponse]:
        messages = self._messages
        deltas = self._assistant.answer(
            messages.analysis(request.facts),
            messages.audience(request.audience),
            messages.history(list(request.history)),
            messages.question(request.question),
        )
        async for response in self._relay(deltas, assistant_pb2.AnswerQuestionResponse):
            yield response

    async def _relay(self, deltas: AsyncIterator[str], response: type[Response]) -> AsyncIterator[Response]:
        # Junto las partes para el mensaje final: el backend guarda ese texto, no las partes sueltas
        parts: list[str] = []
        async for delta in deltas:
            parts.append(delta)
            yield response(text_delta=delta)
        yield response(
            reply=assistant_pb2.AssistantReply(content="".join(parts), prompt_version=self._assistant.prompt_version)
        )
