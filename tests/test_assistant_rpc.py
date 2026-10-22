"""Pruebo AssistantService por la red con un LLM falso: streaming, respuesta final y errores."""

import asyncio
from functools import partial

import grpc
import pytest
from kinesiapp.ai.v1 import assistant_pb2, assistant_pb2_grpc
from kinesiapp.common.v1 import chat_pb2, identity_pb2, movement_pb2

from kinesiapp_ai.chat.analysis_assistant import AnalysisAssistant
from kinesiapp_ai.chat.assessment_details import RiskAssessmentDetails
from kinesiapp_ai.chat.llm import LlmDailyQuotaError, LlmQuotaError, LlmRequest, LlmUnavailableError
from kinesiapp_ai.chat.pattern_catalog import RiskPatternCatalog
from kinesiapp_ai.chat.prompt_builder import MovementAnalysisPromptBuilder
from kinesiapp_ai.chat.system_prompt import SYSTEM_PROMPT_VERSION
from kinesiapp_ai.rpc.assistant_messages import AssistantMessages
from kinesiapp_ai.rpc.assistant_servicer import AssistantServicer
from tests.grpc_harness import WITH_TOKEN, error_reason, serving

FACTS = assistant_pb2.AnalysisFacts(
    movement_type=movement_pb2.MOVEMENT_TYPE_JUMP, risk_score=0.8, dominant_risk_pattern="rigid_landing"
)


class _ScriptedLlm:
    def __init__(self, deltas: list[str], error: Exception | None = None) -> None:
        self.deltas, self.error = deltas, error
        self.requests: list[LlmRequest] = []

    async def stream_reply(self, request: LlmRequest):
        self.requests.append(request)
        for delta in self.deltas:
            yield delta
        if self.error:
            raise self.error


def _registration(llm: _ScriptedLlm):
    assistant = AnalysisAssistant(MovementAnalysisPromptBuilder(RiskPatternCatalog([]), 0.33, 0.66, 5, 10), llm)
    servicer = AssistantServicer(assistant, AssistantMessages(RiskAssessmentDetails(), question_max_chars=50))
    return partial(assistant_pb2_grpc.add_AssistantServiceServicer_to_server, servicer)


async def _stream(llm: _ScriptedLlm, call_name: str, request) -> tuple[list, grpc.aio.AioRpcError | None]:
    received = []
    async with serving([_registration(llm)]) as channel:
        call = getattr(assistant_pb2_grpc.AssistantServiceStub(channel), call_name)
        try:
            async for response in call(request, metadata=WITH_TOKEN):
                received.append(response)
        except grpc.aio.AioRpcError as error:
            return received, error
    return received, None


def test_the_explanation_streams_its_parts_and_ends_with_the_whole_reply() -> None:
    llm = _ScriptedLlm(["Tu aterrizaje ", "fue rígido."])
    request = assistant_pb2.ExplainAnalysisRequest(facts=FACTS, audience=identity_pb2.USER_ROLE_ATHLETE)

    received, error = asyncio.run(_stream(llm, "ExplainAnalysis", request))

    assert error is None
    assert [response.text_delta for response in received[:-1]] == ["Tu aterrizaje ", "fue rígido."]
    assert (received[-1].reply.content, received[-1].reply.prompt_version) == ("Tu aterrizaje fue rígido.", SYSTEM_PROMPT_VERSION)


def test_an_answer_uses_the_question_and_the_history() -> None:
    llm = _ScriptedLlm(["Porque la rodilla casi no se dobló."])
    request = assistant_pb2.AnswerQuestionRequest(
        facts=FACTS,
        audience=identity_pb2.USER_ROLE_COACH,
        history=[
            assistant_pb2.ConversationTurn(role=chat_pb2.CHAT_ROLE_ASSISTANT, content="explicación"),
        ],
        question="  ¿Por qué es rígido?  ",
    )

    received, error = asyncio.run(_stream(llm, "AnswerQuestion", request))

    assert error is None and received[-1].reply.content == "Porque la rodilla casi no se dobló."
    assert llm.requests[0].turns[-1].text == "¿Por qué es rígido?"
    assert [turn.role for turn in llm.requests[0].turns] == ["user", "model", "user"]


def test_a_provider_failure_mid_stream_ends_as_assistant_unavailable_without_a_reply() -> None:
    llm = _ScriptedLlm(["Empiezo"], error=LlmUnavailableError("quota"))
    request = assistant_pb2.ExplainAnalysisRequest(facts=FACTS, audience=identity_pb2.USER_ROLE_ATHLETE)

    received, error = asyncio.run(_stream(llm, "ExplainAnalysis", request))

    assert [response.WhichOneof("event") for response in received] == ["text_delta"]
    assert (error.code(), error_reason(error)) == (grpc.StatusCode.UNAVAILABLE, "ASSISTANT_UNAVAILABLE")


@pytest.mark.parametrize(
    ("error", "reason"),
    [(LlmQuotaError("429"), "ASSISTANT_BUSY"), (LlmDailyQuotaError("429"), "ASSISTANT_DAILY_LIMIT")],
    ids=["per-minute", "daily"],
)
def test_a_spent_quota_ends_as_resource_exhausted_with_its_reason(error: Exception, reason: str) -> None:
    llm = _ScriptedLlm([], error=error)
    request = assistant_pb2.ExplainAnalysisRequest(facts=FACTS, audience=identity_pb2.USER_ROLE_ATHLETE)

    received, rpc_error = asyncio.run(_stream(llm, "ExplainAnalysis", request))

    assert received == []
    assert (rpc_error.code(), error_reason(rpc_error)) == (grpc.StatusCode.RESOURCE_EXHAUSTED, reason)


@pytest.mark.parametrize(
    "request_",
    [
        assistant_pb2.AnswerQuestionRequest(facts=FACTS, audience=identity_pb2.USER_ROLE_COACH, question="   "),
        assistant_pb2.AnswerQuestionRequest(facts=FACTS, audience=identity_pb2.USER_ROLE_COACH, question="x" * 51),
        assistant_pb2.AnswerQuestionRequest(facts=FACTS, question="¿Y?"),
        assistant_pb2.AnswerQuestionRequest(audience=identity_pb2.USER_ROLE_COACH, question="¿Y?"),
    ],
    ids=["blank-question", "long-question", "no-audience", "no-movement"],
)
def test_invalid_requests_never_reach_the_llm(request_) -> None:
    llm = _ScriptedLlm(["no debería"])

    received, error = asyncio.run(_stream(llm, "AnswerQuestion", request_))

    assert received == [] and llm.requests == []
    assert (error.code(), error_reason(error)) == (grpc.StatusCode.INVALID_ARGUMENT, "BAD_REQUEST")
