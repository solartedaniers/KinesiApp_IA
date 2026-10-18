"""Pruebo la validación y traducción de lo que el backend manda al asistente."""

import pytest
from kinesiapp.ai.v1 import assistant_pb2
from kinesiapp.common.v1 import chat_pb2, identity_pb2, movement_pb2
from kinesiapp.common.v1.errors_pb2 import ErrorReason
from pydantic import ValidationError

from kinesiapp_ai.chat.assessment_details import RiskAssessmentDetails
from kinesiapp_ai.chat.prompt_builder import HistoryMessage
from kinesiapp_ai.rpc.assistant_messages import AssistantMessages
from kinesiapp_ai.rpc.domain_error import DomainError

MESSAGES = AssistantMessages(RiskAssessmentDetails(), question_max_chars=20)


def _facts(**fields) -> assistant_pb2.AnalysisFacts:
    return assistant_pb2.AnalysisFacts(movement_type=movement_pb2.MOVEMENT_TYPE_JUMP, risk_score=0.4, **fields)


def test_facts_without_assessment_go_in_reduced_mode() -> None:
    analysis = MESSAGES.analysis(_facts(dominant_risk_pattern="rigid_landing"))

    assert (analysis.movement, analysis.risk_score, analysis.dominant_risk_pattern, analysis.risk_details) == (
        "jump",
        0.4,
        "rigid_landing",
        None,
    )


def test_facts_with_assessment_carry_its_details() -> None:
    assessment = movement_pb2.RiskAssessment(risk_score=0.4)

    analysis = MESSAGES.analysis(_facts(risk_assessment=assessment))

    assert analysis.dominant_risk_pattern is None
    assert analysis.risk_details["repetitions_evaluated"] == 0


def test_roles_and_history_are_translated() -> None:
    turns = [
        assistant_pb2.ConversationTurn(role=chat_pb2.CHAT_ROLE_ASSISTANT, content="explicación"),
        assistant_pb2.ConversationTurn(role=chat_pb2.CHAT_ROLE_USER, content="¿por qué?"),
    ]

    assert MESSAGES.audience(identity_pb2.USER_ROLE_COACH) == "coach"
    assert MESSAGES.history(turns) == [
        HistoryMessage(role="assistant", content="explicación"),
        HistoryMessage(role="user", content="¿por qué?"),
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda: MESSAGES.analysis(assistant_pb2.AnalysisFacts(risk_score=0.4)),
        lambda: MESSAGES.audience(identity_pb2.USER_ROLE_UNSPECIFIED),
        lambda: MESSAGES.history([assistant_pb2.ConversationTurn(content="sin rol")]),
    ],
    ids=["movement", "audience", "history-role"],
)
def test_unspecified_enums_are_bad_requests(call) -> None:
    with pytest.raises(DomainError) as raised:
        call()

    assert raised.value.reason == ErrorReason.ERROR_REASON_BAD_REQUEST


def test_the_question_is_trimmed() -> None:
    assert MESSAGES.question("  ¿Y la rodilla?  ") == "¿Y la rodilla?"


@pytest.mark.parametrize("question", ["", "   ", "x" * 21], ids=["empty", "blank", "too-long"])
def test_empty_or_too_long_questions_are_rejected_on_the_question_field(question: str) -> None:
    with pytest.raises(ValidationError) as raised:
        MESSAGES.question(question)

    assert raised.value.errors()[0]["loc"] == ("question",)
