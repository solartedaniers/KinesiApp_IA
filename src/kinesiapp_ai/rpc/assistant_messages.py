from typing import Annotated

from kinesiapp.ai.v1 import assistant_pb2
from kinesiapp.common.v1 import chat_pb2, identity_pb2, movement_pb2
from kinesiapp.common.v1.errors_pb2 import ErrorReason
from pydantic import StringConstraints, create_model

from kinesiapp_ai.analysis.movement_strategy import Movement
from kinesiapp_ai.chat.assessment_details import RiskAssessmentDetails
from kinesiapp_ai.chat.prompt_builder import AnalysisPromptInput, HistoryMessage
from kinesiapp_ai.rpc.domain_error import DomainError

_MOVEMENTS = {
    movement_pb2.MOVEMENT_TYPE_JUMP: Movement.JUMP.value,
    movement_pb2.MOVEMENT_TYPE_SQUAT: Movement.SQUAT.value,
}
_AUDIENCES = {
    identity_pb2.USER_ROLE_ATHLETE: "athlete",
    identity_pb2.USER_ROLE_COACH: "coach",
    identity_pb2.USER_ROLE_ADMIN: "admin",
}
_ROLES = {
    chat_pb2.CHAT_ROLE_USER: "user",
    chat_pb2.CHAT_ROLE_ASSISTANT: "assistant",
}


class AssistantMessages:
    """Valido y traduzco lo que manda el backend al dominio del chat, y armo las respuestas del contrato."""

    def __init__(self, details: RiskAssessmentDetails, question_max_chars: int) -> None:
        self._details = details
        # Armo el esquema con el tope configurado para que el error diga "question" y su límite
        self._question_schema = create_model(
            "QuestionInput",
            question=(Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=question_max_chars)], ...),
        )

    def analysis(self, facts: assistant_pb2.AnalysisFacts) -> AnalysisPromptInput:
        movement = _required(_MOVEMENTS, facts.movement_type, "facts.movement_type")
        return AnalysisPromptInput(
            movement=movement,
            risk_score=facts.risk_score,
            dominant_risk_pattern=facts.dominant_risk_pattern if facts.HasField("dominant_risk_pattern") else None,
            # Sin RiskAssessment (análisis antiguos) el prompt va en modo reducido
            risk_details=self._details.to_details(facts.risk_assessment) if facts.HasField("risk_assessment") else None,
        )

    @staticmethod
    def audience(role: identity_pb2.UserRole.ValueType) -> str:
        return _required(_AUDIENCES, role, "audience")

    @staticmethod
    def history(turns: list[assistant_pb2.ConversationTurn]) -> list[HistoryMessage]:
        return [HistoryMessage(role=_required(_ROLES, turn.role, "history.role"), content=turn.content) for turn in turns]

    def question(self, raw: str) -> str:
        return self._question_schema(question=raw).question


def _required(options: dict, value: int, field: str) -> str:
    if value not in options:
        raise DomainError(ErrorReason.ERROR_REASON_BAD_REQUEST, f"{field} is missing or invalid")
    return options[value]
