"""Pruebo que el asistente arma el prompt correcto para cada caso y deja pasar el streaming del LLM."""

import asyncio

from kinesiapp_ai.chat.analysis_assistant import AnalysisAssistant
from kinesiapp_ai.chat.llm import LlmRequest
from kinesiapp_ai.chat.pattern_catalog import RiskPatternCatalog
from kinesiapp_ai.chat.prompt_builder import AnalysisPromptInput, HistoryMessage, MovementAnalysisPromptBuilder
from kinesiapp_ai.chat.system_prompt import OPENING_REQUEST, SYSTEM_PROMPT_VERSION

ANALYSIS = AnalysisPromptInput(movement="squat", risk_score=0.8, dominant_risk_pattern="hip_hinge_squat", risk_details=None)


class RecordingLlm:
    """Respondo siempre las mismas partes y guardo cada pedido que recibo."""

    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.requests: list[LlmRequest] = []

    async def stream_reply(self, request: LlmRequest):
        self.requests.append(request)
        for delta in self.deltas:
            yield delta


def _assistant(llm: RecordingLlm, max_history_messages: int = 2) -> AnalysisAssistant:
    prompts = MovementAnalysisPromptBuilder(RiskPatternCatalog([]), 0.33, 0.66, 5, max_history_messages)
    return AnalysisAssistant(prompts, llm)


async def _collect(stream) -> list[str]:
    return [delta async for delta in stream]


def test_the_explanation_asks_for_the_opening_with_the_analysis_data() -> None:
    llm = RecordingLlm(["Bajaste ", "doblando la cadera."])

    deltas = asyncio.run(_collect(_assistant(llm).explain(ANALYSIS, "athlete")))

    assert deltas == ["Bajaste ", "doblando la cadera."]
    request = llm.requests[0]
    assert [(turn.role, turn.text) for turn in request.turns] == [("user", OPENING_REQUEST)]
    assert '"hip_hinge_squat"' in request.system_instruction and '"athlete"' in request.system_instruction


def test_an_answer_sends_the_trimmed_history_and_the_question() -> None:
    llm = RecordingLlm(["Sí."])
    history = [
        HistoryMessage(role="assistant", content="explicación"),
        HistoryMessage(role="user", content="q1"),
        HistoryMessage(role="assistant", content="a1"),
    ]

    asyncio.run(_collect(_assistant(llm).answer(ANALYSIS, "coach", history, "¿Y la rodilla?")))

    assert [(turn.role, turn.text) for turn in llm.requests[0].turns] == [
        ("user", "q1"),
        ("model", "a1"),
        ("user", "¿Y la rodilla?"),
    ]


def test_it_reports_the_version_of_its_instructions() -> None:
    assert _assistant(RecordingLlm([])).prompt_version == SYSTEM_PROMPT_VERSION
