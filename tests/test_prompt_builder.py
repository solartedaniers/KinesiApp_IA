"""JumpAnalysisPromptBuilder con datos simples: modo reducido, historial y catálogo, sin red ni base de datos."""
import pytest

from kinesiapp_ai.chat.pattern_catalog import RiskPatternCatalog
from kinesiapp_ai.chat.prompt_builder import AnalysisPromptInput, HistoryMessage, JumpAnalysisPromptBuilder

RISK_LEVEL_MODERATE = 0.33
RISK_LEVEL_HIGH = 0.66
MAX_REPETITIONS = 5


def _builder(max_history_messages: int = 10) -> JumpAnalysisPromptBuilder:
    return JumpAnalysisPromptBuilder(
        catalog=RiskPatternCatalog([]),
        risk_level_moderate=RISK_LEVEL_MODERATE,
        risk_level_high=RISK_LEVEL_HIGH,
        max_repetitions=MAX_REPETITIONS,
        max_history_messages=max_history_messages,
    )


def _reduced(risk_score: float, pattern: str | None) -> AnalysisPromptInput:
    return AnalysisPromptInput(movement="jump", risk_score=risk_score, dominant_risk_pattern=pattern, risk_details=None)


def test_reduced_mode_for_analyses_without_risk_details():
    data = _builder().analysis_data(_reduced(0.8, "rigid_landing"), "coach")
    assert (data["audience"], data["movement"]) == ("coach", "jump")
    assert data["details_available"] is False
    assert data["dominant_pattern"]["code"] == "rigid_landing"
    assert data["dominant_pattern"]["score"] == 0.8
    assert "signals" not in data and "per_repetition" not in data


def test_reduced_mode_for_simulated_analyses_without_pattern():
    data = _builder().analysis_data(_reduced(0.4, None), "coach")
    # Con score > 0 y sin patrón no se sabe qué pasó: no se afirma que no hubo riesgo
    assert data["dominant_pattern"] is None and "no_pattern" not in data
    assert data["risk_level"] == "moderate"


def test_reduced_mode_with_zero_score_says_there_was_no_pattern():
    data = _builder().analysis_data(_reduced(0, None), "athlete")
    assert data["risk_level"] == "low"
    assert "Ningún patrón" in data["no_pattern"]["meaning"]


def test_history_drops_unanswered_messages_and_keeps_the_most_recent():
    history = [
        HistoryMessage(role="user", content="q1"),
        HistoryMessage(role="assistant", content="a1"),
        HistoryMessage(role="user", content="sin respuesta"),
        HistoryMessage(role="user", content="q2"),
        HistoryMessage(role="assistant", content="a2"),
    ]
    request = _builder(max_history_messages=2).build(_reduced(0.8, "rigid_landing"), "coach", history, "q3")
    assert [(turn.role, turn.text) for turn in request.turns] == [("user", "q2"), ("model", "a2"), ("user", "q3")]


def test_history_starting_with_the_opening_keeps_it_after_its_request():
    history = [HistoryMessage(role="assistant", content="explicación"), HistoryMessage(role="user", content="q1")]
    request = _builder().build(_reduced(0.8, "rigid_landing"), "coach", history, "q2")
    assert [turn.role for turn in request.turns] == ["user", "model", "user"]
    assert request.turns[1].text == "explicación"


def test_opening_uses_the_same_system_prompt_as_the_chat():
    builder, analysis = _builder(), _reduced(0.8, "rigid_landing")
    opening = builder.build_opening(analysis, "coach")
    assert opening.system_instruction == builder.build(analysis, "coach", [], "x").system_instruction
    assert [turn.role for turn in opening.turns] == ["user"]


def test_catalog_rejects_pattern_codes_without_text():
    with pytest.raises(ValueError):
        RiskPatternCatalog([{"new_pattern": ["knee_rigid"]}])
