"""MovementAnalysisPromptBuilder con datos simples: modo reducido, historial y catálogo, sin red ni base de datos."""
import pytest

from kinesiapp_ai.chat.pattern_catalog import RiskPatternCatalog
from kinesiapp_ai.chat.prompt_builder import AnalysisPromptInput, HistoryMessage, MovementAnalysisPromptBuilder
from kinesiapp_ai.settings.risk import RiskSettings

RISK_LEVEL_MODERATE = 0.33
RISK_LEVEL_HIGH = 0.66
MAX_REPETITIONS = 5


def _builder(max_history_messages: int = 10) -> MovementAnalysisPromptBuilder:
    return MovementAnalysisPromptBuilder(
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


def _frontal_details() -> dict:
    signal = {
        "signal": "knee_valgus",
        "direction": "higher_is_riskier",
        "onset_deg": 10.0,
        "saturation_deg": 25.0,
        "measured_deg": {"median": 21.0, "min": 18.0, "max": 24.0},
        "score_median": 0.73,
        "side": "left",
    }
    return {
        "format_version": 1,
        "risk_score": 0.73,
        "dominant_pattern": "knee_valgus_left",
        "camera_view": "front",
        "repetitions_evaluated": 1,
        "detection_methods": {"landing": 1},
        "patterns": {"knee_valgus_left": {"score": 0.73, "repetitions_triggered": 1, "signals": {"knee_valgus_left": signal}}},
        "repetitions": [
            {
                "start_ms": 1500,
                "end_ms": 1800,
                "detected_by": "landing",
                "pattern_scores": {"knee_valgus_left": 0.73},
                "partials": {"knee_valgus_left": {"measured_deg": 21.0, "score": 0.73, "side": "left", "peak_ms": 1633}},
            }
        ],
    }


def test_the_valgus_reaches_the_prompt_with_its_leg_moment_and_view_limits():
    analysis = AnalysisPromptInput(movement="jump", risk_score=0.73, dominant_risk_pattern="knee_valgus_left", risk_details=_frontal_details())
    data = _builder().analysis_data(analysis, "coach")
    assert data["camera_view"] == "front"
    assert data["dominant_pattern"]["code"] == "knee_valgus_left"
    assert data["signals"][0]["side"] == "left"
    assert data["per_repetition"][0]["knee_valgus_left"] == {"measured_deg": 21.0, "score": 0.73, "side": "left", "peak_time_s": 1.63}
    assert any("Vista frontal" in limit for limit in data["measurement_limits"])


def test_an_inconclusive_view_tells_the_assistant_to_record_from_the_front():
    details = _frontal_details() | {"camera_view": "inconclusive"}
    analysis = AnalysisPromptInput(movement="jump", risk_score=0.73, dominant_risk_pattern=None, risk_details=details)
    assert any("grabar de frente" in limit for limit in _builder().analysis_data(analysis, "athlete")["measurement_limits"])


def test_analyses_without_details_were_recorded_from_the_side():
    data = _builder().analysis_data(_reduced(0.8, "rigid_landing"), "coach")
    assert data["camera_view"] == "side"
    assert any("Vista lateral" in limit for limit in data["measurement_limits"])


def test_the_catalog_explains_every_frontal_pattern():
    RiskPatternCatalog([RiskSettings().RISK_FRONT_PATTERNS])
