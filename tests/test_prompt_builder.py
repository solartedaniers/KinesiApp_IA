"""JumpAnalysisPromptBuilder: qué datos recibe Gemini, sin llamar a la red."""
import json

import pytest

from app.analysis.movement_windows import DetectionMethod, MovementWindow
from app.analysis.risk_details import RiskDetailsSerializer
from app.chat.pattern_catalog import RiskPatternCatalog
from app.chat.system_prompt import SYSTEM_PROMPT_TEMPLATE
from app.core.config import settings
from app.models.chat import ChatMessage, ChatRole
from app.models.jump_analysis import JumpAnalysis, JumpAnalysisStatus, MovementType
from app.models.user import UserRole
from app.services.chat_factory import build_prompt_builder
from app.services.jump_video_analyzer_factory import build_jump_risk_profile, build_squat_risk_profile
from tests.pose_fixtures import angle_series, pose_series, standing_points


def _analysis(movement: MovementType, profile, knee: list[float], trunk: list[float], detected_by=DetectionMethod.LANDING):
    windows = [MovementWindow(frame, frame, detected_by) for frame in range(len(knee))]
    risk = profile.aggregator.aggregate(angle_series(knee, trunk), windows)
    return JumpAnalysis(
        athlete_id=1,
        video_reference="v.mp4",
        movement_type=movement,
        status=JumpAnalysisStatus.PROCESSED,
        risk_score=round(risk.risk_score, 3),
        dominant_risk_pattern=risk.dominant_pattern,
        risk_details=RiskDetailsSerializer().serialize(risk, pose_series(standing_points(len(knee)))),
    )


def _data(analysis, audience=UserRole.COACH):
    return build_prompt_builder(settings).analysis_data(analysis, audience)


def test_forward_collapse_sends_both_triggering_signals_with_measured_values():
    analysis = _analysis(MovementType.JUMP, build_jump_risk_profile(settings), [110, 105], [70, 60])
    data = _data(analysis)

    assert data["details_available"] is True
    assert data["risk_level"] == "high"
    assert data["dominant_pattern"]["code"] == "forward_collapse"
    assert "tronco cae hacia adelante" in data["dominant_pattern"]["meaning"]
    signals = {signal["code"]: signal for signal in data["signals"]}
    assert set(signals) == {"knee_deep", "trunk_lean"}
    assert signals["trunk_lean"]["measured_deg"] == {"median": 65.0, "min": 60.0, "max": 70.0}
    assert signals["trunk_lean"]["onset_deg"] == settings.RISK_JUMP_TRUNK_LEAN_ONSET_DEG
    assert data["repetitions"] == {"evaluated": 2, "detection_methods": {"landing": 2}, "triggering_dominant_pattern": 2}
    assert {other["code"] for other in data["other_patterns"]} == {"rigid_landing", "trunk_lean"}


def test_rigid_landing_sends_only_the_knee_signal_with_its_direction():
    data = _data(_analysis(MovementType.JUMP, build_jump_risk_profile(settings), [25, 30], [5, 5]))
    assert data["dominant_pattern"]["code"] == "rigid_landing"
    assert [(s["code"], s["direction"]) for s in data["signals"]] == [("knee_rigid", "lower_is_riskier")]


def test_hip_hinge_found_by_the_trunk_fallback_says_how_it_was_detected():
    analysis = _analysis(
        MovementType.SQUAT, build_squat_risk_profile(settings), [10], [95], detected_by=DetectionMethod.TRUNK_HINGE
    )
    data = _data(analysis, UserRole.ATHLETE)
    assert (data["audience"], data["movement"]) == ("athlete", "squat")
    assert data["dominant_pattern"]["code"] == "hip_hinge_squat"
    assert data["repetitions"]["detection_methods"] == {"trunk_hinge": 1}
    assert data["per_repetition"][0]["detected_by"] == "trunk_hinge"
    assert data["per_repetition"][0]["squat_knee_shallow"] == {"measured_deg": 10.0, "score": 1.0}


def test_no_pattern_explains_with_every_measured_signal():
    data = _data(_analysis(MovementType.SQUAT, build_squat_risk_profile(settings), [105], [58]))
    assert data["dominant_pattern"] is None
    assert data["risk_level"] == "low"
    assert "Ningún patrón" in data["no_pattern"]["meaning"]
    assert {s["code"] for s in data["signals"]} == {"squat_trunk_lean", "squat_knee_shallow"}


def test_per_repetition_keeps_the_highest_scoring_ones_in_time_order(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_PROMPT_MAX_REPETITIONS", 2)
    # Tres aterrizajes rígidos (frames 1, 3 y 4) y dos que no lo son
    knee = [85, 20, 85, 25, 22]
    analysis = _analysis(MovementType.JUMP, build_jump_risk_profile(settings), knee, [5] * 5)
    times = [rep["time_s"] for rep in _data(analysis)["per_repetition"]]
    # Se quedan dos rígidas, ninguna de las que no disparan el patrón
    assert times == sorted(times) and len(times) == 2
    assert times == [round(round(frame * 1000 / 30) / 1000, 2) for frame in (1, 3)]


def test_reduced_mode_for_analyses_without_risk_details():
    analysis = JumpAnalysis(
        athlete_id=1, video_reference="v.mp4", movement_type=MovementType.JUMP, status=JumpAnalysisStatus.PROCESSED,
        risk_score=0.8, dominant_risk_pattern="rigid_landing", risk_details=None,
    )
    data = _data(analysis)
    assert data["details_available"] is False
    assert data["dominant_pattern"]["code"] == "rigid_landing"
    assert data["dominant_pattern"]["score"] == 0.8
    assert "signals" not in data and "per_repetition" not in data


def test_reduced_mode_for_simulated_analyses_without_pattern():
    analysis = JumpAnalysis(
        athlete_id=1, video_reference="v.mp4", movement_type=MovementType.JUMP, status=JumpAnalysisStatus.PROCESSED,
        risk_score=0.4, dominant_risk_pattern=None, risk_details=None,
    )
    data = _data(analysis)
    # Con score > 0 y sin patrón no se sabe qué pasó: no se afirma que no hubo riesgo
    assert data["dominant_pattern"] is None and "no_pattern" not in data
    assert data["risk_level"] == "moderate"


def test_system_prompt_is_the_design_text_with_the_data_and_no_personal_data():
    analysis = _analysis(MovementType.JUMP, build_jump_risk_profile(settings), [110], [70])
    request = build_prompt_builder(settings).build(analysis, UserRole.COACH, [], "¿Qué corrijo?")
    prefix = SYSTEM_PROMPT_TEMPLATE.split("{analysis_json}")[0]
    assert request.system_instruction.startswith(prefix)
    payload = json.loads(request.system_instruction[len(prefix):])
    assert payload["dominant_pattern"]["code"] == "forward_collapse"
    assert set(payload) <= {
        "audience", "movement", "risk_score", "risk_level", "details_available", "dominant_pattern", "no_pattern",
        "other_patterns", "repetitions", "signals", "per_repetition", "measurement_limits",
    }
    assert [(turn.role, turn.text) for turn in request.turns] == [("user", "¿Qué corrijo?")]


def test_history_drops_unanswered_messages_and_keeps_the_most_recent(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_HISTORY_MAX_MESSAGES", 2)
    history = [
        ChatMessage(role=ChatRole.USER, content="q1"),
        ChatMessage(role=ChatRole.ASSISTANT, content="a1"),
        ChatMessage(role=ChatRole.USER, content="sin respuesta"),
        ChatMessage(role=ChatRole.USER, content="q2"),
        ChatMessage(role=ChatRole.ASSISTANT, content="a2"),
    ]
    analysis = _analysis(MovementType.JUMP, build_jump_risk_profile(settings), [110], [70])
    request = build_prompt_builder(settings).build(analysis, UserRole.COACH, history, "q3")
    assert [(turn.role, turn.text) for turn in request.turns] == [("user", "q2"), ("model", "a2"), ("user", "q3")]


def test_catalog_rejects_pattern_codes_without_text():
    with pytest.raises(ValueError):
        RiskPatternCatalog([{"new_pattern": ["knee_rigid"]}])
    RiskPatternCatalog([settings.RISK_JUMP_PATTERNS, settings.RISK_SQUAT_PATTERNS])
