"""Pruebo la traducción del resultado del análisis a los mensajes del contrato."""

import math

import numpy as np
from kinesiapp.common.v1 import movement_pb2

from kinesiapp_ai.analysis.movement_analyzer import MovementAnalysis
from kinesiapp_ai.analysis.movement_windows import DetectionMethod, MovementWindow
from kinesiapp_ai.analysis.risk import AggregatedRisk, PartialScore, RepetitionRisk
from kinesiapp_ai.analysis.risk_level import RiskLevelClassifier
from kinesiapp_ai.rpc.analysis_messages import AnalysisMessages
from tests.pose_fixtures import angle_series, pose_series, standing_points

MESSAGES = AnalysisMessages(RiskLevelClassifier(0.33, 0.66), "pose-v1", "risk-v1")


def _analysis(partials: list[dict[str, PartialScore]], pattern_scores: list[float]) -> MovementAnalysis:
    series = pose_series(standing_points(30))
    angles = angle_series([10.0] * 29 + [math.nan], [5.0] * 30)
    repetitions = [
        RepetitionRisk(MovementWindow(3 + 10 * index, 9 + 10 * index, DetectionMethod.LANDING), partial, {"rigid_landing": score})
        for index, (partial, score) in enumerate(zip(partials, pattern_scores, strict=True))
    ]
    measured = [score for score in pattern_scores if not math.isnan(score)]
    pattern = {"rigid_landing": float(np.median(measured))} if measured else {}
    risk = AggregatedRisk(
        risk_score=pattern.get("rigid_landing", 0.0),
        dominant_pattern="rigid_landing" if pattern else None,
        pattern_scores=pattern,
        patterns={"rigid_landing": ["knee_rigid"]},
        repetitions=repetitions,
    )
    return MovementAnalysis(series=series, angles=angles, risk=risk)


def _rigid(score: float, measured: float) -> dict[str, PartialScore]:
    return {"knee_rigid": PartialScore("knee_flexion", score, measured, onset_deg=60.0, saturation_deg=30.0)}


def test_the_result_carries_the_assessment_samples_duration_and_versions() -> None:
    result = MESSAGES.result(_analysis([_rigid(0.8, 36.0), _rigid(0.7, 39.04)], [0.8, 0.7])).result

    assessment = result.risk_assessment
    assert (assessment.risk_score, assessment.risk_level, assessment.dominant_pattern) == (0.75, movement_pb2.RISK_LEVEL_HIGH, "rigid_landing")
    signal = assessment.patterns[0].signals[0]
    assert (signal.joint, signal.direction) == (movement_pb2.JOINT_ANGLE_KNEE_FLEXION, movement_pb2.RISK_DIRECTION_LOWER_IS_RISKIER)
    assert (signal.measured.min_degrees, signal.measured.max_degrees, signal.score_median) == (36.0, 39.0, 0.75)
    assert assessment.patterns[0].repetitions_triggered == 2
    assert [(rep.start_ms, rep.end_ms) for rep in assessment.repetitions] == [(100, 300), (433, 633)]
    assert (result.duration_ms, result.pose_model_version, result.risk_model_version) == (1000, "pose-v1", "risk-v1")
    # Salto el frame sin rodilla medida: 29 muestras de rodilla y 30 de tronco
    joints = [sample.joint for sample in result.angle_samples]
    assert (joints.count(movement_pb2.JOINT_ANGLE_KNEE_FLEXION), joints.count(movement_pb2.JOINT_ANGLE_TRUNK_INCLINATION)) == (29, 30)


def test_unmeasured_values_stay_absent_instead_of_zero() -> None:
    assessment = MESSAGES.result(_analysis([_rigid(math.nan, math.nan)], [math.nan])).result.risk_assessment

    pattern = assessment.patterns[0]
    assert not pattern.HasField("score")
    assert not pattern.signals[0].HasField("measured") and not pattern.signals[0].HasField("score_median")
    signal_score = assessment.repetitions[0].signal_scores[0]
    assert not signal_score.HasField("score") and not signal_score.HasField("measured_degrees")
    assert not assessment.HasField("dominant_pattern")
    assert assessment.risk_level == movement_pb2.RISK_LEVEL_LOW


def test_progress_messages_carry_stage_and_frames() -> None:
    progress = AnalysisMessages.progress(movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE, 30, 1200).progress

    assert (progress.stage, progress.frames_processed, progress.frames_total) == (movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE, 30, 1200)
