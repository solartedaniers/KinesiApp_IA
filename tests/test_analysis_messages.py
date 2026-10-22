"""Pruebo la traducción del resultado del análisis a los mensajes del contrato."""

import math
from pathlib import Path

import numpy as np
from kinesiapp.common.v1 import movement_pb2

from kinesiapp_ai.analysis.angles import AngleCalculator
from kinesiapp_ai.analysis.camera_view import CameraView
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage, MovementAnalysis, MovementAnalyzer
from kinesiapp_ai.analysis.movement_strategy import Movement
from kinesiapp_ai.analysis.pose_series import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER, Side
from kinesiapp_ai.analysis.preprocessing import LandmarkSeriesPreprocessor
from kinesiapp_ai.pipeline.strategy_factory import MovementStrategyFactory
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.risk import RiskSettings
from kinesiapp_ai.analysis.movement_windows import DetectionMethod, MovementWindow
from kinesiapp_ai.analysis.risk import AggregatedRisk, PartialScore, RepetitionRisk
from kinesiapp_ai.analysis.risk_level import RiskLevelClassifier
from kinesiapp_ai.rpc.analysis_messages import AnalysisMessages
from tests.pose_fixtures import (
    VIEW_CLASSIFIER,
    add_jump,
    angle_series,
    bend_knee_inward,
    frontal_points,
    pose_series,
    standing_points,
)

MESSAGES = AnalysisMessages(RiskLevelClassifier(0.33, 0.66), "pose-v1", "risk-v1")


class _Fixed:
    def __init__(self, series) -> None:
        self._series = series

    def extract(self, video_path, on_frame=None):
        return self._series


class _Silent:
    def on_stage(self, stage) -> None:
        pass

    def on_frames(self, processed: int) -> None:
        pass


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
    return MovementAnalysis(series=series, angles=angles, risk=risk, view=CameraView.SIDE)


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


def test_every_stage_has_its_contract_value() -> None:
    assert {AnalysisMessages.stage(stage) for stage in AnalysisStage} == set(movement_pb2.AnalysisStage.values()) - {
        movement_pb2.ANALYSIS_STAGE_UNSPECIFIED
    }


def test_progress_messages_carry_stage_and_frames() -> None:
    progress = AnalysisMessages.progress(movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE, 30, 1200).progress

    assert (progress.stage, progress.frames_processed, progress.frames_total) == (movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE, 30, 1200)


def _analyzed(points: np.ndarray) -> MovementAnalysis:
    analyzer = MovementAnalyzer(_Fixed(pose_series(points)), LandmarkSeriesPreprocessor(0.1, 3), AngleCalculator(), VIEW_CLASSIFIER)
    strategies = MovementStrategyFactory(MovementPhaseSettings(), RiskSettings()).build_all()
    return analyzer.analyze(Path("video.mp4"), strategies[Movement.JUMP], _Silent())


def _frontal_jump() -> np.ndarray:
    points = frontal_points(90)
    add_jump(points, takeoff=30, landing=45, height=0.15)
    bend_knee_inward(points, slice(45, 55), Side.RIGHT, 30.0)
    return points


def test_a_frontal_result_carries_the_view_each_leg_and_the_pose_at_the_peak() -> None:
    result = MESSAGES.result(_analyzed(_frontal_jump())).result
    assessment = result.risk_assessment

    assert assessment.camera_view == movement_pb2.CAMERA_VIEW_FRONT
    assert list(assessment.warnings) == []
    assert assessment.dominant_pattern == "knee_valgus_right"
    assert {sample.joint for sample in result.angle_samples} == {movement_pb2.JOINT_ANGLE_KNEE_VALGUS}
    assert {sample.side for sample in result.angle_samples} == {movement_pb2.BODY_SIDE_LEFT, movement_pb2.BODY_SIDE_RIGHT}
    signal = next(s for p in assessment.patterns for s in p.signals if s.code == "knee_valgus_right")
    assert signal.side == movement_pb2.BODY_SIDE_RIGHT and signal.joint == movement_pb2.JOINT_ANGLE_KNEE_VALGUS
    repetition = assessment.repetitions[0]
    right = next(score for score in repetition.signal_scores if score.code == "knee_valgus_right")
    assert right.side == movement_pb2.BODY_SIDE_RIGHT and right.HasField("peak_ms")
    keyframe = next(frame for frame in repetition.keyframes if frame.timestamp_ms == right.peak_ms)
    assert len(keyframe.keypoints) == 12
    assert all(0 <= point.x <= 1 and 0 <= point.y <= 1 for point in keyframe.keypoints)


def test_an_inconclusive_view_warns_and_keeps_the_side_samples() -> None:
    points = _frontal_jump()
    for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        points[:, index, 0] = 0.5 + (points[:, index, 0] - 0.5) * 0.5

    result = MESSAGES.result(_analyzed(points)).result

    assert result.risk_assessment.camera_view == movement_pb2.CAMERA_VIEW_INCONCLUSIVE
    assert list(result.risk_assessment.warnings) == [movement_pb2.ANALYSIS_WARNING_CAMERA_VIEW_INCONCLUSIVE]
    assert {sample.joint for sample in result.angle_samples} == {
        movement_pb2.JOINT_ANGLE_KNEE_FLEXION,
        movement_pb2.JOINT_ANGLE_TRUNK_INCLINATION,
    }


def test_a_side_result_marks_the_measured_knee_and_the_moment_of_each_peak() -> None:
    points = standing_points(90)
    add_jump(points, takeoff=30, landing=45, height=0.15)

    assessment = MESSAGES.result(_analyzed(points)).result.risk_assessment

    assert assessment.camera_view == movement_pb2.CAMERA_VIEW_SIDE
    knee = next(score for score in assessment.repetitions[0].signal_scores if score.code == "knee_rigid")
    trunk = next(score for score in assessment.repetitions[0].signal_scores if score.code == "trunk_lean")
    assert knee.side in (movement_pb2.BODY_SIDE_LEFT, movement_pb2.BODY_SIDE_RIGHT) and knee.HasField("peak_ms")
    assert trunk.side == movement_pb2.BODY_SIDE_UNSPECIFIED
    assert {frame.timestamp_ms for frame in assessment.repetitions[0].keyframes} == {knee.peak_ms, trunk.peak_ms}
