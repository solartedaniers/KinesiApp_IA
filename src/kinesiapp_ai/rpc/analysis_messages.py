import math
from statistics import median

import numpy as np

from kinesiapp.ai.v1 import movement_analysis_pb2 as analysis_pb2
from kinesiapp.common.v1 import movement_pb2

from kinesiapp_ai.analysis.camera_view import CameraView
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage, MovementAnalysis
from kinesiapp_ai.analysis.movement_windows import DetectionMethod
from kinesiapp_ai.analysis.pose_series import (
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    Side,
)
from kinesiapp_ai.analysis.risk import (
    AggregatedRisk,
    KneeFlexionRiskStrategy,
    KneeValgusRiskStrategy,
    PartialScore,
    RepetitionRisk,
    TrunkFlexionRiskStrategy,
)
from kinesiapp_ai.analysis.risk_level import RiskLevel, RiskLevelClassifier

_STAGES = {
    AnalysisStage.QUEUED: movement_pb2.ANALYSIS_STAGE_QUEUED,
    AnalysisStage.DOWNLOADING_VIDEO: movement_pb2.ANALYSIS_STAGE_DOWNLOADING_VIDEO,
    AnalysisStage.EXTRACTING_POSE: movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE,
    AnalysisStage.DETECTING_REPETITIONS: movement_pb2.ANALYSIS_STAGE_DETECTING_REPETITIONS,
    AnalysisStage.SCORING_RISK: movement_pb2.ANALYSIS_STAGE_SCORING_RISK,
}
_JOINTS = {
    KneeFlexionRiskStrategy.SIGNAL: movement_pb2.JOINT_ANGLE_KNEE_FLEXION,
    TrunkFlexionRiskStrategy.SIGNAL: movement_pb2.JOINT_ANGLE_TRUNK_INCLINATION,
    KneeValgusRiskStrategy.SIGNAL: movement_pb2.JOINT_ANGLE_KNEE_VALGUS,
}
_SIDES = {Side.LEFT: movement_pb2.BODY_SIDE_LEFT, Side.RIGHT: movement_pb2.BODY_SIDE_RIGHT}
_VIEWS = {
    CameraView.SIDE: movement_pb2.CAMERA_VIEW_SIDE,
    CameraView.FRONT: movement_pb2.CAMERA_VIEW_FRONT,
    CameraView.INCONCLUSIVE: movement_pb2.CAMERA_VIEW_INCONCLUSIVE,
}
_WARNINGS = {CameraView.INCONCLUSIVE: [movement_pb2.ANALYSIS_WARNING_CAMERA_VIEW_INCONCLUSIVE]}
_KEYPOINTS = {
    LEFT_SHOULDER: movement_pb2.BODY_LANDMARK_LEFT_SHOULDER,
    RIGHT_SHOULDER: movement_pb2.BODY_LANDMARK_RIGHT_SHOULDER,
    LEFT_HIP: movement_pb2.BODY_LANDMARK_LEFT_HIP,
    RIGHT_HIP: movement_pb2.BODY_LANDMARK_RIGHT_HIP,
    LEFT_KNEE: movement_pb2.BODY_LANDMARK_LEFT_KNEE,
    RIGHT_KNEE: movement_pb2.BODY_LANDMARK_RIGHT_KNEE,
    LEFT_ANKLE: movement_pb2.BODY_LANDMARK_LEFT_ANKLE,
    RIGHT_ANKLE: movement_pb2.BODY_LANDMARK_RIGHT_ANKLE,
    LEFT_HEEL: movement_pb2.BODY_LANDMARK_LEFT_HEEL,
    RIGHT_HEEL: movement_pb2.BODY_LANDMARK_RIGHT_HEEL,
    LEFT_FOOT_INDEX: movement_pb2.BODY_LANDMARK_LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX: movement_pb2.BODY_LANDMARK_RIGHT_FOOT_INDEX,
}
_DETECTION_METHODS = {
    DetectionMethod.LANDING: movement_pb2.DETECTION_METHOD_LANDING,
    DetectionMethod.KNEE_BOTTOM: movement_pb2.DETECTION_METHOD_KNEE_BOTTOM,
    DetectionMethod.TRUNK_HINGE: movement_pb2.DETECTION_METHOD_TRUNK_HINGE,
    DetectionMethod.HIP_BOTTOM: movement_pb2.DETECTION_METHOD_HIP_BOTTOM,
}
_RISK_LEVELS = {
    RiskLevel.LOW: movement_pb2.RISK_LEVEL_LOW,
    RiskLevel.MODERATE: movement_pb2.RISK_LEVEL_MODERATE,
    RiskLevel.HIGH: movement_pb2.RISK_LEVEL_HIGH,
}
_SCORE_DIGITS = 3
_ANGLE_DIGITS = 1
# Cuatro decimales de coordenada normalizada son menos de un píxel incluso en 4K
_KEYPOINT_DIGITS = 4


class AnalysisMessages:
    """Traduzco el progreso y el resultado del análisis a los mensajes de kinesiapp.ai.v1.

    Dejo ausentes los campos optional cuando el valor es NaN: así el cliente distingue "no medido" de 0.
    """

    def __init__(self, risk_levels: RiskLevelClassifier, pose_model_version: str, risk_model_version: str) -> None:
        self._risk_levels = risk_levels
        self._pose_model_version = pose_model_version
        self._risk_model_version = risk_model_version

    @staticmethod
    def progress(
        stage: movement_pb2.AnalysisStage.ValueType, frames_processed: int = 0, frames_total: int = 0
    ) -> analysis_pb2.AnalyzeVideoResponse:
        return analysis_pb2.AnalyzeVideoResponse(
            progress=movement_pb2.AnalysisProgress(
                stage=stage, frames_processed=frames_processed, frames_total=frames_total
            )
        )

    @staticmethod
    def stage(stage: AnalysisStage) -> movement_pb2.AnalysisStage.ValueType:
        return _STAGES[stage]

    def result(self, analysis: MovementAnalysis) -> analysis_pb2.AnalyzeVideoResponse:
        series = analysis.series
        return analysis_pb2.AnalyzeVideoResponse(
            result=analysis_pb2.MovementAnalysisResult(
                risk_assessment=self._assessment(analysis),
                angle_samples=self._angle_samples(analysis),
                duration_ms=round(series.frame_count * 1000 / series.fps),
                pose_model_version=self._pose_model_version,
                risk_model_version=self._risk_model_version,
            )
        )

    def _assessment(self, analysis: MovementAnalysis) -> movement_pb2.RiskAssessment:
        risk = analysis.risk
        return movement_pb2.RiskAssessment(
            risk_score=round(risk.risk_score, _SCORE_DIGITS),
            risk_level=_RISK_LEVELS[self._risk_levels.classify(risk.risk_score)],
            dominant_pattern=risk.dominant_pattern,
            patterns=[self._pattern(pattern, codes, risk) for pattern, codes in risk.patterns.items()],
            repetitions=[self._repetition(rep, analysis) for rep in risk.repetitions],
            camera_view=_VIEWS[analysis.view],
            warnings=_WARNINGS.get(analysis.view, []),
        )

    def _pattern(self, pattern: str, codes: list[str], risk: AggregatedRisk) -> movement_pb2.RiskPattern:
        return movement_pb2.RiskPattern(
            code=pattern,
            score=_score(risk.pattern_scores.get(pattern, math.nan)),
            repetitions_triggered=sum(1 for rep in risk.repetitions if rep.pattern_scores[pattern] > 0),
            signals=[self._signal(code, [rep.partials[code] for rep in risk.repetitions]) for code in codes],
        )

    @staticmethod
    def _signal(code: str, partials: list[PartialScore]) -> movement_pb2.RiskSignal:
        reference = partials[0]
        measured = [partial.measured_deg for partial in partials if not math.isnan(partial.measured_deg)]
        scores = [partial.score for partial in partials if not math.isnan(partial.score)]
        return movement_pb2.RiskSignal(
            code=code,
            joint=_JOINTS[reference.signal],
            direction=(
                movement_pb2.RISK_DIRECTION_HIGHER_IS_RISKIER
                if reference.higher_is_riskier
                else movement_pb2.RISK_DIRECTION_LOWER_IS_RISKIER
            ),
            onset_degrees=reference.onset_deg,
            saturation_degrees=reference.saturation_deg,
            measured=(
                movement_pb2.AngleSummary(
                    median_degrees=round(median(measured), _ANGLE_DIGITS),
                    min_degrees=round(min(measured), _ANGLE_DIGITS),
                    max_degrees=round(max(measured), _ANGLE_DIGITS),
                )
                if measured
                else None
            ),
            score_median=round(median(scores), _SCORE_DIGITS) if scores else None,
            side=_side(reference.side),
        )

    @staticmethod
    def _repetition(rep: RepetitionRisk, analysis: MovementAnalysis) -> movement_pb2.Repetition:
        series = analysis.series
        return movement_pb2.Repetition(
            start_ms=series.frame_timestamp_ms(rep.window.start_frame),
            end_ms=series.frame_timestamp_ms(rep.window.end_frame),
            detected_by=_DETECTION_METHODS[rep.window.detected_by],
            pattern_scores=[
                movement_pb2.RepetitionScore(code=code, score=_score(score)) for code, score in rep.pattern_scores.items()
            ],
            signal_scores=[
                movement_pb2.RepetitionScore(
                    code=code,
                    score=_score(partial.score),
                    measured_degrees=_angle(partial.measured_deg),
                    side=_side(partial.side),
                    peak_ms=None if partial.peak_frame is None else series.frame_timestamp_ms(partial.peak_frame),
                )
                for code, partial in rep.partials.items()
            ],
            keyframes=[
                AnalysisMessages._keyframe(analysis, frame)
                for frame in sorted({partial.peak_frame for partial in rep.partials.values() if partial.peak_frame is not None})
            ],
        )

    @staticmethod
    def _keyframe(analysis: MovementAnalysis, frame: int) -> movement_pb2.PoseKeyframe:
        points = analysis.series.points[frame]
        return movement_pb2.PoseKeyframe(
            timestamp_ms=analysis.series.frame_timestamp_ms(frame),
            keypoints=[
                movement_pb2.Keypoint(
                    landmark=landmark,
                    x=round(float(points[index, 0]), _KEYPOINT_DIGITS),
                    y=round(float(points[index, 1]), _KEYPOINT_DIGITS),
                )
                for index, landmark in _KEYPOINTS.items()
                if not np.isnan(points[index]).any()
            ],
        )

    @staticmethod
    def _angle_samples(analysis: MovementAnalysis) -> list[movement_pb2.JointAngleSample]:
        series, angles = analysis.series, analysis.angles
        # Mando solo las series que la vista mide bien: de frente la flexión y el tronco salen deformados
        if analysis.view is CameraView.FRONT:
            measured = [
                (movement_pb2.JOINT_ANGLE_KNEE_VALGUS, _SIDES[side], values) for side, values in angles.knee_valgus.items()
            ]
        else:
            measured = [
                (movement_pb2.JOINT_ANGLE_KNEE_FLEXION, movement_pb2.BODY_SIDE_UNSPECIFIED, angles.knee_flexion),
                (movement_pb2.JOINT_ANGLE_TRUNK_INCLINATION, movement_pb2.BODY_SIDE_UNSPECIFIED, angles.trunk_inclination),
            ]
        return [
            movement_pb2.JointAngleSample(
                joint=joint,
                angle_degrees=round(float(value), _ANGLE_DIGITS),
                frame_timestamp_ms=series.frame_timestamp_ms(frame),
                side=side,
            )
            for joint, side, values in measured
            for frame, value in enumerate(values)
            if not math.isnan(value)
        ]


def _side(side: Side | None) -> movement_pb2.BodySide.ValueType:
    return movement_pb2.BODY_SIDE_UNSPECIFIED if side is None else _SIDES[side]


def _score(value: float) -> float | None:
    return None if math.isnan(value) else round(value, _SCORE_DIGITS)


def _angle(value: float) -> float | None:
    return None if math.isnan(value) else round(value, _ANGLE_DIGITS)
