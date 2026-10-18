import math
from statistics import median

from kinesiapp.ai.v1 import movement_analysis_pb2 as analysis_pb2
from kinesiapp.common.v1 import movement_pb2

from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage, MovementAnalysis
from kinesiapp_ai.analysis.movement_windows import DetectionMethod
from kinesiapp_ai.analysis.risk import AggregatedRisk, KneeFlexionRiskStrategy, PartialScore, RepetitionRisk, TrunkFlexionRiskStrategy
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
}
_DETECTION_METHODS = {
    DetectionMethod.LANDING: movement_pb2.DETECTION_METHOD_LANDING,
    DetectionMethod.KNEE_BOTTOM: movement_pb2.DETECTION_METHOD_KNEE_BOTTOM,
    DetectionMethod.TRUNK_HINGE: movement_pb2.DETECTION_METHOD_TRUNK_HINGE,
}
_RISK_LEVELS = {
    RiskLevel.LOW: movement_pb2.RISK_LEVEL_LOW,
    RiskLevel.MODERATE: movement_pb2.RISK_LEVEL_MODERATE,
    RiskLevel.HIGH: movement_pb2.RISK_LEVEL_HIGH,
}
_SCORE_DIGITS = 3
_ANGLE_DIGITS = 1


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
                    code=code, score=_score(partial.score), measured_degrees=_angle(partial.measured_deg)
                )
                for code, partial in rep.partials.items()
            ],
        )

    @staticmethod
    def _angle_samples(analysis: MovementAnalysis) -> list[movement_pb2.JointAngleSample]:
        series, angles = analysis.series, analysis.angles
        return [
            movement_pb2.JointAngleSample(
                joint=joint, angle_degrees=round(float(value), _ANGLE_DIGITS), frame_timestamp_ms=series.frame_timestamp_ms(frame)
            )
            for joint, values in (
                (movement_pb2.JOINT_ANGLE_KNEE_FLEXION, angles.knee_flexion),
                (movement_pb2.JOINT_ANGLE_TRUNK_INCLINATION, angles.trunk_inclination),
            )
            for frame, value in enumerate(values)
            if not math.isnan(value)
        ]


def _score(value: float) -> float | None:
    return None if math.isnan(value) else round(value, _SCORE_DIGITS)


def _angle(value: float) -> float | None:
    return None if math.isnan(value) else round(value, _ANGLE_DIGITS)
