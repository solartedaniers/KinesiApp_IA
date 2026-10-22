from collections import Counter
from typing import Any

from kinesiapp.common.v1 import movement_pb2

from kinesiapp_ai.analysis.camera_view import CameraView
from kinesiapp_ai.analysis.movement_windows import DetectionMethod
from kinesiapp_ai.analysis.pose_series import Side
from kinesiapp_ai.analysis.risk import KneeFlexionRiskStrategy, KneeValgusRiskStrategy, TrunkFlexionRiskStrategy

# Cambia si cambia la forma de los detalles: el constructor del prompt sabe qué esperar
RISK_DETAILS_FORMAT_VERSION = 1

_SIGNALS = {
    movement_pb2.JOINT_ANGLE_KNEE_FLEXION: KneeFlexionRiskStrategy.SIGNAL,
    movement_pb2.JOINT_ANGLE_TRUNK_INCLINATION: TrunkFlexionRiskStrategy.SIGNAL,
    movement_pb2.JOINT_ANGLE_KNEE_VALGUS: KneeValgusRiskStrategy.SIGNAL,
}
_DETECTED_BY = {
    movement_pb2.DETECTION_METHOD_LANDING: DetectionMethod.LANDING.value,
    movement_pb2.DETECTION_METHOD_KNEE_BOTTOM: DetectionMethod.KNEE_BOTTOM.value,
    movement_pb2.DETECTION_METHOD_TRUNK_HINGE: DetectionMethod.TRUNK_HINGE.value,
    movement_pb2.DETECTION_METHOD_HIP_BOTTOM: DetectionMethod.HIP_BOTTOM.value,
}
# Los análisis anteriores a la detección de vista se hicieron todos como de lado
_VIEWS = {
    movement_pb2.CAMERA_VIEW_UNSPECIFIED: CameraView.SIDE.value,
    movement_pb2.CAMERA_VIEW_SIDE: CameraView.SIDE.value,
    movement_pb2.CAMERA_VIEW_FRONT: CameraView.FRONT.value,
    movement_pb2.CAMERA_VIEW_INCONCLUSIVE: CameraView.INCONCLUSIVE.value,
}
_SIDES = {movement_pb2.BODY_SIDE_LEFT: Side.LEFT.value, movement_pb2.BODY_SIDE_RIGHT: Side.RIGHT.value}
_DIRECTIONS = {
    movement_pb2.RISK_DIRECTION_HIGHER_IS_RISKIER: "higher_is_riskier",
    movement_pb2.RISK_DIRECTION_LOWER_IS_RISKIER: "lower_is_riskier",
}


class RiskAssessmentDetails:
    """Adapter: paso el RiskAssessment del contrato al formato de detalles que lee el constructor del prompt.

    Así el prompt no cambia según de dónde vengan los datos, y un campo optional ausente vuelve a ser None.
    """

    def to_details(self, assessment: movement_pb2.RiskAssessment) -> dict[str, Any]:
        return {
            "format_version": RISK_DETAILS_FORMAT_VERSION,
            "risk_score": assessment.risk_score,
            "dominant_pattern": _optional(assessment, "dominant_pattern"),
            "camera_view": _VIEWS[assessment.camera_view],
            "repetitions_evaluated": len(assessment.repetitions),
            "detection_methods": dict(Counter(_DETECTED_BY[rep.detected_by] for rep in assessment.repetitions)),
            "patterns": {pattern.code: self._pattern(pattern) for pattern in assessment.patterns},
            "repetitions": [self._repetition(rep) for rep in assessment.repetitions],
        }

    @staticmethod
    def _pattern(pattern: movement_pb2.RiskPattern) -> dict[str, Any]:
        return {
            "score": _optional(pattern, "score"),
            "repetitions_triggered": pattern.repetitions_triggered,
            "signals": {
                signal.code: {
                    "signal": _SIGNALS[signal.joint],
                    "direction": _DIRECTIONS[signal.direction],
                    "onset_deg": signal.onset_degrees,
                    "saturation_deg": signal.saturation_degrees,
                    "measured_deg": (
                        {
                            "median": signal.measured.median_degrees,
                            "min": signal.measured.min_degrees,
                            "max": signal.measured.max_degrees,
                        }
                        if signal.HasField("measured")
                        else None
                    ),
                    "score_median": _optional(signal, "score_median"),
                    "side": _SIDES.get(signal.side),
                }
                for signal in pattern.signals
            },
        }

    @staticmethod
    def _repetition(rep: movement_pb2.Repetition) -> dict[str, Any]:
        return {
            "start_ms": rep.start_ms,
            "end_ms": rep.end_ms,
            "detected_by": _DETECTED_BY[rep.detected_by],
            "pattern_scores": {score.code: _optional(score, "score") for score in rep.pattern_scores},
            "partials": {
                score.code: {
                    "measured_deg": _optional(score, "measured_degrees"),
                    "score": _optional(score, "score"),
                    "side": _SIDES.get(score.side),
                    "peak_ms": _optional(score, "peak_ms"),
                }
                for score in rep.signal_scores
            },
        }


def _optional(message: Any, field: str) -> Any:
    return getattr(message, field) if message.HasField(field) else None
