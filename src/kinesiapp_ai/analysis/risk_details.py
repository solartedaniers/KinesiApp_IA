import math
from collections import Counter
from statistics import median
from typing import Any

from app.analysis.pose_series import PoseSeries
from app.analysis.risk import AggregatedRisk, PartialScore, RepetitionRisk

# Cambia si cambia la forma del JSON: quien lo lea (el chat) sabe qué esperar
RISK_DETAILS_FORMAT_VERSION = 1


class RiskDetailsSerializer:
    """AggregatedRisk → JSON persistible en jump_analyses.risk_details.

    Por patrón: su score, cuántas repeticiones lo dispararon y, por cada señal, el ángulo medido
    (mediana, mínimo y máximo entre repeticiones), la rampa vigente y la dirección. Por
    repetición: su ventana en ms, cómo se detectó y sus valores. NaN se guarda como null.
    """

    def serialize(self, risk: AggregatedRisk, series: PoseSeries) -> dict[str, Any]:
        return {
            "format_version": RISK_DETAILS_FORMAT_VERSION,
            "risk_score": _round_score(risk.risk_score),
            "dominant_pattern": risk.dominant_pattern,
            "repetitions_evaluated": len(risk.repetitions),
            "detection_methods": dict(Counter(rep.window.detected_by.value for rep in risk.repetitions)),
            "patterns": {
                pattern: self._pattern(pattern, partial_codes, risk)
                for pattern, partial_codes in risk.patterns.items()
            },
            "repetitions": [self._repetition(rep, series) for rep in risk.repetitions],
        }

    def _pattern(self, pattern: str, partial_codes: list[str], risk: AggregatedRisk) -> dict[str, Any]:
        scores = [rep.pattern_scores[pattern] for rep in risk.repetitions]
        return {
            "score": _round_score(risk.pattern_scores.get(pattern, math.nan)),
            "repetitions_triggered": sum(1 for score in scores if score > 0),
            "signals": {
                code: self._signal([rep.partials[code] for rep in risk.repetitions]) for code in partial_codes
            },
        }

    @staticmethod
    def _signal(partials: list[PartialScore]) -> dict[str, Any]:
        reference = partials[0]
        measured = [partial.measured_deg for partial in partials if not math.isnan(partial.measured_deg)]
        scores = [partial.score for partial in partials if not math.isnan(partial.score)]
        return {
            "signal": reference.signal,
            "direction": "higher_is_riskier" if reference.higher_is_riskier else "lower_is_riskier",
            "onset_deg": reference.onset_deg,
            "saturation_deg": reference.saturation_deg,
            "measured_deg": (
                {"median": _round_angle(median(measured)), "min": _round_angle(min(measured)), "max": _round_angle(max(measured))}
                if measured
                else None
            ),
            "score_median": _round_score(median(scores)) if scores else None,
        }

    @staticmethod
    def _repetition(rep: RepetitionRisk, series: PoseSeries) -> dict[str, Any]:
        return {
            "start_ms": series.frame_timestamp_ms(rep.window.start_frame),
            "end_ms": series.frame_timestamp_ms(rep.window.end_frame),
            "detected_by": rep.window.detected_by.value,
            "pattern_scores": {pattern: _round_score(score) for pattern, score in rep.pattern_scores.items()},
            "partials": {
                code: {"measured_deg": _round_angle(partial.measured_deg), "score": _round_score(partial.score)}
                for code, partial in rep.partials.items()
            },
        }


def _round_angle(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 1)


def _round_score(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 3)
