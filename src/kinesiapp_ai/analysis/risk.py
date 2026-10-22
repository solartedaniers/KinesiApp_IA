import math
from dataclasses import dataclass
from statistics import median
from typing import Protocol

import numpy as np

from kinesiapp_ai.analysis.angles import AngleSeries
from kinesiapp_ai.analysis.movement_windows import MovementWindow
from kinesiapp_ai.analysis.pose_series import Side


@dataclass(frozen=True)
class LinearRamp:
    """Convierte un ángulo en un parcial de 0 a 1: 0 hasta `onset`, 1 desde `saturation`.

    Si `saturation` < `onset` la rampa es descendente (menos ángulo = más riesgo).
    """

    onset: float
    saturation: float

    def __post_init__(self) -> None:
        if self.onset == self.saturation:
            raise ValueError("onset and saturation must differ")

    def score(self, value: float) -> float:
        if math.isnan(value):
            return math.nan
        return min(1.0, max(0.0, (value - self.onset) / (self.saturation - self.onset)))


@dataclass(frozen=True)
class PartialScore:
    """Score de una señal en una repetición, con lo necesario para explicarlo: el ángulo medido y
    la rampa vigente al procesar (así no se relee de Settings, que puede haber cambiado). También el frame
    del pico y la pierna medida, para mostrar en qué instante y en qué lado ocurrió."""

    signal: str
    score: float
    measured_deg: float
    onset_deg: float
    saturation_deg: float
    peak_frame: int | None = None
    side: Side | None = None

    @property
    def higher_is_riskier(self) -> bool:
        return self.saturation_deg > self.onset_deg


class RiskScoringStrategy(Protocol):
    """Una señal medida en la ventana; devuelve uno o más parciales nombrados. No decide sola si
    hay riesgo: eso lo hace el aggregator al combinar parciales por patrones."""

    @property
    def partial_codes(self) -> frozenset[str]: ...

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, PartialScore]: ...


@dataclass(frozen=True)
class _Peak:
    value: float
    frame: int | None


def _peak(values: np.ndarray, window: MovementWindow) -> _Peak:
    in_window = values[window.slice()]
    if np.isnan(in_window).all():
        return _Peak(math.nan, None)
    offset = int(np.nanargmax(in_window))
    return _Peak(float(in_window[offset]), window.start_frame + offset)


def _score_peak(signal: str, peak: _Peak, ramps: dict[str, LinearRamp], side: Side | None) -> dict[str, PartialScore]:
    return {
        code: PartialScore(signal, ramp.score(peak.value), peak.value, ramp.onset, ramp.saturation, peak.frame, side)
        for code, ramp in ramps.items()
    }


class KneeFlexionRiskStrategy:
    """Flexión máxima de rodilla en la ventana."""

    SIGNAL = "knee_flexion"

    def __init__(self, ramps: dict[str, LinearRamp]) -> None:
        self._ramps = ramps

    @property
    def partial_codes(self) -> frozenset[str]:
        return frozenset(self._ramps)

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, PartialScore]:
        # De lado mido la pierna más visible: es la que señalo como lado en riesgo
        return _score_peak(self.SIGNAL, _peak(angles.knee_flexion, window), self._ramps, angles.side)


class TrunkFlexionRiskStrategy:
    """Inclinación máxima del tronco respecto de la vertical en la ventana."""

    SIGNAL = "trunk_inclination"

    def __init__(self, ramps: dict[str, LinearRamp]) -> None:
        self._ramps = ramps

    @property
    def partial_codes(self) -> frozenset[str]:
        return frozenset(self._ramps)

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, PartialScore]:
        return _score_peak(self.SIGNAL, _peak(angles.trunk_inclination, window), self._ramps, None)


class KneeValgusRiskStrategy:
    """Valgo máximo (FPPA) de cada rodilla en la ventana, vista de frente. Da un parcial por pierna, con
    el código `{code_prefix}_{lado}`, para que cada lado pueda dispararse solo."""

    SIGNAL = "knee_valgus"

    def __init__(self, ramp: LinearRamp, code_prefix: str) -> None:
        self._ramp = ramp
        self._codes = {side: f"{code_prefix}_{side.value}" for side in Side}

    @property
    def partial_codes(self) -> frozenset[str]:
        return frozenset(self._codes.values())

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, PartialScore]:
        partials = {}
        for side, code in self._codes.items():
            partials.update(_score_peak(self.SIGNAL, _peak(angles.knee_valgus[side], window), {code: self._ramp}, side))
        return partials


class KneeValgusAsymmetryRiskStrategy:
    """Diferencia entre el valgo máximo de las dos rodillas en la ventana. Marco como lado en riesgo y como
    instante los de la rodilla que más se va hacia adentro."""

    SIGNAL = "knee_valgus"

    def __init__(self, ramp: LinearRamp, code: str) -> None:
        self._ramp = ramp
        self._code = code

    @property
    def partial_codes(self) -> frozenset[str]:
        return frozenset({self._code})

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, PartialScore]:
        peaks = {side: _peak(angles.knee_valgus[side], window) for side in Side}
        if any(peak.frame is None for peak in peaks.values()):
            return _score_peak(self.SIGNAL, _Peak(math.nan, None), {self._code: self._ramp}, None)
        worse = max(peaks, key=lambda side: peaks[side].value)
        gap = _Peak(abs(peaks[Side.LEFT].value - peaks[Side.RIGHT].value), peaks[worse].frame)
        return _score_peak(self.SIGNAL, gap, {self._code: self._ramp}, worse)


@dataclass(frozen=True)
class RepetitionRisk:
    window: MovementWindow
    partials: dict[str, PartialScore]
    # NaN si a la repetición le falta alguna señal del patrón
    pattern_scores: dict[str, float]


@dataclass(frozen=True)
class AggregatedRisk:
    risk_score: float
    dominant_pattern: str | None
    pattern_scores: dict[str, float]
    patterns: dict[str, list[str]]
    repetitions: list[RepetitionRisk]


class RiskScoreAggregator:
    """Combina parciales por patrones.

    Patrón = parciales que deben darse juntos: su score en una ventana es el mínimo de ellos.
    Entre repeticiones, la mediana. El risk_score es el del peor patrón. No promedia: un promedio
    diluye un patrón claro.
    """

    def __init__(self, strategies: list[RiskScoringStrategy], patterns: dict[str, list[str]]) -> None:
        available = frozenset().union(*(strategy.partial_codes for strategy in strategies))
        for pattern, partials in patterns.items():
            missing = set(partials) - available
            if not partials or missing:
                raise ValueError(f"Risk pattern {pattern!r} needs unknown partials {sorted(missing)}")
        self._strategies = strategies
        self._patterns = patterns

    def aggregate(self, angles: AngleSeries, windows: list[MovementWindow]) -> AggregatedRisk:
        repetitions = [self._repetition(angles, window) for window in windows]
        pattern_scores = {}
        for pattern in self._patterns:
            scores = [rep.pattern_scores[pattern] for rep in repetitions if not math.isnan(rep.pattern_scores[pattern])]
            if scores:
                pattern_scores[pattern] = median(scores)
        dominant = max(pattern_scores, key=pattern_scores.get) if pattern_scores else None
        risk_score = pattern_scores[dominant] if dominant else 0.0
        return AggregatedRisk(
            risk_score=risk_score,
            dominant_pattern=dominant if risk_score > 0 else None,
            pattern_scores=pattern_scores,
            patterns=self._patterns,
            repetitions=repetitions,
        )

    def _repetition(self, angles: AngleSeries, window: MovementWindow) -> RepetitionRisk:
        partials: dict[str, PartialScore] = {}
        for strategy in self._strategies:
            partials.update(strategy.score(angles, window))
        pattern_scores = {}
        for pattern, required in self._patterns.items():
            scores = [partials[code].score for code in required]
            pattern_scores[pattern] = math.nan if any(math.isnan(score) for score in scores) else min(scores)
        return RepetitionRisk(window=window, partials=partials, pattern_scores=pattern_scores)
