import math
from dataclasses import dataclass
from statistics import median
from typing import Protocol

import numpy as np

from app.analysis.angles import AngleSeries
from app.analysis.movement_windows import MovementWindow


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


class RiskScoringStrategy(Protocol):
    """Una señal medida en la ventana; devuelve uno o más parciales nombrados. No decide sola si
    hay riesgo: eso lo hace el aggregator al combinar parciales por patrones."""

    @property
    def partial_codes(self) -> frozenset[str]: ...

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, float]: ...


def _peak(values: np.ndarray, window: MovementWindow) -> float:
    in_window = values[window.slice()]
    return math.nan if np.isnan(in_window).all() else float(np.nanmax(in_window))


class KneeFlexionRiskStrategy:
    """Flexión máxima de rodilla en la ventana."""

    def __init__(self, ramps: dict[str, LinearRamp]) -> None:
        self._ramps = ramps

    @property
    def partial_codes(self) -> frozenset[str]:
        return frozenset(self._ramps)

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, float]:
        peak = _peak(angles.knee_flexion, window)
        return {code: ramp.score(peak) for code, ramp in self._ramps.items()}


class TrunkFlexionRiskStrategy:
    """Inclinación máxima del tronco respecto de la vertical en la ventana."""

    def __init__(self, ramps: dict[str, LinearRamp]) -> None:
        self._ramps = ramps

    @property
    def partial_codes(self) -> frozenset[str]:
        return frozenset(self._ramps)

    def score(self, angles: AngleSeries, window: MovementWindow) -> dict[str, float]:
        peak = _peak(angles.trunk_inclination, window)
        return {code: ramp.score(peak) for code, ramp in self._ramps.items()}


@dataclass(frozen=True)
class AggregatedRisk:
    risk_score: float
    dominant_pattern: str | None
    pattern_scores: dict[str, float]


class RiskScoreAggregator:
    """Combina parciales por patrones (docs/design/video-analysis-pipeline.md §5.2).

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
        per_window = [self._partials(angles, window) for window in windows]
        pattern_scores = {}
        for pattern, required in self._patterns.items():
            scores = [
                min(values)
                for values in ([partials[code] for code in required] for partials in per_window)
                if not any(math.isnan(value) for value in values)
            ]
            if scores:
                pattern_scores[pattern] = median(scores)
        if not pattern_scores:
            return AggregatedRisk(risk_score=0.0, dominant_pattern=None, pattern_scores={})
        dominant = max(pattern_scores, key=pattern_scores.get)
        risk_score = pattern_scores[dominant]
        return AggregatedRisk(
            risk_score=risk_score,
            dominant_pattern=dominant if risk_score > 0 else None,
            pattern_scores=pattern_scores,
        )

    def _partials(self, angles: AngleSeries, window: MovementWindow) -> dict[str, float]:
        partials: dict[str, float] = {}
        for strategy in self._strategies:
            partials.update(strategy.score(angles, window))
        return partials
