import enum
from dataclasses import dataclass

from kinesiapp_ai.analysis.movement_windows import MovementWindowDetector
from kinesiapp_ai.analysis.risk import RiskScoreAggregator


class Movement(str, enum.Enum):
    JUMP = "jump"
    SQUAT = "squat"


@dataclass(frozen=True)
class MovementStrategy:
    """Strategy: lo que cambia entre salto y sentadilla, es decir, dónde busco las repeticiones y cómo las puntúo."""

    window_detector: MovementWindowDetector
    aggregator: RiskScoreAggregator
