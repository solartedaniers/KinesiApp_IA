import enum
from dataclasses import dataclass
from typing import Protocol

from kinesiapp_ai.analysis.angles import AngleSeries
from kinesiapp_ai.analysis.pose_series import PoseSeries


class DetectionMethod(str, enum.Enum):
    """Cómo se encontró la repetición: el chat lo usa para explicar el resultado."""

    LANDING = "landing"
    KNEE_BOTTOM = "knee_bottom"
    TRUNK_HINGE = "trunk_hinge"


@dataclass(frozen=True)
class MovementWindow:
    """Frames (inclusive) donde se evalúa el riesgo de una repetición: la amortiguación tras un
    aterrizaje o el fondo de una sentadilla."""

    start_frame: int
    end_frame: int
    detected_by: DetectionMethod

    def slice(self) -> slice:
        return slice(self.start_frame, self.end_frame + 1)


class MovementWindowDetector(Protocol):
    def detect(self, series: PoseSeries, angles: AngleSeries) -> list[MovementWindow]: ...
