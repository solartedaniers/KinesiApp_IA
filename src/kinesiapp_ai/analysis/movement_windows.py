from dataclasses import dataclass
from typing import Protocol

from app.analysis.angles import AngleSeries
from app.analysis.pose_series import PoseSeries


@dataclass(frozen=True)
class MovementWindow:
    """Frames (inclusive) donde se evalúa el riesgo de una repetición: la amortiguación tras un
    aterrizaje o el fondo de una sentadilla."""

    start_frame: int
    end_frame: int

    def slice(self) -> slice:
        return slice(self.start_frame, self.end_frame + 1)


class MovementWindowDetector(Protocol):
    def detect(self, series: PoseSeries, angles: AngleSeries) -> list[MovementWindow]: ...
