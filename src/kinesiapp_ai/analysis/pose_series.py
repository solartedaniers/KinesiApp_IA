import enum
from dataclasses import dataclass

import numpy as np

# Índices del modelo BlazePose de 33 puntos que devuelve PoseLandmarker
LANDMARK_COUNT = 33
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32


class Side(str, enum.Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class SideLandmarks:
    shoulder: int
    hip: int
    knee: int
    ankle: int
    heel: int
    foot_index: int


SIDE_LANDMARKS = {
    Side.LEFT: SideLandmarks(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX),
    Side.RIGHT: SideLandmarks(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX),
}


@dataclass(frozen=True)
class PoseSeries:
    """Landmarks de una persona en todos los frames de un video.

    `points` es (frames, 33, 2) en coordenadas normalizadas de imagen (y crece hacia abajo), con
    NaN donde el landmark no se conoce; `visibility` es (frames, 33).
    """

    points: np.ndarray
    visibility: np.ndarray
    fps: float
    aspect_ratio: float

    @property
    def frame_count(self) -> int:
        return len(self.points)

    def frame_timestamp_ms(self, frame_index: int) -> int:
        return round(frame_index * 1000 / self.fps)

    def seconds_to_frames(self, seconds: float) -> int:
        return round(seconds * self.fps)
