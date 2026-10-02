import numpy as np
from scipy.signal import find_peaks

from app.analysis.angles import AngleSeries
from app.analysis.movement_windows import DetectionMethod, MovementWindow
from app.analysis.pose_series import PoseSeries


class SquatBottomDetector:
    """Encuentra el fondo de cada repetición como un pico de flexión de rodilla.

    ponytail: detecta por la rodilla, como el spike; la recomendación de REPORT.md es pasar a la
    trayectoria de la cadera, que no depende de un umbral de flexión.
    """

    def __init__(
        self,
        min_prominence_deg: float,
        min_flexion_deg: float,
        min_distance_seconds: float,
        window_ms: int,
    ) -> None:
        self._min_prominence_deg = min_prominence_deg
        self._min_flexion_deg = min_flexion_deg
        self._min_distance_seconds = min_distance_seconds
        self._window_ms = window_ms

    def detect(self, series: PoseSeries, angles: AngleSeries) -> list[MovementWindow]:
        flexion = angles.knee_flexion
        if np.isnan(flexion).any():
            return []
        bottoms, _ = find_peaks(
            flexion,
            prominence=self._min_prominence_deg,
            distance=max(1, series.seconds_to_frames(self._min_distance_seconds)),
        )
        half = series.seconds_to_frames(self._window_ms / 2000)
        last_frame = series.frame_count - 1
        return [
            MovementWindow(
                max(0, int(bottom) - half), min(int(bottom) + half, last_frame), DetectionMethod.KNEE_BOTTOM
            )
            for bottom in bottoms
            if flexion[bottom] >= self._min_flexion_deg
        ]
