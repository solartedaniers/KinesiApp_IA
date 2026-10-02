import numpy as np

from app.analysis.angles import AngleSeries
from app.analysis.movement_windows import DetectionMethod, MovementWindow, MovementWindowDetector
from app.analysis.pose_series import PoseSeries


class TrunkHingeDetector:
    """Encuentra repeticiones por el tronco: cada tramo continuo con el tronco por encima de
    `min_trunk_inclination_deg` que dura al menos `min_duration_seconds` es una repetición, y se
    evalúa alrededor de su inclinación máxima.

    Cubre la bisagra de cadera con las piernas casi rectas, que no deja un pico de flexión de
    rodilla. Un tramo termina cuando el tronco vuelve por debajo del umbral, así que dos bisagras
    separadas por una subida son dos repeticiones.
    """

    def __init__(self, min_trunk_inclination_deg: float, min_duration_seconds: float, window_ms: int) -> None:
        self._min_trunk_inclination_deg = min_trunk_inclination_deg
        self._min_duration_seconds = min_duration_seconds
        self._window_ms = window_ms

    def detect(self, series: PoseSeries, angles: AngleSeries) -> list[MovementWindow]:
        trunk = angles.trunk_inclination
        with np.errstate(invalid="ignore"):
            leaning = np.concatenate(([False], trunk >= self._min_trunk_inclination_deg, [False]))
        edges = np.flatnonzero(np.diff(leaning.astype(int)))
        min_frames = max(1, series.seconds_to_frames(self._min_duration_seconds))
        half = series.seconds_to_frames(self._window_ms / 2000)
        last_frame = series.frame_count - 1
        windows = []
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start < min_frames:
                continue
            peak = start + int(np.nanargmax(trunk[start:end]))
            windows.append(
                MovementWindow(max(0, peak - half), min(peak + half, last_frame), DetectionMethod.TRUNK_HINGE)
            )
        return windows


class FallbackWindowDetector:
    """Usa el primer detector que encuentre alguna repetición, en orden."""

    def __init__(self, detectors: list[MovementWindowDetector]) -> None:
        self._detectors = detectors

    def detect(self, series: PoseSeries, angles: AngleSeries) -> list[MovementWindow]:
        for detector in self._detectors:
            windows = detector.detect(series, angles)
            if windows:
                return windows
        return []
