import numpy as np
from scipy.signal import find_peaks

from kinesiapp_ai.analysis.angles import AngleSeries
from kinesiapp_ai.analysis.movement_windows import DetectionMethod, MovementWindow
from kinesiapp_ai.analysis.pose_series import LEFT_ANKLE, LEFT_HIP, LEFT_SHOULDER, RIGHT_ANKLE, RIGHT_HIP, RIGHT_SHOULDER, PoseSeries


class SquatBottomDetector:
    """Encuentro el fondo de cada repetición como un pico de flexión de rodilla.

    Las sentadillas sin pico de rodilla (bisagra de cadera) las cubre TrunkHingeDetector como respaldo.
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


class HipBottomDetector:
    """Encuentro el fondo de cada sentadilla vista de frente como el punto más bajo de la cadera.

    De frente la flexión de rodilla no se mide bien, pero la cadera sube y baja igual que de lado. Pido una
    bajada mínima en fracción del alto del cuerpo (hombros a tobillos) para no contar balanceos.
    """

    def __init__(self, min_drop_body_fraction: float, min_distance_seconds: float, window_ms: int) -> None:
        self._min_drop_body_fraction = min_drop_body_fraction
        self._min_distance_seconds = min_distance_seconds
        self._window_ms = window_ms

    def detect(self, series: PoseSeries, angles: AngleSeries) -> list[MovementWindow]:
        y = series.points[:, :, 1]
        hip = np.nanmean(y[:, [LEFT_HIP, RIGHT_HIP]], axis=1)
        if np.isnan(hip).any():
            return []
        body_height = float(np.nanmedian(
            np.nanmean(y[:, [LEFT_ANKLE, RIGHT_ANKLE]], axis=1) - np.nanmean(y[:, [LEFT_SHOULDER, RIGHT_SHOULDER]], axis=1)
        ))
        if not body_height > 0:
            return []
        bottoms, _ = find_peaks(
            hip,
            prominence=self._min_drop_body_fraction * body_height,
            distance=max(1, series.seconds_to_frames(self._min_distance_seconds)),
        )
        half = series.seconds_to_frames(self._window_ms / 2000)
        last_frame = series.frame_count - 1
        return [
            MovementWindow(max(0, int(bottom) - half), min(int(bottom) + half, last_frame), DetectionMethod.HIP_BOTTOM)
            for bottom in bottoms
        ]
