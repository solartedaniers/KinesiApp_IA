import numpy as np
from scipy.signal import find_peaks

from app.analysis.angles import AngleSeries
from app.analysis.movement_windows import MovementWindow
from app.analysis.pose_series import (
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseSeries,
)


class JumpPhaseDetector:
    """Encuentra cada aterrizaje y devuelve su ventana de amortiguación.

    Las tres reglas confirmadas con videos reales (docs/design/video-analysis-pipeline.md §5.1):
    cada salto se ancla en un ápice de la cadera; el pie está en el aire según su punto más bajo
    (tobillo, talón o punta); y el suelo es local, un percentil del pie en una ventana móvil.
    """

    def __init__(
        self,
        hip_apex_prominence: float,
        hip_apex_min_distance_seconds: float,
        apex_airborne_body_fraction: float,
        airborne_body_fraction: float,
        ground_window_seconds: float,
        ground_percentile: float,
        landing_window_ms: int,
        landing_end_margin_frames: int,
    ) -> None:
        self._hip_apex_prominence = hip_apex_prominence
        self._hip_apex_min_distance_seconds = hip_apex_min_distance_seconds
        self._apex_airborne_body_fraction = apex_airborne_body_fraction
        self._airborne_body_fraction = airborne_body_fraction
        self._ground_window_seconds = ground_window_seconds
        self._ground_percentile = ground_percentile
        self._landing_window_ms = landing_window_ms
        self._landing_end_margin_frames = landing_end_margin_frames

    def detect(self, series: PoseSeries, angles: AngleSeries) -> list[MovementWindow]:
        y = series.points[:, :, 1]
        hip = np.nanmean(y[:, [LEFT_HIP, RIGHT_HIP]], axis=1)
        if np.isnan(hip).any():
            return []
        foot = self._lowest_foot_point(y)
        shoulder = np.nanmean(y[:, [LEFT_SHOULDER, RIGHT_SHOULDER]], axis=1)
        body_height = float(np.nanmedian(np.abs(foot - shoulder)))
        ground = self._local_ground(foot, series)

        def airborne(frame: int, body_fraction: float) -> bool:
            return bool(foot[frame] < ground[frame] - body_fraction * body_height)

        apexes, _ = find_peaks(
            -hip,
            prominence=self._hip_apex_prominence,
            distance=max(1, series.seconds_to_frames(self._hip_apex_min_distance_seconds)),
        )
        last_frame = series.frame_count - 1
        window_frames = series.seconds_to_frames(self._landing_window_ms / 1000)
        windows = []
        for apex in apexes:
            if not airborne(apex, self._apex_airborne_body_fraction):
                continue
            landing = int(apex)
            while landing < last_frame and airborne(landing, self._airborne_body_fraction):
                landing += 1
            # Un "aterrizaje" en los últimos frames es el pie saliendo del cuadro, no un contacto
            if landing >= series.frame_count - self._landing_end_margin_frames:
                continue
            windows.append(MovementWindow(landing, min(landing + window_frames, last_frame)))
        return windows

    @staticmethod
    def _lowest_foot_point(y: np.ndarray) -> np.ndarray:
        left = np.nanmax(y[:, [LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX]], axis=1)
        right = np.nanmax(y[:, [RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX]], axis=1)
        return np.nanmean(np.stack([left, right], axis=1), axis=1)

    def _local_ground(self, foot: np.ndarray, series: PoseSeries) -> np.ndarray:
        half = max(1, series.seconds_to_frames(self._ground_window_seconds))
        return np.array([
            np.nanpercentile(foot[max(0, frame - half): frame + half + 1], self._ground_percentile)
            for frame in range(len(foot))
        ])
