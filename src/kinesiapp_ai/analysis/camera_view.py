import enum
import math

import numpy as np

from kinesiapp_ai.analysis.pose_series import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER, PoseSeries


class CameraView(str, enum.Enum):
    SIDE = "side"
    FRONT = "front"
    INCONCLUSIVE = "inconclusive"


class CameraViewClassifier:
    """Decido desde dónde grabó la cámara con el ancho aparente del cuerpo.

    De frente, hombros y caderas se ven separados (el ancho se acerca al alto del torso); de lado, uno
    tapa al otro y el ancho cae casi a cero. Uso la mediana de todo el video para que un giro breve o
    la inclinación del tronco en una repetición no cambien la vista.
    """

    def __init__(self, side_max_width_ratio: float, front_min_width_ratio: float) -> None:
        if side_max_width_ratio >= front_min_width_ratio:
            raise ValueError("side_max_width_ratio must be lower than front_min_width_ratio")
        self._side_max_width_ratio = side_max_width_ratio
        self._front_min_width_ratio = front_min_width_ratio

    def classify(self, series: PoseSeries) -> CameraView:
        ratio = self.width_ratio(series)
        if math.isnan(ratio):
            return CameraView.INCONCLUSIVE
        if ratio <= self._side_max_width_ratio:
            return CameraView.SIDE
        if ratio >= self._front_min_width_ratio:
            return CameraView.FRONT
        return CameraView.INCONCLUSIVE

    @staticmethod
    def width_ratio(series: PoseSeries) -> float:
        """Promedio del ancho de hombros y de caderas sobre el alto del torso, en la mediana de los frames."""
        points = series.points * np.array([series.aspect_ratio, 1.0])
        shoulders = np.abs(points[:, LEFT_SHOULDER, 0] - points[:, RIGHT_SHOULDER, 0])
        hips = np.abs(points[:, LEFT_HIP, 0] - points[:, RIGHT_HIP, 0])
        torso = np.linalg.norm(
            (points[:, LEFT_SHOULDER] + points[:, RIGHT_SHOULDER]) / 2 - (points[:, LEFT_HIP] + points[:, RIGHT_HIP]) / 2,
            axis=1,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = (shoulders + hips) / (2 * torso)
        ratios = ratios[np.isfinite(ratios)]
        return float(np.median(ratios)) if len(ratios) else math.nan
