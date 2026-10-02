import numpy as np

from app.analysis.pose_series import PoseSeries


class LandmarkSeriesPreprocessor:
    """Limpia la serie cruda antes de medir ángulos o detectar fases.

    Descarta un landmark en un frame si su visibilidad es baja O si cae fuera del cuadro
    (coordenada normalizada fuera de [0, 1]): MediaPipe extrapola los landmarks fuera de cuadro y
    los sigue reportando visibles. Después rellena los huecos por interpolación lineal y suaviza
    cada coordenada con una media móvil centrada.
    """

    def __init__(self, visibility_threshold: float, smoothing_window_frames: int) -> None:
        if smoothing_window_frames < 1 or smoothing_window_frames % 2 == 0:
            raise ValueError("smoothing_window_frames must be a positive odd number")
        self._visibility_threshold = visibility_threshold
        self._smoothing_window_frames = smoothing_window_frames

    def process(self, raw: PoseSeries) -> PoseSeries:
        points = raw.points.copy()
        points[self._discard_mask(raw)] = np.nan
        frames, landmarks, axes = points.shape
        columns = points.reshape(frames, landmarks * axes)
        for column in range(columns.shape[1]):
            columns[:, column] = self._smooth(self._fill_gaps(columns[:, column]))
        return PoseSeries(
            points=columns.reshape(frames, landmarks, axes),
            visibility=raw.visibility,
            fps=raw.fps,
            aspect_ratio=raw.aspect_ratio,
        )

    def _discard_mask(self, raw: PoseSeries) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            out_of_frame = ((raw.points < 0) | (raw.points > 1)).any(axis=2)
            low_visibility = ~(raw.visibility >= self._visibility_threshold)
        return out_of_frame | low_visibility

    @staticmethod
    def _fill_gaps(values: np.ndarray) -> np.ndarray:
        known = ~np.isnan(values)
        if known.all() or not known.any():
            return values
        indices = np.arange(len(values))
        return np.interp(indices, indices[known], values[known])

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        half = self._smoothing_window_frames // 2
        if half == 0 or np.isnan(values).all():
            return values
        padded = np.pad(values, half, mode="edge")
        kernel = np.full(self._smoothing_window_frames, 1 / self._smoothing_window_frames)
        return np.convolve(padded, kernel, mode="valid")
