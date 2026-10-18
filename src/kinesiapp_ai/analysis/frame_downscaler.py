import cv2
import numpy as np


class FrameDownscaler:
    """Reduzco cada frame hasta que su lado mayor mida `max_side`, conservando la proporción.

    Lo hago antes de YOLO y MediaPipe para bajar memoria y tiempo con videos de alta resolución. Los
    landmarks salen normalizados, así que los ángulos no cambian de escala.
    """

    def __init__(self, max_side: int | None) -> None:
        self._max_side = max_side

    def apply(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if self._max_side is None or max(height, width) <= self._max_side:
            return frame
        scale = self._max_side / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        # INTER_AREA es el que mejor conserva los bordes al reducir; los otros dejan aliasing
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
