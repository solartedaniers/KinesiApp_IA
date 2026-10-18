import math


class FrameSampler:
    """Proceso uno de cada `stride` frames para no pasar de `max_fps`.

    Uso un paso entero y no una tasa exacta para que la serie siga con frames equiespaciados: todo el
    análisis de fases asume un fps constante.
    """

    def __init__(self, max_fps: float | None) -> None:
        self._max_fps = max_fps

    def stride(self, source_fps: float) -> int:
        if self._max_fps is None or source_fps <= self._max_fps:
            return 1
        return math.ceil(source_fps / self._max_fps)

    def sampled_count(self, frame_count: int, source_fps: float) -> int:
        return math.ceil(frame_count / self.stride(source_fps))
