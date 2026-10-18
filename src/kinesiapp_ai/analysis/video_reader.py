from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kinesiapp_ai.analysis.errors import UnreadableVideoError
from kinesiapp_ai.analysis.frame_downscaler import FrameDownscaler


@dataclass(frozen=True)
class VideoStream:
    fps: float
    width: int
    height: int
    frames: Iterator[np.ndarray]


class VideoFrameReader:
    """Decodifica un video a frames BGR, en orden, sin guardarlos todos en memoria."""

    def __init__(self, downscaler: FrameDownscaler | None = None) -> None:
        self._downscaler = downscaler or FrameDownscaler(None)

    @contextmanager
    def open(self, video_path: Path) -> Iterator[VideoStream]:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise UnreadableVideoError(f"Cannot open video {video_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise UnreadableVideoError(f"Video {video_path} has no frame rate")
            yield VideoStream(
                fps=fps,
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frames=self._frames(capture),
            )
        finally:
            capture.release()

    def _frames(self, capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield self._downscaler.apply(frame)
