import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from kinesiapp_ai.analysis.errors import UnreadableVideoError, VideoTooLongError


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int

    @property
    def duration_ms(self) -> int:
        return round(self.frame_count * 1000 / self.fps)


class VideoDurationGuard:
    """Verifico que el video no pase la duración máxima antes de gastar CPU en analizarlo."""

    def __init__(self, max_duration_seconds: float) -> None:
        self._max_duration_seconds = max_duration_seconds

    def check(self, video_path: Path) -> VideoMetadata:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise UnreadableVideoError(f"Cannot open video {video_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or math.isnan(fps):
                raise UnreadableVideoError(f"Video {video_path} has no frame rate")
            max_frames = math.floor(self._max_duration_seconds * fps)
            declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            # Los WebM de MediaRecorder no traen la cantidad de frames: en ese caso los cuento sin decodificar
            frame_count = declared if declared > 0 else self._count_frames(capture, max_frames + 1)
        finally:
            capture.release()
        if frame_count == 0:
            raise UnreadableVideoError(f"Video {video_path} has no frames")
        if frame_count > max_frames:
            raise VideoTooLongError(f"Video {video_path} lasts more than {self._max_duration_seconds} s")
        return VideoMetadata(fps=fps, frame_count=frame_count)

    @staticmethod
    def _count_frames(capture: cv2.VideoCapture, limit: int) -> int:
        # Dejo de contar al pasar el límite: no necesito saber cuánto más largo es
        count = 0
        while count < limit and capture.grab():
            count += 1
        return count
