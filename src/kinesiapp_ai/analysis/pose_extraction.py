import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinesiapp_ai.analysis.errors import UnreadableVideoError
from kinesiapp_ai.analysis.person_detection import PersonDetector, select_main_person
from kinesiapp_ai.analysis.pose_estimator import PoseEstimator
from kinesiapp_ai.analysis.pose_series import LANDMARK_COUNT, PoseSeries
from kinesiapp_ai.analysis.video_reader import VideoFrameReader


@dataclass(frozen=True)
class CropRegion:
    """Rectángulo del frame en píxeles enteros; x2 e y2 excluidos, como en un slice de numpy."""

    x1: int
    y1: int
    x2: int
    y2: int

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1 : self.y2, self.x1 : self.x2]

    def to_frame_coordinates(self, pose: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        """Pasa una pose (33, 3) normalizada al recorte a normalizada al frame completo, que es lo que
        espera todo el análisis de ángulos. La visibilidad no cambia."""
        remapped = pose.copy()
        remapped[:, 0] = (pose[:, 0] * (self.x2 - self.x1) + self.x1) / frame_width
        remapped[:, 1] = (pose[:, 1] * (self.y2 - self.y1) + self.y1) / frame_height
        return remapped


class MainPersonCropper:
    """Cuando hay más de una persona en el frame, el recorte a la principal con un margen alrededor."""

    def __init__(self, detector: PersonDetector, margin: float, min_relative_area: float) -> None:
        self._detector = detector
        self._margin = margin
        self._min_relative_area = min_relative_area

    def region(self, frame: np.ndarray) -> CropRegion | None:
        """None si no hace falta recortar (una persona o ninguna): se usa el frame completo."""
        height, width = frame.shape[:2]
        person = select_main_person(self._detector.detect(frame), width, height, self._min_relative_area)
        if person is None:
            return None
        margin_x = (person.x2 - person.x1) * self._margin
        margin_y = (person.y2 - person.y1) * self._margin
        return CropRegion(
            x1=max(0, math.floor(person.x1 - margin_x)),
            y1=max(0, math.floor(person.y1 - margin_y)),
            x2=min(width, math.ceil(person.x2 + margin_x)),
            y2=min(height, math.ceil(person.y2 + margin_y)),
        )


class VideoPoseExtractor:
    """Corre el estimador de pose sobre cada frame de un video y arma la serie cruda.

    Con `cropper`, el estimador solo ve a la persona principal cuando hay otras en el frame.
    """

    def __init__(
        self,
        frame_reader: VideoFrameReader,
        estimator_factory: Callable[[], PoseEstimator],
        cropper: MainPersonCropper | None = None,
    ) -> None:
        self._frame_reader = frame_reader
        self._estimator_factory = estimator_factory
        self._cropper = cropper

    def extract(self, video_path: Path) -> PoseSeries:
        frames: list[np.ndarray] = []
        with self._frame_reader.open(video_path) as stream:
            estimator = self._estimator_factory()
            try:
                for index, frame in enumerate(stream.frames):
                    pose = self._estimate(estimator, frame, round(index * 1000 / stream.fps))
                    frames.append(pose if pose is not None else np.full((LANDMARK_COUNT, 3), np.nan))
            finally:
                estimator.close()
        if not frames:
            raise UnreadableVideoError(f"Video {video_path} has no frames")
        landmarks = np.stack(frames)
        return PoseSeries(
            points=landmarks[:, :, :2],
            visibility=landmarks[:, :, 2],
            fps=stream.fps,
            aspect_ratio=stream.width / stream.height,
        )

    def _estimate(self, estimator: PoseEstimator, frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        region = self._cropper.region(frame) if self._cropper else None
        if region is None:
            return estimator.estimate(frame, timestamp_ms)
        pose = estimator.estimate(region.apply(frame), timestamp_ms)
        return None if pose is None else region.to_frame_coordinates(pose, frame.shape[1], frame.shape[0])
