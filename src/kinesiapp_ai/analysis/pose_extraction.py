import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinesiapp_ai.analysis.errors import UnreadableVideoError
from kinesiapp_ai.analysis.person_detection import (
    BoundingBox,
    PersonDetector,
    most_prominent_person,
    select_main_person,
)
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


class MainPersonTracker:
    """Sigue a la persona principal a lo largo de un video y da la región que ve el estimador de pose.

    Una instancia por video, como el estimador. Mientras nadie compite con el atleta devuelve None y
    el estimador ve el frame completo, como siempre. La primera vez que otra persona compite se fija
    a la principal y desde ahí recorta en todos los frames hasta el final, sin volver nunca al frame
    completo: prender y apagar el recorte desajusta el seguimiento interno de MediaPipe para el resto
    del video. Ya fijado, sigue a la caja que más se solapa con la del atleta (no a la más grande),
    sostiene la región cuando el detector lo pierde por desenfoque y la mueve suavizada.
    """

    def __init__(
        self,
        detector: PersonDetector,
        margin: float,
        min_relative_area: float,
        min_iou: float,
        max_missed_frames: int,
        smoothing: float,
    ) -> None:
        self._detector = detector
        self._margin = margin
        self._min_relative_area = min_relative_area
        self._min_iou = min_iou
        self._max_missed_frames = max_missed_frames
        self._smoothing = smoothing
        self._athlete: BoundingBox | None = None
        self._smoothed: np.ndarray | None = None
        self._missed_frames = 0

    def region(self, frame: np.ndarray) -> CropRegion | None:
        height, width = frame.shape[:2]
        boxes = self._detector.detect(frame)
        if self._athlete is None:
            person = select_main_person(boxes, width, height, self._min_relative_area)
            if person is None:
                return None
            self._follow(person)
        else:
            self._track(boxes, width, height)
        return self._crop(width, height)

    def _track(self, boxes: list[BoundingBox], width: int, height: int) -> None:
        match = max(boxes, key=self._athlete.iou, default=None)
        if match is not None and self._athlete.iou(match) >= self._min_iou:
            self._follow(match)
        elif boxes and self._missed_frames >= self._max_missed_frames:
            # Perdido de verdad (se movió demasiado durante el hueco): se vuelve a elegir al principal
            self._follow(most_prominent_person(boxes, width, height))
        else:
            # Desenfoque u oclusión breve: se sostiene la última región conocida
            self._missed_frames += 1

    def _follow(self, box: BoundingBox) -> None:
        self._athlete = box
        self._missed_frames = 0
        coordinates = np.array([box.x1, box.y1, box.x2, box.y2])
        if self._smoothed is None:
            self._smoothed = coordinates
        else:
            self._smoothed = self._smoothing * coordinates + (1 - self._smoothing) * self._smoothed

    def _crop(self, width: int, height: int) -> CropRegion:
        x1, y1, x2, y2 = self._smoothed
        margin_x, margin_y = (x2 - x1) * self._margin, (y2 - y1) * self._margin
        return CropRegion(
            x1=max(0, math.floor(x1 - margin_x)),
            y1=max(0, math.floor(y1 - margin_y)),
            x2=min(width, math.ceil(x2 + margin_x)),
            y2=min(height, math.ceil(y2 + margin_y)),
        )


class VideoPoseExtractor:
    """Corre el estimador de pose sobre cada frame de un video y arma la serie cruda.

    Con `tracker_factory`, el estimador solo ve a la persona principal cuando hay otras en el video.
    """

    def __init__(
        self,
        frame_reader: VideoFrameReader,
        estimator_factory: Callable[[], PoseEstimator],
        tracker_factory: Callable[[], MainPersonTracker] | None = None,
    ) -> None:
        self._frame_reader = frame_reader
        self._estimator_factory = estimator_factory
        self._tracker_factory = tracker_factory

    def extract(self, video_path: Path) -> PoseSeries:
        frames: list[np.ndarray] = []
        with self._frame_reader.open(video_path) as stream:
            estimator = self._estimator_factory()
            tracker = self._tracker_factory() if self._tracker_factory else None
            try:
                for index, frame in enumerate(stream.frames):
                    pose = self._estimate(estimator, tracker, frame, round(index * 1000 / stream.fps))
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

    @staticmethod
    def _estimate(
        estimator: PoseEstimator, tracker: MainPersonTracker | None, frame: np.ndarray, timestamp_ms: int
    ) -> np.ndarray | None:
        region = tracker.region(frame) if tracker else None
        if region is None:
            return estimator.estimate(frame, timestamp_ms)
        pose = estimator.estimate(region.apply(frame), timestamp_ms)
        return None if pose is None else region.to_frame_coordinates(pose, frame.shape[1], frame.shape[0])
