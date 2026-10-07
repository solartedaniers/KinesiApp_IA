"""Selección y recorte de la persona principal, con cajas y estimador falsos (sin YOLO ni MediaPipe)."""
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from kinesiapp_ai.analysis.person_detection import BoundingBox, select_main_person
from kinesiapp_ai.analysis.pose_extraction import MainPersonCropper, VideoPoseExtractor
from kinesiapp_ai.analysis.pose_series import LANDMARK_COUNT
from kinesiapp_ai.analysis.video_reader import VideoStream

WIDTH, HEIGHT = 640, 480
MARGIN = 0.25
MIN_RELATIVE_AREA = 0.25
ATHLETE = BoundingBox(240, 100, 400, 460)


class _FakeDetector:
    def __init__(self, boxes: list[BoundingBox]) -> None:
        self._boxes = boxes

    def detect(self, frame_bgr: np.ndarray) -> list[BoundingBox]:
        return self._boxes


class _FakeReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    @contextmanager
    def open(self, video_path: Path) -> Iterator[VideoStream]:
        height, width = self._frames[0].shape[:2]
        yield VideoStream(fps=30.0, width=width, height=height, frames=iter(self._frames))


class _RecordingEstimator:
    """Devuelve siempre la misma pose y guarda los frames que recibió."""

    POSE = np.column_stack([np.linspace(0.1, 0.9, LANDMARK_COUNT), np.linspace(0.9, 0.1, LANDMARK_COUNT), np.ones(LANDMARK_COUNT)])

    def __init__(self) -> None:
        self.received: list[np.ndarray] = []

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        self.received.append(frame_bgr)
        return self.POSE.copy()

    def close(self) -> None:
        pass


def _select(*boxes: BoundingBox) -> BoundingBox | None:
    return select_main_person(list(boxes), WIDTH, HEIGHT, MIN_RELATIVE_AREA)


def _extract(boxes: list[BoundingBox], frame: np.ndarray) -> tuple[_RecordingEstimator, np.ndarray]:
    estimator = _RecordingEstimator()
    cropper = MainPersonCropper(_FakeDetector(boxes), MARGIN, MIN_RELATIVE_AREA)
    series = VideoPoseExtractor(_FakeReader([frame]), lambda: estimator, cropper).extract(Path("video.mp4"))
    return estimator, series.points[0]


def test_the_largest_and_most_centered_person_is_the_main_one():
    passerby_at_the_edge = BoundingBox(0, 60, 170, 470)
    small_in_the_center = BoundingBox(280, 200, 360, 380)
    assert _select(passerby_at_the_edge, ATHLETE, small_in_the_center) == ATHLETE


def test_between_two_people_of_the_same_size_the_centered_one_wins():
    same_size_at_the_edge = BoundingBox(0, 100, 160, 460)
    assert _select(same_size_at_the_edge, ATHLETE) == ATHLETE


def test_no_choice_is_needed_with_one_person_or_none():
    assert _select(ATHLETE) is None
    assert _select() is None


def test_a_small_reflection_does_not_compete_with_the_athlete():
    mirror_reflection = BoundingBox(80, 260, 160, 380)
    assert _select(ATHLETE, mirror_reflection) is None


def test_with_one_person_the_estimator_gets_the_full_frame_untouched():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    estimator, points = _extract([ATHLETE], frame)
    assert estimator.received[0] is frame
    np.testing.assert_array_equal(points, _RecordingEstimator.POSE[:, :2])


def test_with_several_people_the_estimator_only_gets_the_main_one_plus_margin():
    passerby = BoundingBox(0, 60, 170, 470)
    estimator, _ = _extract([passerby, ATHLETE], np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    # 160×360 más 25 % por lado: x de 200 a 440, y de 10 a 550 recortado al alto del frame
    assert estimator.received[0].shape == (HEIGHT - 10, 440 - 200, 3)
