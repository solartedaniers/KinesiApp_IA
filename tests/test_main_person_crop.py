"""Selección, seguimiento y recorte de la persona principal, con cajas y estimador falsos (sin YOLO ni MediaPipe)."""
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np

from kinesiapp_ai.analysis.angles import AngleCalculator, AngleSeries
from kinesiapp_ai.analysis.person_detection import BoundingBox, select_main_person
from kinesiapp_ai.analysis.pose_extraction import CropRegion, MainPersonTracker, VideoPoseExtractor
from kinesiapp_ai.analysis.pose_series import LANDMARK_COUNT, LEFT_KNEE, RIGHT_KNEE
from kinesiapp_ai.analysis.video_reader import VideoStream
from tests.pose_fixtures import add_hip_hinge, standing_points

WIDTH, HEIGHT = 640, 480
MARGIN = 0.25
MIN_RELATIVE_AREA = 0.25
MIN_IOU = 0.2
MAX_MISSED_FRAMES = 3
SMOOTHING = 0.5
ATHLETE = BoundingBox(240, 100, 400, 460)


class _FakeDetector:
    def __init__(self, boxes: list[BoundingBox]) -> None:
        self._boxes = boxes

    def detect(self, frame_bgr: np.ndarray) -> list[BoundingBox]:
        return self._boxes


class _ScriptedDetector:
    """Devuelve las cajas de cada frame en orden, como si fueran de frames consecutivos de un video."""

    def __init__(self, boxes_per_frame: list[list[BoundingBox]]) -> None:
        self._boxes = iter(boxes_per_frame)

    def detect(self, frame_bgr: np.ndarray) -> list[BoundingBox]:
        return next(self._boxes)


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


class _PixelPoseEstimator:
    """Estimador que "ve" la pose en los píxeles: el canal i marca el píxel del landmark i. Como un
    modelo real, devuelve coordenadas normalizadas a la imagen que recibe, sea el frame o un recorte."""

    def __init__(self) -> None:
        self.received_shapes: list[tuple[int, ...]] = []

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        self.received_shapes.append(frame_bgr.shape)
        height, width = frame_bgr.shape[:2]
        pose = np.ones((LANDMARK_COUNT, 3))
        for index in range(LANDMARK_COUNT):
            row, column = np.argwhere(frame_bgr[:, :, index])[0]
            pose[index, :2] = (column + 0.5) / width, (row + 0.5) / height
        return pose

    def close(self) -> None:
        pass


def _render(points: np.ndarray) -> list[np.ndarray]:
    """Un frame por pose, con un canal por landmark marcado en su píxel."""
    frames = []
    for pose in points:
        frame = np.zeros((HEIGHT, WIDTH, LANDMARK_COUNT), dtype=np.uint8)
        for index, (x, y) in enumerate(pose):
            frame[int(y * HEIGHT), int(x * WIDTH), index] = 1
        frames.append(frame)
    return frames


def _tracker(detector) -> MainPersonTracker:
    return MainPersonTracker(detector, MARGIN, MIN_RELATIVE_AREA, MIN_IOU, MAX_MISSED_FRAMES, SMOOTHING)


def _regions(*boxes_per_frame: list[BoundingBox]) -> list[CropRegion | None]:
    tracker = _tracker(_ScriptedDetector(list(boxes_per_frame)))
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    return [tracker.region(frame) for _ in boxes_per_frame]


def _shifted(box: BoundingBox, dx: float) -> BoundingBox:
    return BoundingBox(box.x1 + dx, box.y1, box.x2 + dx, box.y2)


def _select(*boxes: BoundingBox) -> BoundingBox | None:
    return select_main_person(list(boxes), WIDTH, HEIGHT, MIN_RELATIVE_AREA)


def _extract(boxes: list[BoundingBox], frame: np.ndarray) -> tuple[_RecordingEstimator, np.ndarray]:
    estimator = _RecordingEstimator()
    tracker = partial(_tracker, _FakeDetector(boxes))
    series = VideoPoseExtractor(_FakeReader([frame]), lambda: estimator, tracker).extract(Path("video.mp4"))
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


def test_angles_of_a_cropped_person_match_the_uncropped_ones():
    points = standing_points(6)
    add_hip_hinge(points, 3, 6)
    for index in (LEFT_KNEE, RIGHT_KNEE):
        points[:, index, 0] += np.linspace(0, 0.08, 6)
    frames = _render(points)
    # La caja del atleta rodea sus landmarks (x de 0.5 a 0.75, y de 0.4 a 0.9); la del que pasa compite
    detector = _FakeDetector([BoundingBox(300, 180, 500, 440), BoundingBox(20, 100, 180, 420)])

    def angles(tracker_factory) -> tuple[AngleSeries, _PixelPoseEstimator]:
        estimator = _PixelPoseEstimator()
        series = VideoPoseExtractor(_FakeReader(frames), lambda: estimator, tracker_factory).extract(Path("video.mp4"))
        return AngleCalculator().calculate(series), estimator

    full, _ = angles(None)
    cropped, estimator = angles(partial(_tracker, detector))

    assert all(shape[:2] != (HEIGHT, WIDTH) for shape in estimator.received_shapes)
    assert np.ptp(full.knee_flexion) > 1 and np.ptp(full.trunk_inclination) > 1
    assert cropped.side == full.side
    np.testing.assert_allclose(cropped.knee_flexion, full.knee_flexion, atol=1e-9)
    np.testing.assert_allclose(cropped.trunk_inclination, full.trunk_inclination, atol=1e-9)


PASSERBY = BoundingBox(0, 60, 170, 470)


def test_without_competition_the_video_is_never_cropped():
    assert _regions([ATHLETE], [], [ATHLETE], [ATHLETE]) == [None] * 4


def test_the_region_is_held_while_the_detector_loses_the_athlete():
    regions = _regions([PASSERBY, ATHLETE], [ATHLETE], [], [], [])
    assert None not in regions
    assert regions[1] == regions[2] == regions[3] == regions[4]


def test_once_cropping_starts_it_never_goes_back_to_the_full_frame():
    regions = _regions([PASSERBY, ATHLETE], [ATHLETE], [], [], [], [], [], [ATHLETE], [])
    assert None not in regions


def test_the_lock_follows_the_overlapping_box_and_not_a_larger_one():
    larger_passerby = BoundingBox(0, 0, 230, 480)
    moved_athlete = _shifted(ATHLETE, 20)
    regions = _regions([PASSERBY, ATHLETE], [larger_passerby, moved_athlete], [larger_passerby, moved_athlete])
    assert all(region.x1 > larger_passerby.x2 - 100 for region in regions)
    assert regions[2].x1 > regions[1].x1 > regions[0].x1


def test_the_region_moves_smoothly_toward_the_athlete():
    regions = _regions([PASSERBY, ATHLETE], [_shifted(ATHLETE, 40)])
    # Con suavizado 0.5 la región se mueve la mitad de lo que se movió la caja
    assert regions[1].x1 - regions[0].x1 == 20


def test_a_box_that_barely_touches_the_athlete_does_not_steal_the_lock():
    touching_passerby = BoundingBox(100, 380, 260, 480)
    assert ATHLETE.iou(touching_passerby) < MIN_IOU
    regions = _regions([PASSERBY, ATHLETE], [touching_passerby], [touching_passerby])
    assert regions[0] == regions[1] == regions[2]


def test_after_losing_the_athlete_for_good_the_main_person_is_chosen_again():
    elsewhere = BoundingBox(20, 50, 180, 410)
    regions = _regions([PASSERBY, ATHLETE], *[[]] * MAX_MISSED_FRAMES, [elsewhere])
    assert regions[-1] != regions[-2]
    assert regions[-1].x1 < regions[-2].x1


def test_someone_else_overlapping_the_held_region_does_not_steal_the_lock():
    # Mismo tamaño que el atleta y a su izquierda; mientras el detector no ve al atleta, se le
    # sigue acercando: se solapa con la región sostenida más que el umbral, pero sigue siendo él
    passerby = BoundingBox(150, 100, 310, 460)
    approaching = _shifted(passerby, 10)
    assert ATHLETE.iou(approaching) >= MIN_IOU
    assert approaching.iou(passerby) > approaching.iou(ATHLETE)
    regions = _regions([passerby, ATHLETE], [approaching], [_shifted(approaching, 10)])
    assert regions[0] == regions[1] == regions[2]
