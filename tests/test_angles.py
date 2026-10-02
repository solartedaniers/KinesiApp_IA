"""AngleCalculator: geometría pura sobre landmarks con ángulos conocidos."""
import numpy as np
import pytest

from app.analysis.angles import AngleCalculator
from app.analysis.pose_series import LANDMARK_COUNT, SIDE_LANDMARKS, Side
from tests.pose_fixtures import pose_series

LEFT = SIDE_LANDMARKS[Side.LEFT]


def _single_frame(**coordinates: tuple[float, float]) -> np.ndarray:
    points = np.full((1, LANDMARK_COUNT, 2), 0.5)
    for name, xy in coordinates.items():
        points[0, getattr(LEFT, name)] = xy
    return points


def test_straight_leg_and_upright_trunk_are_zero():
    points = _single_frame(shoulder=(0.5, 0.2), hip=(0.5, 0.5), knee=(0.5, 0.7), ankle=(0.5, 0.9))
    angles = AngleCalculator().calculate(pose_series(points))
    assert angles.knee_flexion[0] == pytest.approx(0, abs=1e-6)
    assert angles.trunk_inclination[0] == pytest.approx(0, abs=1e-6)


def test_right_angle_knee_and_forward_trunk():
    points = _single_frame(shoulder=(0.7, 0.3), hip=(0.5, 0.5), knee=(0.7, 0.5), ankle=(0.7, 0.7))
    angles = AngleCalculator().calculate(pose_series(points))
    assert angles.knee_flexion[0] == pytest.approx(90)
    assert angles.trunk_inclination[0] == pytest.approx(45)


def test_shoulders_below_hip_exceed_ninety_degrees():
    points = _single_frame(shoulder=(0.8, 0.6), hip=(0.5, 0.5), knee=(0.5, 0.7), ankle=(0.5, 0.9))
    assert AngleCalculator().calculate(pose_series(points)).trunk_inclination[0] > 90


def test_aspect_ratio_corrects_portrait_videos():
    # En un video vertical (ancho/alto = 0.5) 0.2 normalizado en x equivale a 0.1 en y
    points = _single_frame(shoulder=(0.7, 0.4), hip=(0.5, 0.5), knee=(0.5, 0.7), ankle=(0.5, 0.9))
    angles = AngleCalculator().calculate(pose_series(points, aspect_ratio=0.5))
    assert angles.trunk_inclination[0] == pytest.approx(45)


def test_uses_the_side_most_visible_to_the_camera():
    series = pose_series(_single_frame())
    series.visibility[:, [LEFT.hip, LEFT.knee, LEFT.ankle]] = 0.2
    assert AngleCalculator().calculate(series).side == Side.RIGHT


def test_unknown_landmarks_give_nan_instead_of_an_angle():
    points = _single_frame()
    points[0, LEFT.knee] = np.nan
    assert np.isnan(AngleCalculator().calculate(pose_series(points)).knee_flexion[0])
