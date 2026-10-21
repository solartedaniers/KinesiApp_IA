"""AngleCalculator: geometría pura sobre landmarks con ángulos conocidos."""
import numpy as np
import pytest

from kinesiapp_ai.analysis.angles import AngleCalculator
from kinesiapp_ai.analysis.pose_series import LANDMARK_COUNT, SIDE_LANDMARKS, Side
from tests.pose_fixtures import bend_knee_inward, frontal_points, pose_series

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


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("fppa", [20.0, -15.0, 0.0])
def test_the_frontal_projection_angle_is_positive_when_the_knee_goes_inward(side: Side, fppa: float):
    points = frontal_points(1)
    bend_knee_inward(points, slice(None), side, fppa)

    valgus = AngleCalculator().calculate(pose_series(points)).knee_valgus

    assert valgus[side][0] == pytest.approx(fppa, abs=1e-6)
    assert valgus[next(other for other in Side if other is not side)][0] == pytest.approx(0, abs=1e-6)


def test_the_valgus_sign_does_not_depend_on_facing_the_camera():
    # De espaldas, la izquierda del deportista queda a la izquierda de la imagen: hacia adentro sigue siendo positivo
    points = frontal_points(1)
    points[:, :, 0] = 1.0 - points[:, :, 0]
    bend_knee_inward(points, slice(None), Side.LEFT, 18.0)

    assert AngleCalculator().calculate(pose_series(points)).knee_valgus[Side.LEFT][0] == pytest.approx(18.0)


def test_the_valgus_of_an_unknown_knee_is_not_measured():
    points = frontal_points(1)
    points[0, SIDE_LANDMARKS[Side.RIGHT].knee] = np.nan

    assert np.isnan(AngleCalculator().calculate(pose_series(points)).knee_valgus[Side.RIGHT][0])
