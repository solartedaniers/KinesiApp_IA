"""LandmarkSeriesPreprocessor: doble criterio de descarte, relleno de huecos y suavizado."""
import numpy as np
import pytest

from kinesiapp_ai.analysis.pose_series import LEFT_ANKLE
from kinesiapp_ai.analysis.preprocessing import LandmarkSeriesPreprocessor
from tests.pose_fixtures import pose_series, standing_points


def _ankle_y(series):
    return series.points[:, LEFT_ANKLE, 1]


def test_out_of_frame_landmark_is_discarded_even_if_reported_visible():
    points = standing_points(5)
    points[2, LEFT_ANKLE, 1] = 1.2
    cleaned = LandmarkSeriesPreprocessor(visibility_threshold=0.5, smoothing_window_frames=1).process(
        pose_series(points, visibility=0.99)
    )
    # Se reemplaza por la interpolación de sus vecinos, no por el valor extrapolado
    assert _ankle_y(cleaned)[2] == pytest.approx(0.9)


def test_low_visibility_landmark_is_discarded():
    points = standing_points(5)
    points[2, LEFT_ANKLE, 1] = 0.5
    raw = pose_series(points)
    raw.visibility[2, LEFT_ANKLE] = 0.1
    cleaned = LandmarkSeriesPreprocessor(visibility_threshold=0.5, smoothing_window_frames=1).process(raw)
    assert _ankle_y(cleaned)[2] == pytest.approx(0.9)


def test_frames_without_pose_are_interpolated():
    points = standing_points(5)
    points[:, LEFT_ANKLE, 1] = [0.5, 0.6, np.nan, 0.8, 0.9]
    cleaned = LandmarkSeriesPreprocessor(0.5, 1).process(pose_series(points))
    assert _ankle_y(cleaned) == pytest.approx([0.5, 0.6, 0.7, 0.8, 0.9])


def test_smoothing_is_a_centered_moving_average_that_keeps_the_length():
    points = standing_points(5)
    points[:, LEFT_ANKLE, 1] = [0.0, 0.0, 0.9, 0.0, 0.0]
    cleaned = LandmarkSeriesPreprocessor(0.5, 3).process(pose_series(points))
    assert _ankle_y(cleaned) == pytest.approx([0.0, 0.3, 0.3, 0.3, 0.0])


def test_rejects_even_smoothing_windows():
    with pytest.raises(ValueError):
        LandmarkSeriesPreprocessor(0.5, 4)
