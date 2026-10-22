"""Pruebo la clasificación de la vista de la cámara con poses sintéticas de ancho conocido."""

import numpy as np
import pytest

from kinesiapp_ai.analysis.camera_view import CameraView, CameraViewClassifier
from kinesiapp_ai.analysis.pose_series import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from tests.pose_fixtures import VIEW_CLASSIFIER, frontal_points, pose_series, standing_points


def _turned(fraction: float) -> np.ndarray:
    """La pose frontal girada: el ancho aparente de hombros y caderas se reduce a esta fracción."""
    points = frontal_points(30)
    for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        points[:, index, 0] = 0.5 + (points[:, index, 0] - 0.5) * fraction
    return points


def test_a_body_seen_from_the_side_is_side() -> None:
    assert VIEW_CLASSIFIER.classify(pose_series(standing_points(30))) is CameraView.SIDE


def test_a_body_seen_from_the_front_is_front() -> None:
    series = pose_series(frontal_points(30))

    assert CameraViewClassifier.width_ratio(series) == pytest.approx(0.76)
    assert VIEW_CLASSIFIER.classify(series) is CameraView.FRONT


def test_a_diagonal_view_is_inconclusive() -> None:
    # A unos 60° del frente el ancho cae a la mitad y queda entre los dos cortes
    assert VIEW_CLASSIFIER.classify(pose_series(_turned(0.5))) is CameraView.INCONCLUSIVE


def test_a_slightly_turned_body_keeps_its_view() -> None:
    assert VIEW_CLASSIFIER.classify(pose_series(_turned(0.85))) is CameraView.FRONT
    assert VIEW_CLASSIFIER.classify(pose_series(_turned(0.2))) is CameraView.SIDE


def test_the_median_ignores_a_brief_turn() -> None:
    points = standing_points(30)
    points[:10] = frontal_points(10)

    assert VIEW_CLASSIFIER.classify(pose_series(points)) is CameraView.SIDE


def test_the_portrait_aspect_ratio_narrows_the_apparent_width() -> None:
    # En un video vertical el ancho normalizado vale menos en píxeles: sin corregirlo, de lado pasaría por frente
    series = pose_series(_turned(0.5), aspect_ratio=0.5)

    assert CameraViewClassifier.width_ratio(series) == pytest.approx(0.19)
    assert VIEW_CLASSIFIER.classify(series) is CameraView.SIDE


def test_without_a_torso_the_view_is_inconclusive() -> None:
    points = frontal_points(30)
    points[:, [LEFT_SHOULDER, RIGHT_SHOULDER]] = np.nan

    assert VIEW_CLASSIFIER.classify(pose_series(points)) is CameraView.INCONCLUSIVE


def test_the_side_cut_must_be_below_the_front_cut() -> None:
    with pytest.raises(ValueError):
        CameraViewClassifier(side_max_width_ratio=0.5, front_min_width_ratio=0.4)
