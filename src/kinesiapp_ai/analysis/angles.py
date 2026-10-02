from dataclasses import dataclass

import numpy as np

from app.analysis.pose_series import SIDE_LANDMARKS, PoseSeries, Side


@dataclass(frozen=True)
class AngleSeries:
    """Ángulos por frame, en grados, del lado del cuerpo más visible para la cámara.

    knee_flexion: 0 = pierna extendida. trunk_inclination: recta cadera→hombro respecto de la
    vertical; 0 = erguido, más de 90 = hombros por debajo de la cadera.
    """

    side: Side
    knee_flexion: np.ndarray
    trunk_inclination: np.ndarray


class AngleCalculator:
    """Geometría 2D pura sobre landmarks ya preprocesados, sin ML."""

    def calculate(self, series: PoseSeries) -> AngleSeries:
        side = self._most_visible_side(series)
        landmarks = SIDE_LANDMARKS[side]
        # x se escala por el aspecto para que los ángulos no se deformen en videos verticales
        points = series.points * np.array([series.aspect_ratio, 1.0])
        hip, knee, ankle, shoulder = (
            points[:, index] for index in (landmarks.hip, landmarks.knee, landmarks.ankle, landmarks.shoulder)
        )
        return AngleSeries(
            side=side,
            knee_flexion=180.0 - self._interior_angle(hip, knee, ankle),
            trunk_inclination=self._angle_from_vertical(shoulder - hip),
        )

    @staticmethod
    def _most_visible_side(series: PoseSeries) -> Side:
        def mean_visibility(side: Side) -> float:
            landmarks = SIDE_LANDMARKS[side]
            return float(np.nanmean(series.visibility[:, [landmarks.hip, landmarks.knee, landmarks.ankle]]))

        return max(Side, key=mean_visibility)

    @staticmethod
    def _interior_angle(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> np.ndarray:
        first, second = a - vertex, c - vertex
        norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.einsum("ij,ij->i", first, second) / norms
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def _angle_from_vertical(vectors: np.ndarray) -> np.ndarray:
        # La vertical "hacia arriba" en coordenadas de imagen es (0, -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = -vectors[:, 1] / np.linalg.norm(vectors, axis=1)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
