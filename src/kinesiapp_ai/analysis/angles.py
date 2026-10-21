from dataclasses import dataclass

import numpy as np

from kinesiapp_ai.analysis.pose_series import LEFT_HIP, RIGHT_HIP, SIDE_LANDMARKS, PoseSeries, Side


@dataclass(frozen=True)
class AngleSeries:
    """Ángulos por frame, en grados, del lado del cuerpo más visible para la cámara.

    knee_flexion: 0 = pierna extendida. trunk_inclination: recta cadera→hombro respecto de la
    vertical; 0 = erguido, más de 90 = hombros por debajo de la cadera.

    knee_valgus: ángulo de proyección frontal (FPPA) de cada pierna, 0 = muslo y pierna alineados,
    positivo = rodilla hacia adentro. Solo tiene sentido con la cámara de frente.
    """

    side: Side
    knee_flexion: np.ndarray
    trunk_inclination: np.ndarray
    knee_valgus: dict[Side, np.ndarray]


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
            knee_valgus={leg: self._frontal_projection_angle(points, leg) for leg in Side},
        )

    @classmethod
    def _frontal_projection_angle(cls, points: np.ndarray, side: Side) -> np.ndarray:
        landmarks = SIDE_LANDMARKS[side]
        hip, knee, ankle = (points[:, index] for index in (landmarks.hip, landmarks.knee, landmarks.ankle))
        deviation = 180.0 - cls._interior_angle(hip, knee, ankle)
        # El signo lo saco de la geometría y no de la etiqueta del lado: así sirve igual si el deportista da
        # la espalda a la cámara. Hacia afuera es alejarse del centro de las caderas
        outward = np.sign(hip[:, 0] - (points[:, LEFT_HIP, 0] + points[:, RIGHT_HIP, 0]) / 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            along = (knee[:, 1] - hip[:, 1]) / (ankle[:, 1] - hip[:, 1])
        line_x = hip[:, 0] + along * (ankle[:, 0] - hip[:, 0])
        inward = (knee[:, 0] - line_x) * outward < 0
        return np.where(inward, deviation, -deviation)

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
