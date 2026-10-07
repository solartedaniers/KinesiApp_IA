"""Series de pose sintéticas con geometría conocida, para probar el dominio sin MediaPipe."""
import numpy as np

from kinesiapp_ai.analysis.angles import AngleSeries
from kinesiapp_ai.analysis.pose_series import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    PoseSeries,
    Side,
)

FPS = 30.0
BODY_HEIGHT = 0.5
FEET = (LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX)


def pose_series(points: np.ndarray, visibility: float = 1.0, aspect_ratio: float = 1.0) -> PoseSeries:
    return PoseSeries(
        points=points,
        visibility=np.full(points.shape[:2], visibility),
        fps=FPS,
        aspect_ratio=aspect_ratio,
    )


def standing_points(frames: int) -> np.ndarray:
    """Persona quieta y de pie: pies en y=0.9, rodillas, cadera media altura más arriba, hombros en
    la cima. Todo alineado en x=0.5: piernas rectas y tronco vertical."""
    points = np.full((frames, LANDMARK_COUNT, 2), 0.5)
    for index in FEET:
        points[:, index, 1] = 0.9
    for index in (LEFT_KNEE, RIGHT_KNEE):
        points[:, index, 1] = 0.9 - BODY_HEIGHT / 4
    for index in (LEFT_HIP, RIGHT_HIP):
        points[:, index, 1] = 0.9 - BODY_HEIGHT / 2
    for index in (LEFT_SHOULDER, RIGHT_SHOULDER):
        points[:, index, 1] = 0.9 - BODY_HEIGHT
    return points


def add_jump(points: np.ndarray, takeoff: int, landing: int, height: float) -> None:
    """Vuelo parabólico entre takeoff y landing: todo el cuerpo sube `height` en el ápice."""
    flight = np.arange(takeoff, landing)
    lift = height * (1 - ((flight - takeoff) / (landing - takeoff) * 2 - 1) ** 2)
    points[flight, :, 1] -= lift[:, None]


def add_hip_hinge(points: np.ndarray, start: int, end: int) -> None:
    """Bisagra de cadera con piernas rectas: los hombros bajan a la altura de la cadera (90°)."""
    for index in (LEFT_SHOULDER, RIGHT_SHOULDER):
        points[start:end, index] = [0.5 + BODY_HEIGHT / 2, 0.9 - BODY_HEIGHT / 2]


def angle_series(knee_flexion: list[float], trunk_inclination: list[float]) -> AngleSeries:
    return AngleSeries(
        side=Side.LEFT,
        knee_flexion=np.array(knee_flexion, dtype=float),
        trunk_inclination=np.array(trunk_inclination, dtype=float),
    )
