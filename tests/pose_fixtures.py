"""Series de pose sintéticas con geometría conocida, para probar el dominio sin MediaPipe."""
import numpy as np

from kinesiapp_ai.analysis.angles import AngleSeries
from kinesiapp_ai.analysis.camera_view import CameraViewClassifier
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
    SIDE_LANDMARKS,
    PoseSeries,
    Side,
)

FPS = 30.0
VIEW_CLASSIFIER = CameraViewClassifier(side_max_width_ratio=0.3, front_min_width_ratio=0.4)
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


def angle_series(
    knee_flexion: list[float],
    trunk_inclination: list[float],
    knee_valgus: dict[Side, list[float]] | None = None,
) -> AngleSeries:
    # Sin valgo explícito lo dejo sin medir, como en una vista lateral
    valgus = knee_valgus or {side: [np.nan] * len(knee_flexion) for side in Side}
    return AngleSeries(
        side=Side.LEFT,
        knee_flexion=np.array(knee_flexion, dtype=float),
        trunk_inclination=np.array(trunk_inclination, dtype=float),
        knee_valgus={side: np.array(values, dtype=float) for side, values in valgus.items()},
    )


SHOULDER_HALF_WIDTH = 0.12
HIP_HALF_WIDTH = 0.07


def frontal_points(frames: int) -> np.ndarray:
    """La misma persona de standing_points pero vista de frente: hombros y caderas separados, cada pierna
    recta bajo su cadera. La izquierda del deportista queda a la derecha de la imagen, como de frente."""
    points = standing_points(frames)
    for index, offset in ((LEFT_SHOULDER, SHOULDER_HALF_WIDTH), (RIGHT_SHOULDER, -SHOULDER_HALF_WIDTH)):
        points[:, index, 0] = 0.5 + offset
    for leg, offset in ((SIDE_LANDMARKS[Side.LEFT], HIP_HALF_WIDTH), (SIDE_LANDMARKS[Side.RIGHT], -HIP_HALF_WIDTH)):
        for index in (leg.hip, leg.knee, leg.ankle, leg.heel, leg.foot_index):
            points[:, index, 0] = 0.5 + offset
    return points


def bend_knee_inward(points: np.ndarray, frames: slice, side: Side, fppa_deg: float) -> None:
    """Llevo la rodilla hacia el centro (o hacia afuera con un ángulo negativo) hasta el FPPA pedido. Con la
    rodilla a media altura entre cadera y tobillo, el FPPA es el doble del ángulo de cada segmento."""
    leg = SIDE_LANDMARKS[side]
    toward_center = -np.sign(points[frames, leg.hip, 0] - 0.5)
    shift = BODY_HEIGHT / 4 * np.tan(np.radians(fppa_deg / 2))
    points[frames, leg.knee, 0] += toward_center * shift
