"""JumpPhaseDetector (las tres reglas de §5.1) y SquatBottomDetector, sobre series sintéticas."""
import numpy as np

from app.analysis.angles import AngleCalculator
from app.analysis.jump_phases import JumpPhaseDetector
from app.analysis.movement_windows import DetectionMethod, MovementWindow
from app.analysis.pose_series import LEFT_ANKLE, LEFT_HEEL, LEFT_HIP, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_HIP
from app.analysis.squat_phases import SquatBottomDetector
from app.analysis.trunk_hinge import FallbackWindowDetector, TrunkHingeDetector
from tests.pose_fixtures import FEET, FPS, add_jump, angle_series, pose_series, standing_points

LANDING_WINDOW_MS = 300


def _jump_detector() -> JumpPhaseDetector:
    return JumpPhaseDetector(
        hip_apex_prominence=0.08,
        hip_apex_min_distance_seconds=0.6,
        apex_airborne_body_fraction=0.06,
        airborne_body_fraction=0.03,
        ground_window_seconds=1.0,
        ground_percentile=90,
        landing_window_ms=LANDING_WINDOW_MS,
        landing_end_margin_frames=3,
    )


def _detect_jumps(points: np.ndarray) -> list[MovementWindow]:
    series = pose_series(points)
    return _jump_detector().detect(series, AngleCalculator().calculate(series))


def test_landing_window_starts_at_ground_contact():
    points = standing_points(90)
    add_jump(points, takeoff=30, landing=45, height=0.15)
    windows = _detect_jumps(points)
    window_frames = round(LANDING_WINDOW_MS / 1000 * FPS)
    assert len(windows) == 1
    assert windows[0].detected_by == DetectionMethod.LANDING
    # La trayectoria vuelve al suelo (menos del 3 % del cuerpo) apenas antes del frame 45
    assert 43 <= windows[0].start_frame <= 45
    assert windows[0].end_frame == windows[0].start_frame + window_frames


def test_two_jumps_give_two_windows():
    points = standing_points(150)
    add_jump(points, takeoff=30, landing=45, height=0.15)
    add_jump(points, takeoff=90, landing=105, height=0.15)
    assert len(_detect_jumps(points)) == 2


def test_rising_on_tiptoes_is_not_a_jump():
    # Tobillos y talones suben con la cadera, pero las puntas siguen en el suelo
    points = standing_points(90)
    add_jump(points, takeoff=30, landing=45, height=0.15)
    still_on_ground = standing_points(90)
    for index in FEET:
        if index not in (LEFT_ANKLE, LEFT_HEEL, RIGHT_ANKLE, RIGHT_HEEL):
            points[:, index] = still_on_ground[:, index]
    assert _detect_jumps(points) == []


def test_ground_is_local_when_the_athlete_approaches_the_camera():
    # Después del salto se acerca a la cámara: los pies bajan en la imagen. Con un suelo único
    # para todo el clip, las fases de pie quedarían "en el aire" y el aterrizaje se correría
    points = standing_points(150)
    add_jump(points, takeoff=30, landing=45, height=0.15)
    points[75:, :, 1] += 0.08
    windows = _detect_jumps(points)
    assert len(windows) == 1
    assert 43 <= windows[0].start_frame <= 45


def test_standing_still_or_squatting_without_flight_is_not_a_jump():
    points = standing_points(90)
    assert _detect_jumps(points) == []
    # La cadera baja y sube (contramovimiento) sin que los pies dejen el suelo
    for index in (LEFT_HIP, RIGHT_HIP):
        points[30:45, index, 1] += 0.12
    assert _detect_jumps(points) == []


def test_landing_in_the_last_frames_is_discarded():
    points = standing_points(48)
    add_jump(points, takeoff=30, landing=47, height=0.15)
    assert _detect_jumps(points) == []


def test_squat_bottoms_are_deep_knee_flexion_peaks():
    flexion = [5.0] * 10 + list(np.linspace(5, 100, 15)) + list(np.linspace(100, 5, 15)) + [5.0] * 10
    flexion += list(np.linspace(5, 40, 10)) + list(np.linspace(40, 5, 10)) + [5.0] * 10
    detector = SquatBottomDetector(min_prominence_deg=25, min_flexion_deg=50, min_distance_seconds=0.6, window_ms=200)
    series = pose_series(standing_points(len(flexion)))
    windows = detector.detect(series, angle_series(flexion, [0.0] * len(flexion)))
    # La segunda bajada (40°) no llega a la flexión mínima: no cuenta como repetición
    half = round(0.1 * FPS)
    assert windows == [MovementWindow(24 - half, 24 + half, DetectionMethod.KNEE_BOTTOM)]


def _hinge_detector() -> TrunkHingeDetector:
    return TrunkHingeDetector(min_trunk_inclination_deg=70, min_duration_seconds=0.3, window_ms=200)


def _trunk_only(trunk: list[float]):
    series = pose_series(standing_points(len(trunk)))
    return series, angle_series([0.0] * len(trunk), trunk)


def test_each_sustained_trunk_lean_is_one_repetition_around_its_peak():
    # Dos bisagras separadas por una subida a 20°; la primera con su máximo en el frame 20
    trunk = [10.0] * 10 + [80.0] * 10 + [95.0] + [80.0] * 9 + [20.0] * 10 + [85.0] * 15 + [10.0] * 5
    windows = _hinge_detector().detect(*_trunk_only(trunk))
    half = round(0.1 * FPS)
    assert len(windows) == 2
    assert windows[0] == MovementWindow(20 - half, 20 + half, DetectionMethod.TRUNK_HINGE)


def test_brief_trunk_lean_is_not_a_repetition():
    trunk = [10.0] * 20 + [90.0] * 5 + [10.0] * 20
    assert _hinge_detector().detect(*_trunk_only(trunk)) == []


def test_fallback_uses_the_trunk_only_when_the_knee_finds_no_repetition():
    flexion = [5.0] * 10 + list(np.linspace(5, 100, 15)) + list(np.linspace(100, 5, 15)) + [5.0] * 10
    trunk = [10.0] * 10 + [80.0] * 30 + [10.0] * 10
    series = pose_series(standing_points(len(flexion)))
    knee = SquatBottomDetector(min_prominence_deg=25, min_flexion_deg=50, min_distance_seconds=0.6, window_ms=200)
    detector = FallbackWindowDetector([knee, _hinge_detector()])

    with_knee_bend = detector.detect(series, angle_series(flexion, trunk))
    assert with_knee_bend == knee.detect(series, angle_series(flexion, trunk))

    straight_legs = detector.detect(series, angle_series([5.0] * len(trunk), trunk))
    assert straight_legs == _hinge_detector().detect(series, angle_series([5.0] * len(trunk), trunk))
    assert len(straight_legs) == 1
