"""Pruebo la detección de la vista con YOLO y MediaPipe reales sobre los videos de Alexander.

Los videos no se versionan: indico su carpeta con KINESIAPP_VIEW_VIDEOS_DIR y, si falta algo, estas pruebas
se saltan. salto2 y sentadilla están grabados de lado; salto1, salto3 y sentadilla3, de frente.
"""

import os
from pathlib import Path

import pytest

from kinesiapp_ai.analysis.camera_view import CameraView
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage, MovementAnalysis
from kinesiapp_ai.analysis.movement_strategy import Movement
from kinesiapp_ai.lifecycle.analysis_models import AnalysisModelLoader
from kinesiapp_ai.pipeline.analyzer_factory import MovementAnalyzerFactory
from kinesiapp_ai.pipeline.strategy_factory import MovementStrategyFactory
from kinesiapp_ai.settings.analysis import AnalysisSettings
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.pose import PoseSettings
from kinesiapp_ai.settings.risk import RiskSettings

VIDEOS_DIR = Path(os.environ.get("KINESIAPP_VIEW_VIDEOS_DIR", "missing"))
VIDEOS = {
    "salto1.mp4": (Movement.JUMP, CameraView.FRONT),
    "salto2.mp4": (Movement.JUMP, CameraView.SIDE),
    "sentadilla.mp4": (Movement.SQUAT, CameraView.SIDE),
    "salto3.mp4": (Movement.JUMP, CameraView.FRONT),
    "sentadilla3.mp4": (Movement.SQUAT, CameraView.FRONT),
}
# Analizo como en producción (ANALYSIS_MAX_FRAME_SIDE_PX de Azure). salto1 dura 41 s: subo el tope solo
# aquí porque lo que pruebo es la vista, no la duración
PRODUCTION_SIDE_PX = 720
MAX_DURATION_SECONDS = 60
POSE = PoseSettings()

pytestmark = pytest.mark.skipif(
    not POSE.POSE_MODEL_PATH.is_file()
    or not POSE.PERSON_DETECTOR_MODEL_PATH.is_file()
    or not all((VIDEOS_DIR / name).is_file() for name in VIDEOS),
    reason="Models or view videos are not available locally",
)


class _Silent:
    def on_stage(self, stage: AnalysisStage) -> None:
        pass

    def on_frames(self, processed: int) -> None:
        pass


@pytest.fixture(scope="module")
def results() -> dict[str, MovementAnalysis]:
    settings = AnalysisSettings(MAX_FRAME_SIDE_PX=PRODUCTION_SIDE_PX, MAX_VIDEO_DURATION_SECONDS=MAX_DURATION_SECONDS)
    analyzer = MovementAnalyzerFactory(POSE, settings).build(AnalysisModelLoader(POSE)())
    strategies = MovementStrategyFactory(MovementPhaseSettings(), RiskSettings()).build_all()
    return {name: analyzer.analyze(VIDEOS_DIR / name, strategies[movement], _Silent()) for name, (movement, _) in VIDEOS.items()}


@pytest.mark.parametrize("name", list(VIDEOS))
def test_each_video_gets_the_view_it_was_recorded_from(results, name: str) -> None:
    assert results[name].view is VIDEOS[name][1]


def test_the_side_videos_keep_the_side_patterns(results) -> None:
    risk = RiskSettings()
    assert set(results["salto2.mp4"].risk.pattern_scores) == set(risk.RISK_JUMP_PATTERNS)
    assert set(results["sentadilla.mp4"].risk.pattern_scores) == set(risk.RISK_SQUAT_PATTERNS)


@pytest.mark.parametrize(("name", "min_repetitions"), [("salto1.mp4", 20), ("salto3.mp4", 8), ("sentadilla3.mp4", 6)])
def test_the_front_videos_measure_the_valgus_of_each_knee_in_every_repetition(results, name: str, min_repetitions: int) -> None:
    risk = results[name].risk
    assert set(risk.pattern_scores) == set(RiskSettings().RISK_FRONT_PATTERNS)
    assert len(risk.repetitions) >= min_repetitions
    for rep in risk.repetitions:
        for code in ("knee_valgus_left", "knee_valgus_right"):
            assert rep.partials[code].peak_frame is not None
            assert rep.partials[code].side is not None
