"""Pruebo el pipeline completo, con YOLO y MediaPipe reales, sobre los videos del spike.

Los videos y los modelos no se versionan: indico la carpeta de videos con KINESIAPP_TEST_VIDEOS_DIR y,
si falta algo, estas pruebas se saltan. El nombre del archivo es la verdad: "bien" es la técnica
correcta y "mal" la incorrecta.
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

VIDEOS_DIR = Path(os.environ.get("KINESIAPP_TEST_VIDEOS_DIR", "missing"))
VIDEOS = {
    "salto_bien_echo.mp4": Movement.JUMP,
    "salto_mal_echo.mp4": Movement.JUMP,
    "sentadilla_bien_echa.mp4": Movement.SQUAT,
    "sentadilla_mal_echa.mp4": Movement.SQUAT,
}
LOW_RISK = 0.33
# Los videos del spike miden 474×850; con este lado mayor cada frame queda en poco más de la mitad
REDUCED_SIDE_PX = 480
POSE = PoseSettings()

pytestmark = pytest.mark.skipif(
    not POSE.POSE_MODEL_PATH.is_file()
    or not POSE.PERSON_DETECTOR_MODEL_PATH.is_file()
    or not all((VIDEOS_DIR / name).is_file() for name in VIDEOS),
    reason="Models or spike videos are not available locally",
)


class _Silent:
    def on_stage(self, stage: AnalysisStage) -> None:
        pass

    def on_frames(self, processed: int) -> None:
        pass


def _analyze_all(max_frame_side_px: int | None) -> dict[str, MovementAnalysis]:
    models = AnalysisModelLoader(POSE)()
    analyzer = MovementAnalyzerFactory(POSE, AnalysisSettings(MAX_FRAME_SIDE_PX=max_frame_side_px)).build(models)
    strategies = MovementStrategyFactory(MovementPhaseSettings(), RiskSettings()).build_all()
    return {name: analyzer.analyze(VIDEOS_DIR / name, strategies[movement], _Silent()) for name, movement in VIDEOS.items()}


@pytest.fixture(scope="module")
def results() -> dict[str, MovementAnalysis]:
    return _analyze_all(max_frame_side_px=None)


@pytest.fixture(scope="module")
def reduced_results() -> dict[str, MovementAnalysis]:
    return _analyze_all(max_frame_side_px=REDUCED_SIDE_PX)


def test_the_bad_jump_is_a_forward_collapse_and_the_good_one_is_low_risk(results) -> None:
    good, bad = results["salto_bien_echo.mp4"].risk, results["salto_mal_echo.mp4"].risk
    assert good.risk_score < LOW_RISK < bad.risk_score
    assert bad.dominant_pattern == "forward_collapse"


def test_the_bad_squat_is_a_hip_hinge_and_the_good_one_is_low_risk(results) -> None:
    good, bad = results["sentadilla_bien_echa.mp4"].risk, results["sentadilla_mal_echa.mp4"].risk
    assert good.risk_score < LOW_RISK < bad.risk_score
    assert bad.dominant_pattern == "hip_hinge_squat"


def test_every_spike_video_is_seen_from_the_side(results, reduced_results) -> None:
    assert {result.view for result in [*results.values(), *reduced_results.values()]} == {CameraView.SIDE}


def test_the_same_video_gives_the_same_result(results) -> None:
    rerun = _analyze_all(max_frame_side_px=None)["salto_mal_echo.mp4"].risk
    assert rerun == results["salto_mal_echo.mp4"].risk


def test_reducing_the_resolution_keeps_every_conclusion(results, reduced_results) -> None:
    for name in VIDEOS:
        full, reduced = results[name].risk, reduced_results[name].risk
        assert reduced.dominant_pattern == full.dominant_pattern, name
        assert (reduced.risk_score < LOW_RISK) == (full.risk_score < LOW_RISK), name
        assert len(reduced.repetitions) == len(full.repetitions), name
