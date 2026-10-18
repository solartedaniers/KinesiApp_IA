"""Pruebo el analizador con poses sintéticas y las estrategias reales de salto y sentadilla."""

from pathlib import Path

import numpy as np
import pytest

from kinesiapp_ai.analysis.angles import AngleCalculator
from kinesiapp_ai.analysis.errors import NoMovementDetectedError
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage, MovementAnalyzer
from kinesiapp_ai.analysis.movement_strategy import Movement
from kinesiapp_ai.analysis.movement_windows import DetectionMethod
from kinesiapp_ai.analysis.pose_series import PoseSeries
from kinesiapp_ai.analysis.preprocessing import LandmarkSeriesPreprocessor
from kinesiapp_ai.pipeline.strategy_factory import MovementStrategyFactory
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.risk import RiskSettings
from tests.pose_fixtures import add_hip_hinge, add_jump, pose_series, standing_points

STRATEGIES = MovementStrategyFactory(MovementPhaseSettings(), RiskSettings()).build_all()


class _FixedExtractor:
    """Devuelvo una serie ya armada y aviso cada frame como lo haría el extractor real."""

    def __init__(self, series: PoseSeries) -> None:
        self._series = series

    def extract(self, video_path: Path, on_frame=None) -> PoseSeries:
        for processed in range(1, self._series.frame_count + 1):
            if on_frame is not None:
                on_frame(processed)
        return self._series


class _RecordingObserver:
    def __init__(self) -> None:
        self.stages: list[AnalysisStage] = []
        self.frames: list[int] = []

    def on_stage(self, stage: AnalysisStage) -> None:
        self.stages.append(stage)

    def on_frames(self, processed: int) -> None:
        self.frames.append(processed)


def _analyze(points: np.ndarray, movement: Movement):
    analyzer = MovementAnalyzer(
        _FixedExtractor(pose_series(points)), LandmarkSeriesPreprocessor(0.1, 3), AngleCalculator()
    )
    observer = _RecordingObserver()
    return analyzer.analyze(Path("video.mp4"), STRATEGIES[movement], observer), observer


def test_a_jump_goes_through_every_stage_and_scores_its_landing() -> None:
    points = standing_points(90)
    add_jump(points, takeoff=30, landing=45, height=0.15)

    analysis, observer = _analyze(points, Movement.JUMP)

    assert observer.stages == list(AnalysisStage)
    assert observer.frames == list(range(1, 91))
    assert [rep.window.detected_by for rep in analysis.risk.repetitions] == [DetectionMethod.LANDING]
    assert set(analysis.risk.pattern_scores) <= set(RiskSettings().RISK_JUMP_PATTERNS)


def test_a_squat_without_knee_bend_falls_back_to_the_trunk_hinge() -> None:
    points = standing_points(60)
    add_hip_hinge(points, 20, 40)

    analysis, _ = _analyze(points, Movement.SQUAT)

    assert [rep.window.detected_by for rep in analysis.risk.repetitions] == [DetectionMethod.TRUNK_HINGE]
    assert analysis.risk.dominant_pattern == "hip_hinge_squat"


def test_standing_still_has_no_movement_and_never_reaches_scoring() -> None:
    observer = _RecordingObserver()
    analyzer = MovementAnalyzer(
        _FixedExtractor(pose_series(standing_points(60))), LandmarkSeriesPreprocessor(0.1, 3), AngleCalculator()
    )

    with pytest.raises(NoMovementDetectedError):
        analyzer.analyze(Path("video.mp4"), STRATEGIES[Movement.JUMP], observer)

    assert AnalysisStage.SCORING_RISK not in observer.stages


def test_the_factory_builds_one_strategy_per_movement() -> None:
    assert set(STRATEGIES) == set(Movement)


def test_a_pattern_with_an_unknown_partial_fails_when_building_the_strategies() -> None:
    risk = RiskSettings(RISK_JUMP_PATTERNS={"broken": ["knee_rigid", "ankle_wobble"]})

    with pytest.raises(ValueError, match="ankle_wobble"):
        MovementStrategyFactory(MovementPhaseSettings(), risk).build_all()
