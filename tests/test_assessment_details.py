"""Compruebo que el RiskAssessment del contrato, al volver a detalles, da lo mismo que el serializador del análisis."""

import numpy as np
import pytest

from kinesiapp_ai.analysis.angles import AngleCalculator
from kinesiapp_ai.analysis.movement_analyzer import MovementAnalysis, MovementAnalyzer
from kinesiapp_ai.analysis.movement_strategy import Movement
from kinesiapp_ai.analysis.preprocessing import LandmarkSeriesPreprocessor
from kinesiapp_ai.analysis.risk_level import RiskLevelClassifier
from kinesiapp_ai.chat.assessment_details import RiskAssessmentDetails
from kinesiapp_ai.pipeline.strategy_factory import MovementStrategyFactory
from kinesiapp_ai.rpc.analysis_messages import AnalysisMessages
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.risk import RiskSettings
from tests.pose_fixtures import add_hip_hinge, add_jump, pose_series, standing_points
from tests.risk_details_reference import RiskDetailsSerializer

STRATEGIES = MovementStrategyFactory(MovementPhaseSettings(), RiskSettings()).build_all()


class _Fixed:
    def __init__(self, points: np.ndarray) -> None:
        self._series = pose_series(points)

    def extract(self, video_path, on_frame=None):
        return self._series


class _Silent:
    def on_stage(self, stage) -> None:
        pass

    def on_frames(self, processed: int) -> None:
        pass


def _jump() -> np.ndarray:
    points = standing_points(150)
    add_jump(points, takeoff=30, landing=45, height=0.15)
    add_jump(points, takeoff=90, landing=105, height=0.15)
    return points


def _hinge() -> np.ndarray:
    points = standing_points(60)
    add_hip_hinge(points, 20, 40)
    return points


def _analysis(points: np.ndarray, movement: Movement) -> MovementAnalysis:
    analyzer = MovementAnalyzer(_Fixed(points), LandmarkSeriesPreprocessor(0.1, 3), AngleCalculator())
    return analyzer.analyze("video.mp4", STRATEGIES[movement], _Silent())


@pytest.mark.parametrize(("points", "movement"), [(_jump(), Movement.JUMP), (_hinge(), Movement.SQUAT)], ids=["jump", "squat"])
def test_the_contract_assessment_gives_back_the_same_details(points: np.ndarray, movement: Movement) -> None:
    analysis = _analysis(points, movement)
    assessment = AnalysisMessages(RiskLevelClassifier(0.33, 0.66), "p", "r").result(analysis).result.risk_assessment

    assert RiskAssessmentDetails().to_details(assessment) == RiskDetailsSerializer().serialize(analysis.risk, analysis.series)
