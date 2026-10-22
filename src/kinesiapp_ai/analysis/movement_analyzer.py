import enum
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kinesiapp_ai.analysis.angles import AngleCalculator, AngleSeries
from kinesiapp_ai.analysis.camera_view import CameraView, CameraViewClassifier
from kinesiapp_ai.analysis.errors import NoMovementDetectedError
from kinesiapp_ai.analysis.movement_strategy import MovementStrategy
from kinesiapp_ai.analysis.pose_extraction import VideoPoseExtractor
from kinesiapp_ai.analysis.pose_series import PoseSeries
from kinesiapp_ai.analysis.preprocessing import LandmarkSeriesPreprocessor
from kinesiapp_ai.analysis.risk import AggregatedRisk


class AnalysisStage(str, enum.Enum):
    QUEUED = "queued"
    DOWNLOADING_VIDEO = "downloading_video"
    EXTRACTING_POSE = "extracting_pose"
    DETECTING_REPETITIONS = "detecting_repetitions"
    SCORING_RISK = "scoring_risk"


class AnalysisObserver(Protocol):
    """Observer del avance de un análisis. Si lanza una excepción, el análisis se detiene ahí."""

    def on_stage(self, stage: AnalysisStage) -> None: ...

    def on_frames(self, processed: int) -> None: ...


@dataclass(frozen=True)
class MovementAnalysis:
    series: PoseSeries
    angles: AngleSeries
    risk: AggregatedRisk
    view: CameraView


class MovementAnalyzer:
    """Video → pose → vista de la cámara → ángulos → repeticiones → riesgo, con la estrategia del
    movimiento elegido para la vista detectada.

    Es síncrono y lento a propósito: siempre lo corro en un hilo aparte, nunca en el event loop.
    """

    def __init__(
        self,
        pose_extractor: VideoPoseExtractor,
        preprocessor: LandmarkSeriesPreprocessor,
        angle_calculator: AngleCalculator,
        view_classifier: CameraViewClassifier,
    ) -> None:
        self._pose_extractor = pose_extractor
        self._preprocessor = preprocessor
        self._angle_calculator = angle_calculator
        self._view_classifier = view_classifier

    def analyze(
        self, video_path: Path, strategies: Mapping[CameraView, MovementStrategy], observer: AnalysisObserver
    ) -> MovementAnalysis:
        observer.on_stage(AnalysisStage.EXTRACTING_POSE)
        series = self._preprocessor.process(self._pose_extractor.extract(video_path, observer.on_frames))
        view = self._view_classifier.classify(series)
        strategy = strategies[view]
        angles = self._angle_calculator.calculate(series)
        observer.on_stage(AnalysisStage.DETECTING_REPETITIONS)
        windows = strategy.window_detector.detect(series, angles)
        if not windows:
            raise NoMovementDetectedError(f"No repetition found in {video_path}")
        observer.on_stage(AnalysisStage.SCORING_RISK)
        return MovementAnalysis(series=series, angles=angles, risk=strategy.aggregator.aggregate(angles, windows), view=view)
