from functools import partial

from kinesiapp_ai.analysis.angles import AngleCalculator
from kinesiapp_ai.analysis.frame_downscaler import FrameDownscaler
from kinesiapp_ai.analysis.frame_sampler import FrameSampler
from kinesiapp_ai.analysis.movement_analyzer import MovementAnalyzer
from kinesiapp_ai.analysis.pose_estimator import MediaPipePoseEstimator
from kinesiapp_ai.analysis.pose_extraction import MainPersonTracker, VideoPoseExtractor
from kinesiapp_ai.analysis.preprocessing import LandmarkSeriesPreprocessor
from kinesiapp_ai.analysis.video_reader import VideoFrameReader
from kinesiapp_ai.lifecycle.analysis_models import AnalysisModels
from kinesiapp_ai.settings.analysis import AnalysisSettings
from kinesiapp_ai.settings.pose import PoseSettings


class MovementAnalyzerFactory:
    """Factory: armo el analizador con los modelos ya cargados y la configuración de pose y de límites."""

    def __init__(self, pose: PoseSettings, analysis: AnalysisSettings) -> None:
        self._pose = pose
        self._analysis = analysis

    def build(self, models: AnalysisModels) -> MovementAnalyzer:
        pose = self._pose
        return MovementAnalyzer(
            pose_extractor=VideoPoseExtractor(
                VideoFrameReader(FrameDownscaler(self._analysis.MAX_FRAME_SIDE_PX), FrameSampler(self._analysis.MAX_FPS)),
                partial(MediaPipePoseEstimator, pose.POSE_MODEL_PATH, pose.POSE_MIN_DETECTION_CONFIDENCE),
                # Comparto el detector entre videos; el seguimiento es uno por video, como el estimador
                partial(
                    MainPersonTracker,
                    models.person_detector,
                    margin=pose.PERSON_CROP_MARGIN,
                    min_relative_area=pose.PERSON_MIN_RELATIVE_AREA,
                    min_iou=pose.PERSON_TRACK_MIN_IOU,
                    max_missed_frames=pose.PERSON_TRACK_MAX_MISSED_FRAMES,
                    smoothing=pose.PERSON_TRACK_SMOOTHING,
                    detection_interval=pose.PERSON_DETECTOR_INTERVAL_FRAMES,
                ),
                max_duration_ms=round(self._analysis.MAX_VIDEO_DURATION_SECONDS * 1000),
            ),
            preprocessor=LandmarkSeriesPreprocessor(
                pose.LANDMARK_VISIBILITY_THRESHOLD, pose.LANDMARK_SMOOTHING_WINDOW_FRAMES
            ),
            angle_calculator=AngleCalculator(),
        )
