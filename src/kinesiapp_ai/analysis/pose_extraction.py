from collections.abc import Callable
from pathlib import Path

import numpy as np

from kinesiapp_ai.analysis.errors import UnreadableVideoError
from kinesiapp_ai.analysis.pose_estimator import PoseEstimator
from kinesiapp_ai.analysis.pose_series import LANDMARK_COUNT, PoseSeries
from kinesiapp_ai.analysis.video_reader import VideoFrameReader


class VideoPoseExtractor:
    """Corre el estimador de pose sobre cada frame de un video y arma la serie cruda."""

    def __init__(self, frame_reader: VideoFrameReader, estimator_factory: Callable[[], PoseEstimator]) -> None:
        self._frame_reader = frame_reader
        self._estimator_factory = estimator_factory

    def extract(self, video_path: Path) -> PoseSeries:
        frames: list[np.ndarray] = []
        with self._frame_reader.open(video_path) as stream:
            estimator = self._estimator_factory()
            try:
                for index, frame in enumerate(stream.frames):
                    pose = estimator.estimate(frame, round(index * 1000 / stream.fps))
                    frames.append(pose if pose is not None else np.full((LANDMARK_COUNT, 3), np.nan))
            finally:
                estimator.close()
        if not frames:
            raise UnreadableVideoError(f"Video {video_path} has no frames")
        landmarks = np.stack(frames)
        return PoseSeries(
            points=landmarks[:, :, :2],
            visibility=landmarks[:, :, 2],
            fps=stream.fps,
            aspect_ratio=stream.width / stream.height,
        )
