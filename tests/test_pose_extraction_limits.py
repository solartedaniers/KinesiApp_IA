"""Pruebo el progreso, la cancelación y el tope de duración del extractor, con un estimador falso."""

from pathlib import Path

import numpy as np
import pytest

from kinesiapp_ai.analysis.errors import AnalysisCancelledError, VideoTooLongError
from kinesiapp_ai.analysis.pose_extraction import VideoPoseExtractor
from kinesiapp_ai.analysis.pose_series import LANDMARK_COUNT
from kinesiapp_ai.analysis.video_reader import VideoFrameReader
from tests.video_fixtures import write_video


class _CountingEstimator:
    def __init__(self) -> None:
        self.frames = 0
        self.closed = False

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        self.frames += 1
        return np.full((LANDMARK_COUNT, 3), 0.5)

    def close(self) -> None:
        self.closed = True


def _extractor(estimator: _CountingEstimator, max_duration_ms: int | None = None) -> VideoPoseExtractor:
    return VideoPoseExtractor(VideoFrameReader(), lambda: estimator, max_duration_ms=max_duration_ms)


def test_the_observer_hears_every_processed_frame(tmp_path: Path) -> None:
    video = write_video(tmp_path / "v.mp4", frame_count=6, fps=10.0)
    heard: list[int] = []

    series = _extractor(_CountingEstimator()).extract(video, heard.append)

    assert heard == [1, 2, 3, 4, 5, 6]
    assert series.frame_count == 6


def test_an_observer_that_cancels_stops_the_extraction_and_releases_the_model(tmp_path: Path) -> None:
    video = write_video(tmp_path / "v.mp4", frame_count=30, fps=10.0)
    estimator = _CountingEstimator()

    def cancel_at_third(processed: int) -> None:
        if processed == 3:
            raise AnalysisCancelledError

    with pytest.raises(AnalysisCancelledError):
        _extractor(estimator).extract(video, cancel_at_third)

    assert (estimator.frames, estimator.closed) == (3, True)


def test_frames_past_the_maximum_duration_stop_the_extraction(tmp_path: Path) -> None:
    video = write_video(tmp_path / "v.mp4", frame_count=15, fps=10.0)
    estimator = _CountingEstimator()

    with pytest.raises(VideoTooLongError):
        _extractor(estimator, max_duration_ms=1000).extract(video)

    # Proceso los 10 frames del primer segundo y paro en el que cae en 1000 ms
    assert (estimator.frames, estimator.closed) == (10, True)


def test_a_video_exactly_at_the_maximum_duration_is_accepted(tmp_path: Path) -> None:
    video = write_video(tmp_path / "v.mp4", frame_count=10, fps=10.0)

    assert _extractor(_CountingEstimator(), max_duration_ms=1000).extract(video).frame_count == 10
