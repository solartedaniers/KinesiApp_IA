"""Pruebo la verificación de duración máxima leyendo videos sintéticos."""

from pathlib import Path

import pytest

from kinesiapp_ai.analysis.errors import UnreadableVideoError, VideoTooLongError
from kinesiapp_ai.analysis.video_duration import VideoDurationGuard
from tests.video_fixtures import write_video


def test_a_video_within_the_limit_returns_its_metadata(tmp_path: Path) -> None:
    video = write_video(tmp_path / "ok.mp4", frame_count=20, fps=10.0)

    metadata = VideoDurationGuard(max_duration_seconds=2.0).check(video)

    assert (metadata.frame_count, metadata.fps, metadata.duration_ms) == (20, 10.0, 2000)


def test_a_video_longer_than_the_limit_is_rejected(tmp_path: Path) -> None:
    video = write_video(tmp_path / "long.mp4", frame_count=21, fps=10.0)

    with pytest.raises(VideoTooLongError):
        VideoDurationGuard(max_duration_seconds=2.0).check(video)


def test_a_file_that_is_not_a_video_is_unreadable(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.webm"
    garbage.write_bytes(b"\x1a\x45\xdf\xa3 not really a webm")

    with pytest.raises(UnreadableVideoError):
        VideoDurationGuard(max_duration_seconds=40.0).check(garbage)


def test_without_declared_frame_count_it_counts_and_stops_past_the_limit() -> None:
    class _Capture:
        def __init__(self, frames: int) -> None:
            self.frames, self.grabbed = frames, 0

        def grab(self) -> bool:
            if self.grabbed == self.frames:
                return False
            self.grabbed += 1
            return True

    short, endless = _Capture(frames=7), _Capture(frames=10_000)

    assert VideoDurationGuard._count_frames(short, limit=31) == 7
    assert VideoDurationGuard._count_frames(endless, limit=31) == 31
    assert endless.grabbed == 31
