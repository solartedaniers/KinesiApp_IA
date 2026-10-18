"""Pruebo la lectura de frames de un video real escrito con OpenCV."""

from pathlib import Path

import pytest

from kinesiapp_ai.analysis.errors import UnreadableVideoError
from kinesiapp_ai.analysis.frame_downscaler import FrameDownscaler
from kinesiapp_ai.analysis.video_reader import VideoFrameReader
from tests.video_fixtures import write_video


def test_frames_come_reduced_when_a_downscaler_is_configured(tmp_path: Path) -> None:
    video = write_video(tmp_path / "wide.mp4", frame_count=4, width=1280, height=720)

    with VideoFrameReader(FrameDownscaler(640)).open(video) as stream:
        shapes = [frame.shape for frame in stream.frames]

    assert shapes == [(360, 640, 3)] * 4


def test_without_downscaler_frames_keep_their_resolution(tmp_path: Path) -> None:
    video = write_video(tmp_path / "small.mp4", frame_count=2, width=320, height=240)

    with VideoFrameReader().open(video) as stream:
        assert [frame.shape for frame in stream.frames] == [(240, 320, 3)] * 2


def test_a_file_that_is_not_a_video_is_unreadable(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video at all")

    with pytest.raises(UnreadableVideoError), VideoFrameReader().open(garbage):
        pass
