"""Pruebo la reducción de resolución opcional de los frames."""

import numpy as np

from kinesiapp_ai.analysis.frame_downscaler import FrameDownscaler


def _frame(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_a_vertical_1080p_frame_is_reduced_by_its_longest_side_keeping_the_ratio() -> None:
    reduced = FrameDownscaler(720).apply(_frame(1920, 1080))

    assert reduced.shape == (720, 405, 3)


def test_a_horizontal_frame_is_reduced_by_its_width() -> None:
    assert FrameDownscaler(640).apply(_frame(720, 1280)).shape == (360, 640, 3)


def test_small_frames_and_the_disabled_option_leave_the_frame_untouched() -> None:
    small = _frame(480, 270)
    large = _frame(1920, 1080)

    assert FrameDownscaler(720).apply(small) is small
    assert FrameDownscaler(None).apply(large) is large
