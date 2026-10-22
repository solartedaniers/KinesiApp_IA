"""Pruebo el paso entero con que muestreo los frames para no pasar del fps máximo."""

import pytest

from kinesiapp_ai.analysis.frame_sampler import FrameSampler


@pytest.mark.parametrize(
    ("max_fps", "source_fps", "stride"),
    [(None, 60.0, 1), (15, 15.0, 1), (15, 30.0, 2), (10, 16.6, 2), (15, 60.0, 4), (10, 29.97, 3)],
)
def test_the_stride_never_lets_the_rate_pass_the_cap(max_fps: float | None, source_fps: float, stride: int) -> None:
    assert FrameSampler(max_fps).stride(source_fps) == stride


def test_the_sampled_count_includes_the_first_frame_of_an_incomplete_stride() -> None:
    assert FrameSampler(10).sampled_count(frame_count=7, source_fps=30.0) == 3
