"""Pruebo que el archivo temporal del video no sobrevive a su bloque, salga como salga."""

import pytest

from kinesiapp_ai.video_source.temporary_video import temporary_video_path


def test_the_file_is_removed_after_use() -> None:
    with temporary_video_path() as path:
        path.write_bytes(b"video")
        assert path.exists()

    assert not path.exists()


def test_the_file_is_removed_when_the_analysis_fails() -> None:
    with pytest.raises(RuntimeError), temporary_video_path() as path:
        path.write_bytes(b"video")
        raise RuntimeError("analysis failed")

    assert not path.exists()


def test_it_tolerates_a_file_already_removed() -> None:
    with temporary_video_path() as path:
        path.unlink()

    assert not path.exists()
