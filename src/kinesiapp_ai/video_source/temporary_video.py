import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

_PREFIX = "kinesiapp-video-"


@contextmanager
def temporary_video_path() -> Iterator[Path]:
    """Reservo un archivo temporal para el video y lo borro al salir, también si falla o se cancela."""
    descriptor, name = tempfile.mkstemp(prefix=_PREFIX)
    os.close(descriptor)
    path = Path(name)
    try:
        yield path
    finally:
        with suppress(FileNotFoundError):
            path.unlink()
