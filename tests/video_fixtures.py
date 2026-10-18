"""Escribo videos sintéticos pequeños con OpenCV para probar la lectura sin depender de archivos reales."""

from pathlib import Path

import cv2
import numpy as np


def write_video(path: Path, frame_count: int, fps: float = 30.0, width: int = 64, height: int = 48) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for index in range(frame_count):
            writer.write(np.full((height, width, 3), index % 256, dtype=np.uint8))
    finally:
        writer.release()
    return path
