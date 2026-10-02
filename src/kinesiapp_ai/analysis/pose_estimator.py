from pathlib import Path
from typing import Protocol

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


class PoseEstimator(Protocol):
    """Frontera hacia el modelo de pose: el resto del dominio no sabe que existe MediaPipe.

    Una instancia por video (ver docs/design/video-analysis-pipeline.md §4.4).
    """

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """(33, 3) con x, y normalizados y visibilidad; None si no hay persona en el frame."""
        ...

    def close(self) -> None: ...


class MediaPipePoseEstimator:
    """PoseLandmarker de la Tasks API en modo VIDEO y CPU, con un modelo .task."""

    # El pipeline sigue a una sola persona: con más poses no hay forma de saber cuál es el atleta
    _TRACKED_POSES = 1

    def __init__(self, model_path: Path, min_pose_detection_confidence: float) -> None:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self._TRACKED_POSES,
            min_pose_detection_confidence=min_pose_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.visibility or 0.0] for lm in result.pose_landmarks[0]])

    def close(self) -> None:
        self._landmarker.close()
