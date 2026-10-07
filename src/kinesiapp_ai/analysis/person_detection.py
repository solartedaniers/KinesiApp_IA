from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime


@dataclass(frozen=True)
class BoundingBox:
    """Caja de una persona en píxeles del frame: (x1, y1) arriba a la izquierda, (x2, y2) abajo a la derecha."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class PersonDetector(Protocol):
    """Frontera hacia el detector de personas: el resto del dominio no sabe que existe YOLO."""

    def detect(self, frame_bgr: np.ndarray) -> list[BoundingBox]: ...


class YoloPersonDetector:
    """YOLO exportado a ONNX (yolo11n), solo la clase "person", en CPU con onnxruntime.

    Sin estado entre frames: una sola instancia sirve para todos los videos.
    """

    _INPUT_SIZE = 640
    _PERSON_CLASS = 0
    # Gris con el que ultralytics rellena el letterbox al entrenar
    _PAD_VALUE = 114
    _NMS_IOU_THRESHOLD = 0.45

    def __init__(self, model_path: Path, min_confidence: float) -> None:
        self._session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._min_confidence = min_confidence

    def detect(self, frame_bgr: np.ndarray) -> list[BoundingBox]:
        height, width = frame_bgr.shape[:2]
        scale = self._INPUT_SIZE / max(height, width)
        resized = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
        # Letterbox anclado arriba a la izquierda: volver a píxeles del frame es solo dividir por la escala
        canvas = np.full((self._INPUT_SIZE, self._INPUT_SIZE, 3), self._PAD_VALUE, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255

        # (84, anclas): cx, cy, w, h en píxeles de la entrada y luego el puntaje de cada una de las 80 clases
        predictions = self._session.run(None, {self._input_name: tensor})[0][0]
        scores = predictions[4 + self._PERSON_CLASS]
        candidates = scores >= self._min_confidence
        cx, cy, w, h = predictions[:4, candidates] / scale
        scores = scores[candidates]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
        kept = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self._min_confidence, self._NMS_IOU_THRESHOLD)
        return [BoundingBox(*map(float, (x, y, x + bw, y + bh))) for x, y, bw, bh in boxes[np.ravel(kept).astype(int)]]
