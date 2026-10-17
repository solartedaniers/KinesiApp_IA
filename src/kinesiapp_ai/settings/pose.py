from pathlib import Path

from pydantic import Field

from kinesiapp_ai.settings.base import EnvSettings

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


class PoseSettings(EnvSettings):
    """Configuro la extracción de pose y la detección de la persona principal.

    Guardo la versión del modelo con cada resultado: si cambio de modelo, cambio también la versión.
    """

    POSE_MODEL_PATH: Path = _MODELS_DIR / "pose_landmarker_full.task"
    POSE_MODEL_VERSION: str = "mediapipe-pose_landmarker_full-float16-v1"
    POSE_MIN_DETECTION_CONFIDENCE: float = Field(default=0.5, ge=0, le=1)
    # Si otra persona compite con el atleta, MediaPipe solo ve el recorte del atleta con este margen por lado.
    # Las cajas por debajo del área relativa a la mayor no compiten: el reflejo del espejo de los videos de
    # prueba ocupa hasta 0.19
    PERSON_DETECTOR_MODEL_PATH: Path = _MODELS_DIR / "yolo11n.onnx"
    PERSON_DETECTOR_MIN_CONFIDENCE: float = Field(default=0.5, ge=0, le=1)
    PERSON_CROP_MARGIN: float = Field(default=0.25, ge=0)
    PERSON_MIN_RELATIVE_AREA: float = Field(default=0.25, ge=0, le=1)
    # En los videos de prueba la caja del atleta se solapa al menos 0.24 con la anterior aun después de un
    # hueco, y el detector lo pierde hasta 7 frames seguidos en el vuelo por el desenfoque
    PERSON_TRACK_MIN_IOU: float = Field(default=0.2, gt=0, le=1)
    PERSON_TRACK_MAX_MISSED_FRAMES: int = Field(default=15, ge=0)
    PERSON_TRACK_SMOOTHING: float = Field(default=0.5, gt=0, le=1)
    LANDMARK_VISIBILITY_THRESHOLD: float = Field(default=0.1, ge=0, le=1)
    LANDMARK_SMOOTHING_WINDOW_FRAMES: int = Field(default=3, ge=1)
