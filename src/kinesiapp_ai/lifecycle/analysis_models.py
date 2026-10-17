from dataclasses import dataclass

from kinesiapp_ai.analysis.person_detection import PersonDetector, YoloPersonDetector
from kinesiapp_ai.analysis.pose_estimator import MediaPipePoseEstimator
from kinesiapp_ai.settings.pose import PoseSettings


@dataclass(frozen=True)
class AnalysisModels:
    """Los modelos que comparten todos los análisis. El estimador de pose no está porque es uno por video."""

    person_detector: PersonDetector


class AnalysisModelLoader:
    """Cargo YOLO y dejo MediaPipe en memoria antes del primer video, para que no lo pague el primer análisis."""

    def __init__(self, settings: PoseSettings) -> None:
        self._settings = settings

    def __call__(self) -> AnalysisModels:
        settings = self._settings
        detector = YoloPersonDetector(settings.PERSON_DETECTOR_MODEL_PATH, settings.PERSON_DETECTOR_MIN_CONFIDENCE)
        # Abro y cierro un estimador: así cargo la librería nativa y el modelo, y falla aquí si falta el archivo
        MediaPipePoseEstimator(settings.POSE_MODEL_PATH, settings.POSE_MIN_DETECTION_CONFIDENCE).close()
        return AnalysisModels(person_detector=detector)
