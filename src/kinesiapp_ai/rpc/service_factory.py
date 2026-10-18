from collections.abc import Callable
from functools import partial

import grpc
from grpc_health.v1 import health, health_pb2_grpc
from kinesiapp.ai.v1 import lifecycle_pb2, lifecycle_pb2_grpc

from kinesiapp_ai.lifecycle.analysis_models import AnalysisModelLoader, AnalysisModels
from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer
from kinesiapp_ai.rpc.lifecycle_servicer import AiLifecycleServicer
from kinesiapp_ai.settings.pose import PoseSettings

ServicerRegistration = Callable[[grpc.aio.Server], None]


class AiServiceFactory:
    """Factory: armo cada servicer de kinesiapp.ai.v1 con sus dependencias a partir de la configuración."""

    def __init__(self, pose_settings: PoseSettings) -> None:
        # Comparto un solo warmer: WarmUp lo dispara y el análisis espera a los mismos modelos
        self.warmer: ModelWarmer[AnalysisModels] = ModelWarmer(AnalysisModelLoader(pose_settings))
        self.health = health.aio.HealthServicer()

    def registrations(self) -> list[ServicerRegistration]:
        return [
            partial(health_pb2_grpc.add_HealthServicer_to_server, self.health),
            partial(lifecycle_pb2_grpc.add_AiLifecycleServiceServicer_to_server, AiLifecycleServicer(self.warmer)),
        ]

    @staticmethod
    def service_names() -> list[str]:
        return [lifecycle_pb2.DESCRIPTOR.services_by_name["AiLifecycleService"].full_name]
