from collections.abc import Callable
from functools import partial

import grpc
from grpc_health.v1 import health, health_pb2_grpc
from kinesiapp.ai.v1 import lifecycle_pb2, lifecycle_pb2_grpc, movement_analysis_pb2, movement_analysis_pb2_grpc

from kinesiapp_ai.analysis.risk_level import RiskLevelClassifier
from kinesiapp_ai.analysis.video_duration import VideoDurationGuard
from kinesiapp_ai.lifecycle.analysis_models import AnalysisModelLoader, AnalysisModels
from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer
from kinesiapp_ai.pipeline.analysis_pipeline import MovementAnalysisPipeline
from kinesiapp_ai.pipeline.analysis_slots import AnalysisSlots
from kinesiapp_ai.pipeline.analyzer_factory import MovementAnalyzerFactory
from kinesiapp_ai.pipeline.strategy_factory import MovementStrategyFactory
from kinesiapp_ai.pipeline.threaded_analysis import ThreadedAnalysisRunner
from kinesiapp_ai.rpc.analysis_messages import AnalysisMessages
from kinesiapp_ai.rpc.lifecycle_servicer import AiLifecycleServicer
from kinesiapp_ai.rpc.movement_analysis_servicer import MovementAnalysisServicer
from kinesiapp_ai.settings.ai import AiSettings
from kinesiapp_ai.video_source.http_video_source import HttpVideoSource, build_http_client
from kinesiapp_ai.video_source.url_policy import AllowedHostPolicy

ServicerRegistration = Callable[[grpc.aio.Server], None]


class AiServiceFactory:
    """Factory: armo cada servicer de kinesiapp.ai.v1 con sus dependencias a partir de la configuración."""

    def __init__(self, settings: AiSettings) -> None:
        # Comparto un solo warmer: WarmUp lo dispara y el análisis espera a los mismos modelos
        self.warmer: ModelWarmer[AnalysisModels] = ModelWarmer(AnalysisModelLoader(settings.pose))
        self.health = health.aio.HealthServicer()
        self._http_client = build_http_client(settings.download)
        self._analysis = self._build_analysis(settings)

    def _build_analysis(self, settings: AiSettings) -> MovementAnalysisServicer:
        pose, analysis, risk, download = settings.pose, settings.analysis, settings.risk, settings.download
        pipeline = MovementAnalysisPipeline(
            slots=AnalysisSlots(analysis.MAX_CONCURRENCY),
            warmer=self.warmer,
            analyzers=MovementAnalyzerFactory(pose, analysis),
            strategies=MovementStrategyFactory(settings.phases, risk).build_all(),
            source=HttpVideoSource(self._http_client, AllowedHostPolicy(download.ALLOWED_HOSTS), download),
            duration_guard=VideoDurationGuard(analysis.MAX_VIDEO_DURATION_SECONDS),
            runner=ThreadedAnalysisRunner(analysis.PROGRESS_EVERY_FRAMES),
        )
        messages = AnalysisMessages(
            RiskLevelClassifier(risk.RISK_LEVEL_MODERATE, risk.RISK_LEVEL_HIGH),
            pose.POSE_MODEL_VERSION,
            risk.RISK_MODEL_VERSION,
        )
        return MovementAnalysisServicer(pipeline, messages)

    def registrations(self) -> list[ServicerRegistration]:
        return [
            partial(health_pb2_grpc.add_HealthServicer_to_server, self.health),
            partial(lifecycle_pb2_grpc.add_AiLifecycleServiceServicer_to_server, AiLifecycleServicer(self.warmer)),
            partial(movement_analysis_pb2_grpc.add_MovementAnalysisServiceServicer_to_server, self._analysis),
        ]

    @staticmethod
    def service_names() -> list[str]:
        return [
            lifecycle_pb2.DESCRIPTOR.services_by_name["AiLifecycleService"].full_name,
            movement_analysis_pb2.DESCRIPTOR.services_by_name["MovementAnalysisService"].full_name,
        ]

    async def aclose(self) -> None:
        await self._http_client.aclose()
