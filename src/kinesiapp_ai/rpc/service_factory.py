from collections.abc import Callable
from functools import partial

import grpc
from grpc_health.v1 import health, health_pb2_grpc
from kinesiapp.ai.v1 import (
    assistant_pb2,
    assistant_pb2_grpc,
    lifecycle_pb2,
    lifecycle_pb2_grpc,
    movement_analysis_pb2,
    movement_analysis_pb2_grpc,
)

from kinesiapp_ai.analysis.frame_sampler import FrameSampler
from kinesiapp_ai.analysis.risk_level import RiskLevelClassifier
from kinesiapp_ai.analysis.video_duration import VideoDurationGuard
from kinesiapp_ai.chat.analysis_assistant import AnalysisAssistant
from kinesiapp_ai.chat.assessment_details import RiskAssessmentDetails
from kinesiapp_ai.chat.fallback_llm import FallbackLlmClient
from kinesiapp_ai.chat.gemini_client import GeminiLlmClient
from kinesiapp_ai.chat.llm import LlmClient
from kinesiapp_ai.chat.pattern_catalog import RiskPatternCatalog
from kinesiapp_ai.chat.prompt_builder import MovementAnalysisPromptBuilder
from kinesiapp_ai.lifecycle.analysis_models import AnalysisModelLoader, AnalysisModels
from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer
from kinesiapp_ai.pipeline.analysis_pipeline import MovementAnalysisPipeline
from kinesiapp_ai.pipeline.analysis_slots import AnalysisSlots
from kinesiapp_ai.pipeline.analyzer_factory import MovementAnalyzerFactory
from kinesiapp_ai.pipeline.strategy_factory import MovementStrategyFactory
from kinesiapp_ai.pipeline.threaded_analysis import ThreadedAnalysisRunner
from kinesiapp_ai.rpc.analysis_messages import AnalysisMessages
from kinesiapp_ai.rpc.assistant_messages import AssistantMessages
from kinesiapp_ai.rpc.assistant_servicer import AssistantServicer
from kinesiapp_ai.rpc.lifecycle_servicer import AiLifecycleServicer
from kinesiapp_ai.rpc.movement_analysis_servicer import MovementAnalysisServicer
from kinesiapp_ai.settings.ai import AiSettings
from kinesiapp_ai.settings.assistant import AssistantSettings
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
        self._assistant = self._build_assistant(settings)

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
            sampler=FrameSampler(analysis.MAX_FPS),
        )
        messages = AnalysisMessages(
            RiskLevelClassifier(risk.RISK_LEVEL_MODERATE, risk.RISK_LEVEL_HIGH),
            pose.POSE_MODEL_VERSION,
            risk.RISK_MODEL_VERSION,
        )
        return MovementAnalysisServicer(pipeline, messages)

    @staticmethod
    def _build_assistant(settings: AiSettings) -> AssistantServicer:
        risk, chat = settings.risk, settings.assistant
        prompts = MovementAnalysisPromptBuilder(
            # Le paso los patrones configurados: si alguno no tiene texto curado, fallo al arrancar
            catalog=RiskPatternCatalog([risk.RISK_JUMP_PATTERNS, risk.RISK_SQUAT_PATTERNS]),
            risk_level_moderate=risk.RISK_LEVEL_MODERATE,
            risk_level_high=risk.RISK_LEVEL_HIGH,
            max_repetitions=chat.CHAT_PROMPT_MAX_REPETITIONS,
            max_history_messages=chat.CHAT_HISTORY_MAX_MESSAGES,
        )
        llm: LlmClient = AiServiceFactory._gemini(chat, chat.GEMINI_MODEL)
        if chat.GEMINI_FALLBACK_MODEL:
            llm = FallbackLlmClient(llm, AiServiceFactory._gemini(chat, chat.GEMINI_FALLBACK_MODEL))
        return AssistantServicer(
            AnalysisAssistant(prompts, llm), AssistantMessages(RiskAssessmentDetails(), chat.CHAT_QUESTION_MAX_CHARS)
        )

    @staticmethod
    def _gemini(chat: AssistantSettings, model_name: str) -> GeminiLlmClient:
        return GeminiLlmClient(
            api_key=chat.GEMINI_API_KEY.get_secret_value(),
            model_name=model_name,
            timeout_seconds=chat.GEMINI_TIMEOUT_SECONDS,
            max_output_tokens=chat.GEMINI_MAX_OUTPUT_TOKENS,
            thinking_level=chat.GEMINI_THINKING_LEVEL,
        )

    def registrations(self) -> list[ServicerRegistration]:
        return [
            partial(health_pb2_grpc.add_HealthServicer_to_server, self.health),
            partial(lifecycle_pb2_grpc.add_AiLifecycleServiceServicer_to_server, AiLifecycleServicer(self.warmer)),
            partial(movement_analysis_pb2_grpc.add_MovementAnalysisServiceServicer_to_server, self._analysis),
            partial(assistant_pb2_grpc.add_AssistantServiceServicer_to_server, self._assistant),
        ]

    @staticmethod
    def service_names() -> list[str]:
        return [
            lifecycle_pb2.DESCRIPTOR.services_by_name["AiLifecycleService"].full_name,
            movement_analysis_pb2.DESCRIPTOR.services_by_name["MovementAnalysisService"].full_name,
            assistant_pb2.DESCRIPTOR.services_by_name["AssistantService"].full_name,
        ]

    async def aclose(self) -> None:
        await self._http_client.aclose()
