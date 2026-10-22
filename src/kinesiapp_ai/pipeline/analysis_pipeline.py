import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.analysis.frame_sampler import FrameSampler
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage, MovementAnalysis, MovementAnalyzer
from kinesiapp_ai.analysis.camera_view import CameraView
from kinesiapp_ai.analysis.movement_strategy import Movement, MovementStrategy
from kinesiapp_ai.analysis.video_duration import VideoDurationGuard
from kinesiapp_ai.lifecycle.analysis_models import AnalysisModels
from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer
from kinesiapp_ai.pipeline.analysis_slots import AnalysisSlots
from kinesiapp_ai.pipeline.threaded_analysis import FramesProcessed, StageReached, ThreadedAnalysisRunner
from kinesiapp_ai.rpc.domain_error import DomainError
from kinesiapp_ai.video_source.http_video_source import VideoSource
from kinesiapp_ai.video_source.temporary_video import temporary_video_path


@dataclass(frozen=True)
class Progress:
    stage: AnalysisStage
    frames_processed: int = 0
    # Lo dejo en 0 mientras no sé cuántos frames trae el video
    frames_total: int = 0


class AnalyzerBuilder(Protocol):
    def build(self, models: AnalysisModels) -> MovementAnalyzer: ...


class MovementAnalysisPipeline:
    """Facade: reservo un cupo, espero los modelos, descargo el video, verifico su duración y lo analizo en un hilo.

    Entrego el avance como un stream que termina con el resultado. Al salir, por error o cancelación,
    borro el video y libero el cupo.
    """

    def __init__(
        self,
        slots: AnalysisSlots,
        warmer: ModelWarmer[AnalysisModels],
        analyzers: AnalyzerBuilder,
        strategies: dict[Movement, dict[CameraView, MovementStrategy]],
        source: VideoSource,
        duration_guard: VideoDurationGuard,
        runner: ThreadedAnalysisRunner,
        sampler: FrameSampler,
    ) -> None:
        self._slots = slots
        self._warmer = warmer
        self._analyzers = analyzers
        self._strategies = strategies
        self._source = source
        self._duration_guard = duration_guard
        self._runner = runner
        self._sampler = sampler

    async def analyze(self, video_url: str, movement: Movement) -> AsyncIterator[Progress | MovementAnalysis]:
        strategies = self._strategies[movement]
        # Valido la URL antes de reservar el cupo: una URL rechazada no debe esperar a los modelos
        self._source.validate(video_url)
        with self._slots.reserve():
            yield Progress(AnalysisStage.QUEUED)
            models = await self._models()
            yield Progress(AnalysisStage.DOWNLOADING_VIDEO)
            with temporary_video_path() as video_path:
                await self._source.download(video_url, video_path)
                metadata = await asyncio.to_thread(self._duration_guard.check, video_path)
                # Reporto el total de frames que de verdad analizo, que con un fps máximo es menor que el del archivo
                frames_total = self._sampler.sampled_count(metadata.frame_count, metadata.fps)
                async for event in self._runner.run(self._analyzers.build(models), video_path, strategies):
                    # Solo la extracción de pose lleva frames, como dice el contrato
                    if isinstance(event, StageReached) and event.stage is AnalysisStage.EXTRACTING_POSE:
                        yield Progress(event.stage, frames_total=frames_total)
                    elif isinstance(event, StageReached):
                        yield Progress(event.stage)
                    elif isinstance(event, FramesProcessed):
                        yield Progress(AnalysisStage.EXTRACTING_POSE, event.processed, frames_total)
                    else:
                        yield event

    async def _models(self) -> AnalysisModels:
        try:
            return await self._warmer.models()
        except Exception as error:
            # El detalle ya lo registró el warmer; al cliente le digo que la IA no está lista
            raise DomainError(ErrorReason.ERROR_REASON_AI_UNAVAILABLE, "Analysis models could not be loaded") from error
