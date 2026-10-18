from collections.abc import AsyncIterator

import grpc
from kinesiapp.ai.v1 import movement_analysis_pb2 as analysis_pb2
from kinesiapp.ai.v1 import movement_analysis_pb2_grpc as analysis_pb2_grpc
from kinesiapp.common.v1 import movement_pb2
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.analysis.movement_strategy import Movement
from kinesiapp_ai.pipeline.analysis_pipeline import MovementAnalysisPipeline, Progress
from kinesiapp_ai.rpc.analysis_messages import AnalysisMessages
from kinesiapp_ai.rpc.domain_error import DomainError

_MOVEMENTS = {
    movement_pb2.MOVEMENT_TYPE_JUMP: Movement.JUMP,
    movement_pb2.MOVEMENT_TYPE_SQUAT: Movement.SQUAT,
}


class MovementAnalysisServicer(analysis_pb2_grpc.MovementAnalysisServiceServicer):
    """Implemento kinesiapp.ai.v1.MovementAnalysisService: traduzco el contrato y dejo el trabajo a la fachada."""

    def __init__(self, pipeline: MovementAnalysisPipeline, messages: AnalysisMessages) -> None:
        self._pipeline = pipeline
        self._messages = messages

    async def AnalyzeVideo(
        self, request: analysis_pb2.AnalyzeVideoRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[analysis_pb2.AnalyzeVideoResponse]:
        movement = _MOVEMENTS.get(request.movement_type)
        if movement is None:
            raise DomainError(ErrorReason.ERROR_REASON_BAD_REQUEST, "movement_type must be jump or squat")
        async for event in self._pipeline.analyze(request.video_url, movement):
            if isinstance(event, Progress):
                yield self._messages.progress(
                    self._messages.stage(event.stage), event.frames_processed, event.frames_total
                )
            else:
                yield self._messages.result(event)
