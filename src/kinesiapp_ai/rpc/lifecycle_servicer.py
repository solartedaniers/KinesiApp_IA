import grpc
from kinesiapp.ai.v1 import lifecycle_pb2, lifecycle_pb2_grpc
from kinesiapp.common.v1 import ai_state_pb2

from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer, WarmState

_STATE_TO_PROTO = {
    WarmState.WARMING_UP: ai_state_pb2.AI_STATE_WARMING_UP,
    WarmState.READY: ai_state_pb2.AI_STATE_READY,
}


class AiLifecycleServicer(lifecycle_pb2_grpc.AiLifecycleServiceServicer):
    """Implemento kinesiapp.ai.v1.AiLifecycleService: disparo la carga de los modelos sin esperarla."""

    def __init__(self, warmer: ModelWarmer) -> None:
        self._warmer = warmer

    async def WarmUp(
        self, request: lifecycle_pb2.WarmUpRequest, context: grpc.aio.ServicerContext
    ) -> lifecycle_pb2.WarmUpResponse:
        return lifecycle_pb2.WarmUpResponse(state=_STATE_TO_PROTO[self._warmer.warm_up()])
