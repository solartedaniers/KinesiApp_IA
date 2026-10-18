"""Levanto un servidor gRPC real en un puerto libre, con los interceptores de producción, para probar por la red."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import grpc
from google.rpc import error_details_pb2
from grpc_status import rpc_status

from kinesiapp_ai.rpc.server import build_interceptors
from kinesiapp_ai.rpc.service_factory import ServicerRegistration
from kinesiapp_ai.settings.ai import AiSettings
from kinesiapp_ai.settings.analysis import AnalysisSettings
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.pose import PoseSettings
from kinesiapp_ai.settings.risk import RiskSettings
from kinesiapp_ai.settings.server import GrpcServerSettings
from kinesiapp_ai.settings.service_auth import ServiceAuthSettings
from kinesiapp_ai.settings.video_download import VideoDownloadSettings

TOKEN = "s" * 32
WITH_TOKEN = (("authorization", f"Bearer {TOKEN}"),)


def isolated_settings() -> AiSettings:
    # Paso el secreto y los hosts a mano para que las pruebas no dependan del .env de quien las corre
    return AiSettings(
        server=GrpcServerSettings(),
        auth=ServiceAuthSettings(AI_SERVICE_TOKEN=TOKEN),
        pose=PoseSettings(),
        analysis=AnalysisSettings(),
        phases=MovementPhaseSettings(),
        risk=RiskSettings(),
        download=VideoDownloadSettings(ALLOWED_HOSTS=["storage.neon.test"]),
    )


@asynccontextmanager
async def serving(registrations: Sequence[ServicerRegistration]) -> AsyncIterator[grpc.aio.Channel]:
    server = grpc.aio.server(interceptors=build_interceptors(TOKEN))
    for register in registrations:
        register(server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield channel
    finally:
        await server.stop(None)


def error_reason(error: grpc.aio.AioRpcError) -> str:
    status = rpc_status.from_call(error)
    info = error_details_pb2.ErrorInfo()
    status.details[0].Unpack(info)
    return info.reason
