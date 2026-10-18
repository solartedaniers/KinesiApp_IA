"""Compruebo que la fábrica registra los servicios del contrato kinesiapp.ai.v1 que ya implementé."""

import asyncio

from kinesiapp.ai.v1 import assistant_pb2, lifecycle_pb2, movement_analysis_pb2

from kinesiapp_ai.rpc.service_factory import AiServiceFactory
from tests.grpc_harness import isolated_settings

# Los quito de aquí a medida que los implemento
PENDING_SERVICES = {"kinesiapp.ai.v1.AssistantService"}


def _contract_services() -> set[str]:
    return {
        service.full_name
        for module in (assistant_pb2, lifecycle_pb2, movement_analysis_pb2)
        for service in module.DESCRIPTOR.services_by_name.values()
    }


def test_every_contract_service_is_registered_or_explicitly_pending() -> None:
    async def main():
        factory = AiServiceFactory(isolated_settings())
        try:
            return set(factory.service_names())
        finally:
            await factory.aclose()

    assert asyncio.run(main()) == _contract_services() - PENDING_SERVICES
