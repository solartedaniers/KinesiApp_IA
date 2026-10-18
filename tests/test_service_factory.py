"""Compruebo que la fábrica registra todos los servicios del contrato kinesiapp.ai.v1."""

import asyncio

from kinesiapp.ai.v1 import assistant_pb2, lifecycle_pb2, movement_analysis_pb2

from kinesiapp_ai.rpc.service_factory import AiServiceFactory
from tests.grpc_harness import isolated_settings


def _contract_services() -> set[str]:
    return {
        service.full_name
        for module in (assistant_pb2, lifecycle_pb2, movement_analysis_pb2)
        for service in module.DESCRIPTOR.services_by_name.values()
    }


def test_every_contract_service_is_registered() -> None:
    async def main():
        factory = AiServiceFactory(isolated_settings())
        try:
            return set(factory.service_names())
        finally:
            await factory.aclose()

    assert asyncio.run(main()) == _contract_services()
