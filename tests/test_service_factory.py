"""Compruebo que la fábrica registra todos los servicios del contrato kinesiapp.ai.v1."""

import asyncio
from dataclasses import replace

from kinesiapp.ai.v1 import assistant_pb2, lifecycle_pb2, movement_analysis_pb2

from kinesiapp_ai.rpc.service_factory import AiServiceFactory
from kinesiapp_ai.settings.assistant import AssistantSettings
from tests.grpc_harness import isolated_settings


def _contract_services() -> set[str]:
    return {
        service.full_name
        for module in (assistant_pb2, lifecycle_pb2, movement_analysis_pb2)
        for service in module.DESCRIPTOR.services_by_name.values()
    }


def _registered_services(settings) -> set[str]:
    async def main():
        factory = AiServiceFactory(settings)
        try:
            return set(factory.service_names())
        finally:
            await factory.aclose()

    return asyncio.run(main())


def test_every_contract_service_is_registered() -> None:
    assert _registered_services(isolated_settings()) == _contract_services()


def test_the_factory_also_builds_with_a_fallback_model() -> None:
    assistant = AssistantSettings(GEMINI_API_KEY="test-key", GEMINI_FALLBACK_MODEL="gemini-3-flash-preview")

    assert _registered_services(replace(isolated_settings(), assistant=assistant)) == _contract_services()
