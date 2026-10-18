"""Pruebo el servidor de la IA por la red: salud, WarmUp y el rechazo de llamadas sin el secreto del backend."""

import asyncio
import threading
from collections.abc import Callable
from functools import partial

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from kinesiapp.ai.v1 import lifecycle_pb2, lifecycle_pb2_grpc
from kinesiapp.common.v1 import ai_state_pb2

from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer
from kinesiapp_ai.rpc.lifecycle_servicer import AiLifecycleServicer
from kinesiapp_ai.rpc.server import mark_serving
from kinesiapp_ai.rpc.service_factory import AiServiceFactory
from tests.grpc_harness import TOKEN, WITH_TOKEN, error_reason, serving, isolated_settings


def _gated_loader() -> tuple[Callable[[], str], threading.Event]:
    release = threading.Event()

    def load() -> str:
        release.wait(timeout=5)
        return "models"

    return load, release


def test_health_reports_serving_only_with_the_service_token() -> None:
    async def main():
        factory = AiServiceFactory(isolated_settings())
        async with serving(factory.registrations()) as channel:
            await mark_serving(factory.health, factory.service_names())
            stub = health_pb2_grpc.HealthStub(channel)
            overall = await stub.Check(health_pb2.HealthCheckRequest(), metadata=WITH_TOKEN)
            lifecycle = await stub.Check(
                health_pb2.HealthCheckRequest(service="kinesiapp.ai.v1.AiLifecycleService"), metadata=WITH_TOKEN
            )
            try:
                await stub.Check(health_pb2.HealthCheckRequest())
            except grpc.aio.AioRpcError as error:
                return overall.status, lifecycle.status, error
        raise AssertionError("la salud respondió sin el secreto")

    overall, lifecycle, anonymous = asyncio.run(main())

    assert overall == lifecycle == health_pb2.HealthCheckResponse.SERVING
    assert anonymous.code() == grpc.StatusCode.UNAUTHENTICATED
    assert error_reason(anonymous) == "UNAUTHORIZED"


def test_wrong_or_malformed_tokens_are_rejected() -> None:
    async def main():
        factory = AiServiceFactory(isolated_settings())
        async with serving(factory.registrations()) as channel:
            stub = lifecycle_pb2_grpc.AiLifecycleServiceStub(channel)
            errors = []
            for metadata in (
                (("authorization", f"Bearer {TOKEN[:-1]}x"),),
                (("authorization", TOKEN),),
                (("authorization", f"Basic {TOKEN}"),),
            ):
                try:
                    await stub.WarmUp(lifecycle_pb2.WarmUpRequest(), metadata=metadata)
                except grpc.aio.AioRpcError as error:
                    errors.append((error.code(), error_reason(error)))
            return errors

    assert asyncio.run(main()) == [(grpc.StatusCode.UNAUTHENTICATED, "UNAUTHORIZED")] * 3


def test_warm_up_answers_without_waiting_and_then_reports_ready() -> None:
    async def main():
        load, release = _gated_loader()
        warmer = ModelWarmer(load)
        registration = partial(lifecycle_pb2_grpc.add_AiLifecycleServiceServicer_to_server, AiLifecycleServicer(warmer))
        async with serving([registration]) as channel:
            stub = lifecycle_pb2_grpc.AiLifecycleServiceStub(channel)
            first = await asyncio.wait_for(stub.WarmUp(lifecycle_pb2.WarmUpRequest(), metadata=WITH_TOKEN), 2)
            again = await stub.WarmUp(lifecycle_pb2.WarmUpRequest(), metadata=WITH_TOKEN)
            release.set()
            await warmer.models()
            ready = await stub.WarmUp(lifecycle_pb2.WarmUpRequest(), metadata=WITH_TOKEN)
            return first.state, again.state, ready.state

    assert asyncio.run(main()) == (
        ai_state_pb2.AI_STATE_WARMING_UP,
        ai_state_pb2.AI_STATE_WARMING_UP,
        ai_state_pb2.AI_STATE_READY,
    )


def test_a_failed_load_is_retried_on_the_next_warm_up() -> None:
    attempts = []

    def flaky_load() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("model file missing")
        return "models"

    async def main():
        warmer = ModelWarmer(flaky_load)
        warmer.warm_up()
        try:
            await warmer.models()
        except OSError:
            pass
        warmer.warm_up()
        return await warmer.models()

    assert asyncio.run(main()) == "models"
    assert len(attempts) == 2
