"""Pruebo por la red que el interceptor de logs registra el status real, también cuando el cliente cancela."""

import asyncio
import logging

import grpc
import pytest

from kinesiapp_ai.rpc.interceptors.logging import LoggingInterceptor

_STREAM = "/test.Logging/Stream"


def _identity(value: bytes) -> bytes:
    return value


async def _cancel_mid_stream() -> None:
    finished = asyncio.Event()

    async def stream(request: bytes, context):
        try:
            yield b"first"
            await asyncio.sleep(30)
        finally:
            finished.set()

    handlers = {"Stream": grpc.unary_stream_rpc_method_handler(stream, _identity, _identity)}
    server = grpc.aio.server(interceptors=[LoggingInterceptor()])
    server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler("test.Logging", handlers)])
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = channel.unary_stream(_STREAM, _identity, _identity)(b"")
            assert await call.read() == b"first"
            call.cancel()
            await asyncio.wait_for(finished.wait(), 5)
            # Le doy al interceptor un turno del event loop para cerrar su scope y escribir el log
            await asyncio.sleep(0.1)
    finally:
        await server.stop(None)


def test_a_stream_cancelled_by_the_client_is_logged_as_cancelled_not_ok(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="kinesiapp_ai.rpc.interceptors.logging"):
        asyncio.run(_cancel_mid_stream())

    assert [record.getMessage().split()[1] for record in caplog.records] == ["CANCELLED"]
