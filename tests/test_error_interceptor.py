"""Pruebo por la red que el interceptor traduce cada error a su status gRPC con el reason del contrato."""

import asyncio

import grpc
import pytest
from google.rpc import error_details_pb2
from grpc_status import rpc_status
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.analysis.errors import NoMovementDetectedError, UnreadableVideoError, VideoAnalysisError, VideoTooLongError
from kinesiapp_ai.chat.llm import LlmOverloadedError, LlmUnavailableError
from kinesiapp_ai.rpc.domain_error import DomainError
from kinesiapp_ai.rpc.interceptors.errors import ErrorInterceptor

_UNARY = "/test.Errors/Unary"
_STREAM = "/test.Errors/Stream"


class _UnmappedAnalysisError(VideoAnalysisError):
    pass


def _identity(value: bytes) -> bytes:
    return value


async def _call(error: Exception, streaming: bool = False) -> tuple[list[bytes], grpc.aio.AioRpcError]:
    async def unary(request: bytes, context) -> bytes:
        raise error

    async def stream(request: bytes, context):
        yield b"first"
        raise error

    handlers = {
        "Unary": grpc.unary_unary_rpc_method_handler(unary, _identity, _identity),
        "Stream": grpc.unary_stream_rpc_method_handler(stream, _identity, _identity),
    }
    server = grpc.aio.server(interceptors=[ErrorInterceptor()])
    server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler("test.Errors", handlers)])
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    received: list[bytes] = []
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            try:
                if streaming:
                    async for message in channel.unary_stream(_STREAM, _identity, _identity)(b""):
                        received.append(message)
                else:
                    await channel.unary_unary(_UNARY, _identity, _identity)(b"")
            except grpc.aio.AioRpcError as rpc_error:
                return received, rpc_error
    finally:
        await server.stop(None)
    raise AssertionError("the call did not fail")


def _reason(error: grpc.aio.AioRpcError) -> str | None:
    status = rpc_status.from_call(error)
    if status is None:
        return None
    info = error_details_pb2.ErrorInfo()
    status.details[0].Unpack(info)
    return info.reason


@pytest.mark.parametrize(
    ("error", "code", "reason"),
    [
        (UnreadableVideoError("broken"), grpc.StatusCode.INVALID_ARGUMENT, "VIDEO_UNREADABLE"),
        (VideoTooLongError("too long"), grpc.StatusCode.INVALID_ARGUMENT, "VIDEO_TOO_LONG"),
        (NoMovementDetectedError("still"), grpc.StatusCode.FAILED_PRECONDITION, "NO_MOVEMENT_DETECTED"),
        (LlmUnavailableError("quota"), grpc.StatusCode.UNAVAILABLE, "ASSISTANT_UNAVAILABLE"),
        (LlmOverloadedError("503"), grpc.StatusCode.UNAVAILABLE, "ASSISTANT_UNAVAILABLE"),
        (DomainError(ErrorReason.ERROR_REASON_AI_BUSY, "busy"), grpc.StatusCode.RESOURCE_EXHAUSTED, "AI_BUSY"),
        (DomainError(ErrorReason.ERROR_REASON_STORAGE_UNAVAILABLE, "down"), grpc.StatusCode.UNAVAILABLE, "STORAGE_UNAVAILABLE"),
        (DomainError(ErrorReason.ERROR_REASON_VIDEO_TOO_LARGE, "big"), grpc.StatusCode.INVALID_ARGUMENT, "VIDEO_TOO_LARGE"),
    ],
)
def test_expected_errors_carry_their_status_and_reason(error: Exception, code: grpc.StatusCode, reason: str) -> None:
    _, rpc_error = asyncio.run(_call(error))

    assert (rpc_error.code(), _reason(rpc_error)) == (code, reason)


def test_an_error_mid_stream_arrives_after_the_messages_already_sent() -> None:
    received, rpc_error = asyncio.run(_call(NoMovementDetectedError("still"), streaming=True))

    assert received == [b"first"]
    assert _reason(rpc_error) == "NO_MOVEMENT_DETECTED"


@pytest.mark.parametrize(
    "error",
    [_UnmappedAnalysisError("new failure"), DomainError(ErrorReason.ERROR_REASON_TEAM_NAME_TAKEN, "x"), RuntimeError("db password=hunter2")],
)
def test_unknown_errors_are_internal_and_leak_nothing(error: Exception) -> None:
    _, rpc_error = asyncio.run(_call(error))

    assert (rpc_error.code(), rpc_error.details()) == (grpc.StatusCode.INTERNAL, "Internal error")


@pytest.mark.parametrize(
    "error",
    [VideoTooLongError("Video /tmp/kinesiapp-video-abc lasts more than 40.0 s"), LlmUnavailableError("Gemini failed: APIError 429")],
)
def test_expected_errors_keep_their_internal_detail_out_of_the_message(error: Exception) -> None:
    _, rpc_error = asyncio.run(_call(error))

    assert "/tmp" not in rpc_error.details() and "Gemini" not in rpc_error.details()
