import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.rpc.domain_error import DomainError
from kinesiapp_ai.rpc.interceptors.call_scope import CallScopeInterceptor

# Uso la misma cabecera estándar que el backend con sus usuarios: Envoy la deja pasar sin configuración extra
AUTHORIZATION_KEY = "authorization"
_BEARER_PREFIX = "bearer "


class ServiceTokenInterceptor(CallScopeInterceptor):
    """Solo dejo pasar las llamadas que traen el secreto compartido con el backend en "authorization: Bearer"."""

    def __init__(self, token: str) -> None:
        self._expected = token.encode("utf-8")

    @asynccontextmanager
    async def scope(self, method: str, context: grpc.aio.ServicerContext) -> AsyncIterator[None]:
        # Comparo en tiempo constante para que el tiempo de respuesta no revele cuántos caracteres acertó
        if not hmac.compare_digest(_bearer_token(context), self._expected):
            raise DomainError(ErrorReason.ERROR_REASON_UNAUTHORIZED, "Missing or invalid service token")
        yield


def _bearer_token(context: grpc.aio.ServicerContext) -> bytes:
    for key, value in context.invocation_metadata() or ():
        if key == AUTHORIZATION_KEY and isinstance(value, str) and value.lower().startswith(_BEARER_PREFIX):
            return value[len(_BEARER_PREFIX):].strip().encode("utf-8")
    return b""
