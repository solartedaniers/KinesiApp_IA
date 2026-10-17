import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc

from kinesiapp_ai.rpc.interceptors.call_scope import CallScopeInterceptor

logger = logging.getLogger(__name__)


class LoggingInterceptor(CallScopeInterceptor):
    """Registro método, status code y duración de cada llamada."""

    @asynccontextmanager
    async def scope(self, method: str, context: grpc.aio.ServicerContext) -> AsyncIterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            code = context.code() or grpc.StatusCode.OK
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("%s %s %.1fms", method, code.name, elapsed_ms)
