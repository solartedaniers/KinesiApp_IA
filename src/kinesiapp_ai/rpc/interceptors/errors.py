import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc
from google.protobuf import any_pb2
from google.rpc import error_details_pb2, status_pb2
from grpc_status import rpc_status
from kinesiapp.common.v1.errors_pb2 import ErrorReason
from pydantic import ValidationError

from kinesiapp_ai.analysis.errors import VideoAnalysisError
from kinesiapp_ai.rpc.domain_error import DomainError
from kinesiapp_ai.chat.llm import LlmUnavailableError
from kinesiapp_ai.rpc.error_status import ERROR_DOMAIN, REASON_BY_EXPECTED_ERROR, STATUS_BY_REASON, reason_name
from kinesiapp_ai.rpc.interceptors.call_scope import CallScopeInterceptor

logger = logging.getLogger(__name__)


class ErrorInterceptor(CallScopeInterceptor):
    """Traduzco las excepciones de los handlers a status gRPC con google.rpc.ErrorInfo en los detalles."""

    @asynccontextmanager
    async def scope(self, method: str, context: grpc.aio.ServicerContext) -> AsyncIterator[None]:
        try:
            yield
        except grpc.aio.AbortError:
            raise
        except DomainError as error:
            if error.reason not in STATUS_BY_REASON:
                logger.error("Reason %s has no status in %s", reason_name(error.reason), method)
                await context.abort(grpc.StatusCode.INTERNAL, "Internal error")
            await self._abort(context, error.reason, error.message)
        except (VideoAnalysisError, LlmUnavailableError) as error:
            reason = REASON_BY_EXPECTED_ERROR.get(type(error))
            if reason is None:
                logger.error("Expected error %s has no reason in %s", type(error).__name__, method)
                await context.abort(grpc.StatusCode.INTERNAL, "Internal error")
            logger.info("Call rejected in %s: %s", method, error)
            await self._abort(context, reason, str(error))
        except ValidationError as error:
            violations = [
                error_details_pb2.BadRequest.FieldViolation(
                    field=".".join(str(part) for part in issue["loc"]), description=issue["msg"]
                )
                for issue in error.errors()
            ]
            await self._abort(
                context,
                ErrorReason.ERROR_REASON_BAD_REQUEST,
                "Invalid request",
                error_details_pb2.BadRequest(field_violations=violations),
            )
        except NotImplementedError:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented")
        except Exception:
            # Dejo el detalle en el log; al cliente no le filtro nada interno
            logger.exception("Unhandled error in %s", method)
            await context.abort(grpc.StatusCode.INTERNAL, "Internal error")

    @staticmethod
    async def _abort(
        context: grpc.aio.ServicerContext, reason: ErrorReason.ValueType, message: str, *extra_details: object
    ) -> None:
        code = STATUS_BY_REASON[reason]
        details = []
        for detail in (error_details_pb2.ErrorInfo(reason=reason_name(reason), domain=ERROR_DOMAIN), *extra_details):
            packed = any_pb2.Any()
            packed.Pack(detail)
            details.append(packed)
        status = rpc_status.to_status(
            status_pb2.Status(code=code.value[0], message=message, details=details)
        )
        await context.abort(status.code, status.details, status.trailing_metadata)
