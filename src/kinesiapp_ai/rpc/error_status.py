from grpc import StatusCode
from kinesiapp.common.v1.errors_pb2 import ErrorReason

ERROR_DOMAIN = "kinesiapp"
_REASON_PREFIX = "ERROR_REASON_"

# Aquí, y solo aquí, convierto un motivo en status code, como lo anota errors.proto. Solo están los que
# emite la IA: si aparece otro, el interceptor responde INTERNAL en vez de inventar un código
STATUS_BY_REASON: dict[ErrorReason.ValueType, StatusCode] = {
    ErrorReason.ERROR_REASON_BAD_REQUEST: StatusCode.INVALID_ARGUMENT,
    ErrorReason.ERROR_REASON_UNAUTHORIZED: StatusCode.UNAUTHENTICATED,
    ErrorReason.ERROR_REASON_STORAGE_UNAVAILABLE: StatusCode.UNAVAILABLE,
    ErrorReason.ERROR_REASON_VIDEO_TOO_LARGE: StatusCode.INVALID_ARGUMENT,
    ErrorReason.ERROR_REASON_VIDEO_TOO_LONG: StatusCode.INVALID_ARGUMENT,
    ErrorReason.ERROR_REASON_VIDEO_UNREADABLE: StatusCode.INVALID_ARGUMENT,
    ErrorReason.ERROR_REASON_NO_MOVEMENT_DETECTED: StatusCode.FAILED_PRECONDITION,
    ErrorReason.ERROR_REASON_AI_UNAVAILABLE: StatusCode.UNAVAILABLE,
    ErrorReason.ERROR_REASON_AI_BUSY: StatusCode.RESOURCE_EXHAUSTED,
    ErrorReason.ERROR_REASON_ASSISTANT_UNAVAILABLE: StatusCode.UNAVAILABLE,
}


def reason_name(reason: ErrorReason.ValueType) -> str:
    return ErrorReason.Name(reason).removeprefix(_REASON_PREFIX)
