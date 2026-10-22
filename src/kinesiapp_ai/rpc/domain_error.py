from kinesiapp.common.v1.errors_pb2 import ErrorReason


class DomainError(Exception):
    """Lo lanzo con el motivo del contrato y el interceptor de errores lo traduce a un status gRPC."""

    def __init__(self, reason: ErrorReason.ValueType, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
