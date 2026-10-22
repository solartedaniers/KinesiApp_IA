from collections.abc import Iterator
from contextlib import contextmanager

from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.rpc.domain_error import DomainError


class AnalysisSlots:
    """Limito los análisis simultáneos y rechazo al instante el que no cabe, en vez de encolarlo sin control.

    No uso un lock porque todo el que reserva corre en el mismo event loop.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._in_use = 0

    @property
    def in_use(self) -> int:
        return self._in_use

    @contextmanager
    def reserve(self) -> Iterator[None]:
        if self._in_use >= self._capacity:
            raise DomainError(ErrorReason.ERROR_REASON_AI_BUSY, "The AI is already running its maximum analyses")
        self._in_use += 1
        try:
            yield
        finally:
            self._in_use -= 1
