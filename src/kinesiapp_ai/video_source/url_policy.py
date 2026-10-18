from collections.abc import Iterable
from urllib.parse import urlsplit

from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.rpc.domain_error import DomainError

_HTTPS_PORT = 443


class AllowedHostPolicy:
    """Solo dejo descargar por https desde los hosts de la lista, para que nadie use la IA contra otros servidores."""

    def __init__(self, allowed_hosts: Iterable[str]) -> None:
        self._allowed_hosts = frozenset(host.lower() for host in allowed_hosts)

    def check(self, url: str) -> None:
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError as error:
            raise _rejected("Malformed video URL") from error
        if parts.scheme != "https":
            raise _rejected("Video URL must use https")
        # Rechazo usuario y contraseña en la URL: "https://host-permitido@otro" engaña a parsers distintos
        if parts.username is not None or parts.password is not None:
            raise _rejected("Video URL cannot carry credentials")
        if parts.hostname not in self._allowed_hosts or port not in (None, _HTTPS_PORT):
            raise _rejected("Video host is not allowed")


def _rejected(message: str) -> DomainError:
    return DomainError(ErrorReason.ERROR_REASON_BAD_REQUEST, message)
