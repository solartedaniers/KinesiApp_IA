import asyncio
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin

import httpx
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.rpc.domain_error import DomainError
from kinesiapp_ai.settings.video_download import VideoDownloadSettings
from kinesiapp_ai.video_source.url_policy import AllowedHostPolicy


class VideoSource(Protocol):
    def validate(self, url: str) -> None:
        """Rechazo la URL sin hacer ninguna petición si no se puede descargar de ella."""
        ...

    async def download(self, url: str, destination: Path) -> None: ...


class HttpVideoSource:
    """Adapter: bajo el video de la URL firmada con httpx, por partes y sin pasar nunca el tope de bytes."""

    def __init__(self, client: httpx.AsyncClient, policy: AllowedHostPolicy, settings: VideoDownloadSettings) -> None:
        self._client = client
        self._policy = policy
        self._settings = settings

    def validate(self, url: str) -> None:
        self._policy.check(url)

    async def download(self, url: str, destination: Path) -> None:
        try:
            async with asyncio.timeout(self._settings.TOTAL_TIMEOUT_SECONDS):
                await self._follow(url, destination)
        except TimeoutError as error:
            raise _storage_unavailable("Video download timed out") from error
        except httpx.HTTPError as error:
            raise _storage_unavailable(f"Video download failed: {type(error).__name__}") from error

    async def _follow(self, url: str, destination: Path) -> None:
        # Sigo las redirecciones a mano para pasar cada destino por la política antes de pedirlo
        for _ in range(self._settings.MAX_REDIRECTS + 1):
            self._policy.check(url)
            async with self._client.stream("GET", url) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                if response.status_code != httpx.codes.OK:
                    raise _storage_unavailable(f"Video download answered {response.status_code}")
                self._reject_declared_size(response)
                await self._save(response, destination)
                return
        raise DomainError(ErrorReason.ERROR_REASON_BAD_REQUEST, "Video URL redirects too many times")

    def _reject_declared_size(self, response: httpx.Response) -> None:
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._settings.MAX_BYTES:
            raise _too_large()

    async def _save(self, response: httpx.Response, destination: Path) -> None:
        received = 0
        with destination.open("wb") as file:
            async for chunk in response.aiter_bytes(self._settings.CHUNK_BYTES):
                received += len(chunk)
                # Cuento lo recibido aunque el servidor haya declarado un tamaño: el encabezado puede mentir
                if received > self._settings.MAX_BYTES:
                    raise _too_large()
                file.write(chunk)


def build_http_client(settings: VideoDownloadSettings) -> httpx.AsyncClient:
    # Ignoro los proxies del entorno para que una variable no desvíe las descargas fuera de la lista
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.TOTAL_TIMEOUT_SECONDS, connect=settings.CONNECT_TIMEOUT_SECONDS),
        follow_redirects=False,
        trust_env=False,
    )


def _storage_unavailable(message: str) -> DomainError:
    return DomainError(ErrorReason.ERROR_REASON_STORAGE_UNAVAILABLE, message)


def _too_large() -> DomainError:
    return DomainError(ErrorReason.ERROR_REASON_VIDEO_TOO_LARGE, "Video exceeds the maximum size")
