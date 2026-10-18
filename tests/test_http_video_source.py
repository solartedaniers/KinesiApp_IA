"""Pruebo la descarga del video contra un transporte falso: límites, redirecciones y fallos del bucket."""

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.rpc.domain_error import DomainError
from kinesiapp_ai.settings.video_download import VideoDownloadSettings
from kinesiapp_ai.video_source.http_video_source import HttpVideoSource
from kinesiapp_ai.video_source.url_policy import AllowedHostPolicy

HOST = "storage.neon.test"
VIDEO_URL = f"https://{HOST}/videos/a.mp4?X-Amz-Signature=abc"
SETTINGS = VideoDownloadSettings(ALLOWED_HOSTS=[HOST], MAX_BYTES=100, CHUNK_BYTES=10, TOTAL_TIMEOUT_SECONDS=1.0)


def _download(handler: Callable[[httpx.Request], httpx.Response], destination: Path, url: str = VIDEO_URL) -> list[str]:
    requested: list[str] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return handler(request)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(recording)) as client:
            await HttpVideoSource(client, AllowedHostPolicy(SETTINGS.ALLOWED_HOSTS), SETTINGS).download(url, destination)

    asyncio.run(main())
    return requested


def _reason_of(handler, destination: Path, url: str = VIDEO_URL) -> ErrorReason.ValueType:
    with pytest.raises(DomainError) as raised:
        _download(handler, destination, url)
    return raised.value.reason


class _CountingBody(httpx.AsyncByteStream):
    """Entrego el cuerpo por partes y cuento cuántas pidió el cliente, para ver si cortó a tiempo."""

    def __init__(self, chunks: int, size: int = 10, delay: float = 0.0) -> None:
        self.chunks, self.size, self.delay, self.sent = chunks, size, delay, 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.chunks):
            await asyncio.sleep(self.delay)
            self.sent += 1
            yield b"v" * self.size


def test_downloads_the_signed_video_to_the_destination(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"

    requested = _download(lambda request: httpx.Response(200, content=b"video-bytes"), destination)

    assert destination.read_bytes() == b"video-bytes"
    assert requested == [VIDEO_URL]


def test_a_disallowed_url_is_rejected_without_any_request(tmp_path: Path) -> None:
    requested: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request)
        return httpx.Response(200)

    reason = _reason_of(handler, tmp_path / "video.mp4", "https://169.254.169.254/latest/meta-data")

    assert reason == ErrorReason.ERROR_REASON_BAD_REQUEST
    assert requested == []


def test_a_declared_size_over_the_limit_stops_before_reading_the_body(tmp_path: Path) -> None:
    body = _CountingBody(chunks=20)

    reason = _reason_of(lambda request: httpx.Response(200, headers={"content-length": "200"}, stream=body), tmp_path / "v.mp4")

    assert reason == ErrorReason.ERROR_REASON_VIDEO_TOO_LARGE
    assert body.sent == 0


def test_an_undeclared_oversized_body_is_cut_as_soon_as_it_passes_the_limit(tmp_path: Path) -> None:
    body = _CountingBody(chunks=1000)

    reason = _reason_of(lambda request: httpx.Response(200, stream=body), tmp_path / "v.mp4")

    assert reason == ErrorReason.ERROR_REASON_VIDEO_TOO_LARGE
    # Con un tope de 100 bytes en partes de 10, corto en la parte 11 y no bajo las otras 989
    assert body.sent == 11


def test_redirects_inside_the_bucket_are_followed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/videos/a.mp4":
            return httpx.Response(307, headers={"location": "/videos/moved.mp4"})
        return httpx.Response(200, content=b"moved")

    requested = _download(handler, tmp_path / "v.mp4")

    assert (tmp_path / "v.mp4").read_bytes() == b"moved"
    assert requested[-1] == f"https://{HOST}/videos/moved.mp4"


def test_a_redirect_to_another_host_is_rejected_before_following_it(tmp_path: Path) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://evil.test/steal"})

    reason = _reason_of(handler, tmp_path / "v.mp4")

    assert reason == ErrorReason.ERROR_REASON_BAD_REQUEST
    assert requested == [HOST]


def test_a_redirect_loop_ends_as_bad_request(tmp_path: Path) -> None:
    reason = _reason_of(lambda request: httpx.Response(302, headers={"location": VIDEO_URL}), tmp_path / "v.mp4")

    assert reason == ErrorReason.ERROR_REASON_BAD_REQUEST


@pytest.mark.parametrize("status", [403, 404, 500])
def test_bucket_errors_are_storage_unavailable(tmp_path: Path, status: int) -> None:
    assert _reason_of(lambda request: httpx.Response(status), tmp_path / "v.mp4") == ErrorReason.ERROR_REASON_STORAGE_UNAVAILABLE


def test_network_failures_are_storage_unavailable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert _reason_of(handler, tmp_path / "v.mp4") == ErrorReason.ERROR_REASON_STORAGE_UNAVAILABLE


def test_a_download_slower_than_the_total_timeout_is_cut(tmp_path: Path) -> None:
    body = _CountingBody(chunks=5, size=1, delay=0.4)

    reason = _reason_of(lambda request: httpx.Response(200, stream=body), tmp_path / "v.mp4")

    assert reason == ErrorReason.ERROR_REASON_STORAGE_UNAVAILABLE
    assert body.sent < 5
