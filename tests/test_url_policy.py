"""Pruebo que la política de hosts solo deja pasar URLs https del bucket permitido."""

import pytest
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.rpc.domain_error import DomainError
from kinesiapp_ai.video_source.url_policy import AllowedHostPolicy

POLICY = AllowedHostPolicy(["storage.neon.test"])


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.neon.test/videos/a.mp4?X-Amz-Signature=abc",
        "https://STORAGE.neon.test/videos/a.mp4",
        "https://storage.neon.test:443/videos/a.mp4",
    ],
)
def test_signed_bucket_urls_pass(url: str) -> None:
    POLICY.check(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://storage.neon.test/videos/a.mp4",
        "file:///etc/passwd",
        "ftp://storage.neon.test/a.mp4",
        "https://evil.test/videos/a.mp4",
        "https://storage.neon.test.evil.test/a.mp4",
        "https://storage.neon.test@evil.test/a.mp4",
        "https://user:pass@storage.neon.test/a.mp4",
        "https://storage.neon.test:8443/a.mp4",
        "https://169.254.169.254/latest/meta-data",
        "https://storage.neon.test:99999/a.mp4",
        "//storage.neon.test/a.mp4",
        "",
    ],
)
def test_everything_else_is_rejected_as_bad_request(url: str) -> None:
    with pytest.raises(DomainError) as raised:
        POLICY.check(url)
    assert raised.value.reason == ErrorReason.ERROR_REASON_BAD_REQUEST
