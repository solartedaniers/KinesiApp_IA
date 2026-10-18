"""Pruebo cómo leo la configuración del entorno: vacíos, secretos y listas de hosts."""

import pytest
from pydantic import ValidationError

from kinesiapp_ai.settings.analysis import AnalysisSettings
from kinesiapp_ai.settings.assistant import AssistantSettings
from kinesiapp_ai.settings.service_auth import ServiceAuthSettings
from kinesiapp_ai.settings.video_download import VideoDownloadSettings

SECRET = "a-secret-that-must-never-reach-the-logs-0123"


def test_an_empty_option_falls_back_to_its_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYSIS_MAX_FRAME_SIDE_PX", "")
    monkeypatch.setenv("ANALYSIS_MAX_CONCURRENCY", "")

    settings = AnalysisSettings(_env_file=None)

    assert (settings.MAX_FRAME_SIDE_PX, settings.MAX_CONCURRENCY) == (None, 1)


def test_an_empty_service_token_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_SERVICE_TOKEN", "")

    with pytest.raises(ValidationError, match="Field required"):
        ServiceAuthSettings(_env_file=None)


def test_validation_errors_never_show_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_SERVICE_TOKEN", SECRET[:20])

    with pytest.raises(ValidationError) as raised:
        ServiceAuthSettings(_env_file=None)

    assert SECRET[:8] not in str(raised.value)


def test_allowed_hosts_come_as_json_and_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDEO_DOWNLOAD_ALLOWED_HOSTS", '[" Bucket.Storage.Neon.Tech "]')

    assert VideoDownloadSettings(_env_file=None).ALLOWED_HOSTS == frozenset({"bucket.storage.neon.tech"})


def test_without_allowed_hosts_the_ai_does_not_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDEO_DOWNLOAD_ALLOWED_HOSTS", raising=False)

    with pytest.raises(ValidationError, match="ALLOWED_HOSTS"):
        VideoDownloadSettings(_env_file=None)


def test_the_fallback_model_is_optional_and_empty_means_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")

    assert AssistantSettings(_env_file=None).GEMINI_FALLBACK_MODEL is None
