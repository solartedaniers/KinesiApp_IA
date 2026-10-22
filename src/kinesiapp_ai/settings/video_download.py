from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from kinesiapp_ai.settings.base import ENV_FILE, EnvSettings


class VideoDownloadSettings(EnvSettings):
    """Configuro desde dónde y cómo descargo los videos, con las variables VIDEO_DOWNLOAD_*."""

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_DOWNLOAD_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # Sin valor por defecto: si no sé cuál es el host de Neon Storage, no arranco en vez de descargar de
    # cualquier lado. Va en JSON, por ejemplo ["xxx.storage.c-6.eu-central-1.aws.neon.tech"]
    ALLOWED_HOSTS: frozenset[str] = Field(min_length=1)
    # Es el mismo tope de subida que aplica el backend
    MAX_BYTES: int = Field(default=100 * 1024 * 1024, gt=0)
    CHUNK_BYTES: int = Field(default=1024 * 1024, gt=0)
    CONNECT_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0)
    # Cubre la descarga completa, no solo cada lectura: un servidor lento no me retiene más que esto
    TOTAL_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0)
    MAX_REDIRECTS: int = Field(default=3, ge=0)

    @field_validator("ALLOWED_HOSTS")
    @classmethod
    def _normalize_hosts(cls, hosts: frozenset[str]) -> frozenset[str]:
        normalized = frozenset(host.strip().lower() for host in hosts)
        if "" in normalized:
            raise ValueError("ALLOWED_HOSTS cannot contain empty hosts")
        return normalized
