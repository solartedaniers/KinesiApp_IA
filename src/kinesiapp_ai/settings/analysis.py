from pydantic import Field
from pydantic_settings import SettingsConfigDict

from kinesiapp_ai.settings.base import ENV_FILE, EnvSettings


class AnalysisSettings(EnvSettings):
    """Configuro los límites de cada análisis con las variables ANALYSIS_*."""

    model_config = SettingsConfigDict(
        env_prefix="ANALYSIS_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # Dejo uno por defecto porque con 512 MB dos análisis a la vez tumban el servicio
    MAX_CONCURRENCY: int = Field(default=1, ge=1)
    # Es el mismo tope que valida el navegador; lo vuelvo a verificar porque los metadatos pueden mentir
    MAX_VIDEO_DURATION_SECONDS: float = Field(default=40.0, gt=0)
    # Si lo defino, reduzco cada frame hasta que su lado mayor mida esto antes de analizarlo. Sin valor
    # analizo a la resolución original
    MAX_FRAME_SIDE_PX: int | None = Field(default=None, ge=160)
    # Reporto el progreso cada tantos frames para no inundar el stream con un mensaje por frame
    PROGRESS_EVERY_FRAMES: int = Field(default=10, ge=1)
