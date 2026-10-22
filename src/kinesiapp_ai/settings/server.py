from pydantic import Field
from pydantic_settings import SettingsConfigDict

from kinesiapp_ai.settings.base import ENV_FILE, EnvSettings


class GrpcServerSettings(EnvSettings):
    """Configuro el servidor gRPC con las variables GRPC_*."""

    model_config = SettingsConfigDict(env_prefix="GRPC_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # Escucho solo en localhost porque lo único expuesto a internet es Envoy, en el mismo contenedor
    HOST: str = Field(default="127.0.0.1", min_length=1)
    PORT: int = Field(default=50051, gt=0, le=65535)
    SHUTDOWN_GRACE_SECONDS: float = Field(default=5.0, gt=0)
