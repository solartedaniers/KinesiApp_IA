from pydantic import Field, SecretStr

from kinesiapp_ai.settings.base import EnvSettings

# Lo exijo largo porque es lo único que separa a la IA, que es pública, de cualquiera que la llame
MIN_SERVICE_TOKEN_LENGTH = 32


class ServiceAuthSettings(EnvSettings):
    """Guardo el secreto que comparto con el backend. No tiene valor por defecto: sin él, la IA no arranca."""

    AI_SERVICE_TOKEN: SecretStr = Field(min_length=MIN_SERVICE_TOKEN_LENGTH)
