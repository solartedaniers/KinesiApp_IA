from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Leo el .env de la raíz de KinesiApp_IA: cada repo tiene el suyo
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class EnvSettings(BaseSettings):
    """Leo las variables de entorno y el .env; la configuración de cada dominio hereda de aquí."""

    # Oculto los valores en los errores de validación: si no, al faltar una variable el log mostraría el
    # inicio de los secretos del .env. Tomo una variable vacía como no definida, así "AI_SERVICE_TOKEN="
    # cuenta como faltante y una opción vacía vuelve a su valor por defecto
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True, env_ignore_empty=True
    )
