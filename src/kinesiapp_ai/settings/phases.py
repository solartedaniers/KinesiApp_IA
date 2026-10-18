from pydantic import Field

from kinesiapp_ai.settings.base import EnvSettings


class MovementPhaseSettings(EnvSettings):
    """Configuro cómo encuentro las repeticiones. Las calibré con un solo par de videos por ejercicio: no son definitivas."""

    JUMP_HIP_APEX_PROMINENCE: float = Field(default=0.08, gt=0)
    JUMP_HIP_APEX_MIN_DISTANCE_SECONDS: float = Field(default=0.6, gt=0)
    JUMP_APEX_AIRBORNE_BODY_FRACTION: float = Field(default=0.06, gt=0)
    JUMP_AIRBORNE_BODY_FRACTION: float = Field(default=0.03, gt=0)
    JUMP_GROUND_WINDOW_SECONDS: float = Field(default=1.0, gt=0)
    JUMP_GROUND_PERCENTILE: float = Field(default=90.0, ge=0, le=100)
    JUMP_LANDING_WINDOW_MS: int = Field(default=300, gt=0)
    JUMP_LANDING_END_MARGIN_FRAMES: int = Field(default=3, ge=0)
    SQUAT_BOTTOM_MIN_PROMINENCE_DEG: float = Field(default=25.0, gt=0)
    SQUAT_BOTTOM_MIN_FLEXION_DEG: float = Field(default=50.0, ge=0)
    SQUAT_BOTTOM_MIN_DISTANCE_SECONDS: float = Field(default=0.6, gt=0)
    SQUAT_BOTTOM_WINDOW_MS: int = Field(default=200, ge=0)
    # Uso este respaldo cuando la rodilla no deja picos (bisagra de cadera con piernas casi rectas). Como
    # umbral de tronco tomo RISK_SQUAT_TRUNK_LEAN_ONSET_DEG, el mismo donde empieza el riesgo
    SQUAT_HINGE_MIN_DURATION_SECONDS: float = Field(default=0.3, gt=0)
