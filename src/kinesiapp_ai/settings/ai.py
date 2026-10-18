from dataclasses import dataclass

from kinesiapp_ai.settings.analysis import AnalysisSettings
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.pose import PoseSettings
from kinesiapp_ai.settings.risk import RiskSettings
from kinesiapp_ai.settings.server import GrpcServerSettings
from kinesiapp_ai.settings.service_auth import ServiceAuthSettings
from kinesiapp_ai.settings.video_download import VideoDownloadSettings


@dataclass(frozen=True)
class AiSettings:
    """Reúno la configuración de cada dominio de la IA para armar el servidor en un solo paso."""

    server: GrpcServerSettings
    auth: ServiceAuthSettings
    pose: PoseSettings
    analysis: AnalysisSettings
    phases: MovementPhaseSettings
    risk: RiskSettings
    download: VideoDownloadSettings

    @classmethod
    def from_environment(cls) -> "AiSettings":
        # Las leo todas al arrancar: si falta una variable obligatoria, la IA no levanta
        return cls(
            server=GrpcServerSettings(),
            auth=ServiceAuthSettings(),
            pose=PoseSettings(),
            analysis=AnalysisSettings(),
            phases=MovementPhaseSettings(),
            risk=RiskSettings(),
            download=VideoDownloadSettings(),
        )
