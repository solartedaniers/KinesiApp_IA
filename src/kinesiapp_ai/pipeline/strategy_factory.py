from kinesiapp_ai.analysis.jump_phases import JumpPhaseDetector
from kinesiapp_ai.analysis.movement_strategy import Movement, MovementStrategy
from kinesiapp_ai.analysis.risk import KneeFlexionRiskStrategy, LinearRamp, RiskScoreAggregator, TrunkFlexionRiskStrategy
from kinesiapp_ai.analysis.squat_phases import SquatBottomDetector
from kinesiapp_ai.analysis.trunk_hinge import FallbackWindowDetector, TrunkHingeDetector
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.risk import RiskSettings


class MovementStrategyFactory:
    """Factory: armo la estrategia de cada movimiento a partir de la configuración de fases y de riesgo."""

    def __init__(self, phases: MovementPhaseSettings, risk: RiskSettings) -> None:
        self._phases = phases
        self._risk = risk

    def build_all(self) -> dict[Movement, MovementStrategy]:
        # Las armo todas al arrancar: si un patrón pide un parcial que no existe, fallo antes del primer video
        return {Movement.JUMP: self._jump(), Movement.SQUAT: self._squat()}

    def _jump(self) -> MovementStrategy:
        phases, risk = self._phases, self._risk
        return MovementStrategy(
            window_detector=JumpPhaseDetector(
                hip_apex_prominence=phases.JUMP_HIP_APEX_PROMINENCE,
                hip_apex_min_distance_seconds=phases.JUMP_HIP_APEX_MIN_DISTANCE_SECONDS,
                apex_airborne_body_fraction=phases.JUMP_APEX_AIRBORNE_BODY_FRACTION,
                airborne_body_fraction=phases.JUMP_AIRBORNE_BODY_FRACTION,
                ground_window_seconds=phases.JUMP_GROUND_WINDOW_SECONDS,
                ground_percentile=phases.JUMP_GROUND_PERCENTILE,
                landing_window_ms=phases.JUMP_LANDING_WINDOW_MS,
                landing_end_margin_frames=phases.JUMP_LANDING_END_MARGIN_FRAMES,
            ),
            aggregator=RiskScoreAggregator(
                strategies=[
                    KneeFlexionRiskStrategy({
                        "knee_rigid": LinearRamp(risk.RISK_JUMP_KNEE_RIGID_ONSET_DEG, risk.RISK_JUMP_KNEE_RIGID_SATURATION_DEG),
                        "knee_deep": LinearRamp(risk.RISK_JUMP_KNEE_DEEP_ONSET_DEG, risk.RISK_JUMP_KNEE_DEEP_SATURATION_DEG),
                    }),
                    TrunkFlexionRiskStrategy({
                        "trunk_lean": LinearRamp(risk.RISK_JUMP_TRUNK_LEAN_ONSET_DEG, risk.RISK_JUMP_TRUNK_LEAN_SATURATION_DEG),
                    }),
                ],
                patterns=risk.RISK_JUMP_PATTERNS,
            ),
        )

    def _squat(self) -> MovementStrategy:
        phases, risk = self._phases, self._risk
        return MovementStrategy(
            window_detector=FallbackWindowDetector([
                SquatBottomDetector(
                    min_prominence_deg=phases.SQUAT_BOTTOM_MIN_PROMINENCE_DEG,
                    min_flexion_deg=phases.SQUAT_BOTTOM_MIN_FLEXION_DEG,
                    min_distance_seconds=phases.SQUAT_BOTTOM_MIN_DISTANCE_SECONDS,
                    window_ms=phases.SQUAT_BOTTOM_WINDOW_MS,
                ),
                TrunkHingeDetector(
                    min_trunk_inclination_deg=risk.RISK_SQUAT_TRUNK_LEAN_ONSET_DEG,
                    min_duration_seconds=phases.SQUAT_HINGE_MIN_DURATION_SECONDS,
                    window_ms=phases.SQUAT_BOTTOM_WINDOW_MS,
                ),
            ]),
            aggregator=RiskScoreAggregator(
                strategies=[
                    KneeFlexionRiskStrategy({
                        "squat_knee_shallow": LinearRamp(
                            risk.RISK_SQUAT_KNEE_SHALLOW_ONSET_DEG, risk.RISK_SQUAT_KNEE_SHALLOW_SATURATION_DEG
                        ),
                    }),
                    TrunkFlexionRiskStrategy({
                        "squat_trunk_lean": LinearRamp(risk.RISK_SQUAT_TRUNK_LEAN_ONSET_DEG, risk.RISK_SQUAT_TRUNK_LEAN_SATURATION_DEG),
                    }),
                ],
                patterns=risk.RISK_SQUAT_PATTERNS,
            ),
        )
