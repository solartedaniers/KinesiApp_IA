from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternText:
    meaning: str
    coaching_focus: str


# Textos curados del diseño del pipeline. Son placeholders sin revisión
# profesional, igual que los umbrales de riesgo: revisarlo antes de mostrarlo a usuarios reales
PATTERN_TEXTS = {
    "rigid_landing": PatternText(
        meaning=(
            "Aterrizaje con poca flexión de rodilla en los primeros 300 ms tras el contacto: el impacto "
            "se absorbe poco."
        ),
        coaching_focus="Aterrizar \"suave\": doblar más rodillas y cadera al tocar el suelo y alargar la amortiguación.",
    ),
    "forward_collapse": PatternText(
        meaning=(
            "Aterrizaje con mucha flexión de rodilla mientras el tronco cae hacia adelante: la bajada no "
            "está controlada."
        ),
        coaching_focus="Frenar la bajada de forma controlada y mantener el pecho más erguido al recibir el peso.",
    ),
    "trunk_lean": PatternText(
        meaning="Inclinación excesiva del tronco hacia adelante al aterrizar, con cualquier flexión de rodilla.",
        coaching_focus="Mantener el tronco más vertical durante la recepción.",
    ),
    "hip_hinge_squat": PatternText(
        meaning=(
            "El cuerpo baja doblándose por la cadera en vez de por la rodilla: tronco muy inclinado con la "
            "rodilla poco flexionada. Es una compensación que tiende a cargar la zona lumbar."
        ),
        coaching_focus="Bajar llevando las rodillas hacia adelante y la cadera hacia abajo, con el tronco más erguido.",
    ),
    "knee_valgus_left": PatternText(
        meaning=(
            "Al recibir el peso, la rodilla izquierda se va hacia adentro (valgo dinámico): vistos de frente, "
            "el muslo y la pierna dejan de estar alineados."
        ),
        coaching_focus="Mantener la rodilla izquierda en línea con la punta del pie al aterrizar o al bajar.",
    ),
    "knee_valgus_right": PatternText(
        meaning=(
            "Al recibir el peso, la rodilla derecha se va hacia adentro (valgo dinámico): vistos de frente, "
            "el muslo y la pierna dejan de estar alineados."
        ),
        coaching_focus="Mantener la rodilla derecha en línea con la punta del pie al aterrizar o al bajar.",
    ),
    "valgus_asymmetry": PatternText(
        meaning="En la misma repetición una rodilla se va hacia adentro bastante más que la otra: las piernas no reciben el peso igual.",
        coaching_focus="Repartir el peso por igual entre las dos piernas y cuidar la alineación de la rodilla que más se desvía.",
    ),
}

NO_PATTERN_TEXT = PatternText(
    meaning="Ningún patrón de riesgo superó su umbral en las repeticiones evaluadas.",
    coaching_focus="Mantener la técnica; no hay un foco de corrección derivado de estos datos.",
)

SIGNAL_DESCRIPTIONS = {
    "knee_rigid": "flexión máxima de rodilla en los 300 ms posteriores al contacto (0° = pierna recta)",
    "knee_deep": "flexión máxima de rodilla en los 300 ms posteriores al contacto (0° = pierna recta)",
    "trunk_lean": (
        "inclinación máxima del tronco respecto de la vertical en los 300 ms posteriores al contacto "
        "(0° = erguido, 90° = horizontal)"
    ),
    "squat_knee_shallow": "flexión máxima de rodilla en el fondo de la repetición (0° = pierna recta)",
    "squat_trunk_lean": (
        "inclinación máxima del tronco respecto de la vertical en el fondo de la repetición "
        "(0° = erguido, 90° = horizontal)"
    ),
    "knee_valgus_left": (
        "ángulo de proyección frontal (FPPA) máximo de la rodilla izquierda en el aterrizaje o en el fondo "
        "(0° = muslo y pierna alineados, positivo = rodilla hacia adentro, negativo = hacia afuera)"
    ),
    "knee_valgus_right": (
        "ángulo de proyección frontal (FPPA) máximo de la rodilla derecha en el aterrizaje o en el fondo "
        "(0° = muslo y pierna alineados, positivo = rodilla hacia adentro, negativo = hacia afuera)"
    ),
    "knee_valgus_asymmetry": "diferencia entre el FPPA máximo de las dos rodillas en la misma repetición",
}

_COMMON_LIMITS = (
    "No se mide la curvatura de la espalda: el tronco se mide como la recta entre cadera y hombro.",
    "Los umbrales son valores de demostración sin validación clínica.",
)
MEASUREMENT_LIMITS = {
    "side": (
        "Vista lateral: no se evalúa el valgo de rodilla (la rodilla hacia adentro) ni la asimetría entre "
        "piernas; para eso hay que grabar de frente.",
        *_COMMON_LIMITS,
    ),
    "front": (
        "Vista frontal: se mide el valgo de cada rodilla y la diferencia entre piernas; la flexión de rodilla "
        "y la inclinación del tronco no se evalúan, se miden grabando de lado.",
        "Los umbrales son valores de demostración sin validación clínica.",
    ),
    "inconclusive": (
        "La cámara no estaba claramente de lado ni de frente: la flexión de rodilla y la inclinación del tronco "
        "son aproximadas y el valgo de rodilla no se midió. Para medir si la rodilla se va hacia adentro hay "
        "que grabar de frente.",
        *_COMMON_LIMITS,
    ),
}


class RiskPatternCatalog:
    """Textos fijos por patrón y por señal. Falla al construirse si la configuración de riesgo usa
    un código sin texto: un patrón nuevo no puede llegar al chat sin su explicación curada."""

    def __init__(self, configured_patterns: Iterable[dict[str, list[str]]]) -> None:
        for patterns in configured_patterns:
            for pattern, partials in patterns.items():
                missing = ({pattern} - PATTERN_TEXTS.keys()) | (set(partials) - SIGNAL_DESCRIPTIONS.keys())
                if missing:
                    raise ValueError(f"Risk codes without chat text: {sorted(missing)}")

    @staticmethod
    def pattern(code: str | None) -> PatternText | None:
        return PATTERN_TEXTS.get(code) if code else None

    @staticmethod
    def no_pattern() -> PatternText:
        return NO_PATTERN_TEXT

    @staticmethod
    def signal_description(code: str) -> str:
        return SIGNAL_DESCRIPTIONS[code]

    @staticmethod
    def measurement_limits(camera_view: str) -> list[str]:
        return list(MEASUREMENT_LIMITS[camera_view])
