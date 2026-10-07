import json
from dataclasses import dataclass
from typing import Any, Literal

from kinesiapp_ai.analysis.risk_details import RISK_DETAILS_FORMAT_VERSION
from kinesiapp_ai.chat.llm import LlmRequest, LlmTurn
from kinesiapp_ai.chat.pattern_catalog import RiskPatternCatalog
from kinesiapp_ai.chat.system_prompt import OPENING_REQUEST, SYSTEM_PROMPT_TEMPLATE

HistoryRole = Literal["user", "assistant"]
ASSISTANT_ROLE: HistoryRole = "assistant"


@dataclass(frozen=True)
class AnalysisPromptInput:
    """Lo que el chat necesita de un análisis procesado, sin saber cómo se guarda."""

    movement: str
    risk_score: float
    dominant_risk_pattern: str | None
    risk_details: dict[str, Any] | None


@dataclass(frozen=True)
class HistoryMessage:
    role: HistoryRole
    content: str


class JumpAnalysisPromptBuilder:
    """Análisis + rol + historial → LlmRequest (docs/design/video-analysis-pipeline.md §6.3-§6.5).

    Sólo usa datos estructurados del análisis: nunca el video, nombres, email ni fecha de
    nacimiento. Los análisis sin risk_details (anteriores a que existiera) van en modo reducido.
    """

    def __init__(
        self,
        catalog: RiskPatternCatalog,
        risk_level_moderate: float,
        risk_level_high: float,
        max_repetitions: int,
        max_history_messages: int,
    ) -> None:
        self._catalog = catalog
        self._risk_level_moderate = risk_level_moderate
        self._risk_level_high = risk_level_high
        self._max_repetitions = max_repetitions
        self._max_history_messages = max_history_messages

    def build(
        self, analysis: AnalysisPromptInput, audience: str, history: list[HistoryMessage], user_message: str
    ) -> LlmRequest:
        data = self.analysis_data(analysis, audience)
        system_instruction = SYSTEM_PROMPT_TEMPLATE.replace(
            "{analysis_json}", json.dumps(data, ensure_ascii=False, indent=2)
        )
        return LlmRequest(
            system_instruction=system_instruction,
            turns=[*self._history_turns(history), LlmTurn(role="user", text=user_message)],
        )

    def build_opening(self, analysis: AnalysisPromptInput, audience: str) -> LlmRequest:
        """Primera explicación del resultado: mismos datos y mismo system prompt que el chat."""
        return self.build(analysis, audience, [], OPENING_REQUEST)

    def analysis_data(self, analysis: AnalysisPromptInput, audience: str) -> dict[str, Any]:
        details = analysis.risk_details
        has_details = isinstance(details, dict) and details.get("format_version") == RISK_DETAILS_FORMAT_VERSION
        data: dict[str, Any] = {
            "audience": audience,
            "movement": analysis.movement,
            "risk_score": analysis.risk_score,
            "risk_level": self._risk_level(analysis.risk_score),
            "details_available": has_details,
        }
        if has_details:
            data.update(self._detailed(details))
        else:
            data.update(self._reduced(analysis))
        data["measurement_limits"] = self._catalog.measurement_limits()
        return data

    def _risk_level(self, risk_score: float) -> str:
        if risk_score >= self._risk_level_high:
            return "high"
        if risk_score >= self._risk_level_moderate:
            return "moderate"
        return "low"

    def _reduced(self, analysis: AnalysisPromptInput) -> dict[str, Any]:
        text = self._catalog.pattern(analysis.dominant_risk_pattern)
        if text is None:
            # Sin patrón conocido: con score 0 es "ningún patrón"; con score > 0 es un análisis
            # simulado anterior a la Fase 1 de IA, del que no se sabe qué patrón hubo
            return {"dominant_pattern": None, **self._no_pattern_if(analysis.risk_score == 0)}
        return {"dominant_pattern": self._pattern_block(analysis.dominant_risk_pattern, analysis.risk_score, text)}

    def _detailed(self, details: dict[str, Any]) -> dict[str, Any]:
        dominant = details["dominant_pattern"]
        patterns: dict[str, Any] = details["patterns"]
        dominant_text = self._catalog.pattern(dominant)
        signal_codes = self._signal_codes(dominant, patterns)
        return {
            "dominant_pattern": (
                self._pattern_block(dominant, patterns[dominant]["score"], dominant_text) if dominant_text else None
            ),
            **self._no_pattern_if(dominant_text is None),
            "other_patterns": [
                {"code": code, "score": pattern["score"], "meaning": self._catalog.pattern(code).meaning}
                for code, pattern in patterns.items()
                if code != dominant and self._catalog.pattern(code)
            ],
            "repetitions": {
                "evaluated": details["repetitions_evaluated"],
                "detection_methods": details["detection_methods"],
                "triggering_dominant_pattern": patterns[dominant]["repetitions_triggered"] if dominant else 0,
            },
            "signals": [self._signal_block(code, patterns) for code in signal_codes],
            "per_repetition": self._per_repetition(details["repetitions"], dominant, signal_codes),
        }

    def _no_pattern_if(self, condition: bool) -> dict[str, Any]:
        if not condition:
            return {}
        text = self._catalog.no_pattern()
        return {"no_pattern": {"meaning": text.meaning, "coaching_focus": text.coaching_focus}}

    @staticmethod
    def _pattern_block(code: str, score: float, text) -> dict[str, Any]:
        return {"code": code, "score": score, "meaning": text.meaning, "coaching_focus": text.coaching_focus}

    @staticmethod
    def _signal_codes(dominant: str | None, patterns: dict[str, Any]) -> list[str]:
        # Con patrón dominante, sus señales; sin él, todas, para explicar por qué no hubo riesgo
        sources = [patterns[dominant]] if dominant else patterns.values()
        return list(dict.fromkeys(code for pattern in sources for code in pattern["signals"]))

    def _signal_block(self, code: str, patterns: dict[str, Any]) -> dict[str, Any]:
        signal = next(pattern["signals"][code] for pattern in patterns.values() if code in pattern["signals"])
        return {
            "code": code,
            "description": self._catalog.signal_description(code),
            "direction": signal["direction"],
            "onset_deg": signal["onset_deg"],
            "saturation_deg": signal["saturation_deg"],
            "measured_deg": signal["measured_deg"],
            "partial_score_median": signal["score_median"],
        }

    def _per_repetition(
        self, repetitions: list[dict[str, Any]], dominant: str | None, signal_codes: list[str]
    ) -> list[dict[str, Any]]:
        # Las de mayor score del patrón dominante, mostradas en orden temporal
        if dominant:
            repetitions = sorted(repetitions, key=lambda rep: rep["pattern_scores"].get(dominant) or 0, reverse=True)
        selected = sorted(repetitions[: self._max_repetitions], key=lambda rep: rep["start_ms"])
        return [
            {
                "time_s": round(rep["start_ms"] / 1000, 2),
                "detected_by": rep["detected_by"],
                **{code: rep["partials"][code] for code in signal_codes if code in rep["partials"]},
            }
            for rep in selected
        ]

    def _history_turns(self, history: list[HistoryMessage]) -> list[LlmTurn]:
        # Un mensaje de usuario que quedó sin respuesta (falló el proveedor) no se reenvía
        answered = [
            message
            for index, message in enumerate(history)
            if message.role == ASSISTANT_ROLE
            or (index + 1 < len(history) and history[index + 1].role == ASSISTANT_ROLE)
        ]
        recent = answered[-self._max_history_messages:] if self._max_history_messages else []
        # El historial debe empezar con un turno de usuario. Si empieza con la explicación inicial
        # (primer mensaje del hilo), se antepone el pedido que la generó para no perderla; si es una
        # respuesta cuya pregunta quedó fuera de la ventana, se descarta como antes
        opening: list[LlmTurn] = []
        if recent and recent[0].role == ASSISTANT_ROLE:
            if recent[0] is history[0]:
                opening = [LlmTurn(role="user", text=OPENING_REQUEST)]
            else:
                recent = recent[1:]
        return opening + [
            LlmTurn(role="model" if message.role == ASSISTANT_ROLE else "user", text=message.content)
            for message in recent
        ]
