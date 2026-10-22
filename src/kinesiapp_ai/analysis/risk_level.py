import enum


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class RiskLevelClassifier:
    """Convierto el puntaje de 0 a 1 en un nivel. Lo hago aquí para que ningún cliente replique los cortes."""

    def __init__(self, moderate_from: float, high_from: float) -> None:
        if not 0 <= moderate_from < high_from <= 1:
            raise ValueError("Risk level cuts must satisfy 0 <= moderate < high <= 1")
        self._moderate_from = moderate_from
        self._high_from = high_from

    def classify(self, risk_score: float) -> RiskLevel:
        if risk_score >= self._high_from:
            return RiskLevel.HIGH
        if risk_score >= self._moderate_from:
            return RiskLevel.MODERATE
        return RiskLevel.LOW
