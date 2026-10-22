"""Pruebo los cortes del nivel de riesgo."""

import pytest

from kinesiapp_ai.analysis.risk_level import RiskLevel, RiskLevelClassifier

CLASSIFIER = RiskLevelClassifier(moderate_from=0.33, high_from=0.66)


@pytest.mark.parametrize(
    ("score", "level"),
    [(0.0, RiskLevel.LOW), (0.329, RiskLevel.LOW), (0.33, RiskLevel.MODERATE), (0.659, RiskLevel.MODERATE), (0.66, RiskLevel.HIGH), (1.0, RiskLevel.HIGH)],
)
def test_each_cut_starts_its_level(score: float, level: RiskLevel) -> None:
    assert CLASSIFIER.classify(score) == level


@pytest.mark.parametrize(("moderate", "high"), [(0.7, 0.6), (0.5, 0.5), (-0.1, 0.5), (0.3, 1.2)])
def test_cuts_out_of_order_or_range_are_rejected(moderate: float, high: float) -> None:
    with pytest.raises(ValueError):
        RiskLevelClassifier(moderate, high)
