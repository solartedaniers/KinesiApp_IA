"""Riesgo por patrones (§5.2 y §5.6): rampas, estrategias y RiskScoreAggregator."""
import math

import pytest

from kinesiapp_ai.analysis.movement_windows import DetectionMethod, MovementWindow
from kinesiapp_ai.analysis.risk import KneeFlexionRiskStrategy, LinearRamp, RiskScoreAggregator, TrunkFlexionRiskStrategy
from tests.pose_fixtures import angle_series

WHOLE = MovementWindow(0, 2, DetectionMethod.LANDING)


def test_linear_ramp_rises_or_falls_and_clamps():
    rising, falling = LinearRamp(40, 60), LinearRamp(60, 30)
    assert [rising.score(v) for v in (30, 50, 70)] == [0, 0.5, 1]
    assert [falling.score(v) for v in (70, 45, 20)] == [0, 0.5, 1]
    assert math.isnan(rising.score(math.nan))


def test_strategies_score_the_peak_inside_the_window_only():
    angles = angle_series([90, 20, 30, 40, 90], [0, 10, 50, 10, 0])
    window = MovementWindow(1, 3, DetectionMethod.LANDING)
    knee = KneeFlexionRiskStrategy({"deep": LinearRamp(0, 80)}).score(angles, window)["deep"]
    trunk = TrunkFlexionRiskStrategy({"lean": LinearRamp(100, 0)}).score(angles, window)["lean"]
    assert (knee.signal, knee.measured_deg, knee.score, knee.higher_is_riskier) == ("knee_flexion", 40, 0.5, True)
    assert (trunk.signal, trunk.measured_deg, trunk.score, trunk.higher_is_riskier) == ("trunk_inclination", 50, 0.5, False)
    assert (trunk.onset_deg, trunk.saturation_deg) == (100, 0)


def test_pattern_needs_all_its_partials_and_uses_the_worst_pattern():
    aggregator = RiskScoreAggregator(
        strategies=[
            KneeFlexionRiskStrategy({"knee_deep": LinearRamp(80, 100)}),
            TrunkFlexionRiskStrategy({"trunk_lean": LinearRamp(25, 45)}),
        ],
        patterns={"forward_collapse": ["knee_deep", "trunk_lean"]},
    )
    deep_upright = aggregator.aggregate(angle_series([110] * 3, [5] * 3), [WHOLE])
    assert (deep_upright.risk_score, deep_upright.dominant_pattern) == (0, None)
    collapse = aggregator.aggregate(angle_series([110] * 3, [60] * 3), [WHOLE])
    assert (collapse.risk_score, collapse.dominant_pattern) == (1, "forward_collapse")


def test_repetitions_are_combined_with_the_median():
    aggregator = RiskScoreAggregator(
        [KneeFlexionRiskStrategy({"knee_deep": LinearRamp(0, 100)})], {"deep": ["knee_deep"]}
    )
    angles = angle_series([10, 90, 30], [0, 0, 0])
    windows = [MovementWindow(frame, frame, DetectionMethod.LANDING) for frame in range(3)]
    assert aggregator.aggregate(angles, windows).risk_score == pytest.approx(0.3)


def test_pattern_with_unknown_partial_is_a_configuration_error():
    with pytest.raises(ValueError):
        RiskScoreAggregator([KneeFlexionRiskStrategy({"knee_deep": LinearRamp(0, 1)})], {"p": ["trunk_lean"]})
