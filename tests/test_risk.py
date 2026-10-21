"""Riesgo por patrones: rampas, estrategias y RiskScoreAggregator."""
import math

import pytest

from kinesiapp_ai.analysis.movement_windows import DetectionMethod, MovementWindow
from kinesiapp_ai.analysis.pose_series import Side
from kinesiapp_ai.analysis.risk import (
    KneeFlexionRiskStrategy,
    KneeValgusAsymmetryRiskStrategy,
    KneeValgusRiskStrategy,
    LinearRamp,
    RiskScoreAggregator,
    TrunkFlexionRiskStrategy,
)
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


def test_the_peak_carries_its_frame_and_the_measured_leg():
    angles = angle_series([90, 20, 30, 40, 90], [0, 10, 50, 10, 0])
    window = MovementWindow(1, 3, DetectionMethod.LANDING)
    knee = KneeFlexionRiskStrategy({"deep": LinearRamp(0, 80)}).score(angles, window)["deep"]
    trunk = TrunkFlexionRiskStrategy({"lean": LinearRamp(0, 80)}).score(angles, window)["lean"]
    assert (knee.peak_frame, knee.side) == (3, Side.LEFT)
    assert (trunk.peak_frame, trunk.side) == (2, None)


def test_each_knee_valgus_is_scored_on_its_own_leg():
    angles = angle_series([0] * 5, [0] * 5, {Side.LEFT: [0, 5, 20, 8, 0], Side.RIGHT: [0, 2, 3, 4, 30]})
    window = MovementWindow(1, 3, DetectionMethod.LANDING)
    partials = KneeValgusRiskStrategy(LinearRamp(10, 30), "knee_valgus").score(angles, window)
    left, right = partials["knee_valgus_left"], partials["knee_valgus_right"]
    assert (left.signal, left.measured_deg, left.score, left.peak_frame, left.side) == ("knee_valgus", 20, 0.5, 2, Side.LEFT)
    assert (right.measured_deg, right.score, right.peak_frame, right.side) == (4, 0, 3, Side.RIGHT)


def test_the_asymmetry_points_to_the_knee_that_goes_further_inward():
    angles = angle_series([0] * 3, [0] * 3, {Side.LEFT: [0, 4, 2], Side.RIGHT: [1, 19, 3]})
    partial = KneeValgusAsymmetryRiskStrategy(LinearRamp(10, 20), "asym").score(angles, WHOLE)["asym"]
    assert (partial.measured_deg, partial.score, partial.side, partial.peak_frame) == (15, 0.5, Side.RIGHT, 1)


def test_the_asymmetry_is_not_measured_without_both_knees():
    angles = angle_series([0] * 3, [0] * 3, {Side.LEFT: [0, 4, 2], Side.RIGHT: [math.nan] * 3})
    partial = KneeValgusAsymmetryRiskStrategy(LinearRamp(10, 20), "asym").score(angles, WHOLE)["asym"]
    assert math.isnan(partial.score) and partial.peak_frame is None and partial.side is None
