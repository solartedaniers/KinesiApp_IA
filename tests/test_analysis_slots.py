"""Pruebo el límite de análisis simultáneos."""

import pytest
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.pipeline.analysis_slots import AnalysisSlots
from kinesiapp_ai.rpc.domain_error import DomainError


def test_a_full_capacity_rejects_at_once_with_ai_busy() -> None:
    slots = AnalysisSlots(capacity=1)

    with slots.reserve(), pytest.raises(DomainError) as raised, slots.reserve():
        pass

    assert raised.value.reason == ErrorReason.ERROR_REASON_AI_BUSY
    assert slots.in_use == 0


def test_the_slot_is_freed_even_when_the_analysis_fails() -> None:
    slots = AnalysisSlots(capacity=1)

    with pytest.raises(RuntimeError), slots.reserve():
        raise RuntimeError("analysis failed")

    with slots.reserve():
        assert slots.in_use == 1


def test_a_larger_capacity_allows_that_many_at_once() -> None:
    slots = AnalysisSlots(capacity=2)

    with slots.reserve(), slots.reserve():
        assert slots.in_use == 2
        with pytest.raises(DomainError), slots.reserve():
            pass


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AnalysisSlots(capacity=0)
