"""Focused tests for the MirroredTestMixin wrapper-owned attribute behavior."""

import pytest
from statistical_comparison_core.base import (
    Decision,
    Hypothesis,
    MirroredTestMixin,
    SequentialTestBase,
    TestResult,
)


class DummyOneSidedTest(SequentialTestBase):
    """Minimal one-sided test for exercising the mixin."""

    def __init__(self, alternative, value=0):
        self.alternative = alternative
        self.value = value
        self._internal = "child"
        self._p_value = 1.0

    def step(self, datum_0, datum_1):
        return TestResult(decision=Decision.FailToDecide)

    def reset(self):
        self.value = 0

    def mutating_method(self):
        """A mutator reachable through __getattr__."""
        self.value = 99
        return self.value


class PlainMirrored(MirroredTestMixin):
    """Mirrored test with default empty _wrapper_owned_attributes."""

    _base_class = DummyOneSidedTest


class WrapperOwnedMirrored(MirroredTestMixin):
    """Mirrored test declaring wrapper-owned names."""

    _base_class = DummyOneSidedTest
    _wrapper_owned_attributes = frozenset({"_p_value", "_p_type"})


# --- existing identity re-export still passes ---

def test_identity_reexport():
    import statistical_comparison_core as scc
    import sequentialized_barnard_tests.base as sbt_base
    assert scc.MirroredTestMixin is sbt_base.MirroredTestMixin


# --- default empty declaration fans ordinary writes to both children ---

def test_default_fanout():
    m = PlainMirrored(Hypothesis.P0MoreThanP1, value=5)
    assert m._test_for_alternative.value == 5
    assert m._test_for_null.value == 5

    m.value = 42
    assert m._test_for_alternative.value == 42
    assert m._test_for_null.value == 42


# --- declared wrapper-owned assignment stays on wrapper, preserves child ---

def test_wrapper_owned_stays_on_wrapper():
    m = WrapperOwnedMirrored(Hypothesis.P0MoreThanP1, value=10)
    m._p_value = 0.03
    m._p_type = "alternative"

    assert m._p_value == 0.03
    assert m._p_type == "alternative"
    # Child _p_value must be untouched (still 1.0 from __init__).
    assert m._test_for_alternative._p_value == 1.0
    assert m._test_for_null._p_value == 1.0
    assert m._test_for_alternative.value == 10
    assert m._test_for_null.value == 10


# --- read-before-first-write for declared wrapper-owned raises,
#     even when the child already has the attribute ---

def test_read_before_write_raises():
    m = WrapperOwnedMirrored(Hypothesis.P0MoreThanP1)
    # Child has _p_value=1.0 from __init__, but wrapper-owned must not forward.
    assert m._test_for_alternative._p_value == 1.0
    with pytest.raises(AttributeError, match="wrapper-owned"):
        _ = m._p_value


# --- wrapper read after wrapper write returns wrapper value ---

def test_wrapper_read_after_write():
    m = WrapperOwnedMirrored(Hypothesis.P0MoreThanP1)
    m._p_value = 0.01
    assert m._p_value == 0.01
    m._p_value = 0.005
    assert m._p_value == 0.005


# --- instance-level _wrapper_owned_attributes assignment raises ---

def test_instance_wrapper_owned_attributes_assignment_raises():
    m = PlainMirrored(Hypothesis.P0MoreThanP1)
    with pytest.raises(AttributeError, match="class level"):
        m._wrapper_owned_attributes = frozenset({"foo"})

    m2 = WrapperOwnedMirrored(Hypothesis.P0MoreThanP1)
    with pytest.raises(AttributeError, match="class level"):
        m2._wrapper_owned_attributes = frozenset({"bar"})


# --- alternative assignment still flips children correctly ---

def test_alternative_flips():
    m = PlainMirrored(Hypothesis.P0MoreThanP1)
    assert m._test_for_alternative.alternative == Hypothesis.P0MoreThanP1
    assert m._test_for_null.alternative == Hypothesis.P0LessThanP1

    m.alternative = Hypothesis.P0LessThanP1
    assert m._test_for_alternative.alternative == Hypothesis.P0LessThanP1
    assert m._test_for_null.alternative == Hypothesis.P0MoreThanP1


# --- un-overridden mutator via __getattr__ binds only to alternative child ---

def test_mutator_binds_only_to_alternative():
    m = PlainMirrored(Hypothesis.P0MoreThanP1, value=0)
    result = m.mutating_method()
    assert result == 99
    assert m._test_for_alternative.value == 99
    # Null child is NOT mutated -- this documents the hazard.
    assert m._test_for_null.value == 0
