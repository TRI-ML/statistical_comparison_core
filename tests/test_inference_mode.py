"""Tests for the ``inference_mode`` keyword on :class:`MirroredTestMixin`.

These tests exercise the child-alpha halving behavior, wrapper-level ``alpha``
read/write semantics, invalid-mode rejection, and the base-class signature guard
for ``inference_mode='ranking'``.
"""

import pytest

from statistical_comparison_core.base import (
    Decision,
    Hypothesis,
    MirroredAlphaSignatureError,
    MirroredTestMixin,
    SequentialTestBase,
    TestResult,
)


class DummyOneSidedTestNoAlpha(SequentialTestBase):
    """One-sided test whose signature does not accept ``alpha``."""

    def __init__(self, alternative, value=0):
        self.alternative = alternative
        self.value = value

    def step(self, datum_0, datum_1):
        return TestResult(decision=Decision.FailToDecide)


class DummyOneSidedTestWithAlpha(SequentialTestBase):
    """One-sided test whose signature exposes ``alpha`` positionally after ``alternative``."""

    def __init__(self, alternative, alpha, extra=None):
        self.alternative = alternative
        self.alpha = alpha
        self.extra = extra

    def step(self, datum_0, datum_1):
        return TestResult(decision=Decision.FailToDecide)


class DummyOneSidedTestNonconforming(SequentialTestBase):
    """One-sided test whose signature does not expose ``alpha`` in a conforming shape."""

    def __init__(self, alternative, threshold, extra=None):
        self.alternative = alternative
        self.threshold = threshold
        self.extra = extra

    def step(self, datum_0, datum_1):
        return TestResult(decision=Decision.FailToDecide)


class MirroredWithAlpha(MirroredTestMixin):
    _base_class = DummyOneSidedTestWithAlpha


class MirroredWithoutAlpha(MirroredTestMixin):
    _base_class = DummyOneSidedTestNoAlpha


class MirroredNonconforming(MirroredTestMixin):
    _base_class = DummyOneSidedTestNonconforming


# --- default construction is unchanged ---

def test_default_mode_passes_full_alpha_to_children():
    m = MirroredWithAlpha(Hypothesis.P0MoreThanP1, alpha=0.05)
    assert m._test_for_alternative.alpha == pytest.approx(0.05)
    assert m._test_for_null.alpha == pytest.approx(0.05)
    assert m.alpha == pytest.approx(0.05)
    assert m._inference_mode == "comparison"


def test_default_mode_via_positional_alpha():
    m = MirroredWithAlpha(Hypothesis.P0LessThanP1, 0.05)
    assert m._test_for_alternative.alpha == pytest.approx(0.05)
    assert m._test_for_null.alpha == pytest.approx(0.05)
    assert m.alpha == pytest.approx(0.05)


def test_default_mode_without_alpha_kwarg_still_works():
    # Base class that does not accept ``alpha`` at all must still work under
    # the default mode without triggering the signature guard.
    m = MirroredWithoutAlpha(Hypothesis.P0MoreThanP1, value=7)
    assert m._test_for_alternative.value == 7
    assert m._test_for_null.value == 7
    assert m._inference_mode == "comparison"


# --- ranking halves at construction and wrapper reads unhalved value ---

def test_ranking_halves_child_alpha_at_construction():
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking"
    )
    assert m._test_for_alternative.alpha == pytest.approx(0.025)
    assert m._test_for_null.alpha == pytest.approx(0.025)
    assert m.alpha == pytest.approx(0.05)


def test_ranking_halves_via_positional_alpha():
    m = MirroredWithAlpha(
        Hypothesis.P0LessThanP1, 0.04, inference_mode="ranking"
    )
    assert m._test_for_alternative.alpha == pytest.approx(0.02)
    assert m._test_for_null.alpha == pytest.approx(0.02)
    assert m.alpha == pytest.approx(0.04)


# --- ranking halves on reassignment ---

def test_ranking_halves_child_alpha_on_reassignment():
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking"
    )
    m.alpha = 0.10
    assert m._test_for_alternative.alpha == pytest.approx(0.05)
    assert m._test_for_null.alpha == pytest.approx(0.05)
    assert m.alpha == pytest.approx(0.10)


def test_comparison_alpha_reassignment_matches_ranking_no_ties():
    m_cmp = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="comparison"
    )
    m_rnt = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking_no_ties"
    )
    for m in (m_cmp, m_rnt):
        m.alpha = 0.02
        assert m._test_for_alternative.alpha == pytest.approx(0.02)
        assert m._test_for_null.alpha == pytest.approx(0.02)
        assert m.alpha == pytest.approx(0.02)


# --- comparison and ranking_no_ties are behaviorally identical for alpha ---

def test_comparison_and_ranking_no_ties_child_alpha_are_identical():
    m_cmp = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="comparison"
    )
    m_rnt = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking_no_ties"
    )
    assert (
        m_cmp._test_for_alternative.alpha
        == m_rnt._test_for_alternative.alpha
        == pytest.approx(0.05)
    )
    assert (
        m_cmp._test_for_null.alpha
        == m_rnt._test_for_null.alpha
        == pytest.approx(0.05)
    )


# --- wrapper p-value composition is explicit and mode-aware ---

@pytest.mark.parametrize("mode", ["comparison", "ranking_no_ties"])
def test_compose_wrapper_p_value_plain_min_for_non_halving_modes(mode):
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode=mode
    )
    assert m.compose_wrapper_p_value(0.03, 0.20) == pytest.approx(0.03)


def test_compose_wrapper_p_value_ranking_doubles_min():
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking"
    )
    assert m.compose_wrapper_p_value(0.03, 0.20) == pytest.approx(0.06)


def test_compose_wrapper_p_value_ranking_caps_at_one():
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking"
    )
    assert m.compose_wrapper_p_value(0.70, 0.90) == pytest.approx(1.0)


def test_compose_wrapper_p_value_not_invoked_implicitly(monkeypatch):
    def fail_if_called(self, p_value_less, p_value_more):
        raise AssertionError("compose_wrapper_p_value should be called explicitly")

    monkeypatch.setattr(MirroredWithAlpha, "compose_wrapper_p_value", fail_if_called)
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking"
    )
    with pytest.raises(AttributeError):
        _ = m._p_value


# --- inference_mode is not forwarded to child tests ---

def test_inference_mode_is_not_forwarded_to_child_tests():
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking"
    )
    assert not hasattr(m._test_for_alternative, "inference_mode")
    assert not hasattr(m._test_for_null, "inference_mode")


# --- invalid inference_mode ---

def test_invalid_inference_mode_raises_value_error_listing_valid_modes():
    with pytest.raises(ValueError) as excinfo:
        MirroredWithAlpha(
            Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="bogus"
        )
    message = str(excinfo.value)
    assert "bogus" in message
    for mode in ("comparison", "ranking", "ranking_no_ties"):
        assert mode in message


# --- signature guard for nonconforming base classes under ranking ---

def test_ranking_raises_named_error_for_nonconforming_base_class():
    with pytest.raises(MirroredAlphaSignatureError):
        MirroredNonconforming(
            Hypothesis.P0MoreThanP1, threshold=0.5, inference_mode="ranking"
        )


def test_ranking_raises_named_error_for_nonconforming_base_class_even_with_alpha_kwarg():
    # Codex review fix: the strict signature guard must fire even when the
    # caller supplies ``alpha`` as a keyword argument. Previously the guard
    # was skipped for kwarg calls and the failure surfaced as a child-side
    # ``TypeError`` for an unexpected keyword.
    with pytest.raises(MirroredAlphaSignatureError):
        MirroredNonconforming(
            Hypothesis.P0MoreThanP1,
            threshold=0.5,
            alpha=0.1,
            inference_mode="ranking",
        )


def test_ranking_raises_named_error_when_alpha_value_missing():
    with pytest.raises(MirroredAlphaSignatureError):
        MirroredWithAlpha(
            Hypothesis.P0MoreThanP1, inference_mode="ranking"
        )


def test_nonconforming_base_still_works_in_default_mode():
    # Under the default comparison mode the signature guard must not fire.
    m = MirroredNonconforming(
        Hypothesis.P0MoreThanP1, threshold=0.5
    )
    assert m._test_for_alternative.threshold == 0.5
    assert m._test_for_null.threshold == 0.5


# --- default comparison forwards alpha reads/writes and never stores a
#     wrapper-level shadow (Codex review fix) ---

def test_comparison_alpha_read_reflects_child_mutation():
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="comparison"
    )
    # Simulate a child normalizing/mutating its own alpha. Under comparison
    # the wrapper must NOT cache a stale value; wrapper.alpha reads the
    # alternative child.
    m._test_for_alternative.alpha = 0.999
    assert m.alpha == pytest.approx(0.999)


def test_ranking_no_ties_alpha_read_reflects_child_mutation():
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking_no_ties"
    )
    m._test_for_alternative.alpha = 0.777
    assert m.alpha == pytest.approx(0.777)


def test_ranking_alpha_read_uses_wrapper_shadow_not_child():
    # Under ranking the wrapper is authoritative for the unhalved alpha; a
    # child mutation must NOT change what the wrapper reports.
    m = MirroredWithAlpha(
        Hypothesis.P0MoreThanP1, alpha=0.05, inference_mode="ranking"
    )
    m._test_for_alternative.alpha = 0.999
    assert m.alpha == pytest.approx(0.05)


def test_mirrored_alpha_signature_error_is_reexported_from_package():
    import statistical_comparison_core as scc
    assert scc.MirroredAlphaSignatureError is MirroredAlphaSignatureError
