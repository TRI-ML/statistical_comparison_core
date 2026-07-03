"""Identity tests: verify statistical_comparison_core objects are the same
as those re-exported from sequentialized_barnard_tests.base."""

import statistical_comparison_core as scc
import statistical_comparison_core.base as scc_base
import sequentialized_barnard_tests.base as sbt_base


def test_decision_identity():
    assert scc.Decision is sbt_base.Decision


def test_hypothesis_identity():
    assert scc.Hypothesis is sbt_base.Hypothesis


def test_two_sample_mean_hypothesis_identity():
    assert scc.TwoSampleMeanHypothesis is sbt_base.TwoSampleMeanHypothesis
    assert scc.Hypothesis is scc.TwoSampleMeanHypothesis
    assert sbt_base.Hypothesis is sbt_base.TwoSampleMeanHypothesis


def test_two_sample_binomial_hypothesis_identity():
    assert scc.TwoSampleBinomialHypothesis is sbt_base.TwoSampleBinomialHypothesis
    assert scc.TwoSampleBinomialHypothesis is scc.TwoSampleMeanHypothesis
    assert sbt_base.TwoSampleBinomialHypothesis is sbt_base.TwoSampleMeanHypothesis


def test_test_result_identity():
    assert scc.TestResult is sbt_base.TestResult


def test_test_base_identity():
    assert scc.TestBase is sbt_base.TestBase


def test_two_sample_test_base_identity():
    assert scc.TwoSampleTestBase is sbt_base.TwoSampleTestBase


def test_sequential_test_base_identity():
    assert scc.SequentialTestBase is sbt_base.SequentialTestBase


def test_sequential_two_sample_test_base_identity():
    assert scc.SequentialTwoSampleTestBase is sbt_base.SequentialTwoSampleTestBase


def test_mirrored_test_mixin_identity():
    assert scc.MirroredTestMixin is sbt_base.MirroredTestMixin


def test_base_module_identity():
    """All names exported from scc.base match sbt_base."""
    assert scc_base.Decision is sbt_base.Decision
    assert scc_base.Hypothesis is sbt_base.Hypothesis
    assert scc_base.TwoSampleMeanHypothesis is sbt_base.TwoSampleMeanHypothesis
    assert (
        scc_base.TwoSampleBinomialHypothesis
        is sbt_base.TwoSampleBinomialHypothesis
    )
    assert scc_base.TestResult is sbt_base.TestResult
    assert scc_base.TestBase is sbt_base.TestBase
    assert scc_base.SequentialTestBase is sbt_base.SequentialTestBase
    assert scc_base.MirroredTestMixin is sbt_base.MirroredTestMixin
