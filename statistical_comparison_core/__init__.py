"""Statistical Comparison Core: shared test interfaces."""

from .base import (
    Decision,
    Hypothesis,
    MirroredAlphaSignatureError,
    MirroredTestMixin,
    SequentialTestBase,
    SequentialTwoSampleTestBase,
    TestBase,
    TestResult,
    TwoSampleBinomialHypothesis,
    TwoSampleMeanHypothesis,
    TwoSampleTestBase,
)

__all__ = [
    "Decision",
    "Hypothesis",
    "MirroredAlphaSignatureError",
    "MirroredTestMixin",
    "SequentialTestBase",
    "SequentialTwoSampleTestBase",
    "TestBase",
    "TestResult",
    "TwoSampleBinomialHypothesis",
    "TwoSampleMeanHypothesis",
    "TwoSampleTestBase",
]
