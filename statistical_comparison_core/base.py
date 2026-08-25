"""Base class definitions.

This module defines base classes for hypothesis tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Type, Union

from numpy.typing import ArrayLike


class Decision(Enum):
    """Enum class to represent the decision of a hypothesis test."""

    AcceptNull = 0
    AcceptAlternative = 1
    FailToDecide = 2


@dataclass
class TestResult:
    """Data class defining the result of a hypothesis test.

    A result must contain a decision. Any auxiliary information will be thrown into
    info as an optional dictionary.
    """

    __test__ = False

    decision: Decision
    info: Optional[dict] = None


class TwoSampleMeanHypothesis(Enum):
    """Enum class to represent directional two-sample mean hypotheses.

    We assume two data-generating distributions with scalar performance parameters
    p_0 and p_1. For Bernoulli data, these parameters are success probabilities. For
    generic bounded or partial-credit scores, they are the means of random variables
    X_0 and X_1. The first hypothesis, `P0LessThanP1`, represents `p_0 < p_1`. The
    second hypothesis, `P0MoreThanP1`, represents `p_0 > p_1`. Note that both of
    these are possible alternative hypotheses in a statistical test, and the user
    will be tasked with choosing one as their alternative when instantiating a test.
    """

    P0LessThanP1 = 0
    P0MoreThanP1 = 1


Hypothesis = TwoSampleMeanHypothesis
TwoSampleBinomialHypothesis = TwoSampleMeanHypothesis


class TwoSampleTestBase(ABC):
    """Abstract base class for a two-sample hypothesis test."""

    @abstractmethod
    def run_on_sequence(
        self,
        sequence_0: ArrayLike,
        sequence_1: ArrayLike,
        *args,
        **kwargs,
    ) -> TestResult:
        """Runs the test on a pair of sequential data.

        Args:
            sequence_0: Sequence of data from the first source.
            sequence_1: Sequence of data from the second source.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).


        Returns:
            TestResult: Result of the hypothesis test.
        """
        pass


TestBase = TwoSampleTestBase


class SequentialTwoSampleTestBase(TwoSampleTestBase):
    """Base class for a family of sequential two-sample hypothesis tests."""

    def run_on_sequence(
        self,
        sequence_0: ArrayLike,
        sequence_1: ArrayLike,
        *args,
        **kwargs,
    ) -> TestResult:
        """Runs the test on a pair of sequential data.

        Args:
            sequence_0: Sequence of data from the first source.
            sequence_1: Sequence of data from the second source.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).

        Returns:
            TestResult: Result of the hypothesis test.

        Raises:
            ValueError: If the two sequences do not have the same length.
        """
        self.reset()
        if not (len(sequence_0) == len(sequence_1)):
            raise (ValueError("The two input sequences must have the same size."))

        result = TestResult(decision=Decision.FailToDecide)
        for idx in range(len(sequence_0)):
            result = self.step(sequence_0[idx], sequence_1[idx], *args, **kwargs)
            if not result.decision == Decision.FailToDecide:
                break
        return result

    def reset(self) -> None:
        """Resets internal memory states of the test."""
        pass

    @abstractmethod
    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        *args,
        **kwargs,
    ) -> TestResult:
        """Runs the test on a single pair of data.

        Args:
            datum_0: Datum from the first source.
            datum_1: Datum from the second source.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).

        Returns:
            TestResult: Result of the hypothesis test.
        """
        pass


SequentialTestBase = SequentialTwoSampleTestBase


class MirroredTestMixin:
    """A mixin class to define mirrored hypothesis tests.

    A mirrored test runs two one-sided tests simultaneously with flipped
    alternative hypotheses, so it can yield Decision.AcceptNull,
    Decision.AcceptAlternative, or Decision.FailToDecide.  For example, if the
    alternative is Hypothesis.P0MoreThanP1 and the decision is
    Decision.AcceptNull, it should be interpreted as accepting
    Hypothesis.P0LessThanP1.

    The significance level alpha controls two errors simultaneously:
    (1) probability of wrongly accepting the alternative when the null is true,
    and (2) probability of wrongly accepting the null when the alternative is
    true.  Bonferroni correction is not needed since the null hypothesis for one
    test is the alternative for the other.

    Attribute ownership rules:

    * The wrapper owns two child tests: ``_test_for_alternative`` (for the
      requested alternative hypothesis) and ``_test_for_null`` (for the flipped
      hypothesis interpreted as the null-side test).
    * Reads of ordinary (non-wrapper-owned) attributes are forwarded to
      ``_test_for_alternative`` via ``__getattr__``.  This is intentional for
      read-only properties exposed by child tests.
    * Writes of ordinary attributes fan out to both child tests when the
      attribute already exists on ``_test_for_alternative``.
    * Assigning ``alternative`` is special: the alternative child receives the
      requested hypothesis and the null child receives the flipped hypothesis.
    * ``_wrapper_owned_attributes`` is an opt-in, **class-level** frozenset
      declaring names whose state belongs to the mirrored wrapper rather than
      either child test.  Empty by default, so existing mirrored classes are
      unaffected.
    * Instance-level assignment to ``_wrapper_owned_attributes`` is rejected
      because it would not affect dispatch.
    * Declared wrapper-owned names are stored on the wrapper and are **not**
      forwarded on read-before-first-write; ``__getattr__`` raises
      ``AttributeError`` instead.
    * Mutating methods inherited through ``__getattr__`` bind only to the
      alternative child.  Callers that need to mutate both children should
      prefer wrapper-level methods or property assignment.

    Attributes:
        _base_class: The one-sided test class.  Must be set by subclasses.
        _wrapper_owned_attributes: Class-level frozenset of names stored on the
            wrapper instead of being fanned out to child tests.  Empty by
            default.
    """

    _base_class: Type[Union[TestBase, SequentialTestBase]] = (
        None  # To be set by subclasses.
    )
    _wrapper_owned_attributes: frozenset = frozenset()
    _always_wrapper_owned_attributes: frozenset = frozenset(
        {
            "_test_for_alternative",
            "_test_for_null",
            "_base_class",
            "_always_wrapper_owned_attributes",
        }
    )

    def __init__(self, alternative: Hypothesis, *args, **kwargs) -> None:
        """Initializes the MirroredTestMixin object.

        It instantiates two private member variables, _test_for_alternative and
        _test_for_null, with flipped alternative hypotheses.

        Args:
            alternative: Specification of the alternative hypothesis.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).

        Raises:
            AttributeError: If alternative is not a valid Hypothesis for a mirrored test.
        """
        if alternative == Hypothesis.P0MoreThanP1:
            null = Hypothesis.P0LessThanP1
        elif alternative == Hypothesis.P0LessThanP1:
            null = Hypothesis.P0MoreThanP1
        else:
            raise (
                AttributeError(f"{alternative} is not a valid value for alternative.")
            )

        self._test_for_alternative = self._base_class(alternative, *args, **kwargs)
        self._test_for_null = self._base_class(null, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute reads to ``_test_for_alternative``.

        Declared wrapper-owned names (listed in
        ``_wrapper_owned_attributes``) are **not** forwarded.  Reading a
        wrapper-owned name before it has been written raises
        ``AttributeError`` instead of silently returning child state, so the
        lifecycle is unambiguous.

        Args:
            name: Name of the attribute.

        Raises:
            AttributeError: If *name* is declared wrapper-owned (prevents
                silent forwarding of child state) or if the attribute does
                not exist on the alternative child test.
        """
        wrapper_owned = type(self)._wrapper_owned_attributes
        if name in wrapper_owned:
            raise AttributeError(
                f"'{type(self).__name__}' declares '{name}' as wrapper-owned "
                f"but it has not been set on the wrapper yet"
            )
        if hasattr(self._test_for_alternative, name):
            return getattr(self._test_for_alternative, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Route attribute writes.

        Dispatch rules, checked in order:

        1. ``alternative`` -- flip the hypothesis and assign to each child.
        2. ``_wrapper_owned_attributes`` -- reject; must be declared at the
           class level to affect dispatch.
        3. Always-wrapper-owned names (``_test_for_alternative``,
           ``_test_for_null``, ``_base_class``,
           ``_always_wrapper_owned_attributes``) and names listed in the
           class-level ``_wrapper_owned_attributes`` -- store on the wrapper
           via ``super().__setattr__``.
        4. Otherwise, fan out to both child tests if the attribute exists on
           ``_test_for_alternative``; raise ``AttributeError`` if not.

        Args:
            name: Name of the attribute.
            value: Value of the attribute.

        Raises:
            AttributeError: If the attribute cannot be set.
        """
        if name == "alternative":
            if value == Hypothesis.P0MoreThanP1:
                null = Hypothesis.P0LessThanP1
            elif value == Hypothesis.P0LessThanP1:
                null = Hypothesis.P0MoreThanP1
            else:
                raise (AttributeError(f"{value} is not a valid value for alternative."))
            self._test_for_alternative.alternative = value
            self._test_for_null.alternative = null
        elif name == "_wrapper_owned_attributes":
            raise AttributeError(
                "_wrapper_owned_attributes must be declared at the class level, "
                "not assigned on an instance, because only the class-level "
                "declaration affects __getattr__/__setattr__ dispatch."
            )
        else:
            always_owned = type(self)._always_wrapper_owned_attributes
            wrapper_owned = type(self)._wrapper_owned_attributes
            if name in always_owned or name in wrapper_owned:
                super().__setattr__(name, value)
            elif hasattr(self._test_for_alternative, name):
                setattr(self._test_for_alternative, name, value)
                setattr(self._test_for_null, name, value)
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
