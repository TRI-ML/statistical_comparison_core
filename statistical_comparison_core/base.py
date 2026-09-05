"""Base class definitions.

This module defines base classes for hypothesis tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import inspect
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


class MirroredAlphaSignatureError(TypeError):
    """Raised when ``inference_mode='ranking'`` requires an ``alpha`` parameter
    that the mirrored base class does not expose in the expected shape.

    The mirrored mixin needs to halve ``alpha`` for both child tests when
    ``inference_mode='ranking'``. That requires the base class ``__init__`` to
    accept ``alpha`` either as the first positional parameter after
    ``alternative`` or as a keyword argument, and requires the caller to
    provide a value. This exception is raised when either condition fails, so
    the failure mode is a clear named error rather than a silent misconfiguration.
    """


class MirroredTestMixin:
    """A mixin class to define mirrored hypothesis tests.

    A mirrored test runs two one-sided tests simultaneously with flipped
    alternative hypotheses, so it can yield Decision.AcceptNull,
    Decision.AcceptAlternative, or Decision.FailToDecide.

    The ``alternative`` argument orients the mirrored pair: it defines the strict
    direction reported as ``Decision.AcceptAlternative``. The semantics of
    ``Decision`` differ depending on the ``inference_mode``:

    * ``inference_mode="comparison"`` (default) - A/B-style comparison where the
      null side includes equality. ``Decision.AcceptAlternative`` means the
      strict alternative was accepted. ``Decision.AcceptNull`` means the
      null/baseline side was accepted, not necessarily that the opposite strict
      ordering was proven. Each child test receives the full wrapper ``alpha``.

    * ``inference_mode="ranking"`` - pairwise ranking where exact ties make either
      strict ordering wrong. ``Decision.AcceptAlternative`` means the oriented
      strict direction was accepted; ``Decision.AcceptNull`` means the opposite
      strict direction was accepted. ``Decision.FailToDecide`` includes
      unresolved exact-tie cases. Each child test receives ``alpha / 2``.

    * ``inference_mode="ranking_no_ties"`` - pairwise ranking under the assumption
      ``p_0 != p_1``. Decisions have the same strict-direction interpretation as
      ranking mode, but equality is excluded from the formal guarantee. Each
      child test receives the full wrapper ``alpha``.

    Practical user guidance on ``inference_mode``:

    * Use ``inference_mode="comparison"`` for an A/B-style improvement claim
      against a baseline: "is policy 1 better than policy 0?" This preserves the
      original NSCORE interpretation and does not split the child-test ``alpha``.

    * Use ``inference_mode="ranking"`` for AnyRank or other
      policy-ranking workflows that report full or partial rankings (for
      example, via CLD), when exact ties in the true performance parameters are
      possible and either strict order would be considered a false ranking. This
      is the conservative setting; each child receives ``alpha / 2`` and more
      policy rollouts are needed to reach a decision.

    * Use ``inference_mode="ranking_no_ties"`` for ranking workflows where the
      user is willing to restrict the formal guarantee to the parameter space
      ``p_0 != p_1``. This may match settings where exact ties in the true
      performance parameters are considered unlikely or outside consideration,
      but the guarantee is explicitly restricted to non-equal parameters.
      This mode does not split the child-test alpha and may reach decisions
      sooner because each child uses the full ``alpha``.


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
    * ``alpha`` is special under ``inference_mode="ranking"``: assignment
      stores the unhalved value on the wrapper and pushes ``alpha / 2`` to
      each child, and reads return the stored wrapper value. Under the other
      inference modes ``alpha`` reads and writes flow through the standard
      forward-and-fan-out path.
    * Ranking mode is intentionally asymmetric on ``alpha``: ``wrapper.alpha``
      is wrapper-authoritative, so a direct child ``alpha`` mutation is not
      reflected by ``wrapper.alpha``; under ``comparison`` and
      ``ranking_no_ties``, ``wrapper.alpha`` reads through the alternative
      child and does reflect such mutations.
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
            "_inference_mode",
            "_wrapper_alpha",
            "_has_wrapper_alpha",
        }
    )
    _ALLOWED_INFERENCE_MODES: frozenset = frozenset(
        {"comparison", "ranking", "ranking_no_ties"}
    )

    def __init__(
        self,
        alternative: Hypothesis,
        *args,
        inference_mode: str = "comparison",
        **kwargs,
    ) -> None:
        """Initializes the MirroredTestMixin object.

        It instantiates two private member variables, ``_test_for_alternative``
        and ``_test_for_null``, with flipped alternative hypotheses.

        Args:
            alternative: Specification of the alternative hypothesis.
            *args: Additional positional arguments forwarded to the base class.
            inference_mode: Which inference the mirrored test supports. Must be
                one of ``"comparison"`` (default; equality belongs to the null
                side and each child receives the full ``alpha``),
                ``"ranking"`` (conservative two-sided ranking; each child
                receives ``alpha / 2`` while the wrapper still reports the
                unhalved ``alpha``), or ``"ranking_no_ties"`` (ranking over the
                restricted parameter space with ``mu_0 != mu_1``; each child
                receives the full ``alpha``). ``inference_mode`` is not
                forwarded to the base class.
            **kwargs: Additional optional or keyword arguments forwarded to the
                base class.

        Raises:
            ValueError: If ``inference_mode`` is not one of the allowed values.
            AttributeError: If ``alternative`` is not a valid ``Hypothesis``
                for a mirrored test.
            MirroredAlphaSignatureError: If ``inference_mode='ranking'`` but
                the base class does not accept ``alpha`` positionally after
                ``alternative`` or as a keyword argument, or if the caller
                did not provide an ``alpha`` value.
        """
        allowed = type(self)._ALLOWED_INFERENCE_MODES
        if inference_mode not in allowed:
            raise ValueError(
                f"Invalid inference_mode {inference_mode!r}. "
                f"Valid values are: {sorted(allowed)}."
            )

        if alternative == Hypothesis.P0MoreThanP1:
            null = Hypothesis.P0LessThanP1
        elif alternative == Hypothesis.P0LessThanP1:
            null = Hypothesis.P0MoreThanP1
        else:
            raise (
                AttributeError(f"{alternative} is not a valid value for alternative.")
            )

        # Wrapper-only state is stored via ``super().__setattr__`` because the
        # child tests do not exist yet and the custom ``__setattr__`` would
        # otherwise try to fan out to them.
        super().__setattr__("_inference_mode", inference_mode)

        if inference_mode == "ranking":
            alpha_kind, alpha_pos, alpha_value = self._find_alpha_in_call(
                args, kwargs, strict=True
            )
            # ``_find_alpha_in_call`` with ``strict=True`` guarantees a value.
            child_alpha = alpha_value / 2.0
            super().__setattr__("_has_wrapper_alpha", True)
            super().__setattr__("_wrapper_alpha", alpha_value)
            if alpha_kind == "kwarg":
                kwargs = dict(kwargs)
                kwargs["alpha"] = child_alpha
            else:
                args = list(args)
                args[alpha_pos] = child_alpha
                args = tuple(args)
        else:
            # comparison and ranking_no_ties: forward ``alpha`` reads/writes
            # unchanged. No wrapper-level ``alpha`` shadow is stored so
            # ``wrapper.alpha`` cannot go stale relative to child state.
            super().__setattr__("_has_wrapper_alpha", False)
            super().__setattr__("_wrapper_alpha", None)

        self._test_for_alternative = self._base_class(alternative, *args, **kwargs)
        self._test_for_null = self._base_class(null, *args, **kwargs)

    def _find_alpha_in_call(self, args, kwargs, *, strict):
        """Locate ``alpha`` in a caller's ``args``/``kwargs`` relative to the
        base class signature.

        Under ``strict=True`` (used for ``inference_mode='ranking'``), the base
        class ``__init__`` signature is always inspected - even when ``alpha``
        is supplied as a keyword argument - and
        :class:`MirroredAlphaSignatureError` is raised when the signature does
        not accept ``alpha`` either as the first positional parameter after
        ``alternative`` or as a keyword argument, or when no ``alpha`` value
        was provided.

        Under ``strict=False``, this method never raises: it returns
        ``(None, None, None)`` when a conforming ``alpha`` cannot be located,
        which lets non-ranking modes accept base classes that do not expose
        ``alpha`` at all.

        Returns:
            ``(kind, positional_index, value)``:

            * ``kind`` is ``"kwarg"`` when ``alpha`` was passed as a keyword,
              ``"positional"`` when it was passed as the first positional
              argument after ``alternative``, or ``None`` when it was not
              provided.
            * ``positional_index`` is the index into ``args`` when ``kind`` is
              ``"positional"``; otherwise ``None``.
            * ``value`` is the caller-supplied ``alpha`` value when ``kind`` is
              not ``None``; otherwise ``None``.

        Raises:
            MirroredAlphaSignatureError: If ``strict`` is ``True`` and either
                the base class signature is nonconforming or no ``alpha``
                value was provided.
        """
        if strict:
            try:
                sig = inspect.signature(self._base_class)
            except (TypeError, ValueError) as e:
                raise MirroredAlphaSignatureError(
                    f"inference_mode='ranking' requires an introspectable base "
                    f"class signature but inspecting "
                    f"{self._base_class.__name__!r} failed: {e}"
                ) from e
            params = list(sig.parameters.values())
            conforming = (
                len(params) >= 2
                and params[0].name == "alternative"
                and params[1].name == "alpha"
            )
            if not conforming:
                raise MirroredAlphaSignatureError(
                    f"inference_mode='ranking' requires that the base class "
                    f"{self._base_class.__name__!r} expose 'alpha' as the "
                    "first positional parameter after 'alternative' or as a "
                    f"keyword argument. Signature: {sig}."
                )
            if "alpha" in kwargs:
                return "kwarg", None, kwargs["alpha"]
            if len(args) >= 1:
                return "positional", 0, args[0]
            raise MirroredAlphaSignatureError(
                f"inference_mode='ranking' requires an 'alpha' value; none "
                f"provided as a positional or keyword argument to "
                f"{type(self).__name__}. Base signature: {sig}."
            )

        if "alpha" in kwargs:
            return "kwarg", None, kwargs["alpha"]
        try:
            sig = inspect.signature(self._base_class)
        except (TypeError, ValueError):
            return None, None, None
        params = list(sig.parameters.values())
        if (
            len(params) >= 2
            and params[0].name == "alternative"
            and params[1].name == "alpha"
            and len(args) >= 1
        ):
            return "positional", 0, args[0]
        return None, None, None

    def compose_wrapper_p_value(self, p_value_less, p_value_more):
        """Combine directional p-values into a wrapper-level p-value.

        Under ``inference_mode="ranking"``, each child runs at ``alpha / 2``.
        The wrapper rejects exactly when ``min(p_value_less, p_value_more) <=
        alpha / 2``, which is equivalent to ``2 * min(...) <= alpha``.
        Returning the doubled, capped value keeps the invariant that
        ``wrapper_p_value <= wrapper.alpha`` means the wrapper rejects at its
        declared level.

        Under ``comparison`` and ``ranking_no_ties``, each child runs at the
        wrapper's full ``alpha``, so the wrapper p-value is the plain minimum
        of the two directional p-values.
        """
        p_value = min(p_value_less, p_value_more)
        if self._inference_mode_is_ranking():
            return min(1.0, 2.0 * p_value)
        return p_value

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
        if name == "alpha":
            try:
                has_flag = object.__getattribute__(self, "_has_wrapper_alpha")
            except AttributeError:
                has_flag = False
            if has_flag:
                return object.__getattribute__(self, "_wrapper_alpha")
        if hasattr(self._test_for_alternative, name):
            return getattr(self._test_for_alternative, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def _inference_mode_is_ranking(self) -> bool:
        """Return ``True`` iff ``_inference_mode`` is set and equals ``ranking``.

        Safe to call before ``_inference_mode`` is stored on the wrapper (during
        construction of built-in attributes), in which case ``False`` is
        returned so that the standard fan-out path is used.
        """
        try:
            mode = object.__getattribute__(self, "_inference_mode")
        except AttributeError:
            return False
        return mode == "ranking"

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
        elif name == "alpha" and self._inference_mode_is_ranking():
            # Ranking mode: wrapper stores the unhalved value and children
            # receive ``alpha / 2``. Under comparison and ranking_no_ties,
            # ``alpha`` falls through to the standard fan-out below.
            super().__setattr__("_wrapper_alpha", value)
            super().__setattr__("_has_wrapper_alpha", True)
            if not hasattr(self._test_for_alternative, "alpha"):
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute 'alpha'"
                )
            setattr(self._test_for_alternative, "alpha", value / 2.0)
            setattr(self._test_for_null, "alpha", value / 2.0)
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
