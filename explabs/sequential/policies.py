"""Stopping policies for sequential A/B tests.

Every policy answers one question: given the traffic seen so far, should we
stop and declare a winner? All of them share the same two-sided Welch z-test
and differ only in

* **when** they are allowed to look (``peekable``), and
* **how strict** the threshold is at each look (``alpha``).

That split is deliberate. "Peek more often" and "correct for peeking" become
two independent dials, which is exactly what this project wants to measure:
the cost of looking is a property of the *schedule*, and the repair is a
property of the *threshold*.

Policies are stateless: they read a pre-computed :class:`~simulate.Simulation`
and return, for every replicate, the day on which they would stop. That makes
them cheap to evaluate in bulk and trivial to plot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from ..sim import Simulation
from .boundaries import schedule_boundaries

# Default effect scales. Both mixing methods need a prior on the effect size,
# and both are sensitive to it — see the sensitivity sweep in run.py. The values
# are "typical absolute effects for a 10% baseline metric", i.e. 0.5pp and 1pp.
DEFAULT_TAU = 0.005
DEFAULT_PRIOR_SD = 0.01


def welch_two_sided_p(sim: Simulation) -> np.ndarray:
    """Two-sided p-value for a difference in means, shape ``(reps, horizon)``.

    Uses a normal reference distribution rather than Student's t. With the
    sample sizes involved here (thousands of users per arm) the difference is
    negligible, and it keeps the arithmetic vectorised. If you shrink traffic
    below a few hundred users per arm, switch to a t reference.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_c = sim.c_s1 / sim.c_n
        mean_t = sim.t_s1 / sim.t_n

        # Sample variance recovered from the sufficient statistics.
        var_c = (sim.c_s2 - sim.c_n * mean_c**2) / (sim.c_n - 1.0)
        var_t = (sim.t_s2 - sim.t_n * mean_t**2) / (sim.t_n - 1.0)

        # Float cancellation can push a near-zero variance slightly negative.
        var_c = np.maximum(var_c, 1e-12)
        var_t = np.maximum(var_t, 1e-12)

        se = np.sqrt(var_c / sim.c_n + var_t / sim.t_n)
        z = (mean_t - mean_c) / se

    p = 2.0 * norm.sf(np.abs(z))
    return np.clip(p, 0.0, 1.0)


def precision_and_effect(sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(precision, effect)``: the inverse variance of the estimated
    difference, and the difference itself.

    ``precision`` is the information content ``V_t`` used by mSPRT and by the
    Bayesian stopping rule. Both methods operate on ``Z = effect * sqrt(V)``,
    which is why they can share this helper despite disagreeing about
    everything else.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_c = sim.c_s1 / sim.c_n
        mean_t = sim.t_s1 / sim.t_n
        var_c = np.maximum((sim.c_s2 - sim.c_n * mean_c**2) / (sim.c_n - 1.0), 1e-12)
        var_t = np.maximum((sim.t_s2 - sim.t_n * mean_t**2) / (sim.t_n - 1.0), 1e-12)
        precision = 1.0 / (var_c / sim.c_n + var_t / sim.t_n)

    return precision, mean_t - mean_c


class Policy(ABC):
    """Base class: a peeking schedule plus a per-look threshold.

    Parameters
    ----------
    min_days:
        No decisions before this many days. Real experiment platforms enforce
        a minimum runtime to avoid acting on day-one noise; it also keeps the
        variance estimates sane.
    family_alpha:
        Overall false-positive budget for the whole experiment.
    """

    def __init__(self, min_days: int = 3, family_alpha: float = 0.05) -> None:
        if min_days < 2:
            raise ValueError("min_days must be at least 2")
        self.min_days = min_days
        self.family_alpha = family_alpha
        self.name = type(self).__name__.lower()

    @abstractmethod
    def peekable(self, horizon: int) -> np.ndarray:
        """Boolean mask of length ``horizon`` marking allowed look days."""

    def alpha(self, horizon: int) -> np.ndarray:
        """Per-look significance threshold, length ``horizon``."""
        return np.full(horizon, self.family_alpha, dtype=float)

    def p_values(self, sim: Simulation) -> np.ndarray:
        """p-value at every day for every replicate, shape ``(reps, horizon)``."""
        return welch_two_sided_p(sim)

    def stop_days(self, sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(day_index, stopped)``.

        ``day_index`` is the zero-based day of the decision; replicates that
        never cross a threshold are censored at the horizon and flagged
        ``stopped=False``.
        """
        p = self.p_values(sim)
        horizon = p.shape[1]

        allowed = np.broadcast_to(self.peekable(horizon), (p.shape[0], horizon))
        crossed = allowed & (p < self.alpha(horizon)[None, :])

        # Respect the minimum runtime.
        crossed[:, : self.min_days - 1] = False

        stopped = crossed.any(axis=1)
        first_day = crossed.argmax(axis=1)  # 0 where nothing was crossed
        return np.where(stopped, first_day, horizon - 1), stopped

    def describe(self) -> str:
        """One-line summary for report headers."""
        looks = int(self.peekable(1000).sum())
        return f"{self.name} (min_days={self.min_days}, family_alpha={self.family_alpha:g})"


class FixedHorizon(Policy):
    """A single look at the end of the planned window.

    The classical baseline. It has no peeking problem by construction, and it
    is the benchmark every sequential policy has to beat on speed without
    losing it on error control.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "fixed_horizon"

    def peekable(self, horizon: int) -> np.ndarray:
        mask = np.zeros(horizon, dtype=bool)
        mask[-1] = True
        return mask


class NaivePeek(Policy):
    """Look every ``peek_every`` days, stop at the first p < alpha.

    This is what a dashboard with a "significant!" badge encourages, and it is
    the behaviour the project is trying to price. Each individual look is a
    valid 5% test; the problem is that there are many of them.
    """

    def __init__(self, peek_every: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        if peek_every < 1:
            raise ValueError("peek_every must be at least 1")
        self.peek_every = peek_every
        self.name = f"naive_peek_e{peek_every}"

    def peekable(self, horizon: int) -> np.ndarray:
        mask = np.zeros(horizon, dtype=bool)
        mask[self.peek_every - 1 :: self.peek_every] = True
        mask[-1] = True  # always take a final look
        return mask


class BonferroniPeek(NaivePeek):
    """Same peeking schedule, alpha split evenly across looks.

    Keeps the family-wise error rate in check with no modelling assumptions,
    at the price of a much stricter bar at every individual look. Included
    because it is the correction most teams reach for first, and the power
    loss it causes is the argument for alpha-spending instead.
    """

    def __init__(self, peek_every: int = 1, **kwargs) -> None:
        super().__init__(peek_every=peek_every, **kwargs)
        self.name = f"bonferroni_e{peek_every}"

    def alpha(self, horizon: int) -> np.ndarray:
        looks = int(self.peekable(horizon).sum())
        return np.full(horizon, self.family_alpha / looks, dtype=float)


class GroupSequentialPeek(NaivePeek):
    """O'Brien-Fleming or Pocock boundaries via Lan-DeMets alpha spending.

    Reuses the look schedule of :class:`NaivePeek` and replaces the flat
    threshold with the boundary implied by a spending function. The boundary is
    converted to an equivalent per-look p-value threshold, so the base class
    machinery needs no changes: ``p < 2 * Phi(-b)`` is exactly ``|z| > b``.

    Look counts above roughly ten are numerically degenerate for OBF — the spend
    at the first look underflows to zero and the boundary runs off to infinity.
    That is not a bug, it is the method telling you that a daily OBF look can
    never fire.
    """

    def __init__(self, peek_every: int = 5, spending: str = "obf", **kwargs) -> None:
        super().__init__(peek_every=peek_every, **kwargs)
        self.spending = spending
        self.name = f"{spending}_e{peek_every}"
        self._bounds_cache: dict[int, np.ndarray] = {}

    def boundaries(self, horizon: int) -> np.ndarray:
        """z-boundaries for the looks this policy is allowed to take."""
        if horizon not in self._bounds_cache:
            days = np.flatnonzero(self.peekable(horizon))
            self._bounds_cache[horizon] = schedule_boundaries(
                days, horizon, alpha=self.family_alpha, spending=self.spending
            )
        return self._bounds_cache[horizon]

    def alpha(self, horizon: int) -> np.ndarray:
        thresholds = np.zeros(horizon)
        days = np.flatnonzero(self.peekable(horizon))
        thresholds[days] = 2.0 * norm.sf(self.boundaries(horizon))
        return thresholds


class MSPRTPeek(Policy):
    """mSPRT (Johari, Koomen, Pekelis & Walsh 2017), always-valid p-values.

    Mixes the likelihood ratio over a ``N(0, tau^2)`` prior on the absolute
    effect. With information ``V`` and estimate ``d``, the mixture likelihood
    ratio is

        Lambda_t = exp(tau^2 V^2 d^2 / (2 (1 + V tau^2))) / sqrt(1 + V tau^2)

    and the always-valid p-value is ``1 / max_{s <= t} Lambda_s``. Validity
    comes from Lambda being a nonnegative martingale with expectation 1 under
    H0 plus Ville's inequality — it does not come from correcting for the number
    of looks, which is why no look schedule has to be declared in advance.

    ``tau`` is an effect *scale*, not a correction factor. Setting it near the
    effects you actually care about makes the test sharp there; setting it far
    off makes it either sluggish or noisy. The sensitivity sweep in ``run.py``
    exists because this is the parameter people tune until the answer looks
    right, which is not a validation strategy.
    """

    def __init__(self, tau: float = DEFAULT_TAU, **kwargs) -> None:
        super().__init__(**kwargs)
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau
        self.name = f"msprt_tau{tau:g}"

    def peekable(self, horizon: int) -> np.ndarray:
        # Continuous monitoring: every day is a valid look.
        return np.ones(horizon, dtype=bool)

    def p_values(self, sim: Simulation) -> np.ndarray:
        precision, effect = precision_and_effect(sim)
        denom = 1.0 + precision * self.tau**2
        with np.errstate(over="ignore", invalid="ignore"):
            lam = np.exp(self.tau**2 * precision**2 * effect**2 / (2.0 * denom)) / np.sqrt(denom)
        running_max = np.maximum.accumulate(lam, axis=1)
        with np.errstate(divide="ignore"):
            p = 1.0 / running_max
        return np.clip(np.nan_to_num(p, nan=1.0), 0.0, 1.0)


class BayesPeek(Policy):
    """Bayesian monitoring: stop once the posterior excludes zero.

    Normal-normal conjugate update with prior ``N(0, prior_sd^2)``. The reported
    "p-value" is twice the smaller posterior tail, so thresholding at 0.05 is
    the same as "95% credible interval excludes zero".

    Included to be measured, not endorsed. A fixed posterior threshold is a
    stopping rule like any other, and a stopping rule evaluated repeatedly does
    not control the frequentist error rate just because it is Bayesian. The FPR
    curve shows how much it inflates.
    """

    def __init__(self, prior_sd: float = DEFAULT_PRIOR_SD, **kwargs) -> None:
        super().__init__(**kwargs)
        if prior_sd <= 0:
            raise ValueError("prior_sd must be positive")
        self.prior_sd = prior_sd
        self.name = f"bayes_sd{prior_sd:g}"

    def peekable(self, horizon: int) -> np.ndarray:
        return np.ones(horizon, dtype=bool)

    def p_values(self, sim: Simulation) -> np.ndarray:
        precision, effect = precision_and_effect(sim)
        post_precision = precision + 1.0 / self.prior_sd**2
        post_mean = effect * precision / post_precision
        post_sd = np.sqrt(1.0 / post_precision)
        return np.clip(2.0 * norm.sf(np.abs(post_mean) / post_sd), 0.0, 1.0)


def default_policies(
    min_days: int = 3,
    family_alpha: float = 0.05,
    tau: float = DEFAULT_TAU,
    prior_sd: float = DEFAULT_PRIOR_SD,
) -> list[Policy]:
    """The policy set for the method comparison.

    Four families, each at several peeking frequencies where the frequency is a
    free parameter:

    - unadjusted (fixed horizon, naive peek, Bonferroni)
    - group sequential (O'Brien-Fleming via Lan-DeMets)
    - always-valid (mSPRT)
    - Bayesian (posterior exclusion of zero)
    """
    common = dict(min_days=min_days, family_alpha=family_alpha)
    policies: list[Policy] = [FixedHorizon(**common)]

    for peek_every in (1, 5, 10, 30):
        policies.append(NaivePeek(peek_every=peek_every, **common))
    for peek_every in (1, 5, 10, 30):
        policies.append(BonferroniPeek(peek_every=peek_every, **common))
    # Daily OBF looks are numerically degenerate, so the schedule starts at 5.
    for peek_every in (5, 10, 30):
        policies.append(GroupSequentialPeek(peek_every=peek_every, **common))

    policies.append(MSPRTPeek(tau=tau, **common))
    policies.append(BayesPeek(prior_sd=prior_sd, **common))
    return policies
