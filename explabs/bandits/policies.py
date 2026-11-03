"""Assignment policies, and the engine that runs them.

Every policy answers the same question each step: *with what probability does
the next user get the treatment?* That number is the propensity, and it is the
object that makes or breaks post-hoc inference. The policy is written as an
explicit Bernoulli draw against that probability rather than as a direct
argmax, which is distributionally identical for Thompson sampling and makes the
propensity a first-class output instead of something reconstructed afterwards.

Two random streams are kept separate on purpose:

``rng_policy``
    consumed by the allocation rule;
``rng_reward``
    consumed by the outcome draw, one uniform per user per step.

Sharing the reward stream across policies is what lets a regret comparison
attribute a difference to the policy rather than to a different sequence of
lucky users -- the same common-random-numbers trick used elsewhere in this
project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import betaln, betainc

from .dgp import ContextualDGP

# Numerical floor on a propensity. A policy that has all but abandoned an arm
# produces a win probability that underflows to exactly 0 in float64, and a
# denominator of zero is worse than a denominator of 1e-9: the former destroys
# the estimate, the latter buries it in a variance the data cannot support. The
# floor is far below any propensity that a finite sample can resolve, so it
# changes no reported number -- it only keeps the arithmetic defined.
PROPENSITY_FLOOR = 1e-9


def beta_win_probability(
    a1: np.ndarray, b1: np.ndarray, a0: np.ndarray, b0: np.ndarray, nodes: int = 64
) -> np.ndarray:
    """``P(theta_1 > theta_0)`` for independent Beta posteriors.

    Thompson sampling needs this number every step and it has no closed form, so
    it is computed by Gauss-Legendre quadrature on ``P(theta_1 > theta_0) =
    int f_1(x) F_0(x) dx`` over a range truncated at ten posterior standard
    deviations. The integrand is analytic and the interval is scaled to the
    posterior, so 64 nodes reach machine-level accuracy even when the posterior
    is concentrated on a hundredth of the unit interval -- the regime that a
    fixed grid on ``[0, 1]`` gets wrong.

    The alternative -- estimating the probability from finitely many posterior
    draws -- would put noise in a denominator, and noise in a denominator biases
    every inverse-propensity estimate that follows.
    """
    a1, b1, a0, b0 = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (a1, b1, a0, b0))
    total = a1 + b1
    mean = a1 / total
    sd = np.sqrt(a1 * b1 / (total**2 * (total + 1.0)))

    lo = np.maximum(mean - 10.0 * sd, 0.0)
    hi = np.minimum(mean + 10.0 * sd, 1.0)
    # Map the reference nodes onto each posterior's own interval.
    unit, weights = np.polynomial.legendre.leggauss(nodes)
    unit = 0.5 * (unit + 1.0)
    weights = 0.5 * weights

    x = lo[:, None] + (hi - lo)[:, None] * unit[None, :]
    log_density = (
        (a1 - 1.0)[:, None] * np.log(np.clip(x, 1e-300, None))
        + (b1 - 1.0)[:, None] * np.log(np.clip(1.0 - x, 1e-300, None))
        - betaln(a1, b1)[:, None]
    )
    cdf = betainc(a0[:, None], b0[:, None], x)
    integral = (hi - lo) * np.sum(weights[None, :] * np.exp(log_density) * cdf, axis=1)
    return np.clip(integral, PROPENSITY_FLOOR, 1.0 - PROPENSITY_FLOOR)


class _Counts:
    """Sufficient statistics shared by the policies: pulls and reward sums per bucket and arm."""

    n_buckets: int = 1
    name: str = "policy"
    label: str = "policy"

    def reset(self, reps: int, rng: np.random.Generator) -> None:
        self.reps = reps
        self.n = np.zeros((reps, self.n_buckets, 2))
        self.obs = np.zeros((reps, self.n_buckets, 2))

    def observe(self, buckets: np.ndarray, arms: np.ndarray, rewards: np.ndarray) -> None:
        idx = np.arange(buckets.size)
        # Each (replication, bucket) pair appears once per step, so element-wise
        # fancy-index assignment is exact here and avoids the cost of bincount.
        self.n[idx, buckets, arms] += 1.0
        self.obs[idx, buckets, arms] += rewards

    def _slice(self, buckets: np.ndarray):
        idx = np.arange(buckets.size)
        return idx, self.n[idx, buckets], self.obs[idx, buckets]


@dataclass
class FixedShare(_Counts):
    """A fixed A/B split. The baseline that adaptation is measured against."""

    share: float = 0.5
    n_buckets: int = 1
    name: str = "fixed"

    def __post_init__(self) -> None:
        self.key = "fixed"
        self.label = f"fixed {self.share:.0%}/{1 - self.share:.0%}" if self.share != 0.5 else "fixed 50/50"

    def propensity(self, buckets: np.ndarray) -> np.ndarray:
        return np.full(buckets.size, self.share)

    def sample(self, buckets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return (rng.random(buckets.size) < self.share).astype(np.int8)


@dataclass
class ThompsonSampling(_Counts):
    """Beta-Bernoulli Thompson sampling, optionally run per feature bucket.

    ``n_buckets = 1`` ignores the user's features entirely: allocation punishes
    the worse arm but cannot condition on who is arriving. ``n_buckets = 4``
    keeps a separate posterior per quadrant, so a user's features determine how
    likely they are to see the treatment. That difference is the whole story of
    Section 3: it is what turns adaptation from harmless into confounding.
    """

    n_buckets: int = 1
    name: str = "thompson"

    def __post_init__(self) -> None:
        self.key = "ts" if self.n_buckets == 1 else f"ts{self.n_buckets}"
        self.label = ("Thompson sampling (pooled)" if self.n_buckets == 1
                      else f"Thompson sampling ({self.n_buckets} segments)")

    def _posterior(self, buckets: np.ndarray):
        idx, n, obs = self._slice(buckets)
        return idx, 1.0 + obs, 1.0 + n - obs

    def sample(self, buckets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        idx, a, b = self._posterior(buckets)
        theta = rng.beta(a, b)  # (reps, 2): one draw per arm for the arrived bucket
        return (theta[:, 1] >= theta[:, 0]).astype(np.int8)

    def propensity(self, buckets: np.ndarray) -> np.ndarray:
        idx, a, b = self._posterior(buckets)
        out = np.empty(buckets.size)
        for seg in range(self.n_buckets):
            rows = idx[buckets == seg]
            if rows.size == 0:
                continue
            out[rows] = beta_win_probability(a[rows, 1], b[rows, 1], a[rows, 0], b[rows, 0])
        return out


@dataclass
class UCB1(_Counts):
    """UCB1: pull the arm with the best optimistic upper bound.

    Deterministic given the data, which is the point of including it -- a
    deterministic policy has propensities of exactly 0 and 1, so the estimators
    in :mod:`explabs.bandits.inference` have nothing to divide by.
    """

    n_buckets: int = 1
    name: str = "ucb1"

    def __post_init__(self) -> None:
        self.key = "ucb1"
        self.label = "UCB1 (pooled)"

    def _index(self, buckets: np.ndarray) -> np.ndarray:
        idx, n, obs = self._slice(buckets)
        total = n.sum(axis=1, keepdims=True)
        mean = np.divide(obs, n, out=np.zeros_like(obs), where=n > 0)
        bonus = np.sqrt(2.0 * np.log(np.maximum(total, 2.0)) / np.maximum(n, 1.0))
        return np.where(n > 0, mean + bonus, np.inf)

    def _decide(self, buckets: np.ndarray) -> np.ndarray:
        _, n, _ = self._slice(buckets)
        index = self._index(buckets)
        arm = (index[:, 1] > index[:, 0]).astype(np.int8)
        # Both arms get one pull before the bound means anything.
        arm = np.where(n[:, 0] == 0, 0, arm).astype(np.int8)
        arm = np.where((n[:, 0] > 0) & (n[:, 1] == 0), 1, arm).astype(np.int8)
        return arm

    def sample(self, buckets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._decide(buckets)

    def propensity(self, buckets: np.ndarray) -> np.ndarray:
        return self._decide(buckets).astype(float)


@dataclass
class EpsilonGreedy(_Counts):
    """Greedy on the running mean, uniform over both arms with probability epsilon.

    Included because epsilon is the one dial that moves regret and statistical
    power in opposite directions *in a controlled way*: epsilon is exactly the
    overlap the policy guarantees, so the inference it supports improves at a
    known rate as epsilon rises.
    """

    epsilon: float = 0.1
    n_buckets: int = 1
    name: str = "epsilon-greedy"

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        self.key = "eg" if self.n_buckets == 1 else f"eg{self.n_buckets}"
        self.label = f"$\\epsilon$-greedy ({self.epsilon:g})"

    def propensity(self, buckets: np.ndarray) -> np.ndarray:
        _, n, obs = self._slice(buckets)
        mean = np.divide(obs, n, out=np.zeros_like(obs), where=n > 0)
        # An exact tie is broken by a fair coin, so it is a 50/50 propensity, not
        # an arbitrary choice. At step one both means are zero by construction.
        greedy = np.where(mean[:, 1] > mean[:, 0], 1.0,
                          np.where(mean[:, 1] < mean[:, 0], 0.0, 0.5))
        return (1.0 - self.epsilon) * greedy + self.epsilon * 0.5

    def sample(self, buckets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return (rng.random(buckets.size) < self.propensity(buckets)).astype(np.int8)


@dataclass
class Run:
    """One batch of adaptive experiments, kept at user granularity."""

    policy: str
    key: str
    label: str
    dgp: ContextualDGP
    buckets: np.ndarray
    arms: np.ndarray
    propensities: np.ndarray
    rewards: np.ndarray
    x: np.ndarray
    regret: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def reps(self) -> int:
        return self.arms.shape[0]

    @property
    def horizon(self) -> int:
        return self.arms.shape[1]

    @property
    def treated_share(self) -> np.ndarray:
        """Share of traffic sent to the treatment arm, per replication."""
        return self.arms.mean(axis=1)

    def cumulative_regret(self) -> np.ndarray:
        """Cumulative regret, shape ``(reps, horizon)``."""
        return np.cumsum(self.regret, axis=1)

    def segments(self) -> np.ndarray:
        return self.buckets


def run_policy(
    policy,
    dgp: ContextualDGP,
    horizon: int,
    reps: int = 200,
    seed: int = 0,
    record_propensity: bool = True,
) -> Run:
    """Run ``reps`` independent copies of one policy to ``horizon`` users each.

    The replications are vectorised: every step updates arrays of length
    ``reps``, so the cost is one Python loop over the horizon rather than
    ``reps`` loops. That is what makes exact propensities affordable -- the
    quadrature behind Thompson sampling costs a few milliseconds per step no
    matter how many replications ride along.
    """
    rng_context = np.random.default_rng([seed, 3])
    rng_policy = np.random.default_rng([seed, 1])
    rng_reward = np.random.default_rng([seed, 2])

    x = rng_context.standard_normal((reps, horizon, 2))
    p0, p1 = dgp.probs(x.reshape(-1, 2))
    p0 = p0.reshape(reps, horizon)
    p1 = p1.reshape(reps, horizon)
    buckets = dgp.segments(x.reshape(-1, 2), policy.n_buckets).reshape(reps, horizon)

    policy.reset(reps, rng_policy)

    arms = np.zeros((reps, horizon), dtype=np.int8)
    props = np.full((reps, horizon), np.nan)
    rewards = np.zeros((reps, horizon))
    regret = np.zeros((reps, horizon))

    for t in range(horizon):
        b = buckets[:, t]
        if record_propensity:
            props[:, t] = policy.propensity(b)
        a = policy.sample(b, rng_policy)
        arms[:, t] = a

        chosen = np.where(a == 1, p1[:, t], p0[:, t])
        # One uniform per user per step, from a stream no policy touches, so the
        # reward draws are common across policies.
        rewards[:, t] = (rng_reward.random(reps) < chosen)
        regret[:, t] = np.maximum(p0[:, t], p1[:, t]) - chosen

        policy.observe(b, a, rewards[:, t])

    return Run(
        policy=policy.name,
        key=policy.key,
        label=policy.label,
        dgp=dgp,
        buckets=buckets,
        arms=arms,
        propensities=props,
        rewards=rewards,
        x=x,
        regret=regret,
        meta={"horizon": horizon, "reps": reps, "seed": seed},
    )
