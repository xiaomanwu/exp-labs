"""The three assignment designs, on top of the market in :mod:`market`.

Each function runs one whole experiment and returns a point estimate plus the
standard error a practitioner would report. They differ only in what gets
randomised -- riders, cities, or time -- which is exactly the choice this study
is about.

The framing that matters
------------------------
None of these designs is "biased" in the abstract. Each is unbiased for the
contrast its randomisation actually creates:

- user-level   -> the within-market contrast at the experiment's treated share
                  (:func:`market.direct_effect`), not the roll-out effect
- cluster      -> the global effect, at the cost of a much larger variance
- switchback   -> the global effect, at the cost of exposure to carryover

Calling the user-level design "biased" is shorthand for "it answers a question
nobody asked". :mod:`run` reports both the gap to the global effect and the gap
to the design's own estimand, so the two claims stay separable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from .market import (
    MarketConfig,
    draw_city,
    draw_city_effect,
    expected_trips,
)

Z95 = 1.959963984540054


@dataclass(frozen=True)
class DesignResult:
    """One experiment's estimate, with the standard errors each design offers.

    ``crit`` is the critical value the interval actually uses. It is not always
    1.96: a cluster design with six clusters needs a t quantile, and reporting
    the normal one is how small cluster experiments end up overconfident.
    """

    tau: float
    se: float
    se_naive: float | None = None
    crit: float = Z95
    n_clusters: int = 0
    units: int = 0
    units_spent: int = 0

    @property
    def lo(self) -> float:
        return self.tau - self.crit * self.se

    @property
    def hi(self) -> float:
        return self.tau + self.crit * self.se

    def covers(self, truth: float, crit: float | None = None) -> bool:
        c = self.crit if crit is None else crit
        return abs(self.tau - truth) <= c * self.se


def _two_sample_se(y_treated: np.ndarray, y_control: np.ndarray) -> float:
    """Welch standard error for a difference in means, unit = one rider."""
    if y_treated.size < 2 or y_control.size < 2:
        return float("nan")
    return float(
        np.sqrt(
            y_treated.var(ddof=1) / y_treated.size
            + y_control.var(ddof=1) / y_control.size
        )
    )


def user_level(
    cfg: MarketConfig,
    n_cities: int,
    rng: np.random.Generator,
    share: float = 0.5,
) -> DesignResult:
    """Randomise riders within each city; pool every city into one contrast.

    The standard error treats riders as independent units, which they are here:
    randomisation is at the rider level and the market state is shared by both
    arms, so it shifts both means together and cancels in the difference. A
    correct standard error around the wrong estimand is the whole problem.
    """
    outcomes, treated = [], []
    for _ in range(n_cities):
        city = draw_city_effect(cfg, rng)
        y, t, _ = draw_city(city, share, rng)
        outcomes.append(y)
        treated.append(t)

    y = np.concatenate(outcomes)
    t = np.concatenate(treated)
    tau = y[t].mean() - y[~t].mean()
    return DesignResult(tau=float(tau), se=_two_sample_se(y[t], y[~t]), units=y.size)


def cluster(
    cfg: MarketConfig,
    n_cities: int,
    rng: np.random.Generator,
) -> DesignResult:
    """Randomise whole cities; estimate from city-level means.

    Returns two standard errors on purpose. ``se`` is the cluster-robust one
    (a two-sample t on city means); ``se_naive`` is what you get by pooling
    riders and ignoring the clustering, which is the mistake this design is
    most often shipped with.
    """
    if n_cities % 2:
        raise ValueError("n_cities must be even for a balanced split")
    is_treated_city = rng.permutation(n_cities) < n_cities // 2

    city_means = np.empty(n_cities)
    outcomes, treated = [], []
    for i in range(n_cities):
        city = draw_city_effect(cfg, rng)
        y, t, _ = draw_city(city, 1.0 if is_treated_city[i] else 0.0, rng)
        city_means[i] = y.mean()
        outcomes.append(y)
        treated.append(t)

    treated_means = city_means[is_treated_city]
    control_means = city_means[~is_treated_city]
    tau = treated_means.mean() - control_means.mean()
    se_cluster = float(
        np.sqrt(
            treated_means.var(ddof=1) / treated_means.size
            + control_means.var(ddof=1) / control_means.size
        )
    )
    # With a handful of clusters the normal quantile is too optimistic. The
    # Welch df keeps the interval honest down to the smallest sizes we run.
    df = treated_means.size + control_means.size - 2
    crit = float(student_t.ppf(0.975, df))

    y = np.concatenate(outcomes)
    t = np.concatenate(treated)
    return DesignResult(
        tau=float(tau),
        se=se_cluster,
        se_naive=_two_sample_se(y[t], y[~t]),
        crit=crit,
        n_clusters=n_cities,
        units=y.size,
    )


def switchback(
    cfg: MarketConfig,
    n_cities: int,
    periods: int,
    rng: np.random.Generator,
    carryover: float = 0.0,
    block_len: int = 1,
    washout: int = 0,
) -> DesignResult:
    """Alternate the whole city between treated and control across periods.

    ``carryover`` is the share of the previous period's outcome that persists
    into the current one: a rider subsidised today keeps some of the extra
    demand tomorrow, and a driver who logged on for today's surge is still
    online tomorrow. This is the crossover model, which is what a switchback
    is, and pure alternation is its worst case. With ``k`` periods per arm the
    estimate is attenuated to

        (1 - lambda * (2k - 1) / k) * global effect

    because every period but the first follows a period of the opposite
    assignment, so both arms are contaminated by the same amount and the
    contamination cancels inside each arm rather than across arms. The factor
    tends to ``1 - 2 * lambda`` for a long experiment; at ``k = 4`` and
    ``lambda = 0.5`` it is close to zero, which is why "the switchback showed
    nothing" is not evidence of no effect.

    ``washout`` discards the first that many periods of every block, which
    removes the contamination exactly. Discarded periods still cost traffic --
    the estimate ignores them, the bill does not.

    The estimate is paired within city (each city contributes its own treated
    minus control mean), so city heterogeneity never enters the standard error.
    """
    if periods % block_len:
        raise ValueError("periods must be a multiple of block_len")
    n_blocks = periods // block_len
    if n_blocks % 2:
        raise ValueError("need an even number of blocks for a balanced design")
    if block_len - washout < 1:
        raise ValueError("washout leaves no usable period in a block")
    if not 0.0 <= carryover < 1.0:
        raise ValueError("carryover must be in [0, 1)")

    per_city = np.empty(n_cities)
    usable = 0
    for i in range(n_cities):
        city = draw_city_effect(cfg, rng)
        treated_periods: list[float] = []
        control_periods: list[float] = []
        # The experiment opens settled, so period 0 inherits nothing.
        previous = expected_trips(city, 1.0)

        for block in range(n_blocks):
            share = 1.0 if block % 2 == 0 else 0.0
            for j in range(block_len):
                steady = expected_trips(city, share)
                mean = (1.0 - carryover) * steady + carryover * previous
                y = rng.poisson(mean, city.riders)
                if j >= washout:
                    (treated_periods if share == 1.0 else control_periods).append(
                        float(y.mean())
                    )
                    usable += 1
                previous = steady

        per_city[i] = np.mean(treated_periods) - np.mean(control_periods)

    return DesignResult(
        tau=float(per_city.mean()),
        se=float(per_city.std(ddof=1) / np.sqrt(n_cities)),
        n_clusters=n_cities,
        units=usable * cfg.riders,
        units_spent=n_cities * periods * cfg.riders,
    )
