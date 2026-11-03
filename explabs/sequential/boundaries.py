"""Group-sequential boundaries by the Armitage-McPherson recursion.

Given look times expressed as information fractions ``t_1 < ... < t_K`` and a
spending function ``alpha*(t)``, compute boundaries ``b_1..b_K`` such that

    P(|Z_k| > b_k and |Z_j| <= b_j for all j < k)
        = alpha*(t_k) - alpha*(t_{k-1})

where ``Z`` follows the canonical joint distribution of sequential test
statistics: independent increments, ``Cov(Z_i, Z_j) = sqrt(t_i / t_j)`` for
``i < j``.

The recursion keeps, on a grid, the sub-density of ``Z_k`` restricted to paths
that have not crossed an earlier boundary, then integrates out the previous
look:

    f_k(z) = integral over |u| <= b_{k-1} of
             f_{k-1}(u) * phi((z - rho*u) / s) / s du,
             with rho = sqrt(t_{k-1} / t_k), s = sqrt(1 - rho^2)

This is the standard numerical approach (Armitage, McPherson & Rowe 1969;
Jennison & Turnbull 2000, ch. 19). Accuracy is roughly 1e-5 on a 600-point grid
for K <= 30, which is enough to check the boundaries against published tables —
see ``validate.py``, which does exactly that.

Why not just simulate the boundaries? Because a boundary that is off by 0.01 in
z is a type-I error rate that is off by a few percent, and that error is
invisible in a Monte Carlo run of a few thousand replications.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def spending_obf(t: np.ndarray, alpha: float) -> np.ndarray:
    """Lan-DeMets approximation to O'Brien-Fleming spending.

    Spends very little early and approaches the nominal level at the end, which
    is what makes early stopping require overwhelming evidence.
    """
    t = np.asarray(t, dtype=float)
    return 2.0 - 2.0 * norm.cdf(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(t))


def spending_pocock(t: np.ndarray, alpha: float) -> np.ndarray:
    """Lan-DeMets approximation to Pocock spending: roughly constant boundaries."""
    t = np.asarray(t, dtype=float)
    return alpha * np.log(1.0 + (np.e - 1.0) * t)


def spending_linear(t: np.ndarray, alpha: float) -> np.ndarray:
    """Spend error evenly across information: the most aggressive schedule."""
    return alpha * np.asarray(t, dtype=float)


SPENDING_FUNCTIONS = {
    "obf": spending_obf,
    "pocock": spending_pocock,
    "linear": spending_linear,
}


def group_sequential_boundaries(
    t: np.ndarray,
    alpha: float = 0.05,
    spending: str = "obf",
    n_grid: int = 601,
    z_max: float = 9.0,
) -> np.ndarray:
    """Two-sided z-boundaries for the look schedule given by ``t``.

    Parameters
    ----------
    t:
        Information fractions of the looks, strictly increasing, last one <= 1.
    alpha:
        Family-wise two-sided error rate for the whole experiment.
    spending:
        Key into :data:`SPENDING_FUNCTIONS`, or a callable ``(t, alpha) -> alpha``.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t must be a non-empty 1-D array of information fractions")
    if np.any(np.diff(t) <= 0):
        raise ValueError("information fractions must be strictly increasing")
    if t[-1] > 1.0 + 1e-12:
        raise ValueError("information fraction cannot exceed 1")

    spend = SPENDING_FUNCTIONS[spending] if isinstance(spending, str) else spending

    # Cumulative spend per look, forced to be non-decreasing (an increasing
    # schedule is a modelling assumption; some spending functions violate it
    # numerically at very small t).
    cumulative = np.minimum(np.maximum.accumulate(spend(t, alpha)), alpha)
    increments = np.diff(np.concatenate([[0.0], cumulative]))

    z = np.linspace(-z_max, z_max, n_grid)
    weights = np.full(n_grid, 2.0 * z_max / (n_grid - 1))
    weights[[0, -1]] /= 2.0  # trapezoid rule

    bounds = np.empty(t.size)
    density = norm.pdf(z)  # look 1: no earlier boundary to condition on

    for k in range(t.size):
        if k > 0:
            rho = np.sqrt(t[k - 1] / t[k])
            s = np.sqrt(1.0 - rho**2)
            transition = norm.pdf((z[:, None] - rho * z[None, :]) / s) / s
            density = transition @ (density * weights)

        total = float(np.sum(density * weights))
        running = np.concatenate([[0.0], np.cumsum((density * weights)[:-1])])

        def tail(b: float) -> float:
            inside = np.interp(b, z, running) - np.interp(-b, z, running)
            return total - inside

        lo, hi = 0.0, z_max
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if tail(mid) > increments[k]:
                lo = mid
            else:
                hi = mid
        bounds[k] = 0.5 * (lo + hi)

        # Keep only the paths that survived this look.
        density = np.where(np.abs(z) <= bounds[k], density, 0.0)

    return bounds


def schedule_boundaries(
    look_days: np.ndarray,
    horizon_days: int,
    alpha: float = 0.05,
    spending: str = "obf",
) -> np.ndarray:
    """Boundaries for looks taken on specific days of a fixed-length experiment.

    Converts days to information fractions assuming information accrues linearly
    in time (constant daily traffic). Unequal spacing is handled exactly: the
    recursion only ever needs ``t``.
    """
    look_days = np.asarray(look_days, dtype=float)
    t = (look_days + 1.0) / float(horizon_days)
    return group_sequential_boundaries(t, alpha=alpha, spending=spending)


def crossing_probabilities(bounds: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Recover the per-look crossing probabilities implied by ``bounds``.

    Used by the validation suite: the numbers should sum to ``alpha`` and match
    the spending-function increments, which is a much sharper check than
    comparing a single boundary.
    """
    bounds = np.asarray(bounds, dtype=float)
    t = np.asarray(t, dtype=float)

    z = np.linspace(-9.0, 9.0, 1201)
    weights = np.full(z.size, 18.0 / (z.size - 1))
    weights[[0, -1]] /= 2.0

    density = norm.pdf(z)
    probs = np.empty(bounds.size)

    for k in range(bounds.size):
        if k > 0:
            rho = np.sqrt(t[k - 1] / t[k])
            s = np.sqrt(1.0 - rho**2)
            density = (norm.pdf((z[:, None] - rho * z[None, :]) / s) / s) @ (density * weights)

        outside = np.abs(z) > bounds[k]
        probs[k] = float(np.sum((density * weights)[outside]))
        density = np.where(~outside, density, 0.0)

    return probs


if __name__ == "__main__":
    # Sanity output: with one look the boundary must be the fixed-sample 1.96,
    # and the five-look OBF boundaries should reproduce the published table.
    print("K = 1:", np.round(group_sequential_boundaries(np.array([1.0])), 4))
    for name in ("obf", "pocock"):
        b = group_sequential_boundaries(np.linspace(0.2, 1.0, 5), spending=name)
        print(f"K = 5, {name}:", np.round(b, 4))
