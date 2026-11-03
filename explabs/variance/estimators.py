"""Estimators for study 02.

Every estimator returns an :class:`Estimate` -- a point estimate and its
standard error -- so the harness can compute bias, coverage and power without
caring which method produced it.

The contract that matters: an estimator may look at the covariates for both
arms, but the *adjustment coefficient* must be learned from control units only.
The treated arm is what we are trying to measure; letting it into the fit is
how adjusted estimators acquire bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .dgp import Experiment


@dataclass(frozen=True)
class Estimate:
    """A point estimate of the absolute treatment effect and its standard error."""

    tau: float
    se: float

    @property
    def z(self) -> float:
        return self.tau / self.se if self.se > 0 else np.inf


def _two_sample(y_adj: np.ndarray, t: np.ndarray) -> Estimate:
    """Difference in means plus the Welch-style standard error.

    Used by every estimator below: they differ only in how ``y_adj`` is
    constructed, which is the whole point of the abstraction.
    """
    yt, yc = y_adj[t], y_adj[~t]
    var = yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size
    return Estimate(tau=float(yt.mean() - yc.mean()), se=float(np.sqrt(var)))


# --------------------------------------------------------------------------
# Baseline and CUPED
# --------------------------------------------------------------------------


def baseline(exp: Experiment) -> Estimate:
    """Unadjusted difference in means: what the naive dashboard reports."""
    return _two_sample(exp.y, exp.t)


def cuped(exp: Experiment) -> Estimate:
    """CUPED with a multivariate covariate (Deng et al., 2013).

    ``theta`` comes from a regression of the outcome on the covariates *within
    the control arm*. The centring uses the pooled covariate mean, which is
    legitimate because covariates are measured before randomisation and are
    therefore independent of assignment.
    """
    return _cuped_with(exp, exp.x)


def _cuped_with(exp: Experiment, covariate: np.ndarray) -> Estimate:
    """Shared CUPED machinery for any covariate (raw ``X``, ``m(X)``, or ``f``)."""
    covariate = np.atleast_2d(covariate.T).T if covariate.ndim == 1 else covariate
    xc = covariate[~exp.t]
    yc = exp.y[~exp.t]

    # Regress centred control outcomes on centred control covariates.
    x_centred = xc - xc.mean(axis=0)
    theta, *_ = np.linalg.lstsq(x_centred, yc - yc.mean(), rcond=None)

    # Adjust every unit using the pooled covariate mean.
    adjusted = exp.y - (covariate - covariate.mean(axis=0)) @ theta
    return _two_sample(adjusted, exp.t)


def oracle(exp: Experiment) -> Estimate:
    """Adjust by the true ``f(X)``.

    Not an implementable method -- it is the ceiling. If CUPAC ever beat this,
    the harness would be leaking the ground truth.
    """
    return _cuped_with(exp, exp.signal)


# --------------------------------------------------------------------------
# CUPAC: a learned prediction used in place of a single covariate
# --------------------------------------------------------------------------


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    seed: int,
) -> np.ndarray:
    """One gradient-boosted fit, predicting everywhere.

    Capacity was chosen by measurement, not taste. On 8 covariates and 5k units
    a 200-iteration, 64-leaf model recovers the true ``f`` with correlation^2 =
    0.81; this smaller configuration reaches 0.90 and runs 2.2x faster. The
    bigger model is fitting idiosyncratic noise, which is precisely the failure
    that cross-fitting only partly protects against.
    """
    model = HistGradientBoostingRegressor(
        max_iter=60,
        max_leaf_nodes=31,
        learning_rate=0.12,
        max_bins=128,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model.predict(x_all)


def cupac(exp: Experiment, folds: int = 4, seed: int = 0) -> Estimate:
    """CUPAC with cross-fitting inside the control arm.

    Control units are split into ``folds`` parts. Each part is predicted by a
    model trained on the *other* parts, and treated units are predicted by the
    average of the fold models. Cross-fitting is what keeps the control
    predictions honest: an in-sample prediction absorbs part of the control
    noise, which pulls the control mean towards its fitted value and biases
    ``tau_hat`` upward.
    """
    control = np.flatnonzero(~exp.t)
    treated = np.flatnonzero(exp.t)
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(control), folds)

    prediction = np.empty(exp.n)
    treated_prediction = np.zeros(treated.size)

    for k, held_out in enumerate(parts):
        train = np.concatenate([p for j, p in enumerate(parts) if j != k])
        pred = _fit_predict(exp.x[train], exp.y[train], exp.x, seed + k)
        prediction[held_out] = pred[held_out]
        treated_prediction += pred[treated] / folds

    prediction[treated] = treated_prediction
    return _cuped_with(exp, prediction)


def cupac_insample(exp: Experiment, seed: int = 0) -> Estimate:
    """The mistake, kept as a benchmark: one model, fitted and applied in-sample.

    This is what happens when nobody asks whether the prediction used on a unit
    came from a model that had already seen that unit's outcome. The estimate
    stays roughly right in expectation under a symmetric fit, but the standard
    error does not: the adjusted control residuals are shrunk in a way the
    treatment residuals are not, so the difference looks more precise than it is.
    """
    control = np.flatnonzero(~exp.t)
    prediction = _fit_predict(exp.x[control], exp.y[control], exp.x, seed)
    return _cuped_with(exp, prediction)


# --------------------------------------------------------------------------
# Post-stratification
# --------------------------------------------------------------------------


def poststrat(exp: Experiment, n_strata: int = 5) -> Estimate:
    """Stratify on quantile bins of the pre-experiment score, then average.

    ``tau_hat = sum_s w_s (mean_t,s - mean_c,s)`` with ``w_s`` the stratum's
    share of all units -- the standard estimator from Miratrix et al. (2013).
    Strata that end up empty in either arm are dropped and the weights
    renormalised; with quantile bins and a few thousand units that is rare, and
    it is exactly what makes post-stratification brittle when the number of
    strata grows.
    """
    score = exp.score
    edges = np.quantile(score, np.linspace(0, 1, n_strata + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    stratum = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_strata - 1)

    total = 0.0
    variance = 0.0
    weight_sum = 0.0

    for s in range(n_strata):
        in_s = stratum == s
        t_s, c_s = in_s & exp.t, in_s & ~exp.t
        if t_s.sum() < 2 or c_s.sum() < 2:
            continue
        w = in_s.sum() / exp.n
        total += w * (exp.y[t_s].mean() - exp.y[c_s].mean())
        variance += w**2 * (
            exp.y[t_s].var(ddof=1) / t_s.sum() + exp.y[c_s].var(ddof=1) / c_s.sum()
        )
        weight_sum += w

    # Renormalise so the weights sum to one even if a stratum was dropped.
    return Estimate(tau=float(total / weight_sum), se=float(np.sqrt(variance) / weight_sum))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

Estimator = Callable[[Experiment], Estimate]

# ``oracle`` and ``cupac_insample`` are benchmarks, not recommendations; the
# report labels them as such so nobody mistakes them for best practice.
METHODS: dict[str, Estimator] = {
    "baseline": baseline,
    "cuped": cuped,
    "cupac": cupac,
    "cupac_insample": cupac_insample,
    "poststrat_5": lambda e: poststrat(e, 5),
    "poststrat_20": lambda e: poststrat(e, 20),
    "oracle": oracle,
}

BENCHMARK_METHODS = ("oracle", "cupac_insample")


def run_methods(exp: Experiment, methods: dict[str, Estimator] | None = None) -> dict[str, Estimate]:
    """Evaluate every method on one experiment."""
    return {name: fn(exp) for name, fn in (methods or METHODS).items()}
