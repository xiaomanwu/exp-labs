"""Semi-synthetic data generator for study 02 (variance reduction).

The point of this module is that the *signal* is known. We construct

    Y = mu + tau * T + signal_sd * f(X) + noise_sd * eps

with ``f`` normalised to mean 0 and variance 1, so the population quantities
that the estimators are supposed to recover are available in closed form:

    Var(Y)          = signal_sd^2 + noise_sd^2
    rho^2           = signal_sd^2 / Var(Y)          <- variance reduction ceiling
    best linear R^2 of f on X, call it ``r2_linear``

CUPED is a *linear* adjustment, so it cannot remove more than
``rho^2 * r2_linear`` of the variance. An oracle that adjusts by the true
``f(X)`` removes ``rho^2``. The gap between those two numbers is exactly the
room that CUPAC is trying to fill, which is why both are reported.

``signal`` controls how much structure is left after linear projection:

``linear``       f is linear in X, so r2_linear = 1 and CUPED is optimal
``interaction``  f has an interaction, a periodic term and a hinge; CUPED
                 leaves real structure on the table
``threshold``    f is mostly a step function in one covariate; a step cannot be
                 fit by a line, but quantile bins can approximate it

Everything stochastic in the covariate distribution and in ``f`` is derived
from fixed seeds, so a given ``(p, n_factors, signal)`` always describes the
same population. Only the sampling of units is fresh per replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SIGNAL_KINDS = ("linear", "interaction", "threshold")

# Size of the fixed draw used to normalise f and to measure its linear R^2.
# Large enough that the constants are stable to ~3 significant figures.
_CALIBRATION_N = 200_000


@dataclass(frozen=True)
class DGPConfig:
    """Shape and ground truth of one simulated experiment.

    Parameters
    ----------
    n_per_arm:
        Units per arm. 50/50 allocation throughout.
    p, n_factors:
        Covariate dimension and the number of latent factors behind it. The
        factors create correlation between covariates, which matters for
        post-stratification and for a tree model (a tree splits on whichever
        covariate is informative; with 8 independent covariates it would have
        to work harder).
    rho:
        Target correlation between the outcome and its *explainable* part.
        With ``signal="linear"`` this is the achieved correlation, and the
        theoretical variance reduction is exactly ``rho**2``.
    signal:
        One of :data:`SIGNAL_KINDS`.
    mean:
        Control-arm mean of the metric, in natural units (revenue per user).
    cv:
        Coefficient of variation of the outcome, i.e. ``sd / mean``. Revenue
        metrics in practice sit between 0.5 and 3; 0.5 keeps Monte Carlo noise
        low without making the problem trivial.
    tau_rel:
        True relative effect on the treatment arm. ``0.0`` puts us under H0.
    """

    n_per_arm: int = 5_000
    p: int = 8
    n_factors: int = 2
    rho: float = 0.7
    signal: str = "linear"
    mean: float = 100.0
    cv: float = 0.5
    tau_rel: float = 0.01

    def __post_init__(self) -> None:
        if self.signal not in SIGNAL_KINDS:
            raise ValueError(f"signal must be one of {SIGNAL_KINDS}, got {self.signal!r}")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if self.n_per_arm < 50:
            raise ValueError("n_per_arm must be at least 50")
        if self.cv <= 0:
            raise ValueError("cv must be positive")

    @property
    def total_sd(self) -> float:
        return self.mean * self.cv

    @property
    def signal_sd(self) -> float:
        """Standard deviation of the explainable component ``f(X)``."""
        return self.rho * self.total_sd

    @property
    def noise_sd(self) -> float:
        """Standard deviation of the idiosyncratic component."""
        return self.total_sd * np.sqrt(1.0 - self.rho**2)

    @property
    def tau(self) -> float:
        """True absolute effect on the treatment arm."""
        return self.tau_rel * self.mean

    def label(self) -> str:
        return (
            f"{self.signal:<11} rho={self.rho:.2f} n={self.n_per_arm//1000}k/arm "
            f"tau={self.tau_rel:+.1%}"
        )


# --------------------------------------------------------------------------
# Population constants: covariate covariance, signal shape, linear R^2
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _loadings(p: int, n_factors: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor-model loadings for a ``p``-dimensional covariate vector.

    Returns ``(A, d)`` such that ``X = Z @ A.T + E * d`` has unit-variance
    columns, with ``Z`` latent factors and ``E`` idiosyncratic noise.
    """
    rng = np.random.default_rng(20_240 + 10 * p + n_factors)
    loadings = rng.normal(size=(p, n_factors)) * 0.8
    uniqueness = rng.uniform(0.3, 1.0, size=p)
    cov = loadings @ loadings.T + np.diag(uniqueness)
    scale = np.sqrt(np.diag(cov))
    return loadings / scale[:, None], np.sqrt(uniqueness) / scale


@lru_cache(maxsize=None)
def _weights(p: int) -> np.ndarray:
    """Fixed unit-norm weight vector defining the linear part of ``f``."""
    rng = np.random.default_rng(991 + p)
    w = rng.normal(size=p)
    return w / np.linalg.norm(w)


def _draw_covariates(
    rng: np.random.Generator, n: int, p: int, n_factors: int
) -> np.ndarray:
    """Draw ``n`` covariates from the factor model, shape ``(n, p)``."""
    a, d = _loadings(p, n_factors)
    return rng.normal(size=(n, n_factors)) @ a.T + rng.normal(size=(n, p)) * d


def _raw_signal(x: np.ndarray, w: np.ndarray, kind: str) -> np.ndarray:
    """The un-normalised signal ``f`` before it is scaled to unit variance."""
    if kind == "linear":
        return x @ w
    if kind == "interaction":
        # A linear backbone plus three structures a line cannot express: an
        # interaction, a periodic term, a hinge. The backbone matters -- a
        # signal with no linear part would make CUPED a straw man, and real
        # metrics always have *some* linear component.
        return (
            1.6 * (x @ w)
            + 1.0 * (x @ w) * x[:, 0]
            + 0.8 * np.sin(1.5 * x[:, 1])
            + 1.2 * np.maximum(x[:, 2], 0.0) * x[:, 3]
        )
    # "threshold": mostly a step in x[:, 0], plus mild smooth terms.
    return (
        2.0 * (x[:, 0] > 0.6).astype(float)
        + 0.6 * x[:, 1]
        + 0.4 * x[:, 2] ** 2
    )


@lru_cache(maxsize=None)
def signal_constants(
    p: int, n_factors: int, kind: str
) -> tuple[float, float, np.ndarray, float]:
    """Normalisation and linear fit of ``f``, measured on a fixed large draw.

    Returns ``(centre, scale, beta, r2_linear)`` where ``beta`` are the OLS
    coefficients of ``f`` on ``X`` (used as the "pre-experiment score" that
    post-stratification bins on) and ``r2_linear`` is the population R^2 of
    that regression.
    """
    x = _draw_covariates(np.random.default_rng(4_242), _CALIBRATION_N, p, n_factors)
    raw = _raw_signal(x, _weights(p), kind)
    centre, scale = float(raw.mean()), float(raw.std())

    target = (raw - centre) / scale
    design = np.column_stack([np.ones_like(target), x])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    r2 = 1.0 - residual.var() / target.var()
    return centre, scale, beta[1:], float(r2)


def theoretical_reduction(cfg: DGPConfig) -> dict[str, float]:
    """Population variance-reduction ratios, in closed form.

    ``cuped`` is what a linear adjustment can reach;
    ``oracle`` is what adjusting by the true ``f(X)`` reaches.
    """
    _, _, _, r2_linear = signal_constants(cfg.p, cfg.n_factors, cfg.signal)
    return {
        "rho_squared": cfg.rho**2,
        "r2_linear": r2_linear,
        "cuped": cfg.rho**2 * r2_linear,
        "oracle": cfg.rho**2,
    }


# --------------------------------------------------------------------------
# Sampling one experiment
# --------------------------------------------------------------------------


@dataclass
class Experiment:
    """One realisation: outcomes, assignment, covariates and the true signal.

    ``signal`` is the standardised ``f(X)``. It is returned so that the
    oracle estimator can adjust by it; no honest estimator gets to see it.
    """

    cfg: DGPConfig
    y: np.ndarray
    t: np.ndarray
    x: np.ndarray
    signal: np.ndarray
    score: np.ndarray

    @property
    def n(self) -> int:
        return self.y.size


def draw_experiment(cfg: DGPConfig, rng: np.random.Generator) -> Experiment:
    """Draw one two-arm experiment with a known effect."""
    n = 2 * cfg.n_per_arm
    x = _draw_covariates(rng, n, cfg.p, cfg.n_factors)
    centre, scale, _, _ = signal_constants(cfg.p, cfg.n_factors, cfg.signal)
    signal = (_raw_signal(x, _weights(cfg.p), cfg.signal) - centre) / scale

    treatment = np.zeros(n, dtype=bool)
    treatment[rng.permutation(n)[: cfg.n_per_arm]] = True

    y = (
        cfg.mean
        + cfg.tau * treatment
        + cfg.signal_sd * signal
        + cfg.noise_sd * rng.normal(size=n)
    )

    # The "pre-experiment score": a linear index of the covariates, estimated
    # once on the calibration draw. This is the analyst's stand-in for a
    # pre-period metric -- available before randomisation, unaffected by it.
    _, _, beta, _ = signal_constants(cfg.p, cfg.n_factors, cfg.signal)
    score = x @ beta

    return Experiment(cfg=cfg, y=y, t=treatment, x=x, signal=signal, score=score)


if __name__ == "__main__":
    for kind in SIGNAL_KINDS:
        cfg = DGPConfig(signal=kind, rho=0.7, n_per_arm=2_000)
        stats = theoretical_reduction(cfg)
        print(f"{kind:<12} rho^2={stats['rho_squared']:.3f}  "
              f"R^2_linear={stats['r2_linear']:.3f}  "
              f"ceiling(CUPED)={stats['cuped']:.3f}  "
              f"ceiling(oracle)={stats['oracle']:.3f}")

    # Sanity: the covariate columns should be unit variance and correlated.
    x = _draw_covariates(np.random.default_rng(0), 100_000, 8, 2)
    print("\ncovariate sd :", np.round(x.std(axis=0), 3))
    print("max |corr|   :", round(float(np.abs(np.corrcoef(x.T) - np.eye(8)).max()), 3))
