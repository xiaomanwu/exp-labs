"""Post-adaptive inference: what a fixed-horizon estimator reports after a bandit ran.

Three estimators, all aimed at the same estimand -- the average treatment
effect over the arrival distribution:

``naive``
    difference in arm means. Correct when allocation ignores the covariates,
    because then the two arm samples are draws from the same population.
``ipw``
    Horvitz-Thompson with the propensity the policy actually used. Weights each
    user by the inverse of their probability of receiving the arm they got.
``aipw``
    IPW plus an outcome model, cross-fitted. The model soaks up the part of the
    variance that the weights leave behind, and the estimator stays consistent if
    *either* the propensities or the model are right.

Standard errors come from the influence function of each estimator. For IPW and
AIPW that influence function is exact given the recorded propensities; treating
propensities as known rather than estimated is what makes these intervals
conservative rather than invalid -- a distinction Hadad et al. (2021) is careful
about and this study reproduces numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

# A replication with fewer than this many effective observations on either arm
# has no estimate worth reporting. "Effective" means the sum of inverse
# propensities, which equals the arm's sample size under a fixed split and
# collapses when a policy has all but abandoned an arm.
MIN_EFFECTIVE = 5.0

# A step whose propensity is 0 or 1 contributes nothing to identification: the
# arm either was certain or impossible, so the inverse weight is 1 or undefined
# rather than a correction. A policy that spends most of its steps in that state
# -- UCB1, a greedy rule -- has no overlap, and Horvitz-Thompson silently stops
# estimating a mean and starts reporting a share-scaled sum. The share of
# degenerate steps is the diagnostic that catches it.
MAX_DEGENERATE_SHARE = 0.5


@dataclass(frozen=True)
class Result:
    """Per-replication point estimates and standard errors."""

    method: str
    estimate: np.ndarray
    se: np.ndarray

    def defined(self) -> np.ndarray:
        return np.isfinite(self.estimate) & np.isfinite(self.se)

    def summary(self, truth: float, alpha: float = 0.05) -> dict:
        """Bias, mean SE, coverage and two-sided rejection rate over replications."""
        ok = self.defined()
        est, se = self.estimate[ok], self.se[ok]
        if est.size == 0:
            return {"method": self.method, "reps": 0, "bias": np.nan, "mc_se": np.nan,
                    "se_mean": np.nan, "coverage": np.nan, "reject": np.nan}
        z = est / se
        crit = float(ndtri(1.0 - alpha / 2.0))
        covered = np.abs(est - truth) <= crit * se
        return {
            "method": self.method,
            "reps": int(est.size),
            "bias": float(est.mean() - truth),
            "mc_se": float(est.std(ddof=1) / np.sqrt(est.size)),
            "se_mean": float(se.mean()),
            "coverage": float(covered.mean()),
            "reject": float((np.abs(z) > crit).mean()),
        }


def _effective_counts(arm: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum of inverse propensities on each arm -- the diagnostic for overlap.

    A propensity of exactly 0 or 1 (UCB1, a greedy policy) makes the reciprocal
    infinite; the ``where`` keeps that from leaking into the other arm's count,
    and ``errstate`` keeps it from printing a warning for something that is a
    deliberate part of the measurement.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        n1 = np.where(arm == 1, 1.0 / e, 0.0).sum(axis=1)
        n0 = np.where(arm == 0, 1.0 / (1.0 - e), 0.0).sum(axis=1)
    return n0, n1


def _identified(arm: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Which replications have enough overlap for an inverse-propensity estimate."""
    degenerate = ((e <= 1e-12) | (e >= 1.0 - 1e-12)).mean(axis=1)
    n0, n1 = _effective_counts(arm, e)
    return (degenerate < MAX_DEGENERATE_SHARE) & (n0 >= MIN_EFFECTIVE) & (n1 >= MIN_EFFECTIVE)


def _mask(est: np.ndarray, ok: np.ndarray) -> np.ndarray:
    return np.where(ok, est, np.nan)


def naive(arm: np.ndarray, reward: np.ndarray) -> Result:
    """Difference in arm means with the textbook two-sample standard error."""
    treated = arm == 1
    n1 = treated.sum(axis=1).astype(float)
    n0 = arm.shape[1] - n1
    mean1 = np.where(n1 > 0, (reward * treated).sum(axis=1) / np.maximum(n1, 1), np.nan)
    mean0 = np.where(n0 > 0, (reward * ~treated).sum(axis=1) / np.maximum(n0, 1), np.nan)
    var1 = ((reward - mean1[:, None]) ** 2 * treated).sum(axis=1) / np.maximum(n1 - 1, 1)
    var0 = ((reward - mean0[:, None]) ** 2 * ~treated).sum(axis=1) / np.maximum(n0 - 1, 1)
    se = np.sqrt(np.where(n1 > 1, var1 / np.maximum(n1, 1), np.nan)
                 + np.where(n0 > 1, var0 / np.maximum(n0, 1), np.nan))
    return Result("naive", mean1 - mean0, se)


def ipw(arm: np.ndarray, reward: np.ndarray, e: np.ndarray) -> Result:
    """Horvitz-Thompson inverse-propensity estimator for the ATE.

    The term for a user is ``1{A=1} Y / e - 1{A=0} Y / (1-e)``; its mean over the
    horizon is the estimate and its standard deviation over ``sqrt(T)`` is the
    standard error. Written with ``where`` rather than a product so that a
    propensity of exactly 1 never produces ``0 * inf``.
    """
    horizon = arm.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(arm == 1, reward / e, 0.0)
        term0 = np.where(arm == 0, reward / (1.0 - e), 0.0)
    psi = term1 - term0
    est = psi.mean(axis=1)
    se = psi.std(axis=1, ddof=1) / np.sqrt(horizon)

    ok = _identified(arm, e)
    return Result("ipw", _mask(est, ok), _mask(se, ok))


def hajek(arm: np.ndarray, reward: np.ndarray, e: np.ndarray) -> Result:
    """Self-normalised IPW: divide by the realised weight mass instead of ``T``.

    Slightly lower variance than Horvitz-Thompson and the same limit; reported
    as a robustness column because it behaves much better when a few
    propensities are small.
    """
    horizon = arm.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.where(arm == 1, 1.0 / e, 0.0)
        w0 = np.where(arm == 0, 1.0 / (1.0 - e), 0.0)
    mass1, mass0 = w1.sum(axis=1), w0.sum(axis=1)
    mean1 = (w1 * reward).sum(axis=1) / np.maximum(mass1, 1e-12)
    mean0 = (w0 * reward).sum(axis=1) / np.maximum(mass0, 1e-12)

    # Ratio linearisation: the influence of user t on the arm mean is
    # ``w_t (y_t - mean) / E[mass]``, and E[mass] = T because E[1/e] = 1.
    psi = w1 * (reward - mean1[:, None]) - w0 * (reward - mean0[:, None])
    est = mean1 - mean0
    se = psi.std(axis=1, ddof=1) / np.sqrt(horizon)

    ok = _identified(arm, e)
    return Result("hajek", _mask(est, ok), _mask(se, ok))


def _fit_arm_model(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Logistic fit on the training rows, predictions on the test rows.

    Falls back to the training-set mean whenever a logistic fit is unavailable
    or degenerate -- a fold in which one arm has almost no data must not be
    allowed to emit a coefficient the size of a mountain.
    """
    from sklearn.linear_model import LogisticRegression

    fallback = np.full(test.sum(), float(y[train].mean()) if train.any() else 0.0)
    if train.sum() < 20 or np.unique(y[train]).size < 2:
        return fallback
    model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    try:
        model.fit(x[train], y[train])
        return model.predict_proba(x[test])[:, 1]
    except Exception:  # pragma: no cover - numerical guard, not a code path we expect
        return fallback


def aipw(
    arm: np.ndarray,
    reward: np.ndarray,
    e: np.ndarray,
    x: np.ndarray,
    folds: int = 2,
    seed: int = 0,
) -> Result:
    """Augmented IPW with cross-fitted outcome models.

    ``x`` has shape ``(reps, horizon, features)``. The outcome model is fit on
    one fold and evaluated on the other, which keeps the model's own errors from
    leaking into the estimate -- an in-sample fit would understate the residual
    and make the standard error too small.

    The influence function is evaluated with the *recorded* propensities. That
    is the conservative choice: propensities that depend on past data are
    themselves estimated, and ignoring that makes the interval slightly wider
    than it needs to be, never narrower.
    """
    reps, horizon = arm.shape
    estimates = np.full(reps, np.nan)
    errors = np.full(reps, np.nan)
    ok = _identified(arm, e)
    rng = np.random.default_rng(seed)

    for r in range(reps):
        if not ok[r]:
            continue
        a, y, e_r, x_r = arm[r], reward[r], e[r], x[r]
        fold = np.arange(horizon) % folds
        rng.shuffle(fold)

        m1 = np.empty(horizon)
        m0 = np.empty(horizon)
        for f in range(folds):
            test = fold == f
            train = ~test
            for value, out in ((1, m1), (0, m0)):
                sel = train & (a == value)
                out[test] = _fit_arm_model(x_r, y, sel, test)

        psi = (m1 - m0
               + np.where(a == 1, (y - m1) / e_r, 0.0)
               - np.where(a == 0, (y - m0) / (1.0 - e_r), 0.0))
        estimates[r] = psi.mean()
        errors[r] = psi.std(ddof=1) / np.sqrt(horizon)

    return Result("aipw", estimates, errors)


def composition_bias(run, n_buckets: int | None = None) -> np.ndarray:
    """Naive bias implied *purely* by who ended up in which arm.

    Write the naive estimate as a weighted average of segment means. If the
    policy sends a different share of each segment to the treatment, the arm
    means are averages over differently composed populations, and the gap
    between them is a bias that has nothing to do with sampling noise.

    This function computes that gap from the realised within-segment treated
    share and the true segment rates, so it needs no replications and no
    estimator: it is arithmetic on the allocation the policy actually produced.
    Comparing it with the measured naive bias separates *composition*
    (who got treated) from *selection* (which users a policy abandons early).
    """
    buckets = run.buckets
    n_buckets = int(buckets.max()) + 1 if n_buckets is None else n_buckets
    rates = run.dgp.segment_rates(n_buckets)
    reps, horizon = run.arms.shape

    treated = np.zeros(reps)
    control = np.zeros(reps)
    mass_t = np.zeros(reps)
    mass_c = np.zeros(reps)
    for s in range(n_buckets):
        mask = buckets == s
        n_s = mask.sum(axis=1).astype(float)
        n_t = (mask & (run.arms == 1)).sum(axis=1).astype(float)
        n_c = n_s - n_t
        treated += n_t * rates[s, 1]
        control += n_c * rates[s, 0]
        mass_t += n_t
        mass_c += n_c

    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = np.where(mass_t > 0, treated / mass_t, np.nan)
        mean0 = np.where(mass_c > 0, control / mass_c, np.nan)
    return mean1 - mean0 - run.dgp.ate


def design_matrix(run, kind: str = "full") -> np.ndarray:
    """Covariates handed to the AIPW outcome model.

    ``full`` uses the two continuous features, which is the model the data were
    generated from. ``segments`` uses only the bucket the policy saw -- the
    misspecified model, included to show that a wrong outcome model costs
    efficiency but does not reintroduce bias, because the weights are still
    right.
    """
    if kind == "full":
        return run.x
    if kind == "segments":
        buckets = run.buckets
        n = int(buckets.max()) + 1
        one_hot = np.stack([(buckets == j).astype(float) for j in range(n)], axis=2)
        return one_hot[:, :, 1:] if n > 1 else np.zeros(buckets.shape + (0,))
    raise ValueError(f"unknown feature set {kind!r}")
