"""Correctness checks for study 02.

The claims this study makes are numerical, so the checks are numerical too:

1. **The generator is what it says it is.** Covariates have the advertised
   moments, the signal has unit variance, and the linear R^2 measured on the
   calibration draw agrees with an independent draw. Every theoretical ceiling
   in the report is derived from that R^2, so if it drifts, the targets drift
   with it.

2. **The estimators hit the ceiling theory says they should hit.** CUPED's
   variance ratio must equal ``1 - rho^2 R^2_lin`` and the oracle's must equal
   ``1 - rho^2``. These are the only checks that distinguish "our CUPED is
   correct" from "our CUPED is close to something".

3. **Nothing is biased, and the intervals cover.** Unbiasedness on every method,
   type-I error at nominal, and coverage of the nominal 95% interval.

4. **The known failure mode fails the way it is supposed to.** In-sample CUPAC
   must understate its standard error; if it ever came out calibrated, the
   demonstration in the write-up would be wrong.

Usage::

    python -m explabs.variance.validate                # ~5 min on 8 cores
    python -m explabs.variance.validate --quick        # ~40 s
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np

from ..checks import (
    Checker,
    bootstrap_se_of_ratio,
    bootstrap_se_of_reduction_difference,
)
from .dgp import (
    DGPConfig,
    _draw_covariates,
    _raw_signal,
    _weights,
    draw_experiment,
    signal_constants,
    theoretical_reduction,
)
from .estimators import METHODS, baseline, cupac, cuped, oracle, poststrat
from .run import ALPHA, Z_ALPHA, _one_rep


def collect(
    cfg: DGPConfig,
    methods: tuple[str, ...],
    reps: int,
    seed: int = 0,
    n_jobs: int = 8,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run ``reps`` replications and return ``{method: (taus, ses)}``."""
    from joblib import Parallel, delayed, parallel_backend

    tasks = [(cfg, seed + i, methods) for i in range(reps)]
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(t) for t in tasks)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in methods:
        out[name] = (
            np.array([r[name][0] for r in results]),
            np.array([r[name][1] for r in results]),
        )
    return out


# --------------------------------------------------------------------------
# 1. Generator
# --------------------------------------------------------------------------


def check_generator(ck: Checker) -> None:
    x = _draw_covariates(np.random.default_rng(1_234), 200_000, 8, 2)
    sd = x.std(axis=0)
    ck.check(
        "dgp/covariate-scale",
        f"column sd min={sd.min():.4f} max={sd.max():.4f} (target 1)",
        bool(np.all(np.abs(sd - 1) < 0.01)),
    )

    corr = np.corrcoef(x.T)
    off_diagonal = np.abs(corr - np.eye(8)).max()
    ck.check(
        "dgp/covariate-correlation",
        f"max |off-diagonal correlation| = {off_diagonal:.3f} "
        f"(factor model, so > 0 is expected; 0 would mean a broken loading matrix)",
        off_diagonal > 0.05,
    )

    for kind in ("linear", "interaction", "threshold"):
        centre, scale, _, _ = signal_constants(8, 2, kind)
        fresh = _draw_covariates(np.random.default_rng(555), 200_000, 8, 2)
        signal = (_raw_signal(fresh, _weights(8), kind) - centre) / scale
        ck.check(
            f"dgp/signal-moments[{kind}]",
            f"mean={signal.mean():+.4f} sd={signal.std():.4f} (target 0, 1)",
            abs(signal.mean()) < 0.01 and abs(signal.std() - 1) < 0.01,
        )


def check_theory_is_measurable(ck: Checker) -> None:
    """Recompute the linear R^2 on an independent draw.

    The ceilings quoted in the report are ``rho^2 * R^2_lin``. If ``R^2_lin``
    were an artefact of the calibration draw, every "CUPED is at its ceiling"
    claim downstream would be comparing against the wrong number.
    """
    for kind in ("linear", "interaction", "threshold"):
        centre, scale, _, r2_calibration = signal_constants(8, 2, kind)
        x = _draw_covariates(np.random.default_rng(90_090), 150_000, 8, 2)
        target = (_raw_signal(x, _weights(8), kind) - centre) / scale
        design = np.column_stack([np.ones_like(target), x])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ beta
        r2_fresh = 1.0 - residual.var() / target.var()
        ck.check(
            f"theory/r2-linear[{kind}]",
            f"calibration {r2_calibration:.4f} vs independent draw {r2_fresh:.4f}",
            abs(r2_calibration - r2_fresh) < 0.01,
        )


# --------------------------------------------------------------------------
# 2. Ceilings
# --------------------------------------------------------------------------


def check_ceilings(ck: Checker, reps: int, n_jobs: int) -> None:
    for kind in ("linear", "interaction"):
        cfg = DGPConfig(signal=kind, rho=0.7, n_per_arm=5_000, tau_rel=0.01)
        theory = theoretical_reduction(cfg)
        got = collect(cfg, ("baseline", "cuped", "oracle"), reps, n_jobs=n_jobs)
        base = got["baseline"][0]

        for method, target in (("cuped", theory["cuped"]), ("oracle", theory["oracle"])):
            ratio = got[method][0].var(ddof=1) / base.var(ddof=1)
            se = bootstrap_se_of_ratio(got[method][0], base)
            observed = 1.0 - ratio
            # 3 bootstrap SE, floored so a lucky batch cannot make this strict.
            tol = max(3.0 * se, 0.01)
            ck.check(
                f"ceiling/{method}[{kind}]",
                f"variance reduction {observed:.4f} vs theory {target:.4f} "
                f"(3SE = {tol:.4f})",
                abs(observed - target) <= tol,
            )


# --------------------------------------------------------------------------
# 3. Bias, type-I error, coverage
# --------------------------------------------------------------------------


def check_unbiased(ck: Checker, reps: int, n_jobs: int) -> None:
    cfg = DGPConfig(signal="interaction", rho=0.7, n_per_arm=5_000, tau_rel=0.01)
    got = collect(cfg, tuple(METHODS), reps, n_jobs=n_jobs)

    for name, (tau, se) in got.items():
        mc_se = tau.std(ddof=1) / np.sqrt(reps)
        bias = tau.mean() - cfg.tau
        # 3 Monte Carlo standard errors. Every method here is unbiased by
        # construction; a failure means an estimator is using information it
        # should not have.
        ck.check(
            f"bias/{name}",
            f"bias {bias:+.4f} ({bias / mc_se:+.2f} MC SE), 3SE = {3 * mc_se:.4f}",
            abs(bias) <= 3 * mc_se,
        )


def check_type_one(ck: Checker, reps: int, n_jobs: int) -> None:
    cfg = DGPConfig(signal="interaction", rho=0.7, n_per_arm=5_000, tau_rel=0.0)
    got = collect(cfg, tuple(METHODS), reps, n_jobs=n_jobs)

    for name, (tau, se) in got.items():
        rate = float(np.mean(np.abs(tau / se) > Z_ALPHA))
        # Wilson half-width at the observed rate; a binomial tolerance would
        # collapse to zero if a method never rejected.
        p = min(max(rate, 0.02), 0.98)
        tol = 4.0 * np.sqrt(p * (1 - p) / reps)
        ck.check(
            f"type-i/{name}",
            f"rejection rate {rate:.3f} under H0 (4SE = {tol:.3f})",
            rate <= ALPHA + tol,
        )


def check_coverage(ck: Checker, reps: int, n_jobs: int) -> None:
    cfg = DGPConfig(signal="linear", rho=0.7, n_per_arm=5_000, tau_rel=0.01)
    got = collect(cfg, ("baseline", "cuped", "oracle"), reps, n_jobs=n_jobs)
    tol = 4.0 * np.sqrt(0.95 * 0.05 / reps)

    for name, (tau, se) in got.items():
        coverage = float(np.mean(np.abs(tau - cfg.tau) <= Z_ALPHA * se))
        ck.check(
            f"coverage/{name}",
            f"95% interval covers {coverage:.3f} of the time (4SE = {tol:.3f})",
            abs(coverage - 0.95) <= tol,
        )


# --------------------------------------------------------------------------
# 4. The failure modes, and no leakage
# --------------------------------------------------------------------------


def check_no_leakage(ck: Checker) -> None:
    """Permuting treated outcomes must not move the estimate at all.

    The treated mean is preserved by a permutation within the treated arm, so a
    method whose adjustment coefficient is estimated on control units only is
    exactly invariant. Any movement means the treated arm leaked into the fit.

    Post-stratification is deliberately excluded: it is a design-based
    estimator that legitimately uses treated values *within each stratum*, so a
    permutation that moves units across strata is supposed to change the answer.
    Its unbiasedness is covered by ``bias/poststrat_*`` instead.
    """
    cfg = DGPConfig(signal="interaction", rho=0.7, n_per_arm=2_000, tau_rel=0.01)
    exp = draw_experiment(cfg, np.random.default_rng(4))
    shuffled = exp.y.copy()
    idx = np.flatnonzero(exp.t)
    shuffled[idx] = shuffled[np.random.default_rng(5).permutation(idx)]
    exp2 = replace(exp, y=shuffled)

    for name, fn in (("cuped", cuped), ("oracle", oracle), ("cupac", cupac),
                     ("cupac_insample", lambda e: METHODS["cupac_insample"](e))):
        delta = abs(fn(exp).tau - fn(exp2).tau)
        ck.check(
            f"leakage/{name}",
            f"|estimate change| = {delta:.2e} when treated outcomes are permuted",
            delta < 1e-8,
        )


def check_cupac_pitfall(ck: Checker, reps: int, n_jobs: int) -> None:
    """In-sample CUPAC must understate its standard error, and CUPAC must beat CUPED.

    Both halves matter. The first is the failure being demonstrated; the second
    is the reason anyone would use a learned covariate at all.
    """
    cfg = DGPConfig(signal="interaction", rho=0.7, n_per_arm=5_000, tau_rel=0.01)
    theory = theoretical_reduction(cfg)
    got = collect(
        cfg,
        ("baseline", "cuped", "cupac", "cupac_insample", "oracle"),
        reps,
        n_jobs=n_jobs,
    )
    base_var = got["baseline"][0].var(ddof=1)

    def reduction(name: str) -> float:
        return 1.0 - got[name][0].var(ddof=1) / base_var

    def se_calibration(name: str) -> float:
        tau, se = got[name]
        return float(se.mean() / tau.std(ddof=1))

    cross_fit_cal = se_calibration("cupac")
    in_sample_cal = se_calibration("cupac_insample")

    ck.check(
        "cupac/cross-fit-calibrated",
        f"mean SE / empirical SD = {cross_fit_cal:.3f} (target 1)",
        abs(cross_fit_cal - 1.0) < 0.10,
    )
    ck.check(
        "cupac/in-sample-understates-se",
        f"mean SE / empirical SD = {in_sample_cal:.3f} vs cross-fit {cross_fit_cal:.3f} "
        f"-> {100 * (1 - in_sample_cal / cross_fit_cal):.0f}% understated",
        in_sample_cal < cross_fit_cal - 0.04,
    )
    ck.check(
        "cupac/beats-cuped",
        f"CUPAC {reduction('cupac'):.3f} vs CUPED {reduction('cuped'):.3f} "
        f"(CUPED ceiling {theory['cuped']:.3f})",
        reduction("cupac") > reduction("cuped") + 0.05,
    )
    # Paired bootstrap: the two reductions come from the same replications, so
    # the difference is far better determined than either level.
    se_diff = bootstrap_se_of_reduction_difference(
        got["oracle"][0], got["cupac"][0], got["baseline"][0]
    )
    gap = reduction("cupac") - reduction("oracle")
    ck.check(
        "cupac/below-oracle",
        f"CUPAC - oracle = {gap:+.3f} (paired 3SE = {3 * se_diff:.3f}); "
        f"a learned covariate must not beat the true signal",
        gap <= 3 * se_diff,
    )


def check_poststrat(ck: Checker) -> None:
    """More strata should buy more variance reduction, on a linear outcome."""
    cfg = DGPConfig(signal="linear", rho=0.7, n_per_arm=20_000, tau_rel=0.01)
    reps = 400
    taus = {k: np.empty(reps) for k in (5, 20)}
    base = np.empty(reps)
    for i in range(reps):
        exp = draw_experiment(cfg, np.random.default_rng(i))
        base[i] = baseline(exp).tau
        for k in taus:
            taus[k][i] = poststrat(exp, k).tau

    base_var = base.var(ddof=1)
    r5 = 1.0 - taus[5].var(ddof=1) / base_var
    r20 = 1.0 - taus[20].var(ddof=1) / base_var
    ck.check(
        "poststrat/monotone-in-strata",
        f"5 strata {r5:.3f} -> 20 strata {r20:.3f} (both below the CUPED ceiling "
        f"{theoretical_reduction(cfg)['cuped']:.3f})",
        r20 > r5,
    )
    ck.check(
        "poststrat/under-ceiling",
        f"20 strata {r20:.3f} vs CUPED ceiling {theoretical_reduction(cfg)['cuped']:.3f}",
        r20 <= theoretical_reduction(cfg)["cuped"] + 0.05,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1_200, help="replications, cheap methods")
    parser.add_argument("--gbm-reps", type=int, default=300, help="replications, CUPAC scenarios")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.reps, args.gbm_reps, args.n_jobs = 300, 40, 4

    ck = Checker()
    check_generator(ck)
    check_theory_is_measurable(ck)
    check_ceilings(ck, args.reps, args.n_jobs)
    check_coverage(ck, args.reps, args.n_jobs)
    check_poststrat(ck)
    check_no_leakage(ck)
    check_unbiased(ck, args.gbm_reps, args.n_jobs)
    check_type_one(ck, args.gbm_reps, args.n_jobs)
    check_cupac_pitfall(ck, args.gbm_reps, args.n_jobs)
    sys.exit(ck.summary())


if __name__ == "__main__":
    main()
