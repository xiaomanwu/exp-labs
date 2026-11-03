"""Correctness checks for the simulator and the test machinery.

Three kinds of check:

1. **Simulator recovers the truth.** The realized lift must match the configured
   lift, and the realized variance must match the binomial variance implied by
   the configured rate. If this fails, every downstream number is decoration.

2. **The p-value machinery agrees with an outside implementation.**
   ``statsmodels.stats.proportion.proportions_ztest`` is used as the external
   reference. The two are not supposed to be identical: statsmodels defaults to
   a pooled-variance score test, this repo uses an unpooled Welch-style
   variance, which is what A/B platforms use for mean metrics. The check
   therefore asserts *closeness* and reports the gap, rather than demanding
   equality — a silently large gap would mean the sufficient-statistic recovery
   is wrong.

3. **The calibration results hold.** Fixed horizon should land on alpha,
   Bonferroni should stay under it, and the false-positive rate should increase
   with the number of looks.

Usage::

    python validation.py
    python validation.py --reps 5000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..checks import Checker, tolerance
from .boundaries import (
    crossing_probabilities,
    group_sequential_boundaries,
    spending_obf,
)
from .policies import BonferroniPeek, FixedHorizon, NaivePeek, welch_two_sided_p
from ..sim import ExperimentConfig, simulate


def check_simulator(ck: Checker, reps: int, traffic: int, horizon: int, lift: float) -> None:
    """Does the simulator reproduce the config it was given?"""
    cfg = ExperimentConfig(
        relative_lift=lift, daily_traffic_per_arm=traffic, horizon_days=horizon
    )
    sim = simulate(cfg, reps=reps, seed=11)

    # Realized treatment rate over all traffic, against the configured value.
    # Use the final cumulative column, not a sum over all columns: the arrays
    # are cumulative, so summing them would count every user many times over.
    observed_t = sim.t_s1[:, -1].sum() / sim.t_n[:, -1].sum()
    observed_c = sim.c_s1[:, -1].sum() / sim.c_n[:, -1].sum()
    total = sim.t_n[:, -1].sum()
    se_t = np.sqrt(observed_t * (1 - observed_t) / total)

    ck.check(
        "simulator/lift",
        f"treatment rate {observed_t:.5f} vs configured {cfg.treatment_rate:.5f} "
        f"(4SE = {4 * se_t:.5f})",
        abs(observed_t - cfg.treatment_rate) < 4 * se_t,
    )
    ck.check(
        "simulator/control",
        f"control rate {observed_c:.5f} vs configured {cfg.baseline_rate:.5f}",
        abs(observed_c - cfg.baseline_rate) < 4 * np.sqrt(observed_c * (1 - observed_c) / total),
    )

    # Bernoulli check: sum of squares must equal the sum (values are 0/1).
    ck.check(
        "simulator/sufficient-stats",
        "sum of squares equals sum for a 0/1 metric",
        np.allclose(sim.c_s1, sim.c_s2) and np.allclose(sim.t_s1, sim.t_s2),
    )


def check_pvalue_against_statsmodels(ck: Checker, reps: int, traffic: int, horizon: int) -> None:
    """Compare with statsmodels' two-proportion score test on the final day."""
    try:
        from statsmodels.stats.proportion import proportions_ztest
    except ImportError:
        ck.skip(
            "pvalue/statsmodels",
            "statsmodels not installed (`pip install statsmodels`), cross-check not run",
        )
        return

    cfg = ExperimentConfig(
        relative_lift=0.05, daily_traffic_per_arm=traffic, horizon_days=horizon
    )
    sim = simulate(cfg, reps=reps, seed=21)
    last = horizon - 1

    p_ours = welch_two_sided_p(sim)[:, last]
    p_ref = np.array(
        [
            proportions_ztest(
                count=[sim.t_s1[i, last], sim.c_s1[i, last]],
                nobs=[sim.t_n[i, last], sim.c_n[i, last]],
            )[1]
            for i in range(reps)
        ]
    )

    gap = np.abs(p_ours - p_ref)
    ck.check(
        "pvalue/agreement",
        f"max |dp| = {gap.max():.2e}, median |dp| = {np.median(gap):.2e} "
        "(unpooled vs pooled variance, so not expected to be zero)",
        gap.max() < 0.01,
    )
    ck.check(
        "pvalue/ordering",
        f"correlation with reference = {np.corrcoef(p_ours, p_ref)[0, 1]:.6f}",
        np.corrcoef(p_ours, p_ref)[0, 1] > 0.999,
    )


def check_calibration(ck: Checker, reps: int, traffic: int, horizon: int) -> None:
    """The results the README quotes should hold on a fresh sample."""
    cfg = ExperimentConfig(relative_lift=0.0, daily_traffic_per_arm=traffic, horizon_days=horizon)
    sim = simulate(cfg, reps=reps, seed=31)

    fixed = FixedHorizon().stop_days(sim)[1].mean()
    ck.check(
        "calibration/fixed-horizon",
        f"type-I error {fixed:.4f} vs nominal 0.05",
        abs(fixed - 0.05) < tolerance(reps),
    )

    bonf = BonferroniPeek(peek_every=1).stop_days(sim)[1].mean()
    ck.check(
        "calibration/bonferroni",
        f"type-I error {bonf:.4f} <= 0.05 (conservative by construction)",
        bonf <= 0.05 + tolerance(reps),
    )

    rates = [NaivePeek(peek_every=k).stop_days(sim)[1].mean() for k in (30, 10, 5, 1)]
    ck.check(
        "calibration/monotone-in-looks",
        "FPR by looks: " + ", ".join(f"{k}->{r:.3f}" for k, r in zip((1, 3, 6, 30), rates)),
        all(later >= earlier for earlier, later in zip(rates, rates[1:])),
    )

    ck.check(
        "calibration/peeking-is-costly",
        f"daily peek FPR {rates[-1]:.3f} exceeds fixed horizon {fixed:.3f} by >10 points",
        rates[-1] - fixed > 0.10,
    )


def check_boundaries(ck: Checker) -> None:
    """Validate the group-sequential solver against closed-form and published values.

    Three independent anchors:

    1. With a single look the boundary must be the fixed-sample 1.96.
    2. The classic O'Brien-Fleming table for K=5 (4.8769, 3.3567, 2.6802,
       2.2894, 2.0310) must cross at alpha = 0.05 when pushed through *our*
       recursion. This checks the recursion against a published construction
       rather than checking our solver against itself.
    3. Lan-DeMets Pocock spending should average to the classic Pocock constant
       of about 2.413 for K=5.
    """
    one = group_sequential_boundaries(np.array([1.0]), alpha=0.05)[0]
    ck.check(
        "boundaries/single-look",
        f"K=1 boundary {one:.4f} vs fixed-sample 1.9600",
        abs(one - 1.959964) < 0.005,
    )

    t5 = np.linspace(0.2, 1.0, 5)
    published = np.array([4.8769, 3.3567, 2.6802, 2.2894, 2.0310])
    total = crossing_probabilities(published, t5).sum()
    ck.check(
        "boundaries/published-obf-table",
        f"classic OBF K=5 boundaries cross at {total:.5f} (target 0.05)",
        abs(total - 0.05) < 0.002,
    )

    pocock = group_sequential_boundaries(t5, alpha=0.05, spending="pocock")
    ck.check(
        "boundaries/pocock-constant",
        f"mean boundary {pocock.mean():.4f} vs published constant 2.413",
        abs(pocock.mean() - 2.413) < 0.02,
    )

    lan = group_sequential_boundaries(t5, alpha=0.05, spending="obf")
    recovered = crossing_probabilities(lan, t5)
    target = np.diff(np.concatenate([[0.0], spending_obf(t5, 0.05)]))
    ck.check(
        "boundaries/spending-recovered",
        f"per-look crossing probabilities match spending increments "
        f"(max |diff| = {np.abs(recovered - target).max():.2e}, "
        f"total = {recovered.sum():.5f})",
        np.abs(recovered - target).max() < 5e-4 and abs(recovered.sum() - 0.05) < 0.002,
    )


def check_policy_interface(ck: Checker) -> None:
    """Structural checks: the abstraction should behave as documented."""
    horizon = 30
    ck.check(
        "policy/fixed-horizon-looks-once",
        "FixedHorizon peeks only on the final day",
        FixedHorizon().peekable(horizon).sum() == 1
        and bool(FixedHorizon().peekable(horizon)[-1]),
    )
    ck.check(
        "policy/peek-schedule",
        "NaivePeek(k) looks ceil(horizon/k) times including the final day",
        int(NaivePeek(peek_every=5).peekable(horizon).sum()) == 6,
    )
    ck.check(
        "policy/bonferroni-threshold",
        "Bonferroni threshold is alpha divided by the number of looks",
        np.isclose(BonferroniPeek(peek_every=5).alpha(horizon)[0], 0.05 / 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=2_000)
    parser.add_argument("--traffic", type=int, default=2_000)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    print(f"Validation run: {args.reps} reps, {args.traffic} users/day/arm, {args.horizon} days\n")

    ck = Checker()
    check_simulator(ck, args.reps, args.traffic, args.horizon, lift=0.05)
    check_pvalue_against_statsmodels(ck, min(args.reps, 500), args.traffic, args.horizon)
    check_boundaries(ck)
    check_policy_interface(ck)
    check_calibration(ck, args.reps, args.traffic, args.horizon)
    sys.exit(ck.summary())


if __name__ == "__main__":
    main()
