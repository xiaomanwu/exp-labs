"""Correctness checks for study 03.

The design comparisons are only meaningful if the market behaves as advertised
and each estimator does what it claims. These checks pin both down:

- the market obeys its own accounting (capacity, utilisation, no free lunch)
- with the externality switched off all three designs agree
- with it switched on, the user-level gap is the estimand gap, not noise
- cluster coverage needs *both* a cluster-robust SE and a small-sample quantile
- the switchback attenuation matches its closed form
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np

from ..checks import Checker
from .designs import cluster, switchback, user_level
from .market import (
    MarketConfig,
    draw_city,
    expected_trips,
    global_effect,
    market_state,
    population_direct_effect,
    population_global_effect,
)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def check_market_accounting(ck: Checker) -> None:
    cfg = MarketConfig()
    state = market_state(cfg, 0.0)
    ck.check(
        "market/baseline-utilisation",
        f"util={state.utilization:.6f} vs baseline {cfg.baseline_utilization:.6f}, "
        f"fill={state.fill_control:.6f} vs {cfg.baseline_fill:.6f}",
        abs(state.utilization - cfg.baseline_utilization) < 1e-12
        and abs(state.fill_control - cfg.baseline_fill) < 1e-12,
    )

    worst = 0.0
    for phi in (0.0, 1.0, 3.0):
        for share in (0.0, 0.25, 0.5, 1.0):
            for priority in (0.0, 0.5, 1.0):
                c = replace(cfg, congestion=phi, priority=priority)
                s = market_state(c, share)
                rides = s.requests_treated * s.fill_treated + s.requests_control * s.fill_control
                worst = max(worst, rides / s.capacity - 1.0, -s.rides / max(s.capacity, 1e-9))
                worst = max(worst, abs(s.rides - rides) / max(rides, 1e-9))
    ck.check(
        "market/capacity-respected",
        f"completed rides never exceed capacity (worst excess {worst:.2e})",
        worst < 1e-9,
    )

    # A subsidy that does not move demand must not move the outcome.
    flat = replace(cfg, demand_elasticity=0.0)
    ck.check(
        "market/no-demand-response",
        f"global effect with elasticity 0 is {global_effect(flat):+.3e}",
        abs(global_effect(flat)) < 1e-12,
    )

    effects = [global_effect(replace(cfg, congestion=p)) for p in (0.0, 0.5, 1.0, 2.0, 3.0)]
    ck.check(
        "market/effect-decreasing-in-congestion",
        "global effect " + " > ".join(f"{e:.4f}" for e in effects),
        all(a > b for a, b in zip(effects, effects[1:])),
    )

    # Priority redistributes; with everyone treated there is nobody to crowd out.
    tau_by_priority = [
        global_effect(replace(cfg.strained(), congestion=1.0, priority=p))
        for p in (0.0, 0.5, 1.0)
    ]
    ck.check(
        "market/priority-invariance",
        f"global effect under priority 0/0.5/1 = "
        f"{tau_by_priority[0]:.6f} / {tau_by_priority[1]:.6f} / {tau_by_priority[2]:.6f}",
        max(tau_by_priority) - min(tau_by_priority) < 1e-12,
    )


def check_simulator(ck: Checker, reps: int) -> None:
    """The Poisson draw must be centred on the market's expected trips."""
    cfg = replace(MarketConfig(), elasticity_sd=0.0)
    rng = np.random.default_rng(101)
    for share in (0.0, 1.0):
        target = expected_trips(cfg, share)
        means = []
        for _ in range(reps):
            y, _, _ = draw_city(cfg, share, rng)
            means.append(y.mean())
        observed, se = _mean_se(np.array(means))
        ck.check(
            f"simulator/mean-at-share-{share:.0f}",
            f"{observed:.4f} vs expected {target:.4f} (4SE = {4 * se:.4f})",
            abs(observed - target) <= 4 * se,
        )


def check_sutva_off(ck: Checker, reps: int) -> None:
    """With no congestion the three designs must all recover the same number."""
    cfg = replace(MarketConfig(), congestion=0.0)
    truth = population_global_effect(cfg)
    rng = np.random.default_rng(202)

    estimates = {
        "user-level": np.array([user_level(cfg, 8, rng).tau for _ in range(reps)]),
        "cluster": np.array([cluster(cfg, 16, rng).tau for _ in range(reps)]),
        "switchback": np.array([switchback(cfg, 8, 8, rng).tau for _ in range(reps)]),
    }
    ck.check(
        "sutva/effect-is-nonzero",
        f"global effect with the externality off is {truth:.4f}, so the check is not vacuous",
        truth > 0.5,
    )
    for name, values in estimates.items():
        mean, se = _mean_se(values)
        ck.check(
            f"sutva/{name}",
            f"{mean:.4f} vs {truth:.4f} (4SE = {4 * se:.4f})",
            abs(mean - truth) <= 4 * se,
        )


def check_estimand_gap(ck: Checker, reps: int) -> None:
    """The user-level gap must be the estimand gap, not a numerical accident."""
    cfg = replace(MarketConfig(), congestion=1.0)
    truth = population_global_effect(cfg)
    direct = population_direct_effect(cfg, 0.5)
    rng = np.random.default_rng(303)
    values = np.array([user_level(cfg, 8, rng).tau for _ in range(reps)])
    mean, se = _mean_se(values)

    ck.check(
        "estimand/user-level-matches-direct",
        f"{mean:.4f} vs within-market effect {direct:.4f} (4SE = {4 * se:.4f})",
        abs(mean - direct) <= 4 * se,
    )
    ck.check(
        "estimand/gap-is-real",
        f"roll-out effect {truth:.4f} is {abs(mean - truth) / se:.1f} SE away",
        abs(mean - truth) > 6 * se,
    )


def check_bias_monotone(ck: Checker) -> None:
    cfg = MarketConfig()
    rng = np.random.default_rng(404)
    reps = 120
    biases, ses = [], []
    for phi in (0.0, 1.0, 2.0, 3.0):
        c = replace(cfg, congestion=phi)
        truth = population_global_effect(c)
        values = np.array([user_level(c, 8, rng).tau for _ in range(reps)])
        biases.append(values.mean() - truth)
        ses.append(values.std(ddof=1) / np.sqrt(reps))
    ck.check(
        "bias/monotone-in-congestion",
        "user-level bias " + " -> ".join(f"{b:+.4f}" for b in biases),
        all(a < b for a, b in zip(biases, biases[1:])),
    )
    ck.check(
        "bias/zero-at-no-externality",
        f"bias at phi=0 is {biases[0]:+.4f} (3 MC SE = {3 * ses[0]:.4f})",
        abs(biases[0]) <= 3 * ses[0],
    )


def check_cluster_inference(ck: Checker, reps: int) -> None:
    cfg = replace(MarketConfig(), congestion=1.0)
    truth = population_global_effect(cfg)

    for cities, label in ((16, "16 cities"), (6, "6 cities")):
        rng = np.random.default_rng(505 + cities)
        taus = np.empty(reps)
        ses = np.empty(reps)
        crits = np.empty(reps)
        naives = np.empty(reps)
        for i in range(reps):
            r = cluster(cfg, cities, rng)
            taus[i], ses[i], crits[i], naives[i] = r.tau, r.se, r.crit, r.se_naive

        cover_t = float(np.mean(np.abs(taus - truth) <= crits * ses))
        cover_z = float(np.mean(np.abs(taus - truth) <= 1.959963984540054 * ses))
        cover_naive = float(np.mean(np.abs(taus - truth) <= 1.959963984540054 * naives))
        mean, se = _mean_se(taus)

        ck.check(
            f"cluster/unbiased-{cities}",
            f"{mean:.4f} vs {truth:.4f} (4SE = {4 * se:.4f})",
            abs(mean - truth) <= 4 * se,
        )
        # The t correction fixes most, not all, of the small-cluster problem:
        # with three clusters per arm the interval is still a little short.
        # That residual is a finding, so the threshold is honest rather than
        # aspirational -- below ~10 clusters the real answer is randomisation
        # inference, which this repo does not implement.
        threshold = 0.92 if cities >= 12 else 0.88
        ck.check(
            f"cluster/robust-coverage-{label}",
            f"t-quantile coverage {cover_t:.3f}, normal quantile {cover_z:.3f} "
            f"(threshold {threshold})",
            cover_t >= threshold,
        )
        ck.check(
            f"cluster/naive-coverage-{label}",
            f"rider-level SE coverage {cover_naive:.3f} (should be far below 0.95)",
            cover_naive < 0.85,
        )
        if cities == 6:
            ck.check(
                "cluster/small-sample-correction-matters",
                f"6 clusters: t {cover_t:.3f} vs normal {cover_z:.3f}",
                cover_t > cover_z + 0.01,
            )


def check_switchback(ck: Checker) -> None:
    cfg = replace(MarketConfig(), congestion=1.0, elasticity_sd=0.0)
    truth = global_effect(cfg)
    periods = 8
    k = periods // 2
    rng = np.random.default_rng(606)
    reps = 200

    for lam in (0.0, 0.2, 0.5):
        values = np.array(
            [switchback(cfg, 8, periods, rng, carryover=lam).tau for _ in range(reps)]
        )
        mean, se = _mean_se(values)
        predicted = truth * (1.0 - lam * (2 * k - 1) / k)
        ck.check(
            f"switchback/attenuation-lambda{lam}",
            f"measured {mean:.4f} vs closed form {predicted:.4f} (4SE = {4 * se:.4f})",
            abs(mean - predicted) <= 4 * se,
        )

    # Washout removes the contamination outright, at any carryover.
    values = np.array(
        [
            switchback(cfg, 8, periods, rng, carryover=0.8, block_len=2, washout=1).tau
            for _ in range(reps)
        ]
    )
    mean, se = _mean_se(values)
    ck.check(
        "switchback/washout-removes-bias",
        f"carryover 0.8 with washout: {mean:.4f} vs {truth:.4f} (4SE = {4 * se:.4f})",
        abs(mean - truth) <= 4 * se,
    )

    # Longer blocks dilute contamination without discarding traffic.
    values = np.array(
        [switchback(cfg, 8, periods, rng, carryover=0.5, block_len=2).tau for _ in range(reps)]
    )
    mean, se = _mean_se(values)
    ck.check(
        "switchback/blocks-dilute",
        f"carryover 0.5 with 2-period blocks: {mean:.4f} "
        f"(alternation would give {truth * (1 - 0.5 * (2 * k - 1) / k):.4f})",
        mean > truth * (1 - 0.5 * (2 * k - 1) / k) + 4 * se,
    )


def check_crowdout(ck: Checker) -> None:
    """Scarce capacity is the second channel, and it needs scarcity to exist."""
    strained = replace(MarketConfig().strained(), congestion=1.0)
    truth = population_global_effect(strained)
    rng = np.random.default_rng(707)
    reps = 150

    biases = {}
    for priority in (0.0, 0.5):
        cfg = replace(strained, priority=priority)
        values = np.array([user_level(cfg, 8, rng).tau for _ in range(reps)])
        biases[priority] = values.mean() - truth

    ck.check(
        "crowdout/raises-user-level-bias",
        f"bias {biases[0.0]:+.4f} at priority 0, {biases[0.5]:+.4f} at priority 0.5",
        biases[0.5] > biases[0.0] + 0.1,
    )

    open_market = replace(MarketConfig(), congestion=1.0, priority=0.5)
    a = market_state(open_market, 0.5)
    ck.check(
        "crowdout/inactive-without-scarcity",
        f"slack market: fill_treated {a.fill_treated:.4f} == fill_control {a.fill_control:.4f}",
        abs(a.fill_treated - a.fill_control) < 1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    reps = 50 if args.quick else args.reps

    ck = Checker()
    check_market_accounting(ck)
    check_simulator(ck, max(30, reps // 5))
    check_sutva_off(ck, reps)
    check_estimand_gap(ck, reps)
    check_bias_monotone(ck)
    check_cluster_inference(ck, max(100, reps // 2))
    check_switchback(ck)
    check_crowdout(ck)
    sys.exit(ck.summary())


if __name__ == "__main__":
    main()
