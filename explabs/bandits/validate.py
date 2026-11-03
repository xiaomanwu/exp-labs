"""Correctness checks for study 04.

Six groups, each aimed at a claim the report makes:

``dgp``         the truth is the truth (quadrature, and the two worlds)
``propensity``  the recorded propensities are the policy's actual probabilities
``regret``      the allocation rules behave the way the theory says
``estimator``   the three estimators do what they claim under a fixed split
``adaptive``    what breaks under adaptive allocation, and what repairs it
``artefacts``   the report's files exist and the run is reproducible
"""

from __future__ import annotations

import argparse

import numpy as np

from ..checks import Checker
from .dgp import ContextualDGP, additive_dgp, heterogeneous_dgp
from .inference import aipw, composition_bias, design_matrix, hajek, ipw, naive
from .policies import (
    EpsilonGreedy,
    FixedShare,
    ThompsonSampling,
    UCB1,
    beta_win_probability,
    run_policy,
)

_RUNS: dict[str, object] = {}


def _run(name: str, factory, dgp, horizon: int, reps: int, seed: int, record: bool = True):
    """Memoise runs: several checks interrogate the same simulation."""
    if name not in _RUNS:
        _RUNS[name] = run_policy(factory, dgp, horizon, reps=reps, seed=seed,
                                 record_propensity=record)
    return _RUNS[name]


def _mc_se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan


# --------------------------------------------------------------------------
# Truth
# --------------------------------------------------------------------------


def check_dgp(ck: Checker) -> None:
    dgp = heterogeneous_dgp(0.02)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4_000_000, 2))
    p0, p1 = dgp.probs(x)
    for arm, sample in ((0, p0), (1, p1)):
        exact = dgp.arm_rate(arm)
        se = sample.std(ddof=1) / np.sqrt(sample.size)
        ck.check(
            f"dgp/quadrature-vs-mc-arm{arm}",
            f"quadrature {exact:.6f} vs 4M draws {sample.mean():.6f} ({abs(exact - sample.mean()) / se:.1f} MC SE)",
            abs(exact - sample.mean()) < 4 * se,
        )

    ck.check(
        "dgp/solved-ate",
        f"target 0.02 -> {dgp.ate:+.10f}; target 0.0 -> {heterogeneous_dgp(0.0).ate:+.2e}",
        abs(dgp.ate - 0.02) < 1e-9 and abs(heterogeneous_dgp(0.0).ate) < 1e-9,
    )

    for label, d in (("heterogeneous", dgp), ("null", heterogeneous_dgp(0.0))):
        shares = d.segment_shares(4)
        contribution = float(np.sum(shares * d.segment_ates(4)))
        ck.check(
            f"dgp/segments-sum-to-ate/{label}",
            f"weighted segment effects {contribution:+.10f} vs ATE {d.ate:+.10f}, "
            f"shares {np.round(shares, 3)}",
            abs(contribution - d.ate) < 1e-9 and abs(shares.sum() - 1) < 1e-12,
        )

    flat = additive_dgp(0.02)
    spread = float(np.ptp(flat.segment_ates(4)))
    null = heterogeneous_dgp(0.0)
    ck.check(
        "dgp/additive-is-homogeneous",
        f"segment effects spread {spread:.2e} (should be zero)",
        spread < 1e-12,
    )
    ck.check(
        "dgp/null-world-is-heterogeneous",
        f"ATE {null.ate:+.2e} but segment effects span "
        f"[{null.segment_ates(4).min():+.3f}, {null.segment_ates(4).max():+.3f}]",
        abs(null.ate) < 1e-12 and np.abs(null.segment_ates(4)).max() > 0.05,
    )

    try:
        ContextualDGP(shift=0.01, target_ate=0.01)
        raised = False
    except ValueError:
        raised = True
    ck.check("dgp/rejects-ambiguous-config", "two intercept specifications raise", raised)


# --------------------------------------------------------------------------
# Propensities
# --------------------------------------------------------------------------


def check_propensity(ck: Checker) -> None:
    rng = np.random.default_rng(3)
    worst, worst_z, z_list = 0.0, 0.0, []
    for a1, b1, a0, b0 in ((1, 1, 1, 1), (5, 3, 2, 6), (60, 40, 55, 45),
                           (500, 20, 480, 40), (300, 100, 280, 120)):
        q = float(beta_win_probability(np.array([a1]), np.array([b1]),
                                       np.array([a0]), np.array([b0]), nodes=48)[0])
        draws = 400_000
        m = float((rng.beta(a1, b1, draws) > rng.beta(a0, b0, draws)).mean())
        se = np.sqrt(q * (1 - q) / draws)
        z_list.append(abs(q - m) / se)
        worst = max(worst, abs(q - m))
        worst_z = max(worst_z, z_list[-1])
    ck.check(
        "propensity/quadrature-vs-mc",
        f"worst absolute gap {worst:.2e}, worst z {worst_z:.1f} over {len(z_list)} posteriors",
        worst < 2e-3 and worst_z < 4.0,
    )

    nodes = [float(beta_win_probability(np.array([300.0]), np.array([100.0]),
                                        np.array([280.0]), np.array([120.0]), nodes=n)[0])
             for n in (36, 48, 96)]
    ck.check(
        "propensity/quadrature-converges",
        "nodes 36/48 vs reference 96 -> " + " ".join(f"{v:.9f}" for v in nodes),
        max(abs(nodes[2] - v) for v in nodes) < 5e-8,
    )

    dgp = heterogeneous_dgp(0.02)
    ts = _run("ts", ThompsonSampling(n_buckets=1), dgp, 1200, 250, 11)
    share, mean_prop = float(ts.arms.mean()), float(ts.propensities.mean())
    se = np.sqrt(0.25 / ts.arms.size)
    ck.check(
        "propensity/sampler-matches-quadrature",
        f"realised treated share {share:.5f} vs mean recorded propensity {mean_prop:.5f} "
        f"({abs(share - mean_prop) / se:.1f} MC SE)",
        abs(share - mean_prop) < 5 * se,
    )

    for name, run in (("ts", ts),
                      ("ts4", _run("ts4", ThompsonSampling(n_buckets=4), dgp, 1200, 250, 12)),
                      ("eg", _run("eg", EpsilonGreedy(0.1), dgp, 1200, 250, 13))):
        e = run.propensities
        # E[1{A=a}/e_a] = 1 per step, so the per-replication average over the
        # horizon of the inverse weights sits at 1. This is the identity that
        # makes the estimator unbiased, so it is worth checking on its own.
        b1 = float(np.mean(np.where(run.arms == 1, 1.0 / e, 0.0).sum(axis=1) / run.horizon))
        b0 = float(np.mean(np.where(run.arms == 0, 1.0 / (1.0 - e), 0.0).sum(axis=1) / run.horizon))
        ck.check(
            f"propensity/balance-identity/{name}",
            f"mean inverse weight per step: treated {b1:.3f}, control {b0:.3f} (target 1)",
            abs(b1 - 1) < 0.03 and abs(b0 - 1) < 0.03,
        )

    e = ts.propensities
    ck.check(
        "propensity/strictly-interior",
        f"TS propensities in [{e.min():.2e}, {e.max():.6f}]",
        bool((e > 0).all() and (e < 1).all()),
    )

    eg = _run("eg", EpsilonGreedy(0.1), dgp, 1200, 250, 13)
    floor = float(eg.propensities.min())
    ck.check(
        "propensity/epsilon-floor",
        f"$\\epsilon$-greedy (0.1) minimum propensity {floor:.4f} >= epsilon/2 = 0.05",
        floor >= 0.05 - 1e-12,
    )

    ucb = _run("ucb1", UCB1(), dgp, 1200, 250, 14)
    e = ucb.propensities
    degenerate = float(((e <= 1e-12) | (e >= 1 - 1e-12)).mean())
    ck.check(
        "propensity/ucb1-is-deterministic",
        f"{degenerate:.1%} of steps have a propensity of exactly 0 or 1",
        degenerate > 0.99,
    )


# --------------------------------------------------------------------------
# Regret
# --------------------------------------------------------------------------


def check_regret(ck: Checker) -> None:
    lift = 0.02
    dgp = additive_dgp(lift)
    fixed = _run("fixed-regret", FixedShare(), dgp, 1500, 200, 21, record=False)
    curve = fixed.cumulative_regret()[:, -1]
    expected = 1500 * lift / 2
    ck.check(
        "regret/fixed-matches-TDelta/2",
        f"mean regret {curve.mean():.2f} vs analytic {expected:.2f} "
        f"({abs(curve.mean() - expected) / _mc_se(curve):.1f} MC SE, sd {curve.std(ddof=1):.2f})",
        abs(curve.mean() - expected) < 4 * _mc_se(curve),
    )

    ts = _run("ts-regret", ThompsonSampling(n_buckets=1), dgp, 1500, 200, 21, record=False)
    ucb = _run("ucb1-regret", UCB1(), dgp, 1500, 200, 21, record=False)
    eg = _run("eg-regret", EpsilonGreedy(0.1), dgp, 1500, 200, 21, record=False)
    scores = {k: float(v.cumulative_regret()[:, -1].mean())
              for k, v in (("fixed", fixed), ("ts", ts), ("ucb1", ucb), ("eg", eg))}
    ck.check(
        "regret/adaptation-beats-fixed",
        "  ".join(f"{k}={v:.1f}" for k, v in scores.items()),
        scores["ts"] < scores["fixed"] and scores["ucb1"] < scores["fixed"]
        and scores["eg"] < scores["fixed"],
    )

    # Regret counted in *mistakes* (regret / price-per-mistake) is what scales
    # with 1/Delta; the raw conversion count grows with Delta because each
    # mistake costs more. Counting mistakes is the comparison that means
    # something, so the check uses that.
    mistakes = {}
    for lift in (0.01, 0.2):
        d = additive_dgp(lift)
        regret = float(
            run_policy(ThompsonSampling(n_buckets=1), d, 800, reps=150, seed=31,
                       record_propensity=False).cumulative_regret()[:, -1].mean()
        )
        mistakes[lift] = regret / lift
    ck.check(
        "regret/shrinks-as-the-gap-widens",
        f"lift 0.01 -> {mistakes[0.01]:.0f} wasted users, lift 0.20 -> {mistakes[0.2]:.0f} "
        f"wasted users at 800 users",
        mistakes[0.2] < mistakes[0.01] / 3,
    )

    # UCB1's guarantee is a bound, not a growth rate: the regret is bounded by
    # a constant times log(T)/Delta. The bound holds with a wide margin here,
    # and the margin is the honest way to report "logarithmic".
    bound = []
    for h in (1000, 4000, 16000):
        r = run_policy(UCB1(), additive_dgp(0.10), h, reps=100, seed=41,
                       record_propensity=False)
        regret = float(r.cumulative_regret()[:, -1].mean())
        bound.append(regret / (8.0 * np.log(h) / 0.10))
    ck.check(
        "regret/ucb1-within-its-bound",
        "UCB1 regret as a share of the 8 log(T)/Delta bound at T=1000/4000/16000: "
        + " ".join(f"{b:.3f}" for b in bound),
        all(0.0 < b <= 0.5 for b in bound),
    )

    small = additive_dgp(0.02)
    ucb_small = float(run_policy(UCB1(), small, 1500, reps=200, seed=43,
                                 record_propensity=False).cumulative_regret()[:, -1].mean())
    ts_small = float(run_policy(ThompsonSampling(n_buckets=1), small, 1500, reps=200,
                                seed=43, record_propensity=False)
                     .cumulative_regret()[:, -1].mean())
    fixed_small = 1500 * 0.02 / 2
    ck.check(
        "regret/ucb1-cannot-separate-a-small-gap",
        f"lift 2%: fixed {fixed_small:.1f}, UCB1 {ucb_small:.1f} "
        f"({1 - ucb_small / fixed_small:.0%} saved), TS {ts_small:.1f} "
        f"({1 - ts_small / fixed_small:.0%} saved)",
        ucb_small > 0.8 * fixed_small and ts_small < 0.75 * fixed_small,
    )

    # The *percentage* saved is a ratio of two small numbers at tiny gaps, and
    # a handful of replications that lock onto the wrong arm move it by several
    # points. The claim that survives is about conversions: what a bandit saves
    # grows with the gap, and below half a point it is a fraction of one
    # conversion per 1,500 users -- real, but not a reason to run a bandit.
    saved_conv = {}
    for gap in (0.005, 0.02, 0.04):
        r = _run(f"ts-gap-{gap}", ThompsonSampling(n_buckets=1), additive_dgp(gap),
                 1500, 200, 21, record=False)
        saved_conv[gap] = 1500 * gap / 2 - float(r.cumulative_regret()[:, -1].mean())
    ck.check(
        "regret/saving-scales-with-the-gap",
        "conversions saved per 1,500 users: "
        + "  ".join(f"{g:.3f}->{v:.2f}" for g, v in saved_conv.items()),
        saved_conv[0.005] < 1.0
        and saved_conv[0.005] < saved_conv[0.02] < saved_conv[0.04],
    )


# --------------------------------------------------------------------------
# Estimators under a known-good design
# --------------------------------------------------------------------------


def check_estimators(ck: Checker) -> None:
    dgp = heterogeneous_dgp(0.02)
    fixed = _run("fixed-inf", FixedShare(), dgp, 1200, 300, 51)

    h = hajek(fixed.arms, fixed.rewards, fixed.propensities).estimate
    n = naive(fixed.arms, fixed.rewards).estimate
    ck.check(
        "estimator/hajek-reduces-to-naive",
        f"max |difference| {np.nanmax(np.abs(h - n)):.2e} under a constant propensity",
        np.nanmax(np.abs(h - n)) < 1e-12,
    )

    rows = {}
    for name, res in (("naive", naive(fixed.arms, fixed.rewards)),
                      ("ipw", ipw(fixed.arms, fixed.rewards, fixed.propensities)),
                      ("aipw", aipw(fixed.arms, fixed.rewards, fixed.propensities,
                                    design_matrix(fixed, "full")))):
        s = res.summary(dgp.ate)
        rows[name] = s
        ck.check(
            f"estimator/unbiased-under-fixed-split/{name}",
            f"bias {s['bias']:+.5f} ({abs(s['bias']) / s['mc_se']:.1f} MC SE), "
            f"coverage {s['coverage']:.3f}",
            abs(s["bias"]) < 4 * s["mc_se"],
        )
    ck.check(
        "estimator/fixed-split-coverage",
        f"naive {rows['naive']['coverage']:.3f}, IPW {rows['ipw']['coverage']:.3f}, "
        f"AIPW {rows['aipw']['coverage']:.3f}",
        all(0.93 <= rows[m]["coverage"] <= 0.98 for m in rows),
    )


# --------------------------------------------------------------------------
# What adaptivity does
# --------------------------------------------------------------------------


def check_adaptive(ck: Checker) -> None:
    """The heart of the study: what adaptivity does, and what repairs it."""
    dgp = heterogeneous_dgp(0.02)
    ts = _run("ts-inf", ThompsonSampling(n_buckets=1), dgp, 1500, 300, 11)
    ts4_inf = _run("ts4-inf", ThompsonSampling(n_buckets=4), dgp, 1500, 300, 12)

    results = {
        "naive": naive(ts.arms, ts.rewards).summary(dgp.ate),
        "ipw": ipw(ts.arms, ts.rewards, ts.propensities).summary(dgp.ate),
        "aipw": aipw(ts.arms, ts.rewards, ts.propensities, design_matrix(ts, "full")).summary(dgp.ate),
        "aipw-seg": aipw(ts.arms, ts.rewards, ts.propensities,
                         design_matrix(ts, "segments")).summary(dgp.ate),
    }
    nv = results["naive"]
    ck.check(
        "adaptive/naive-is-biased",
        f"bias {nv['bias']:+.5f} ({nv['bias'] / nv['mc_se']:.1f} MC SE), "
        f"{nv['bias'] / dgp.ate:+.0%} of the true effect",
        nv["bias"] > 3 * nv["mc_se"] and nv["bias"] > 0.2 * dgp.ate,
    )

    for name in ("ipw", "aipw", "aipw-seg"):
        s = results[name]
        ck.check(
            f"adaptive/{name}-removes-the-bias",
            f"bias {s['bias']:+.5f} ({abs(s['bias']) / s['mc_se']:.1f} MC SE)",
            abs(s["bias"]) < 3 * s["mc_se"],
        )

    ratio = results["aipw"]["se_mean"] / results["ipw"]["se_mean"]
    ck.check(
        "adaptive/aipw-is-tighter-than-ipw",
        f"mean SE: AIPW {results['aipw']['se_mean']:.4f} vs IPW {results['ipw']['se_mean']:.4f} "
        f"(ratio {ratio:.3f}; modest, because a binary outcome leaves the outcome "
        f"model little to explain)",
        0.90 < ratio < 0.97,
    )

    misspecified = results["aipw-seg"]
    ck.check(
        "adaptive/aipw-double-robust",
        f"with outcome models that only see the policy's own buckets, bias "
        f"{misspecified['bias']:+.5f} ({abs(misspecified['bias']) / misspecified['mc_se']:.1f} MC SE)",
        abs(misspecified["bias"]) < 4 * misspecified["mc_se"],
    )

    ucb = _run("ucb1-inf", UCB1(), dgp, 1500, 300, 14)
    defined = np.isfinite(ipw(ucb.arms, ucb.rewards, ucb.propensities).estimate).mean()
    ck.check(
        "adaptive/no-overlap-is-flagged",
        f"UCB1 leaves IPW undefined in {1 - defined:.0%} of replications",
        defined < 0.05,
    )

    # Bias decomposition: a pooled policy cannot tilt the arm composition
    # (assignment ignores the segment), a contextual policy can and does.
    # Composition needs only the allocation, so these runs skip the propensity
    # bookkeeping and can afford enough replications to resolve a small mean.
    pooled = composition_bias(run_policy(FixedShare(), dgp, 1500, reps=800, seed=71,
                                         record_propensity=False))
    contextual = composition_bias(run_policy(ThompsonSampling(n_buckets=4), dgp, 1500,
                                             reps=800, seed=71, record_propensity=False))
    ck.check(
        "adaptive/composition-bias-only-with-context",
        f"mean composition bias: pooled {pooled.mean():+.5f} vs contextual "
        f"{contextual.mean():+.5f} on an effect of {dgp.ate:+.3f}",
        np.nanmean(np.abs(pooled)) < 0.002 and abs(contextual.mean()) > 0.003,
    )

    # The null world: nothing to find, so every rejection is a false positive.
    # The naive estimator fails here even when its bias is small, because its
    # standard error does not know the allocation was adaptive.
    null = heterogeneous_dgp(0.0)
    fixed_null = run_policy(FixedShare(), null, 1500, reps=250, seed=61)
    eg4_null = run_policy(EpsilonGreedy(0.1, n_buckets=4), null, 1500, reps=250, seed=61)
    naive_fixed = naive(fixed_null.arms, fixed_null.rewards).summary(0.0)
    naive_eg4 = naive(eg4_null.arms, eg4_null.rewards).summary(0.0)
    aipw_eg4 = aipw(eg4_null.arms, eg4_null.rewards, eg4_null.propensities,
                    design_matrix(eg4_null, "full")).summary(0.0)
    comp_eg4 = float(np.nanmean(composition_bias(eg4_null)))
    ck.check(
        "adaptive/null-world-naive-se-invalid",
        "true ATE = 0, naive coverage: fixed split "
        f"{naive_fixed['coverage']:.3f} vs contextual bandit {naive_eg4['coverage']:.3f} "
        f"(mean claimed SE {naive_eg4['se_mean']:.4f}, actual sd "
        f"{naive_eg4['mc_se'] * np.sqrt(naive_eg4['reps']):.4f})",
        naive_fixed["coverage"] > 0.92 and naive_eg4["coverage"] < 0.80,
    )
    ck.check(
        "adaptive/null-world-false-positives",
        f"true ATE = 0: naive rejects {naive_eg4['reject']:.3f} of the time after a "
        f"contextual bandit (composition bias {comp_eg4:+.4f}), AIPW rejects "
        f"{aipw_eg4['reject']:.3f}",
        naive_eg4["reject"] > 0.25 and aipw_eg4["reject"] < 0.10,
    )


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def check_reproducible(ck: Checker) -> None:
    dgp = additive_dgp(0.02)
    a = run_policy(ThompsonSampling(n_buckets=2), dgp, 200, reps=8, seed=99)
    b = run_policy(ThompsonSampling(n_buckets=2), dgp, 200, reps=8, seed=99)
    c = run_policy(ThompsonSampling(n_buckets=2), dgp, 200, reps=8, seed=100)
    ck.check(
        "artefacts/same-seed-same-run",
        "identical arms, propensities and rewards for a repeated seed",
        np.array_equal(a.arms, b.arms)
        and np.allclose(a.propensities, b.propensities)
        and np.array_equal(a.rewards, b.rewards),
    )
    ck.check(
        "artefacts/seed-changes-the-run",
        "a different seed produces different allocation",
        not np.array_equal(a.arms, c.arms),
    )


CHECKS = (
    ("dgp", check_dgp),
    ("propensity", check_propensity),
    ("regret", check_regret),
    ("estimators", check_estimators),
    ("adaptive", check_adaptive),
    ("artefacts", check_reproducible),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    ck = Checker()
    for name, fn in CHECKS:
        if args.only and name not in args.only:
            continue
        print(f"\n--- {name} ---")
        fn(ck)
    raise SystemExit(ck.summary())


if __name__ == "__main__":
    main()
