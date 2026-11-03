"""Correctness checks for study 05.

Groups, each aimed at a claim the report makes:

``data``      the synthetic truth is the truth (ATE quadrature, benchmark loaders)
``qini``      the shared evaluation code path is correct and its null is null
``constant``  anchor 1: with constant effects the models' Qini curves agree
``segment``   anchor 2: with one strong segment every model ranks it to the top
``models``    the constructions do what their papers promise (X under imbalance,
              honesty in the forest)
``targeting`` budget-allocation arithmetic is monotone and exact at 100%
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..checks import Checker
from .data import constant_dgp, heterogeneous_dgp, load_data, single_segment_dgp
from .models import HonestCausalForest, SLearner, TLearner, XLearner
from .qini import auqc, capture_curve, paired_bootstrap_auqc, qini_curve

ORDER = ("s", "t", "x", "forest")


def _fit_all(dgp, n: int, train_n: int, seed: int, n_trees: int = 100):
    """Fit every model once on one draw; memoised by the caller's naming."""
    d = dgp.draw(n, seed=seed)
    x_tr, t_tr, y_tr = d["x"][:train_n], d["t"][:train_n], d["y"][:train_n]
    x_ev = d["x"][train_n:]
    truth = dgp.tau(x_ev)
    scores = {}
    for model in (SLearner(seed=seed), TLearner(seed=seed), XLearner(seed=seed),
                  HonestCausalForest(seed=seed, n_trees=n_trees)):
        model.fit(x_tr, t_tr, y_tr, e=dgp.treat_share)
        scores[model.key] = model.tau(x_ev)
    return d, x_ev, truth, scores


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def check_data(ck: Checker) -> None:
    for name, dgp in (("heterogeneous", heterogeneous_dgp()),
                      ("single-segment", single_segment_dgp()),
                      ("constant", constant_dgp())):
        d = dgp.draw(400_000, seed=5)
        taus = dgp.tau(d["x"])
        mc = float(taus.mean())
        se = float(taus.std(ddof=1) / np.sqrt(taus.size))
        # A constant effect has zero MC noise: the comparison is exact, down
        # to a floor that absorbs float noise (std() of identical values is
        # ~1e-17, not 0).
        tol = max(4 * se, 1e-12)
        detail = (f"{abs(dgp.ate - mc) / se:.1f} MC SE" if se > 1e-15
                  else f"exact match, |diff| {abs(dgp.ate - mc):.1e} (zero MC noise)")
        ck.check(
            f"data/ate-quadrature/{name}",
            f"quadrature ATE {dgp.ate:.6f} vs 400k-draw MC {mc:.6f} ({detail})",
            abs(dgp.ate - mc) <= tol,
        )

    # Benchmark CSVs round-trip through the loader (one file per directory,
    # since the loader picks the first recognised CSV it finds).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(0)
        n = 500
        pd.DataFrame({
            **{f"f{i}": rng.normal(size=n) for i in range(12)},
            "treatment": (rng.random(n) < 0.85).astype(int),
            "visit": (rng.random(n) < 0.05).astype(int),
            "conversion": (rng.random(n) < 0.01).astype(int),
        }).to_csv(root / "criteo.csv", index=False)
        data = load_data(root)
        ck.check(
            "data/loader-criteo",
            f"{data.source}, n={data.x.shape[0]}, features={data.x.shape[1]}, "
            f"treated={data.t.mean():.2f}",
            "Criteo" in data.source and data.x.shape == (n, 12)
            and abs(data.t.mean() - 0.85) < 0.05,
        )

        hill = root / "hillstrom"
        hill.mkdir()
        pd.DataFrame({
            "recency": rng.integers(1, 12, n).astype(float),
            "history_segment": ["1) $0-100"] * n,
            "history": rng.uniform(30, 300, n),
            "segment": ["Mens Email"] * (n // 2) + ["No Email"] * (n - n // 2),
            "visit": (rng.random(n) < 0.1).astype(int),
            "spend": np.zeros(n),
        }).to_csv(hill / "hillstrom.csv", index=False)
        data = load_data(hill)
        ck.check(
            "data/loader-hillstrom",
            f"{data.source}, n={data.x.shape[0]}, treated={data.t.mean():.2f}",
            "Hillstrom" in data.source and data.x.shape[0] == n
            and abs(data.t.mean() - 0.5) < 0.05,
        )


# --------------------------------------------------------------------------
# Qini machinery
# --------------------------------------------------------------------------


def check_qini(ck: Checker) -> None:
    # Hand-computed curve on a tiny example: the formula, not the vibe.
    t = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    score = np.array([3.0, 1.0, 2.0, 0.0])
    _, curve = qini_curve(t, y, score)
    expected = np.array([1.0, 1.0, -1.0, 0.0])
    ck.check(
        "qini/curve-formula",
        f"curve {np.round(curve, 3).tolist()} vs hand-computed {expected.tolist()}",
        np.allclose(curve, expected),
    )

    # The null: a random ranking has zero expected area above its diagonal.
    dgp = heterogeneous_dgp()
    d = dgp.draw(20_000, seed=9)
    ev = slice(10_000, None)
    t_ev, y_ev = d["t"][ev], d["y"][ev]
    rng = np.random.default_rng(0)
    nulls = np.array([auqc(t_ev, y_ev, rng.random(t_ev.size)) for _ in range(60)])
    se = nulls.std(ddof=1) / np.sqrt(nulls.size)
    ck.check(
        "qini/random-ranker-is-null",
        f"mean auQC of random rankings {nulls.mean():.2f} ({abs(nulls.mean()) / se:.1f} SE from 0)",
        abs(nulls.mean()) < 3 * se,
    )

    # Oracle and models share the code path, so the check is the ordering.
    # A balanced-arm world is used deliberately: Qini's noise is dominated by
    # the smaller arm, and at the Criteo-like 85% treated share a 10k-user
    # holdout cannot statistically separate an oracle from random targeting.
    # That fact is itself part of the report (section on evaluation power).
    eval_dgp = heterogeneous_dgp()
    eval_dgp.treat_share = 0.5
    d = eval_dgp.draw(50_000, seed=9)
    t_ev, y_ev = d["t"][20_000:], d["y"][20_000:]
    truth = eval_dgp.tau(d["x"][20_000:])
    scores = {"oracle": truth,
              "random": rng.random(t_ev.size),
              "anti": -truth}          # targeting the backfired segment first
    _, draws = paired_bootstrap_auqc(scores, t_ev, y_ev, boot=300, seed=1)
    diffs = {f"{a} - {b}": (float((draws[a] - draws[b]).mean()),
                            float((draws[a] - draws[b]).std(ddof=1)))
             for a, b in (("oracle", "random"), ("random", "anti"),
                          ("oracle", "anti"))}
    ck.check(
        "qini/oracle-beats-random-beats-anti",
        "paired auQC differences: "
        + "  ".join(f"{k}={m:+.1f}+-{s:.1f}" for k, (m, s) in diffs.items()),
        diffs["oracle - random"][0] > 2 * diffs["oracle - random"][1]
        and diffs["random - anti"][0] > 2 * diffs["random - anti"][1],
    )


# --------------------------------------------------------------------------
# Anchor 1: constant effects
# --------------------------------------------------------------------------


def check_constant(ck: Checker) -> None:
    dgp = constant_dgp(0.02)
    d, _, truth, scores = _fit_all(dgp, n=30_000, train_n=20_000, seed=13)
    t_ev, y_ev = d["t"][20_000:], d["y"][20_000:]

    rng = np.random.default_rng(0)
    null_se = float(np.std([auqc(t_ev, y_ev, rng.random(t_ev.size))
                            for _ in range(40)], ddof=1))
    areas = {k: auqc(t_ev, y_ev, sc) for k, sc in scores.items()}
    worst = max(abs(v) for v in areas.values())
    spread = max(areas.values()) - min(areas.values())
    ck.check(
        "constant/all-models-agree",
        "auQC " + "  ".join(f"{k}={v:.1f}" for k, v in areas.items())
        + f" (random-ranker SE {null_se:.1f})",
        worst < 3.5 * null_se and spread < 5 * null_se,
    )

    # T- and X-learner and the forest should sit on the truth on average...
    calib = {k: float(sc.mean()) for k, sc in scores.items() if k in ("t", "x", "forest")}
    ck.check(
        "constant/mean-tau-calibrated",
        "mean predicted tau " + "  ".join(f"{k}={v:.4f}" for k, v in calib.items())
        + f" vs true {dgp.ate:.4f}",
        all(abs(v - dgp.ate) < 0.008 for v in calib.values()),
    )
    # ...while the S-learner documents its own known failure: the treatment
    # feature is one column among ten, no split uses it, and the model
    # collapses to "no effect anywhere".
    ck.check(
        "constant/s-learner-shrinks-to-zero",
        f"mean predicted tau of the S-learner {scores['s'].mean():.4f} "
        f"vs true {dgp.ate:.4f} (the failure mode Künzel et al. predicted)",
        abs(scores["s"].mean()) < 0.01,
    )


# --------------------------------------------------------------------------
# Anchor 2: one strong segment
# --------------------------------------------------------------------------


def check_segment(ck: Checker) -> None:
    dgp = single_segment_dgp(0.10)
    d, x_ev, truth, scores = _fit_all(dgp, n=30_000, train_n=20_000, seed=17)
    t_ev, y_ev = d["t"][20_000:], d["y"][20_000:]
    top = x_ev.shape[0] // 10
    seg = dgp.seg_tau
    for key in ORDER:
        sc = scores[key]
        picked = truth[np.argsort(-sc)[:top]].mean()
        ck.check(
            f"segment/top-decile/{key}",
            f"top decile by {key} has true mean tau {picked:.4f} "
            f"(segment tau {seg:.2f}, population {dgp.ate:.4f})",
            picked >= 0.6 * seg,
        )
    oracle_top = truth[np.argsort(-truth)[:top]].mean()
    ck.check(
        "segment/oracle-is-the-segment",
        f"oracle top decile {oracle_top:.4f} (== segment tau {seg:.2f})",
        abs(oracle_top - seg) < 1e-9,
    )


# --------------------------------------------------------------------------
# Model constructions
# --------------------------------------------------------------------------


def check_models(ck: Checker) -> None:
    # Künzel's claim: with a rare treated arm the X-learner borrows strength
    # from the big control arm and beats the T-learner.
    dgp = heterogeneous_dgp()
    dgp.treat_share = 0.10
    d, x_ev, truth, scores = _fit_all(dgp, n=40_000, train_n=28_000, seed=23)
    rmse = {k: float(np.sqrt(np.mean((sc - truth) ** 2)))
            for k, sc in scores.items() if k in ("t", "x")}
    corr = {k: float(np.corrcoef(scores[k], truth)[0, 1])
            for k in ("t", "x")}
    ck.check(
        "models/x-beats-t-under-imbalance",
        f"10% treated: X rmse {rmse['x']:.4f} vs T {rmse['t']:.4f}; "
        f"corr {corr['x']:.3f} vs {corr['t']:.3f}",
        rmse["x"] < rmse["t"] and corr["x"] > corr["t"],
    )

    # Athey-Imbens claim: honest leaf estimation beats in-sample estimation.
    d = dgp.draw(40_000, seed=29)  # note: still 10% treated, harder world
    x_tr, t_tr, y_tr = d["x"][:28_000], d["t"][:28_000], d["y"][:28_000]
    x_ev2 = d["x"][28_000:]
    truth2 = dgp.tau(x_ev2)
    res = {}
    for honest in (True, False):
        forest = HonestCausalForest(seed=31, n_trees=100, honest=honest)
        forest.fit(x_tr, t_tr, y_tr)
        res[honest] = float(np.sqrt(np.mean((forest.tau(x_ev2) - truth2) ** 2)))
    ck.check(
        "models/honest-beats-dishonest",
        f"fresh-data rmse: honest {res[True]:.4f} vs in-sample {res[False]:.4f}",
        res[True] < res[False],
    )

    # Calibration of the forest at the default treated share. A single
    # forest's mean prediction wanders by ~+-0.014 (each tree is a noisy
    # effect estimate and trees are correlated), so calibration is judged on
    # the mean over three independent training draws.
    dgp85 = heterogeneous_dgp()
    means = []
    for seed in (37, 41, 43):
        d = dgp85.draw(40_000, seed=seed)
        forest = HonestCausalForest(seed=seed, n_trees=100).fit(
            d["x"][:28_000], d["t"][:28_000], d["y"][:28_000])
        means.append(float(forest.tau(d["x"][28_000:]).mean()))
    ck.check(
        "models/forest-calibrated-to-ate",
        "mean predicted tau over 3 draws: "
        + "  ".join(f"{m:.4f}" for m in means)
        + f" (mean {np.mean(means):.4f}) vs true ATE {dgp85.ate:.4f}",
        abs(float(np.mean(means)) - dgp85.ate) < 0.008,
    )

    # The ranking-quality finding, one more training draw: the forest should
    # capture the most incremental conversions per user targeted (exact truth).
    d, _, truth3, scores3 = _fit_all(dgp85, n=30_000, train_n=20_000, seed=43)
    fractions = (0.1, 0.2)
    cap = {k: capture_curve(truth3, sc, fractions)
           for k, sc in scores3.items() if k in ORDER}
    oracle_cap = capture_curve(truth3, truth3, fractions)
    ck.check(
        "models/forest-captures-most",
        "capture at 10%/20% of the population: "
        + "  ".join(f"{k}={cap[k][0.1]:.0f}/{cap[k][0.2]:.0f}" for k in ORDER)
        + f"; oracle {oracle_cap[0.1]:.0f}/{oracle_cap[0.2]:.0f}",
        all(cap["forest"][f] >= max(cap[k][f] for k in ORDER if k != "forest")
            for f in fractions)
        and cap["forest"][0.1] >= 0.8 * oracle_cap[0.1],
    )


# --------------------------------------------------------------------------
# Targeting arithmetic
# --------------------------------------------------------------------------


def check_targeting(ck: Checker) -> None:
    dgp = heterogeneous_dgp()
    d = dgp.draw(30_000, seed=41)
    truth = dgp.tau(d["x"][15_000:])
    rng = np.random.default_rng(0)
    fractions = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    for name, score in (("forest-like", truth + rng.normal(0, 0.01, truth.size)),
                        ("noisy", truth + rng.normal(0, 0.05, truth.size)),
                        ("random", rng.random(truth.size))):
        cap = capture_curve(truth, score, fractions)
        vals = [cap[f] for f in fractions]
        exact_total = float(truth.sum())
        ck.check(
            f"targeting/capture-{name}",
            f"captured {vals[0]:.1f} -> {vals[-1]:.1f}, total effect {exact_total:.1f}; "
            f"monotone={all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))}",
            all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
            and abs(cap[1.0] - exact_total) < 1e-6,
        )
    # a better ranking must capture at least as much below the 100% point
    # (at 100% both exhaust the population; floats make the tie inexact)
    good = capture_curve(truth, truth + rng.normal(0, 0.01, truth.size), fractions)
    bad = capture_curve(truth, rng.random(truth.size), fractions)
    ck.check(
        "targeting/better-ranking-captures-more",
        "good vs random capture: "
        + "  ".join(f"{f:.0%}:{good[f] - bad[f]:+.1f}" for f in fractions),
        all(good[f] >= bad[f] - 1e-6 for f in fractions if f < 1.0),
    )


GROUPS = (
    ("data", check_data),
    ("qini", check_qini),
    ("constant", check_constant),
    ("segment", check_segment),
    ("models", check_models),
    ("targeting", check_targeting),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help="run a single group, e.g. --only qini")
    args = parser.parse_args()
    ck = Checker()
    for name, fn in GROUPS:
        if args.only and name != args.only:
            continue
        print(f"--- {name} ---")
        fn(ck)
    raise SystemExit(ck.summary())


if __name__ == "__main__":
    main()
