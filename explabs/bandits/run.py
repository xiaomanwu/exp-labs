"""Study 04 evaluation: what adaptation buys, and what it charges for inference.

Four sweeps, one per question:

``regret``     how much traffic a bandit saves over a fixed split
``overlap``    how fast a bandit stops treating anyone differently
``inference``  what a naive / IPW / AIPW estimate reports after the bandit ran
``frontier``   the exchange rate between the two, dialled through epsilon

Every sweep uses the same user stream (common random numbers), so a difference
between two policies is a difference in the policy and not in the traffic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .dgp import additive_dgp, heterogeneous_dgp
from .inference import aipw, composition_bias, design_matrix, hajek, ipw, naive
from .policies import EpsilonGreedy, FixedShare, ThompsonSampling, UCB1, run_policy

OUT = Path(__file__).resolve().parents[2] / "studies" / "04-bandits"

# The policies compared throughout. Keys double as legend order and colour keys.
def policy_set(epsilon: float = 0.1) -> list:
    return [
        FixedShare(),
        ThompsonSampling(n_buckets=1),
        ThompsonSampling(n_buckets=4),
        EpsilonGreedy(epsilon=epsilon, n_buckets=1),
        EpsilonGreedy(epsilon=epsilon, n_buckets=4),
        UCB1(),
    ]


METHODS = ("naive", "ipw", "hajek", "aipw", "aipw-seg")


def _estimates(run) -> dict:
    """All estimators applied to one batch of replications."""
    return {
        "naive": naive(run.arms, run.rewards),
        "ipw": ipw(run.arms, run.rewards, run.propensities),
        "hajek": hajek(run.arms, run.rewards, run.propensities),
        "aipw": aipw(run.arms, run.rewards, run.propensities, design_matrix(run, "full")),
        "aipw-seg": aipw(run.arms, run.rewards, run.propensities,
                         design_matrix(run, "segments")),
    }


def summarise_run(run, world: str) -> list[dict]:
    """Bias, coverage and rejection rate for every estimator on one run."""
    truth = run.dgp.ate
    comp = composition_bias(run)
    rows = []
    for name, res in _estimates(run).items():
        s = res.summary(truth)
        # Bias decomposition on the replications where both sides are defined.
        finite = np.isfinite(res.estimate) & np.isfinite(comp)
        comp_f = comp[finite]
        est_f = res.estimate[finite]
        s.update(
            world=world,
            key=run.key,
            policy=run.label,
            truth=truth,
            treated_share=float(run.arms.mean()),
            composition_bias=float(comp_f.mean()) if comp_f.size else np.nan,
            selection_bias=float(np.mean(est_f - truth - comp_f)) if comp_f.size else np.nan,
        )
        rows.append(s)
    return rows


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def sweep_regret(lifts: tuple[float, ...], horizon: int, reps: int, seed: int) -> pd.DataFrame:
    """Cumulative regret for each policy, at a grid of horizons and lifts."""
    horizons = (125, 250, 500, 1000, 2000, 4000, 8000)
    horizons = [h for h in horizons if h <= horizon]
    if horizons[-1] != horizon:
        horizons.append(horizon)

    rows = []
    for i, lift in enumerate(lifts):
        dgp = additive_dgp(lift)
        for policy in policy_set():
            run = run_policy(policy, dgp, horizon, reps=reps, seed=seed + i,
                             record_propensity=False)
            paths = run.cumulative_regret()
            curve = paths.mean(axis=0)
            # Monte Carlo noise of the mean path. At small lifts the saving is
            # a few percent of a small number, so without this the table's
            # bottom rows read like effects.
            mc_se = paths.std(axis=0, ddof=1) / np.sqrt(paths.shape[0])
            for h in horizons:
                baseline = h * lift / 2.0
                rows.append({
                    "lift": lift, "horizon": h, "key": run.key, "policy": run.label,
                    "regret": float(curve[h - 1]),
                    "regret_se": float(mc_se[h - 1]),
                    "fixed_regret": baseline,
                    "saved_pct": float(1.0 - curve[h - 1] / baseline),
                    "saved_se": float(mc_se[h - 1] / baseline),
                })
    return pd.DataFrame(rows)


def sweep_overlap(dgp, horizon: int, reps: int, seed: int) -> pd.DataFrame:
    """Treated share and overlap floor over time, per policy."""
    rows = []
    for policy in policy_set():
        run = run_policy(policy, dgp, horizon, reps=reps, seed=seed)
        floor = np.minimum(run.propensities, 1.0 - run.propensities)
        share = run.arms.mean(axis=0)
        for t in range(horizon):
            rows.append({
                "key": run.key, "policy": run.label, "step": t + 1,
                "treated_share": float(share[t]),
                "overlap_floor": float(np.median(floor[:, t])),
                "low_overlap_share": float((floor[:, t] < 0.05).mean()),
            })
    return pd.DataFrame(rows)


def sweep_inference(dgp, world: str, horizon: int, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, policy in enumerate(policy_set()):
        run = run_policy(policy, dgp, horizon, reps=reps, seed=seed + i)
        rows.extend(summarise_run(run, world))
    return pd.DataFrame(rows)


def sweep_horizon(dgp, horizons: tuple[int, ...], reps: int, seed: int) -> pd.DataFrame:
    """Does the naive bias shrink as the experiment runs longer? It depends."""
    rows = []
    for i, h in enumerate(horizons):
        for j, policy in enumerate((ThompsonSampling(n_buckets=1), EpsilonGreedy(0.1))):
            run = run_policy(policy, dgp, h, reps=reps, seed=seed + 10 * i + j)
            s = naive(run.arms, run.rewards).summary(dgp.ate)
            rows.append({"horizon": h, "key": run.key, "policy": run.label,
                         "bias": s["bias"], "mc_se": s["mc_se"]})
    return pd.DataFrame(rows)


def sweep_epsilon(dgp, epsilons: tuple[float, ...], horizon: int, reps: int,
                  seed: int) -> pd.DataFrame:
    """Regret saved against inference cost, with epsilon as the exchange rate."""
    reference = run_policy(FixedShare(), dgp, horizon, reps=reps, seed=seed,
                           record_propensity=False)
    fixed_regret = float(reference.cumulative_regret()[:, -1].mean())

    rows = []
    for i, eps in enumerate(epsilons):
        run = run_policy(EpsilonGreedy(epsilon=eps), dgp, horizon, reps=reps, seed=seed + i + 1)
        floor = float(np.median(np.minimum(run.propensities, 1 - run.propensities)))
        regret = float(run.cumulative_regret()[:, -1].mean())
        row = {
            "policy": f"epsilon={eps:g}", "key": f"eg{eps:g}", "epsilon": eps,
            "overlap_floor": floor,
            "regret": regret,
            "saved_pct": float(1 - regret / fixed_regret),
        }
        for name, res in _estimates(run).items():
            s = res.summary(dgp.ate)
            row[f"se_{name}"] = s["se_mean"]
            row[f"bias_{name}"] = s["bias"]
            row[f"cov_{name}"] = s["coverage"]
        rows.append(row)

    # Thompson sampling has no epsilon; place it at its own realised overlap.
    run = run_policy(ThompsonSampling(n_buckets=1), dgp, horizon, reps=reps, seed=seed + 99)
    floor = float(np.median(np.minimum(run.propensities, 1 - run.propensities)))
    regret = float(run.cumulative_regret()[:, -1].mean())
    row = {
        "policy": "thompson", "key": "ts", "epsilon": np.nan, "overlap_floor": floor,
        "regret": regret, "saved_pct": float(1 - regret / fixed_regret),
    }
    for name, res in _estimates(run).items():
        s = res.summary(dgp.ate)
        row[f"se_{name}"] = s["se_mean"]
        row[f"bias_{name}"] = s["bias"]
        row[f"cov_{name}"] = s["coverage"]
    rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

COLORS = {
    "fixed": "#6B6B66",
    "ts": "#1F5C99",
    "ts4": "#0F6E56",
    "eg": "#B07D2B",
    "eg4": "#8C5AA8",
    "ucb1": "#B0453C",
}
LABELS = {
    "fixed": "fixed 50/50",
    "ts": "TS (pooled)",
    "ts4": "TS (4 segments)",
    "eg": "$\\epsilon$-greedy (pooled)",
    "eg4": "$\\epsilon$-greedy (4 segments)",
    "ucb1": "UCB1 (pooled)",
}
ORDER = ["fixed", "ts", "ts4", "eg", "eg4", "ucb1"]


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.22, lw=0.6)


def plot_regret(df: pd.DataFrame, path: Path, headline_lift: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    ax = axes[0]
    sub = df[(df.lift == headline_lift) & (df.horizon >= 125)]
    for key in ORDER:
        s = sub[sub.key == key].sort_values("horizon")
        if s.empty:
            continue
        ax.plot(s.horizon, s.regret, marker="o", ms=3.5, lw=1.6,
                color=COLORS[key], label=LABELS[key])
    # One standard-error band, for the policy the study is about: without it
    # the gap between Thompson sampling and a fixed split looks cleaner than
    # it is at the short horizons.
    ts = sub[sub.key == "ts"].sort_values("horizon")
    if not ts.empty and "regret_se" in ts:
        ax.fill_between(ts.horizon, ts.regret - ts.regret_se, ts.regret + ts.regret_se,
                        color=COLORS["ts"], alpha=0.18, lw=0)
    ax.plot(sub.horizon.max() * np.array([0.06, 1.0]),
            headline_lift / 2 * sub.horizon.max() * np.array([0.06, 1.0]),
            color="#5B5B57", lw=1.0, ls="--", label="fixed: $T\\Delta/2$")
    ax.set_xscale("log")
    ax.set_xlabel("users (log scale)")
    ax.set_ylabel("cumulative regret (conversions)")
    ax.set_title(f"Adaptation saves regret at every horizon (lift {headline_lift:.0%})",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _clean(ax)

    ax = axes[1]
    last = df[df.horizon == df.horizon.max()]
    shown = ("fixed", "ts", "ts4", "eg", "ucb1")
    for key in shown:
        s = last[last.key == key].sort_values("lift")
        if s.empty:
            continue
        style = dict(ls="--", lw=1.2) if key == "fixed" else dict(marker="o", ms=4, lw=1.6)
        ax.plot(s.lift, s.regret, color=COLORS[key], label=LABELS[key], **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true lift $\\Delta$ (log scale)")
    ax.set_ylabel(f"regret at {int(df.horizon.max())} users")
    ax.set_title("A smaller lift is a harder problem", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_overlap(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    ax = axes[0]
    for key in ORDER:
        s = df[df.key == key].sort_values("step")
        if s.empty:
            continue
        # Per-step shares over a few hundred replications are binomial noise;
        # a rolling mean shows the trajectory the noise is hiding.
        smooth = s.treated_share.rolling(41, center=True, min_periods=11).mean()
        ax.plot(s.step, smooth, lw=1.7, color=COLORS[key], label=LABELS[key])
    ax.axhline(0.5, color="#5B5B57", lw=1.0, ls="--")
    ax.set_xlabel("users seen")
    ax.set_ylabel("share of traffic sent to the treatment")
    ax.set_title("Who the policy is still willing to treat", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.set_ylim(0, 1.02)
    _clean(ax)

    ax = axes[1]
    for key in ORDER:
        s = df[df.key == key].sort_values("step")
        if s.empty:
            continue
        ax.plot(s.step, np.clip(s.overlap_floor, 1e-3, None), lw=1.6,
                color=COLORS[key], label=LABELS[key])
    ax.set_yscale("log")
    ax.set_xlabel("users seen")
    ax.set_ylabel("median overlap  $\\min(e, 1-e)$")
    ax.set_title("Overlap decays, and what that does to inverse weights",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_inference(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["naive", "ipw", "hajek", "aipw"]
    labels = {"naive": "naive", "ipw": "IPW (HT)", "hajek": "IPW (Hájek)", "aipw": "AIPW"}
    colors = {"naive": "#B0453C", "ipw": "#1F5C99", "hajek": "#5B8DC0", "aipw": "#0F6E56"}
    keys = [k for k in ORDER if k in set(df.key)]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    width = 0.2
    for i, m in enumerate(methods):
        vals, errs = [], []
        for k in keys:
            row = df[(df.key == k) & (df.method == m)]
            if row.empty or not np.isfinite(row.bias.iloc[0]):
                vals.append(np.nan)
                errs.append(np.nan)
            else:
                vals.append(float(row.bias.iloc[0]))
                errs.append(2 * float(row.mc_se.iloc[0]))
        x = np.arange(len(keys)) + (i - 1.5) * width
        ax.bar(x, vals, width=width * 0.92, color=colors[m], label=labels[m],
               yerr=errs, capsize=2, error_kw={"lw": 0.9})
    # A policy with no overlap has no inverse-propensity estimate at all; say so
    # in place of the missing bar rather than letting the gap speak for itself.
    for j, k in enumerate(keys):
        row = df[(df.key == k) & (df.method == "ipw")]
        if not row.empty and not np.isfinite(row.bias.iloc[0]):
            ax.text(j + 0.12, 0.0008, "no overlap", rotation=90, va="bottom",
                    ha="center", fontsize=6.5, color="#B0453C", alpha=0.9)
    ax.axhline(0, color="#5B5B57", lw=1.0)
    ax.set_xlim(-0.6, len(keys) - 0.2)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([LABELS[k] for k in keys], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("bias (conversion rate)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    _clean(ax)

    ax = axes[1]
    for i, m in enumerate(methods):
        vals = []
        for k in keys:
            row = df[(df.key == k) & (df.method == m)]
            vals.append(float(row.coverage.iloc[0])
                        if not row.empty and np.isfinite(row.coverage.iloc[0]) else np.nan)
        x = np.arange(len(keys)) + (i - 1.5) * width
        ax.bar(x, vals, width=width * 0.92, color=colors[m], label=labels[m])
    ax.axhline(0.95, color="#5B5B57", lw=1.2, ls="--")
    ax.text(len(keys) - 0.5, 0.955, "nominal 95%", fontsize=7.5, ha="right",
            color="#5B5B57")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([LABELS[k] for k in keys], rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("coverage of the 95% interval")
    ax.set_title("Coverage: bias shows up as missed intervals", fontsize=10, loc="left")
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_frontier(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    ordered = df.sort_values("overlap_floor")

    ax = axes[0]
    ax.plot(ordered.overlap_floor, 100 * ordered.saved_pct, marker="o", ms=5, lw=1.6,
            color="#1F5C99", label="$\\epsilon$-greedy")
    ts = df[df.key == "ts"]
    if not ts.empty:
        ax.scatter(ts.overlap_floor, 100 * ts.saved_pct, marker="*", s=150,
                   color="#B0453C", zorder=5, label="Thompson sampling")
    ax.axhline(0, color="#5B5B57", lw=1.0, ls="--")
    ax.set_xlabel("overlap floor  $\\min(e, 1-e)$")
    ax.set_ylabel("regret saved vs fixed A/B (%)")
    ax.set_title("Exploration has an optimum: too little and the policy locks in",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.02, 0.09))
    _clean(ax)

    ax = axes[1]
    ax.plot(ordered.overlap_floor, ordered.se_aipw, marker="o", ms=5, lw=1.6,
            color="#0F6E56", label="AIPW")
    ax.plot(ordered.overlap_floor, ordered.se_ipw, marker="s", ms=4, lw=1.4,
            color="#1F5C99", label="IPW (HT)")
    ax.plot(ordered.overlap_floor, ordered.se_naive, marker="^", ms=4, lw=1.2,
            color="#B0453C", ls="--", label="naive")
    if not ts.empty:
        ax.scatter(ts.overlap_floor, ts.se_aipw, marker="*", s=150, color="#0F6E56",
                   zorder=5)
        ax.scatter(ts.overlap_floor, ts.se_ipw, marker="*", s=150, color="#1F5C99", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("overlap floor  $\\min(e, 1-e)$  (log scale)")
    ax.set_ylabel("standard error of the ATE estimate")
    ax.set_title("...and the naive estimator hides the cost in its bias",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    import time

    clock = time.perf_counter()

    def stamp(message: str) -> None:
        print(f"  [{time.perf_counter() - clock:6.1f}s] {message}", flush=True)

    reps = 60 if args.quick else args.reps
    horizon = 600 if args.quick else args.horizon

    figs = OUT / "figures"
    results = OUT / "results"
    figs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    # 1. Regret -----------------------------------------------------------
    # Regret has no propensities to compute, so replications are cheap here --
    # and they are needed, because a minority of replications lock onto the
    # wrong arm and make the mean saving jumpy at a few hundred reps.
    lifts = (0.0025, 0.005, 0.01, 0.02, 0.04)
    regret = sweep_regret(lifts, horizon, max(reps, 800), args.seed)
    regret.to_csv(results / "regret.csv", index=False)
    plot_regret(regret, figs / "regret_curves.png", headline_lift=0.02)
    stamp("regret")

    # 2. Overlap ----------------------------------------------------------
    mixed = heterogeneous_dgp(0.02)
    overlap = sweep_overlap(mixed, horizon, reps, args.seed + 100)
    overlap.to_csv(results / "overlap.csv", index=False)
    plot_overlap(overlap, figs / "overlap.png")
    stamp("overlap")

    # 3. Inference --------------------------------------------------------
    inference = sweep_inference(mixed, "heterogeneous", horizon, reps, args.seed + 200)
    inference.to_csv(results / "inference.csv", index=False)
    plot_inference(inference, figs / "inference_bias.png",
                   "Bias after the bandit ran (lift world, ATE = +2.0 pts)")
    stamp("inference (heterogeneous)")

    null = heterogeneous_dgp(0.0)
    null_table = sweep_inference(null, "null", horizon, reps, args.seed + 300)
    null_table.to_csv(results / "inference_null.csv", index=False)
    plot_inference(null_table, figs / "inference_null.png",
                   "Bias when the true average effect is zero")
    stamp("inference (null)")

    # 4. Horizon and exploration frontiers --------------------------------
    horizons = (500, 1000, 2000, 4000)
    hz = sweep_horizon(mixed, horizons, max(120, reps // 2), args.seed + 400)
    hz.to_csv(results / "horizon.csv", index=False)
    stamp("horizon")

    frontier = sweep_epsilon(additive_dgp(0.02), (0.02, 0.05, 0.1, 0.2, 0.35, 0.5),
                             horizon, max(150, reps // 2), args.seed + 500)
    frontier.to_csv(results / "frontier.csv", index=False)
    plot_frontier(frontier, figs / "epsilon_frontier.png")
    stamp("frontier")

    # Report the last horizon explicitly. A bare pivot_table over a frame with
    # one row per (lift, horizon) silently averages across horizons, which is
    # not the quantity anyone wants to read off this table.
    final = regret[regret.horizon == regret.horizon.max()]
    print(f"Regret saved vs fixed A/B after {int(regret.horizon.max()):,} users")
    print((final.pivot_table(index="lift", columns="key", values="saved_pct") * 100)
          .round(1).to_string())
    print("\n...and one standard error on that percentage")
    print((final.pivot_table(index="lift", columns="key", values="saved_se") * 100)
          .round(1).to_string())
    print("\nInference after the bandit (heterogeneous world, ATE = +0.020)")
    print(inference.pivot_table(index="key", columns="method",
                                values=["bias", "coverage"]).round(4).to_string())
    print("\nNaive bias vs the null world (rejection rate at ATE = 0)")
    print(null_table.pivot_table(index="key", columns="method", values="reject")
          .round(3).to_string())
    print("\nDoes the naive bias shrink with the horizon?")
    print(hz.pivot_table(index="horizon", columns="key", values="bias").round(4).to_string())
    print("\nExploration frontier (additive world, lift 2%)")
    print(frontier[["policy", "overlap_floor", "saved_pct", "se_ipw", "se_aipw", "cov_aipw"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
