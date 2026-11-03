"""Study 03 evaluation: run the three designs against a known global effect.

Four sweeps, one per question:

``externality``   how the bias grows as the congestion externality strengthens
``inference``     whether each design's standard error covers the truth
``carryover``     what slow driver supply does to a switchback estimate
``budget``        precision per unit of traffic, i.e. what each design costs

Plus ``crowdout``, which isolates the second interference channel: scarce
capacity allocated in favour of treated riders.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .designs import DesignResult, cluster, switchback, user_level
from .market import MarketConfig, population_direct_effect, population_global_effect

OUT = Path(__file__).resolve().parents[2] / "studies" / "03-interference"

# Fixed experiment sizes. They are not equal in traffic -- equalising traffic
# across these designs requires choosing how many cities to spend on time,
# which is the ``budget`` sweep's job. These are "what a team would plausibly
# run", and the traffic column keeps the comparison honest.
USER_CITIES = 8
CLUSTER_CITIES = 16
SWITCH_CITIES = 8
SWITCH_PERIODS = 8
RIDERS = MarketConfig().riders
REFERENCE_TRAFFIC = USER_CITIES * RIDERS

DESIGNS = ("user-level", "cluster", "switchback")


def _run(kind: str, cfg: MarketConfig, rng: np.random.Generator, **kw) -> DesignResult:
    if kind == "user-level":
        return user_level(cfg, kw.get("cities", USER_CITIES), rng, share=kw.get("share", 0.5))
    if kind == "cluster":
        return cluster(cfg, kw.get("cities", CLUSTER_CITIES), rng)
    if kind == "switchback":
        return switchback(
            cfg,
            kw.get("cities", SWITCH_CITIES),
            kw.get("periods", SWITCH_PERIODS),
            rng,
            carryover=kw.get("carryover", 0.0),
            block_len=kw.get("block_len", 1),
            washout=kw.get("washout", 0),
        )
    raise ValueError(kind)


def summarise(
    kind: str,
    cfg: MarketConfig,
    reps: int,
    rng: np.random.Generator,
    truth: float,
    own_estimand: float,
    **kw,
) -> dict[str, float | str]:
    """Repeat one design and reduce it to the numbers the report quotes."""
    taus = np.empty(reps)
    ses = np.empty(reps)
    crits = np.empty(reps)
    ses_naive = np.full(reps, np.nan)
    traffic = 0.0
    spent = 0.0

    for i in range(reps):
        result = _run(kind, cfg, rng, **kw)
        taus[i] = result.tau
        ses[i] = result.se
        crits[i] = result.crit
        if result.se_naive is not None:
            ses_naive[i] = result.se_naive
        traffic = result.units
        spent = result.units_spent or result.units

    mean_tau = float(taus.mean())
    sd = float(taus.std(ddof=1))
    covered = np.abs(taus - truth) <= crits * ses
    # Same interval, normal quantile instead of the t correction, to show what
    # the correction is buying.
    covered_z = np.abs(taus - truth) <= 1.959963984540054 * ses
    has_naive = bool(np.isfinite(ses_naive).any())
    covered_naive = (
        np.abs(taus - truth) <= 1.959963984540054 * ses_naive if has_naive else None
    )

    return {
        "design": kind,
        "congestion": cfg.congestion,
        "priority": cfg.priority,
        "elasticity_sd": cfg.elasticity_sd,
        "truth": truth,
        "own_estimand": own_estimand,
        "mean_tau": mean_tau,
        "sd": sd,
        "bias": mean_tau - truth,
        "bias_vs_own": mean_tau - own_estimand,
        "relative_bias": (mean_tau - truth) / truth if truth else np.nan,
        "mc_se": sd / np.sqrt(reps),
        "mean_se": float(np.mean(ses)),
        "mean_se_naive": float(np.nanmean(ses_naive)) if has_naive else np.nan,
        "mean_crit": float(np.mean(crits)),
        "coverage": float(covered.mean()),
        "coverage_z": float(covered_z.mean()),
        "coverage_naive": float(covered_naive.mean()) if has_naive else np.nan,
        "ci_width": float(np.mean(2 * crits * ses)),
        "ci_width_naive": (
            float(np.mean(2 * 1.959963984540054 * ses_naive)) if has_naive else np.nan
        ),
        "traffic": float(traffic),
        "traffic_spent": float(spent),
        "sd_at_reference_traffic": sd * np.sqrt(spent / REFERENCE_TRAFFIC),
        "reps": reps,
    }


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def sweep_externality(cfg: MarketConfig, phis, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for phi in phis:
        c = replace(cfg, congestion=float(phi))
        truth = population_global_effect(c)
        own = population_direct_effect(c, 0.5)
        rng = np.random.default_rng(seed + int(round(phi * 1_000)))
        for kind in DESIGNS:
            rows.append(summarise(kind, c, reps, rng, truth, own))
    return pd.DataFrame(rows)


def sweep_inference(cfg: MarketConfig, sizes, reps: int, seed: int) -> pd.DataFrame:
    c = replace(cfg, congestion=1.0)
    truth = population_global_effect(c)
    own = population_direct_effect(c, 0.5)
    rows = []
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        rows.append(summarise("cluster", c, reps, rng, truth, own, cities=n))
        rows.append(summarise("user-level", c, reps, rng, truth, own, cities=n))
    return pd.DataFrame(rows)


def sweep_carryover(cfg: MarketConfig, values, reps: int, seed: int) -> pd.DataFrame:
    """Carryover at three block structures: pure alternation, blocks, washout.

    Longer blocks dilute contamination -- only the first period after a switch
    is affected -- and washout removes it, at the cost of traffic. Both are
    measured rather than asserted, because the cost is the whole argument.
    """
    schemes = (
        (1, 0, "alternate every period"),
        (2, 0, "2-period blocks"),
        (2, 1, "2-period blocks + 1 washout"),
    )
    c = replace(cfg, congestion=1.0)
    truth = population_global_effect(c)
    own = population_direct_effect(c, 0.5)
    rows = []
    for block_len, washout, label in schemes:
        for carry in values:
            rng = np.random.default_rng(seed + int(round(carry * 100)) + 7 * washout + block_len)
            row = summarise(
                "switchback", c, reps, rng, truth, own,
                carryover=float(carry), block_len=block_len, washout=washout,
            )
            row["scheme"] = label
            row["carryover"] = float(carry)
            rows.append(row)
    return pd.DataFrame(rows)


def sweep_budget(cfg: MarketConfig, reps: int, seed: int) -> pd.DataFrame:
    """Precision as a function of traffic, at two levels of city heterogeneity."""
    rows = []
    for label, sd in (("homogeneous cities", 0.0), ("heterogeneous cities", 0.25)):
        c = replace(cfg, congestion=1.0, elasticity_sd=sd)
        truth = population_global_effect(c)
        own = population_direct_effect(c, 0.5)
        rng = np.random.default_rng(seed + int(sd * 100))
        block = []
        for cities in (4, 8, 16, 32):
            block.append(summarise("user-level", c, reps, rng, truth, own, cities=cities))
            block.append(summarise("cluster", c, reps, rng, truth, own, cities=cities))
            block.append(
                summarise("switchback", c, reps, rng, truth, own,
                          cities=cities, periods=SWITCH_PERIODS)
            )
        for row in block:
            row["regime"] = label
        rows.extend(block)
    return pd.DataFrame(rows)


def sweep_crowdout(cfg: MarketConfig, priorities, reps: int, seed: int) -> pd.DataFrame:
    """The scarce-capacity channel, with the congestion channel held fixed."""
    strained = cfg.strained()
    rows = []
    for pi in priorities:
        c = replace(strained, congestion=1.0, priority=float(pi))
        truth = population_global_effect(c)
        own = population_direct_effect(c, 0.5)
        rng = np.random.default_rng(seed + int(round(pi * 100)))
        rows.append(summarise("user-level", c, reps, rng, truth, own))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

COLORS = {
    "user-level": "#B0453C",
    "cluster": "#1F5C99",
    "switchback": "#0F6E56",
}
LABELS = {
    "user-level": "user-level randomisation",
    "cluster": "cluster (city) randomisation",
    "switchback": "switchback",
}


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.22, lw=0.6)


def plot_bias(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    ax = axes[0]
    for kind in DESIGNS:
        sub = df[df.design == kind].sort_values("congestion")
        ax.errorbar(
            sub.congestion, sub.bias, yerr=2 * sub.mc_se,
            marker="o", ms=4.5, lw=1.6, capsize=2.5, alpha=0.9,
            color=COLORS[kind], label=LABELS[kind],
        )
    ax.axhline(0, color="#5B5B57", lw=1.0, ls="--")
    ax.set_xlabel("congestion externality $\\phi$")
    ax.set_ylabel("bias vs the roll-out effect\n(completed trips per rider)")
    ax.set_title("Only the clustered designs track the roll-out", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean(ax)

    ax = axes[1]
    for kind in DESIGNS:
        sub = df[df.design == kind].sort_values("congestion")
        ax.plot(sub.congestion, sub.mean_tau, marker="o", ms=4.5, lw=1.6,
                color=COLORS[kind], label=LABELS[kind])
    ref = df.drop_duplicates("congestion").sort_values("congestion")
    ax.plot(ref.congestion, ref.truth, color="#5B5B57", lw=1.3, ls="--",
            label="global effect (roll-out)")
    ax.plot(ref.congestion, ref.own_estimand, color="#B0453C", lw=1.1, ls=":",
            label="within-market effect (half treated)")
    ax.set_xlabel("congestion externality $\\phi$")
    ax.set_ylabel("mean estimate")
    ax.set_title("User-level answers a different question, precisely", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_inference(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1))

    ax = axes[0]
    cl = df[df.design == "cluster"].sort_values("traffic")
    ax.plot(cl.traffic / 1000, cl.coverage, "o-", color=COLORS["cluster"], lw=1.6, ms=4.5,
            label="cluster-robust SE, t quantile")
    ax.plot(cl.traffic / 1000, cl.coverage_z, "d--", color="#6B8EB8", lw=1.3, ms=4.0,
            label="cluster-robust SE, normal quantile")
    ax.plot(cl.traffic / 1000, cl.coverage_naive, "s:", color="#C0504D", lw=1.4, ms=4.0,
            label="rider-level SE (ignores clustering)")
    us = df[df.design == "user-level"].sort_values("traffic")
    ax.plot(us.traffic / 1000, us.coverage, "^:", color="#B07A2A", lw=1.2, ms=3.6,
            label="user-level (right SE, wrong estimand)")
    ax.axhline(0.95, color="#5B5B57", lw=1.0, ls="--")
    ax.set_xlabel("traffic (thousand rider-periods)")
    ax.set_ylabel("95% CI coverage of the roll-out effect")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title("Two separate ways to be overconfident", fontsize=10, loc="left")
    # Semi-opaque frame: four lines cross most of the panel, so an invisible
    # box always lands on something.
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.2,
              loc="center left")
    _clean(ax)

    ax = axes[1]
    ax.plot(cl.traffic / 1000, cl.ci_width, "o-", color=COLORS["cluster"], lw=1.6, ms=4.5,
            label="cluster-robust")
    ax.plot(cl.traffic / 1000, cl.ci_width_naive, "s--", color="#C0504D", lw=1.4, ms=4.0,
            label="rider-level (wrong)")
    ax.set_xlabel("traffic (thousand rider-periods)")
    ax.set_ylabel("mean 95% CI width")
    ax.set_title("The honest interval is the wide one", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_carryover(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1))

    ax = axes[0]
    styles = {
        "alternate every period": ("o-", "#0F6E56"),
        "2-period blocks": ("s--", "#B07A2A"),
        "2-period blocks + 1 washout": ("^:", "#1F5C99"),
    }
    for scheme, (marker, color) in styles.items():
        sub = df[df.scheme == scheme].sort_values("carryover")
        ax.errorbar(sub.carryover, sub.bias, yerr=2 * sub.mc_se, fmt=marker, color=color,
                    lw=1.5, ms=4.5, capsize=2.5, label=scheme)
    ax.axhline(0, color="#5B5B57", lw=1.0, ls="--")
    ax.set_xlabel("carryover: share of the previous period's outcome that persists")
    ax.set_ylabel("bias vs the roll-out effect")
    ax.set_title("Alternation lets carryover cancel the signal", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _clean(ax)

    ax = axes[1]
    for scheme, (marker, color) in styles.items():
        sub = df[df.scheme == scheme].sort_values("carryover")
        # RMSE, not sd: past lambda ~ 0.57 the estimate crosses zero and its
        # variance rises again while the answer stays wrong.
        rmse = np.sqrt(sub.bias**2 + sub.sd**2)
        ax.plot(sub.carryover, rmse, marker, color=color, lw=1.5, ms=4.5, label=scheme)
    ax.set_xlabel("carryover")
    ax.set_ylabel("RMSE vs the roll-out effect")
    ax.set_title("Washout is the only scheme that stays accurate", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_budget(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), sharey=True)
    for ax, regime in zip(axes, df.regime.unique()):
        sub = df[df.regime == regime]
        for kind in DESIGNS:
            s = sub[sub.design == kind].sort_values("traffic")
            ax.plot(s.traffic / 1000, s.sd, "o-", color=COLORS[kind], lw=1.6, ms=4.5,
                    label=LABELS[kind])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("traffic (thousand rider-periods)")
        ax.set_title(regime, fontsize=10, loc="left")
        _clean(ax)
    axes[0].set_ylabel("standard deviation of the estimate")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Precision per unit of traffic, by design", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_crowdout(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(df.priority, df.bias, yerr=2 * df.mc_se, fmt="o-", color=COLORS["user-level"],
                lw=1.6, ms=5, capsize=3, label="user-level bias")
    ax.axhline(0, color="#5B5B57", lw=1.0, ls="--")
    base = float(df.iloc[0].bias)
    ax.axhline(base, color="#B07A2A", lw=1.1, ls=":",
               label="congestion channel alone")
    for _, row in df.iterrows():
        ax.annotate(f"{row.bias:.3f}", (row.priority, row.bias),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5,
                    color="#3A3A38")
    ax.set_xlabel("allocation priority given to treated riders")
    ax.set_ylabel("bias vs the roll-out effect")
    ax.set_title("Scarce capacity: who absorbs the shortfall", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--budget-reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="small run for a smoke test")
    args = parser.parse_args()

    reps = 60 if args.quick else args.reps
    budget_reps = 40 if args.quick else args.budget_reps
    figs = OUT / "figures"
    results = OUT / "results"
    figs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    cfg = MarketConfig()

    bias = sweep_externality(cfg, np.round(np.linspace(0, 3, 7), 2), reps, args.seed)
    bias.to_csv(results / "bias_vs_externality.csv", index=False)
    plot_bias(bias, figs / "bias_vs_externality.png")

    inference = sweep_inference(cfg, (4, 6, 10, 16, 24, 32), reps, args.seed + 11)
    inference.to_csv(results / "cluster_inference.csv", index=False)
    plot_inference(inference, figs / "cluster_coverage.png")

    carry = sweep_carryover(cfg, (0.0, 0.2, 0.4, 0.6, 0.8), reps, args.seed + 23)
    carry.to_csv(results / "switchback_carryover.csv", index=False)
    plot_carryover(carry, figs / "switchback_carryover.png")

    budget = sweep_budget(cfg, budget_reps, args.seed + 37)
    budget.to_csv(results / "precision_budget.csv", index=False)
    plot_budget(budget, figs / "precision_budget.png")

    crowd = sweep_crowdout(cfg, (0.0, 0.25, 0.5, 1.0), reps, args.seed + 41)
    crowd.to_csv(results / "crowdout.csv", index=False)
    plot_crowdout(crowd, figs / "crowdout.png")

    print("Bias vs externality")
    print(
        bias.pivot_table(index="congestion", columns="design", values="bias")
        .round(4).to_string()
    )
    print("\nCluster inference (coverage of the roll-out effect)")
    print(
        inference.pivot_table(index="traffic", columns="design",
                              values=["coverage", "coverage_naive"])
        .round(3).to_string()
    )
    print("\nSwitchback under carryover")
    print(
        carry.pivot_table(index="carryover", columns="scheme", values="bias")
        .round(4).to_string()
    )
    print("\nCrowd-out channel (strained market)")
    print(crowd[["priority", "mean_tau", "bias", "coverage"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
