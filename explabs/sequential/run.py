"""Study 01 — sequential testing methods compared.

Three questions, three outputs:

1. **How much does peeking inflate the false-positive rate?** A curve of FPR
   against the number of looks, for each method family.
2. **What does each method cost in power, and what does it save in users?**
3. **How sensitive are the two prior-driven methods to their prior?** Because
   "set tau until the answer looks right" is not a validation strategy.

Run from the repository root::

    python -m explabs.sequential.run
    python -m explabs.sequential.run --reps 5000 --traffic 1000

Outputs land in ``studies/01-sequential/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..sim import ExperimentConfig, Simulation, simulate
from .policies import (
    BayesPeek,
    MSPRTPeek,
    Policy,
    default_policies,
)

STUDY_DIR = Path(__file__).resolve().parents[2] / "studies" / "01-sequential"
RESULTS_DIR = STUDY_DIR / "results"
FIGURES_DIR = STUDY_DIR / "figures"

FAMILY_STYLE = {
    "unadjusted": dict(color="#888780", marker="o"),
    "bonferroni": dict(color="#185FA5", marker="s"),
    "group sequential": dict(color="#0F6E56", marker="^"),
    "always-valid": dict(color="#D85A30", marker="D"),
    "bayesian": dict(color="#993556", marker="v"),
}


def family_of(policy_name: str) -> str:
    """Map a policy name to the method family it belongs to."""
    if policy_name.startswith(("naive_peek", "fixed_horizon")):
        return "unadjusted"
    if policy_name.startswith("bonferroni"):
        return "bonferroni"
    if policy_name.startswith(("obf", "pocock")):
        return "group sequential"
    if policy_name.startswith("msprt"):
        return "always-valid"
    if policy_name.startswith("bayes"):
        return "bayesian"
    return "other"


@dataclass
class PolicyReport:
    """Aggregated outcome for one policy on one experiment configuration."""

    policy: str
    looks: int
    reject_rate: float
    early_stop_rate: float
    mean_day: float
    median_day: float
    mean_users: float
    median_users: float

    def as_row(self, config: ExperimentConfig) -> dict:
        return {
            "policy": self.policy,
            "family": family_of(self.policy),
            "looks": self.looks,
            "config": config.label(),
            "relative_lift": config.relative_lift,
            "horizon_days": config.horizon_days,
            "reject_rate": round(self.reject_rate, 4),
            "early_stop_rate": round(self.early_stop_rate, 4),
            "mean_day": round(self.mean_day, 2),
            "median_day": round(self.median_day, 1),
            "mean_users": int(round(self.mean_users)),
            "median_users": int(round(self.median_users)),
        }


def evaluate(policy: Policy, sim: Simulation) -> PolicyReport:
    """Run one policy against a pre-simulated batch of experiments."""
    day_index, stopped = policy.stop_days(sim)

    day = day_index + 1.0  # report in 1-based days
    rows = np.arange(sim.reps)
    users = sim.c_n[rows, day_index] + sim.t_n[rows, day_index]
    horizon = sim.horizon

    return PolicyReport(
        policy=policy.name,
        looks=int(policy.peekable(horizon).sum()),
        reject_rate=float(stopped.mean()),
        early_stop_rate=float((stopped & (day < horizon)).mean()),
        mean_day=float(day.mean()),
        median_day=float(np.median(day)),
        mean_users=float(users.mean()),
        median_users=float(np.median(users)),
    )


def run_grid(
    configs: list[ExperimentConfig],
    policies: list[Policy],
    reps: int = 2_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate every policy on every configuration.

    Traffic is simulated once per configuration and shared by all policies, so
    differences between policies are not contaminated by different random
    draws (common random numbers).
    """
    rows = []
    for index, config in enumerate(configs):
        sim = simulate(config, reps=reps, seed=seed + index)
        for policy in policies:
            rows.append(evaluate(policy, sim).as_row(config))
    return pd.DataFrame(rows)


def add_savings(df: pd.DataFrame) -> pd.DataFrame:
    """Append sample-size savings relative to the fixed-horizon baseline."""
    baseline = (
        df[df["policy"] == "fixed_horizon"]
        .set_index("config")["mean_users"]
        .rename("baseline_users")
    )
    df = df.join(baseline, on="config")
    df["saved_vs_fixed"] = 1.0 - df["mean_users"] / df["baseline_users"]
    return df


def print_table(df: pd.DataFrame, title: str, columns: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(df[columns].to_string(index=False))


def plot_fpr_curve(fpr: pd.DataFrame, out_path: Path) -> None:
    """The peeking-cost curve: FPR against number of looks, by family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    # Families with a variable number of looks get a line.
    for family in ("unadjusted", "bonferroni", "group sequential"):
        subset = fpr[fpr["family"] == family].sort_values("looks")
        if len(subset) < 2:
            continue
        style = FAMILY_STYLE[family]
        ax.plot(
            subset["looks"],
            subset["reject_rate"],
            linewidth=1.7,
            markersize=6,
            label=family,
            **style,
        )

    # Families evaluated on a single continuous-monitoring schedule get a point.
    for family in ("always-valid", "bayesian"):
        subset = fpr[fpr["family"] == family]
        style = FAMILY_STYLE[family]
        ax.plot(
            subset["looks"],
            subset["reject_rate"],
            linestyle="none",
            markersize=8,
            label=family,
            **style,
        )

    ax.axhline(0.05, color="#5F5E5A", linestyle="--", linewidth=1, label="nominal 5%")
    ax.set_xlabel("number of looks per experiment")
    ax.set_ylabel("false positive rate under H0")
    ax.set_title("Peeking inflates false positives - but not for every method", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_power_cost(df: pd.DataFrame, out_path: Path) -> None:
    """Power against users spent, one point per policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    plotted = df[(df["looks"] > 1) | (df["policy"] == "fixed_horizon")]

    for family, subset in plotted.groupby("family"):
        style = FAMILY_STYLE.get(family, dict(color="#888780", marker="o"))
        ax.scatter(
            subset["mean_users"] / 1000,
            subset["reject_rate"],
            s=55,
            label=family,
            **style,
        )

    for _, row in plotted.iterrows():
        ax.annotate(
            row["policy"].replace("_", " "),
            (row["mean_users"] / 1000, row["reject_rate"]),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7.5,
            color="#3A3A38",
        )

    ax.set_xlabel("mean users at decision (thousands)")
    ax.set_ylabel("power at +5% lift")
    ax.set_title("Speed versus power: what each method actually trades", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _mean_users(policy: Policy, sim: Simulation) -> float:
    day_index, _ = policy.stop_days(sim)
    rows = np.arange(sim.reps)
    return float((sim.c_n[rows, day_index] + sim.t_n[rows, day_index]).mean())


def prior_sensitivity(
    config_null: ExperimentConfig,
    config_alt: ExperimentConfig,
    taus: list[float],
    prior_sds: list[float],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """FPR and power as a function of the two prior scales.

    A method that only behaves at a hand-picked prior has not been validated,
    it has been tuned. This table is the evidence that we checked.
    """
    rows = []
    sim_null = simulate(config_null, reps=reps, seed=seed + 900)
    sim_alt = simulate(config_alt, reps=reps, seed=seed + 901)

    for tau in taus:
        policy = MSPRTPeek(tau=tau)
        rows.append(
            {
                "method": "msprt",
                "prior": tau,
                "fpr": round(float(policy.stop_days(sim_null)[1].mean()), 4),
                "power": round(float(policy.stop_days(sim_alt)[1].mean()), 4),
                "mean_users": int(round(_mean_users(policy, sim_alt))),
            }
        )

    for prior_sd in prior_sds:
        policy = BayesPeek(prior_sd=prior_sd)
        rows.append(
            {
                "method": "bayes",
                "prior": prior_sd,
                "fpr": round(float(policy.stop_days(sim_null)[1].mean()), 4),
                "power": round(float(policy.stop_days(sim_alt)[1].mean()), 4),
                "mean_users": int(round(_mean_users(policy, sim_alt))),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3_000, help="Monte Carlo replications per config")
    parser.add_argument("--traffic", type=int, default=2_000, help="daily users per arm")
    parser.add_argument("--baseline", type=float, default=0.10, help="control conversion rate")
    parser.add_argument("--horizon", type=int, default=30, help="planned experiment length in days")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    common = dict(
        baseline_rate=args.baseline,
        daily_traffic_per_arm=args.traffic,
        horizon_days=args.horizon,
    )
    policies = default_policies()

    null = ExperimentConfig(relative_lift=0.0, **common)
    alt = ExperimentConfig(relative_lift=0.05, **common)

    fpr = add_savings(run_grid([null], policies, reps=args.reps, seed=args.seed))
    power = add_savings(run_grid([alt], policies, reps=args.reps, seed=args.seed + 100))

    print_table(
        fpr.sort_values(["family", "looks"]),
        f"Type-I error under H0 (target 0.05) - {null.label()}",
        ["policy", "family", "looks", "reject_rate", "mean_day", "mean_users"],
    )
    print_table(
        power.sort_values("mean_users"),
        f"Power and cost at lift = +5% - {alt.label()}",
        ["policy", "family", "looks", "reject_rate", "mean_day", "mean_users", "saved_vs_fixed"],
    )

    sensitivity = prior_sensitivity(
        null,
        alt,
        taus=[0.002, 0.005, 0.01, 0.02],
        prior_sds=[0.005, 0.01, 0.02],
        reps=min(args.reps, 1_500),
        seed=args.seed,
    )
    print_table(
        sensitivity,
        "Sensitivity to the prior scale (FPR must stay near 0.05 at every value)",
        ["method", "prior", "fpr", "power", "mean_users"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([fpr, power], ignore_index=True).to_csv(RESULTS_DIR / "policy_grid.csv", index=False)
    sensitivity.to_csv(RESULTS_DIR / "prior_sensitivity.csv", index=False)

    plot_fpr_curve(fpr, FIGURES_DIR / "fpr_inflation.png")
    plot_power_cost(power, FIGURES_DIR / "power_vs_cost.png")

    print(f"\nSaved {RESULTS_DIR / 'policy_grid.csv'}")
    print(f"Saved {RESULTS_DIR / 'prior_sensitivity.csv'}")
    print(f"Saved {FIGURES_DIR / 'fpr_inflation.png'}")
    print(f"Saved {FIGURES_DIR / 'power_vs_cost.png'}")


if __name__ == "__main__":
    main()
