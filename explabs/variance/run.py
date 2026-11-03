"""Monte Carlo harness for study 02.

Three analyses, each answering a different question:

``ceiling``   Is our CUPED implementation actually achieving the theoretical
              ``rho^2``? Sweep the covariate-outcome correlation with a linear
              outcome, where theory is exact.
``methods``   At a fixed correlation, how much of the remaining variance does
              each method remove when the outcome is nonlinear?
``decisions`` The same runs, converted into the units a team argues about: MDE,
              required sample size, false-positive rate, power.

The cheap analyses run many replications; anything touching CUPAC runs fewer,
because each replication costs several gradient-boosted fits. Both numbers are
reported so the Monte Carlo error is visible rather than assumed.

Usage
-----
    python -m explabs.variance.run              # full run, ~6 min on 8 cores
    python -m explabs.variance.run --quick      # ~1 min, for smoke testing
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dgp import DGPConfig, SIGNAL_KINDS, draw_experiment, theoretical_reduction
from .estimators import METHODS, Estimate, run_methods

ALPHA = 0.05
POWER = 0.80
Z_ALPHA = 1.959963984540054
Z_POWER = 0.8416212335729143

OUT = Path(__file__).resolve().parents[2] / "studies" / "02-variance-reduction"


# --------------------------------------------------------------------------
# One batch of replications
# --------------------------------------------------------------------------


def _one_rep(args: tuple[DGPConfig, int, tuple[str, ...]]) -> dict[str, tuple[float, float]]:
    """Draw one experiment and estimate the effect with every requested method.

    Module-level so joblib can pickle it. Returns ``{method: (tau, se)}``.
    """
    cfg, seed, methods = args
    exp = draw_experiment(cfg, np.random.default_rng(seed))
    chosen = {name: METHODS[name] for name in methods}
    return {name: (est.tau, est.se) for name, est in run_methods(exp, chosen).items()}


def replicate(
    cfg: DGPConfig,
    methods: tuple[str, ...] = tuple(METHODS),
    reps: int = 400,
    seed: int = 0,
    n_jobs: int = 8,
    save_to: Path | None = None,
) -> pd.DataFrame:
    """Run ``reps`` replications and return a tidy per-method summary."""
    from joblib import Parallel, delayed, parallel_backend

    tasks = [(cfg, seed + i, methods) for i in range(reps)]
    # One BLAS/OpenMP thread per worker: the parallelism here is across
    # replications, and letting each worker also spawn threads costs more in
    # contention than it buys (measured: 569% CPU for 4 workers before this).
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_rep)(t) for t in tasks
        )

    rows = []
    base_taus = np.array([r["baseline"][0] for r in results])
    base_var = base_taus.var(ddof=1)

    for name in methods:
        tau = np.array([r[name][0] for r in results])
        se = np.array([r[name][1] for r in results])
        rows.append(_summarise(name, cfg, tau, se, base_taus, base_var, reps))

    if save_to is not None:
        # Keep the per-replication estimates: figures and follow-up analyses can
        # then be rebuilt without refitting a single gradient-boosted model.
        np.savez_compressed(
            save_to,
            methods=np.array(methods),
            **{f"{n}_tau": np.array([r[n][0] for r in results]) for n in methods},
            **{f"{n}_se": np.array([r[n][1] for r in results]) for n in methods},
        )

    return pd.DataFrame(rows)


def _vr_interval(
    tau: np.ndarray, base_taus: np.ndarray, boot: int = 1_000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap interval for the variance reduction.

    This is not decoration. The reduction is a difference of two nearly equal
    variances when the covariate is weak, so its *absolute* Monte Carlo error is
    roughly constant (~1 variance point) no matter how large the true effect is.
    At rho = 0.2, where the target is 4 points, that is a 25% relative error --
    which is why the interval is reported instead of three significant figures.
    """
    rng = np.random.default_rng(seed)
    n = tau.size
    idx = rng.integers(0, n, (boot, n))
    base = base_taus[idx].var(axis=1, ddof=1)
    reduction = 1.0 - tau[idx].var(axis=1, ddof=1) / base
    lo, hi = np.percentile(reduction, [2.5, 97.5])
    return float(lo), float(hi)


def _summarise(
    name: str,
    cfg: DGPConfig,
    tau: np.ndarray,
    se: np.ndarray,
    base_taus: np.ndarray,
    base_var: float,
    reps: int,
) -> dict[str, float | str]:
    """Turn a batch of estimates into the metrics the report quotes."""
    mean_tau = float(tau.mean())
    emp_sd = float(tau.std(ddof=1))
    mean_se = float(se.mean())
    theory = theoretical_reduction(cfg)
    vr_lo, vr_hi = _vr_interval(tau, base_taus)

    # Monte Carlo standard error of the mean estimate, used to judge bias.
    mc_se = emp_sd / np.sqrt(reps)
    bias = mean_tau - cfg.tau

    # Coverage of the nominal 95% interval.
    covered = np.abs(tau - cfg.tau) <= Z_ALPHA * se

    # Rejection uses the estimator's own standard error, as a practitioner would.
    reject = np.abs(tau / se) > Z_ALPHA

    variance_ratio = emp_sd**2 / base_var

    return {
        "signal": cfg.signal,
        "rho": cfg.rho,
        "n_per_arm": cfg.n_per_arm,
        "tau_rel": cfg.tau_rel,
        "method": name,
        "tau_true": cfg.tau,
        "mean_tau": mean_tau,
        "bias": bias,
        "bias_t": bias / mc_se if mc_se > 0 else np.nan,
        "emp_sd": emp_sd,
        "mean_se": mean_se,
        "se_calibration": mean_se / emp_sd,
        "variance_ratio": variance_ratio,
        "variance_reduction": 1.0 - variance_ratio,
        "vr_lo": vr_lo,
        "vr_hi": vr_hi,
        "coverage": float(covered.mean()),
        "reject_rate": float(reject.mean()),
        "mde_rel": (Z_ALPHA + Z_POWER) * mean_se / cfg.mean,
        "n_req_ratio": variance_ratio,
        "theory_cuped": theory["cuped"],
        "theory_oracle": theory["oracle"],
        "reps": reps,
    }


# --------------------------------------------------------------------------
# Analysis 1: does CUPED hit the theoretical ceiling?
# --------------------------------------------------------------------------


def ceiling_sweep(reps: int, n_jobs: int) -> pd.DataFrame:
    """Linear outcome, correlation swept from 0.2 to 0.9."""
    frames = []
    for rho in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        cfg = DGPConfig(signal="linear", rho=rho, n_per_arm=5_000, tau_rel=0.01)
        frames.append(
            replicate(
                cfg,
                methods=("baseline", "cuped", "oracle"),
                reps=reps,
                n_jobs=n_jobs,
                save_to=OUT / "results" / "raw" / f"ceiling_rho{rho:.1f}.npz",
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Analysis 2 and 3: method comparison, null and alternative
# --------------------------------------------------------------------------


def method_comparison(reps: int, n_jobs: int) -> pd.DataFrame:
    """Three outcome shapes, each run under H0 and under a +1% effect."""
    frames = []
    for kind in SIGNAL_KINDS:
        for tau_rel, label in ((0.0, "H0"), (0.01, "H1")):
            cfg = DGPConfig(signal=kind, rho=0.7, n_per_arm=5_000, tau_rel=tau_rel)
            t0 = time.time()
            frame = replicate(
                cfg,
                reps=reps,
                n_jobs=n_jobs,
                save_to=OUT / "results" / "raw" / f"methods_{kind}_{label}.npz",
            )
            frame["hypothesis"] = label
            frame["seconds"] = round(time.time() - t0, 1)
            frames.append(frame)
            print(f"  {kind:<12} {label}  {reps} reps  {frame['seconds'].iloc[0]:>5.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_table(df: pd.DataFrame, title: str, columns: list[str], sort_by: str | None = None) -> None:
    if sort_by:
        df = df.sort_values(sort_by)
    view = df[columns].copy()
    for col in ("bias", "emp_sd", "mean_se", "mde_rel"):
        if col in view:
            view[col] = view[col].map(lambda v: f"{v: .4f}")
    for col in ("variance_reduction", "coverage", "reject_rate", "se_calibration", "theory_cuped"):
        if col in view:
            view[col] = view[col].map(lambda v: f"{v:.3f}")
    print(f"\n{title}")
    print(view.to_string(index=False))


def plot_ceiling(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    cuped = df[df.method == "cuped"].sort_values("rho")
    oracle = df[df.method == "oracle"].sort_values("rho")

    grid = np.linspace(0, 0.95, 100)
    ax.plot(grid, grid, color="#9A9A96", lw=1.0, ls="--", label="theory: $\\rho^2$")
    ax.plot(oracle.rho**2, oracle.variance_reduction, "o", ms=5, color="#B07A2A",
            label="oracle (adjust by true $f(X)$)")
    ax.plot(cuped.rho**2, cuped.variance_reduction, "o-", ms=4, lw=1.4, color="#1F5C99",
            label="CUPED")

    # Bootstrap intervals. Their absolute width is roughly constant (~1 variance
    # point) rather than proportional to the effect, which is why the low-rho
    # points look further from the line than they are.
    lower = cuped.variance_reduction - cuped.vr_lo
    upper = cuped.vr_hi - cuped.variance_reduction
    ax.errorbar(cuped.rho**2, cuped.variance_reduction, yerr=[lower, upper], fmt="none",
                ecolor="#1F5C99", elinewidth=1.0, capsize=2.5, alpha=0.7)

    ax.set_xlabel("covariate-outcome correlation squared, $\\rho^2$")
    ax.set_ylabel("realized variance reduction")
    ax.set_title("Linear outcome: CUPED reaches $\\rho^2$ exactly", fontsize=10, loc="left")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_methods(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h1 = df[df.hypothesis == "H1"]
    order = ["baseline", "cuped", "poststrat_5", "poststrat_20", "cupac",
             "cupac_insample", "oracle"]
    labels = {
        "baseline": "unadjusted",
        "cuped": "CUPED",
        "poststrat_5": "post-strat (5)",
        "poststrat_20": "post-strat (20)",
        "cupac": "CUPAC (cross-fit)",
        "cupac_insample": "CUPAC (in-sample)",
        "oracle": "oracle",
    }
    colors = {
        "baseline": "#8C8C88",
        "cuped": "#1F5C99",
        "poststrat_5": "#4E8F6B",
        "poststrat_20": "#2F6E4E",
        "cupac": "#B07A2A",
        "cupac_insample": "#C0504D",
        "oracle": "#5B5B57",
    }

    signals = [s for s in SIGNAL_KINDS if s in set(h1.signal)]
    fig, axes = plt.subplots(1, len(signals), figsize=(11.5, 4.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, kind in zip(axes, signals):
        sub = h1[h1.signal == kind].set_index("method").loc[[m for m in order if m in set(h1.method)]]
        y = np.arange(len(sub))[::-1]
        ax.barh(y, sub.variance_reduction, color=[colors[m] for m in sub.index], height=0.62)
        ax.errorbar(
            sub.variance_reduction,
            y,
            # Clip at zero: the plug-in estimate can sit a hair outside its own
            # bootstrap percentile range when the reduction is near zero.
            xerr=[np.maximum(sub.variance_reduction - sub.vr_lo, 0.0),
                  np.maximum(sub.vr_hi - sub.variance_reduction, 0.0)],
            fmt="none",
            ecolor="#2B2B29",
            elinewidth=0.9,
            capsize=2,
            alpha=0.75,
        )

        theory = sub.theory_cuped.iloc[0]
        ax.axvline(theory, color="#1F5C99", ls=":", lw=1.2)
        ax.axvline(sub.theory_oracle.iloc[0], color="#5B5B57", ls="--", lw=1.2)
        # Both reference labels live at the top of the panel, stacked, so they
        # never collide with a bar or with each other -- on the linear signal
        # the two lines coincide and side-by-side text would overlap.
        ax.text(theory, len(sub) - 0.45, " CUPED ceiling\n $\\rho^2R^2_{lin}$",
                color="#1F5C99", fontsize=7, va="top")
        ax.text(sub.theory_oracle.iloc[0], len(sub) - 1.55, " oracle $\\rho^2$",
                color="#5B5B57", fontsize=7, va="top")

        ax.set_yticks(y)
        ax.set_yticklabels([labels[m] for m in sub.index], fontsize=8)
        ax.set_xlim(0, 0.55)
        ax.set_title(f"{kind}", fontsize=9, loc="left")
        for m, yy, v in zip(sub.index, y, sub.variance_reduction):
            ax.text(v + 0.006, yy, f"{v:.0%}", va="center", fontsize=7, color="#2B2B29")
        ax.grid(axis="x", alpha=0.22, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("")
    fig.suptitle("Variance reduction by method ($\\rho^2=0.49$, 5k/arm, +1% effect)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_decisions(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h0 = df[df.hypothesis == "H0"]
    h1 = df[df.hypothesis == "H1"]
    order = ["baseline", "cuped", "poststrat_5", "poststrat_20", "cupac",
             "cupac_insample", "oracle"]
    labels = ["unadjusted", "CUPED", "post-strat (5)", "post-strat (20)",
              "CUPAC (cross-fit)", "CUPAC (in-sample)", "oracle"]
    keep = [m for m in order if m in set(h1.method)]

    fpr = [h0[h0.method == m].reject_rate.mean() for m in keep]
    power = [h1[h1.method == m].reject_rate.mean() for m in keep]
    se_cal = [h1[h1.method == m].se_calibration.mean() for m in keep]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    x = np.arange(len(keep))
    axes[0].bar(x - 0.2, fpr, width=0.38, color="#B0453C", label="false positive rate (H0)")
    axes[0].bar(x + 0.2, power, width=0.38, color="#1F5C99", label="power at +1% (H1)")
    axes[0].axhline(0.05, color="#B0453C", ls=":", lw=1.2)
    # A +1% effect on a CV=0.5 metric at 5k/arm is deliberately underpowered
    # (~0.35 at best), so the y-limit follows the data instead of the 0.80 line.
    top = max(max(fpr), max(power)) * 1.18
    axes[0].set_ylim(0, top)
    for xi, v in zip(x - 0.2, fpr):
        axes[0].text(xi, v + top * 0.015, f"{v:.1%}", ha="center", fontsize=6.5, color="#B0453C")
    for xi, v in zip(x + 0.2, power):
        axes[0].text(xi, v + top * 0.015, f"{v:.0%}", ha="center", fontsize=6.5, color="#1F5C99")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axes[0].set_title("False positives (labels) and power", fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    axes[0].grid(axis="y", alpha=0.2, lw=0.6)

    axes[1].barh(np.arange(len(keep))[::-1], se_cal,
                 color=["#C0504D" if v < 0.95 else "#1F5C99" for v in se_cal], height=0.6)
    axes[1].axvline(1.0, color="#2B2B29", lw=1.0, ls="--")
    axes[1].set_yticks(np.arange(len(keep))[::-1])
    axes[1].set_yticklabels(labels, fontsize=7.5)
    axes[1].set_xlim(0.7, 1.15)
    axes[1].set_xlabel("mean estimated SE / empirical SD of the estimate")
    axes[1].set_title("Where the in-sample version goes wrong", fontsize=9, loc="left")
    for yy, v in zip(np.arange(len(keep))[::-1], se_cal):
        axes[1].text(v + 0.008, yy, f"{v:.2f}", va="center", fontsize=7)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)
    axes[1].grid(axis="x", alpha=0.2, lw=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=400, help="replications for CUPAC scenarios")
    parser.add_argument("--ceiling-reps", type=int, default=1_000, help="replications for the ceiling sweep")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    args = parser.parse_args()

    if args.quick:
        args.reps, args.ceiling_reps, args.n_jobs = 40, 200, 4

    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "raw").mkdir(parents=True, exist_ok=True)

    print("Analysis 1: ceiling sweep (linear outcome)")
    t0 = time.time()
    ceiling = ceiling_sweep(args.ceiling_reps, args.n_jobs)
    ceiling.to_csv(OUT / "results" / "ceiling_sweep.csv", index=False)
    print_table(
        ceiling,
        "Does CUPED reach rho^2? (linear outcome)",
        ["rho", "method", "variance_reduction", "theory_cuped", "reps"],
        sort_by="rho",
    )
    print(f"  ({time.time() - t0:.0f}s)")

    print("\nAnalysis 2/3: method comparison, H0 and H1")
    t0 = time.time()
    methods = method_comparison(args.reps, args.n_jobs)
    methods.to_csv(OUT / "results" / "method_comparison.csv", index=False)

    h1 = methods[methods.hypothesis == "H1"]
    for kind in SIGNAL_KINDS:
        print_table(
            h1[h1.signal == kind],
            f"Variance reduction and stability - {kind} (rho^2 = 0.49)",
            ["method", "variance_reduction", "mean_se", "se_calibration",
             "bias_t", "coverage", "mde_rel"],
            sort_by="variance_reduction",
        )

    h0 = methods[methods.hypothesis == "H0"]
    print_table(
        h0,
        "H0: type-I error must stay at 5%",
        ["signal", "method", "reject_rate", "coverage", "se_calibration"],
        sort_by="signal",
    )
    print(f"  ({time.time() - t0:.0f}s)")

    plot_ceiling(ceiling, OUT / "figures" / "variance_reduction.png")
    plot_methods(methods, OUT / "figures" / "method_comparison.png")
    plot_decisions(methods, OUT / "figures" / "mde_and_calibration.png")
    print(f"\nWrote results and figures to {OUT}")


if __name__ == "__main__":
    main()
