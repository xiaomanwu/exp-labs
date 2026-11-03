"""Study 05 evaluation: which uplift model turns prediction into targeting.

One randomized dataset, two slices:

``train``  fit four models that rank users by estimated effect
``eval``   score every ranker with the *same* Qini machinery on held-out
           randomized data, and (synthetic world) against the exact truth

Outputs: Qini curves with the oracle and random rankers overlaid, auQC with
paired-bootstrap standard errors, uplift by decile with ground truth, and a
budget-allocation table (exact incremental conversions captured per share of
the population targeted).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset, load_data
from .models import HonestCausalForest, SLearner, TLearner, XLearner
from .qini import (
    auqc,
    capture_curve,
    decile_table,
    paired_bootstrap_auqc,
    qini_curve,
)

OUT = Path(__file__).resolve().parents[2] / "studies" / "05-uplift"

COLORS = {
    "oracle": "#5B5B57",
    "s": "#8A7CA8",
    "t": "#C46A4A",
    "x": "#3D7AB5",
    "forest": "#2E8B57",
    "random": "#AAAAA5",
}
LABELS = {
    "oracle": "oracle (true effect)",
    "s": "S-learner",
    "t": "T-learner",
    "x": "X-learner",
    "forest": "causal forest (honest)",
    "random": "random targeting",
}
ORDER = ("s", "t", "x", "forest")


def model_set(seed: int = 0) -> list:
    return [SLearner(seed=seed), TLearner(seed=seed), XLearner(seed=seed),
            HonestCausalForest(seed=seed)]


def _clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------


def run_study(data: Dataset, train_frac: float = 0.6,
              boot: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = data.x.shape[0]
    perm = rng.permutation(n)
    cut = int(train_frac * n)
    tr, ev = perm[:cut], perm[cut:]
    x_tr, t_tr, y_tr = data.x[tr], data.t[tr], data.y[tr]
    x_ev, t_ev, y_ev = data.x[ev], data.t[ev], data.y[ev]

    truth = None if data.truth is None else data.truth.tau(x_ev)

    models = model_set(seed=seed)
    scores: dict[str, np.ndarray] = {}
    for model in models:
        model.fit(x_tr, t_tr, y_tr, e=data.e)
        scores[model.key] = model.tau(x_ev)
    if truth is not None:
        scores["oracle"] = truth
    # a random ranker is the null every Qini curve must beat
    scores["random"] = rng.random(x_ev.shape[0])

    fracs, curves = {}, {}
    for key, sc in scores.items():
        fracs[key], curves[key] = qini_curve(t_ev, y_ev, sc)
    ses, draws = paired_bootstrap_auqc(scores, t_ev, y_ev, boot=boot,
                                       seed=seed + 1)

    # Paired differences: the comparison anyone actually cares about, with the
    # correlation between rankings respected.
    diffs = {}
    for a, b in (("forest", "random"), ("forest", "s"), ("forest", "t"),
                 ("forest", "x"), ("oracle", "forest")):
        if a in draws and b in draws:
            d = draws[a] - draws[b]
            diffs[f"{a} - {b}"] = (float(d.mean()), float(d.std(ddof=1)))

    auqcs = {k: auqc(t_ev, y_ev, sc) for k, sc in scores.items()}

    # Effect recovery against the exact truth (synthetic world only): RMSE
    # and correlation measure the estimate, Qini and capture measure what
    # targeting does with it. They need not rank models identically.
    recovery = None
    if truth is not None:
        recovery = {
            k: {"rmse": float(np.sqrt(np.mean((sc - truth) ** 2))),
                "corr": float(np.corrcoef(sc, truth)[0, 1]),
                "mean": float(sc.mean())}
            for k, sc in scores.items() if k in ORDER
        }

    fractions = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    capture = ({k: capture_curve(truth, sc, fractions) for k, sc in scores.items()}
               if truth is not None else None)

    deciles = {k: decile_table(t_ev, y_ev, sc, truth=truth)
               for k, sc in scores.items() if k in ORDER + ("oracle",)}

    return {
        "data": data, "scores": scores, "truth": truth,
        "fractions": fracs, "curves": curves, "auqc": auqcs,
        "se": ses, "diffs": diffs, "capture": capture, "deciles": deciles,
        "recovery": recovery, "n_train": cut, "n_eval": n - cut,
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_qini(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fr = result["fractions"]
    for key in ("random", "s", "t", "x", "forest", "oracle"):
        if key not in result["curves"]:
            continue
        ax.plot(fr[key] * 100, result["curves"][key], color=COLORS[key],
                lw=2.2 if key == "forest" else 1.4,
                ls="--" if key in ("random", "oracle") else "-",
                label=LABELS[key], alpha=0.9)
    ax.set_xlabel("share of population targeted (%)")
    ax.set_ylabel("cumulative incremental responses (Qini)")
    ax.set_title("Ranking quality, measured the same way for every model",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_deciles(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    dec = result["deciles"]["forest"]
    xs = np.arange(1, 11)
    width = 0.38
    meas = [row["measured"] for row in result["deciles"]["forest"]]
    truth = [row.get("truth", np.nan) for row in result["deciles"]["forest"]]
    x_score = [row["mean_score"] for row in result["deciles"]["forest"]]
    ax.bar(xs - width / 2, 100 * np.asarray(truth), width,
           color="#9BB8D3", label="true mean effect (exact)")
    ax.bar(xs + width / 2, 100 * np.asarray(x_score), width,
           color="#2E8B57", label="forest estimate")
    ax2 = ax.twinx()
    ax2.plot(xs, 100 * np.asarray(meas), "o--", color="#5B5B57", ms=5,
             lw=1.2, label="measured uplift on eval slice")
    ax2.set_ylabel("measured uplift (pts)", fontsize=9)
    ax2.spines[["top"]].set_visible(False)
    ax.set_xlabel("decile of model score (1 = most targeted)")
    ax.set_ylabel("effect (pts)")
    ax.set_title("Uplift by decile: what the forest claims vs what is true",
                 fontsize=10.5, loc="left")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8,
              loc="upper right")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_capture(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = np.array([100 * f for f in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)])
    for key in ("random", "s", "t", "x", "forest", "oracle"):
        if key not in result["capture"]:
            continue
        ys = np.array([result["capture"][key][f] for f in
                       (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)])
        ax.plot(xs, ys, color=COLORS[key],
                lw=2.2 if key == "forest" else 1.4,
                ls="--" if key in ("random", "oracle") else "-",
                label=LABELS[key], marker="o", ms=3.5)
    ax.set_xlabel("share of population targeted (%)")
    ax.set_ylabel("incremental conversions captured (exact)")
    ax.set_title("A fixed budget buys different outcomes per model",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    import time

    clock = time.perf_counter()

    def stamp(message: str) -> None:
        print(f"  [{time.perf_counter() - clock:6.1f}s] {message}", flush=True)

    figs = OUT / "figures"
    results = OUT / "results"
    figs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    data = load_data()
    print(f"dataset: {data.source}, n={data.x.shape[0]}, "
          f"treated={data.t.mean():.1%}")
    stamp("data")

    result = run_study(data, boot=args.boot, seed=args.seed)
    stamp("models + scoring")

    plot_qini(result, figs / "qini.png")
    stamp("qini figure")
    if result["truth"] is not None:
        plot_deciles(result, figs / "deciles.png")
        plot_capture(result, figs / "budget.png")
        stamp("decile + budget figures")

    # CSVs -----------------------------------------------------------------
    curve_rows = []
    for key, curve in result["curves"].items():
        thin = np.linspace(0, len(curve) - 1, 200).astype(int)
        for i in thin:
            curve_rows.append({"model": key, "fraction": result["fractions"][key][i],
                               "qini": curve[i]})
    pd.DataFrame(curve_rows).to_csv(results / "qini_curves.csv", index=False)

    auqc_rows = [{"model": k, "auqc": v, "se": result["se"].get(k, np.nan)}
                 for k, v in result["auqc"].items()]
    pd.DataFrame(auqc_rows).to_csv(results / "auqc.csv", index=False)

    if result["capture"] is not None:
        cap_rows = []
        fractions = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
        for key, cap in result["capture"].items():
            cap_rows.append({"model": key, **{f"f{f}": cap[f] for f in fractions}})
        pd.DataFrame(cap_rows).to_csv(results / "budget.csv", index=False)

    if result["recovery"] is not None:
        pd.DataFrame([{"model": k, **v} for k, v in result["recovery"].items()]
                     ).to_csv(results / "recovery.csv", index=False)

    dec_rows = []
    for key, table in result["deciles"].items():
        for row in table:
            dec_rows.append({"model": key, **row})
    pd.DataFrame(dec_rows).to_csv(results / "deciles.csv", index=False)

    print("\nArea under the Qini curve (incremental responses; paired bootstrap SE)")
    print(pd.DataFrame(auqc_rows).round(1).to_string(index=False))
    print("\nPaired auQC differences (bootstrap mean +- SE)")
    print(pd.DataFrame([{"pair": k, "diff": round(m, 1), "se": round(s, 1)}
                        for k, (m, s) in result["diffs"].items()]
                       ).to_string(index=False))
    print("\nExact incremental conversions captured by the top-f targeting rule")
    if result["capture"] is not None:
        print(pd.DataFrame(
            [{"model": k, **{f"{f:.0%}": round(v, 1) for f, v in cap.items()}}
             for k, cap in result["capture"].items()]).to_string(index=False))
    if result["recovery"] is not None:
        print("\nEffect recovery against the exact truth")
        print(pd.DataFrame(
            [{"model": k, **{kk: round(vv, 4) for kk, vv in v.items()}}
             for k, v in result["recovery"].items()]).to_string(index=False))


if __name__ == "__main__":
    main()
