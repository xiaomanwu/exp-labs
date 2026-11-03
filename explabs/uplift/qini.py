"""Qini and targeting evaluation for study 05.

One code path evaluates every model -- including the oracle and a random
ranker. This is deliberate and is one of the study's validation anchors:
Qini implementations differ (transformed-outcome IPW gains, count-ratio
curves, normalisation choices), and a difference there leaks straight into
the model comparison. Here every curve is the Radcliffe count-ratio Qini:

    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)

cumulative over users ranked by score. On a randomized evaluation slice the
curve is unbiased for every ranking, which is what lets models be compared.

Also here: paired-bootstrap standard errors (models share one evaluation
slice, so their auQC estimates are correlated and must be resampled
together), decile tables with ground truth, and budget-allocation capture
curves.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Qini
# --------------------------------------------------------------------------


def qini_curve(t: np.ndarray, y: np.ndarray, score: np.ndarray):
    """Radcliffe Qini curve for users ranked by ``score`` (descending).

    Returns ``(fractions, curve)`` with ``curve[k]`` the incremental
    responses after targeting the top ``(k+1)/n`` of users. Early steps with
    no control users yet define ``Q = Y_t`` (the scaling term is undefined
    and the cumulative treated count is the honest value).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    to, yo = t[order], y[order]
    n = to.size
    nt = np.cumsum(to)
    nc = np.cumsum(1.0 - to)
    yt = np.cumsum(yo * to)
    yc = np.cumsum(yo * (1.0 - to))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nc > 0, nt / np.where(nc > 0, nc, 1.0), 0.0)
    curve = yt - yc * ratio
    fractions = np.arange(1, n + 1) / n
    return fractions, curve


def auqc(t: np.ndarray, y: np.ndarray, score: np.ndarray) -> float:
    """Area between the Qini curve and the random-ranking diagonal."""
    fractions, curve = qini_curve(t, y, score)
    baseline = curve[-1] * fractions
    return float(np.trapz(curve - baseline, fractions))


def paired_bootstrap_auqc(scores: dict[str, np.ndarray], t: np.ndarray,
                          y: np.ndarray, boot: int = 500,
                          seed: int = 0) -> tuple[dict[str, float],
                                                  dict[str, np.ndarray]]:
    """Standard error of each auQC, resampling evaluation users.

    All models are scored on the *same* users, so the estimates are
    correlated; resampling users once and re-scoring every model keeps that
    pairing, which is the difference between a comparison and two guesses.
    Returns ``(standard errors, bootstrap draws per model)`` -- the draws let
    the caller form paired differences without re-running the bootstrap.
    """
    rng = np.random.default_rng(seed)
    n = t.size
    names = list(scores)
    draws = np.empty((boot, len(names)))
    for b in range(boot):
        idx = rng.integers(0, n, n)
        for j, name in enumerate(names):
            draws[b, j] = auqc(t[idx], y[idx], scores[name][idx])
    se = {name: float(draws[:, j].std(ddof=1))
          for j, name in enumerate(names)}
    return se, {name: draws[:, j] for j, name in enumerate(names)}


# --------------------------------------------------------------------------
# Targeting arithmetic
# --------------------------------------------------------------------------


def capture_curve(tau_true: np.ndarray, score: np.ndarray,
                  fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5,
                                                  0.6, 0.7, 0.8, 0.9, 1.0)):
    """Incremental conversions captured by targeting the top-``f`` by score.

    Uses the exact individual effects, so this is the *consequence* of a
    ranking, not an estimate of it -- available only where the truth is known.
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    tau_sorted = np.asarray(tau_true)[order]
    captured = np.cumsum(tau_sorted)
    n = tau_sorted.size
    return {f: float(captured[max(1, int(round(f * n))) - 1]) for f in fractions}


def decile_table(t: np.ndarray, y: np.ndarray, score: np.ndarray,
                 truth: np.ndarray | None = None,
                 n_deciles: int = 10) -> list[dict]:
    """Per-decile measured uplift, optional true mean effect and mean score.

    Deciles are by model score (decile 1 = highest score). ``measured`` is
    the within-decile difference of treatment and control response rates --
    unbiased because assignment is randomized.
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    t, y = np.asarray(t, float)[order], np.asarray(y, float)[order]
    truth = None if truth is None else np.asarray(truth, float)[order]
    score = np.asarray(score, float)[order]
    rows = []
    edges = np.linspace(0, t.size, n_deciles + 1).astype(int)
    for d in range(n_deciles):
        sl = slice(edges[d], edges[d + 1])
        t_d, y_d = t[sl], y[sl]
        one, zero = t_d == 1.0, t_d == 0.0
        row = {
            "decile": d + 1,
            "measured": float(y_d[one].mean() - y_d[zero].mean()),
            "mean_score": float(score[sl].mean()),
        }
        if truth is not None:
            row["truth"] = float(truth[sl].mean())
        rows.append(row)
    return rows
