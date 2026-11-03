"""Shared correctness-check plumbing.

A check suite that stops at the first failure hides how much of the claim was
actually verified. This collector records every result and distinguishes three
outcomes -- passed, failed, skipped -- so a missing dependency can never be
reported as a pass.
"""

from __future__ import annotations

import numpy as np


class Checker:
    """Collects pass/fail/skip results instead of raising on the first problem."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def check(self, name: str, detail: str, ok: bool) -> None:
        status = "PASS" if ok else "FAIL"
        self.results.append((name, detail, status))
        print(f"[{status}] {name}: {detail}")

    def skip(self, name: str, detail: str) -> None:
        self.results.append((name, detail, "SKIP"))
        print(f"[SKIP] {name}: {detail}")

    def summary(self) -> int:
        failed = [name for name, _, status in self.results if status == "FAIL"]
        skipped = [name for name, _, status in self.results if status == "SKIP"]
        ran = len(self.results) - len(skipped)
        print()
        if failed:
            print(f"{len(failed)} of {len(self.results)} checks failed: {', '.join(failed)}")
            return 1
        message = f"All {ran} runnable checks passed."
        if skipped:
            message += f" Skipped (missing dependency): {', '.join(skipped)}."
        print(message)
        return 0


def tolerance(reps: int, p: float = 0.05, sigmas: float = 4.0) -> float:
    """Monte Carlo tolerance: ``sigmas`` binomial standard errors."""
    return sigmas * np.sqrt(p * (1.0 - p) / reps)


def bootstrap_se_of_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    boot: int = 2_000,
    seed: int = 0,
) -> float:
    """Standard error of ``var(numerator) / var(denominator)``.

    A variance-ratio check needs a tolerance that reflects the actual
    replications, not a rule of thumb: the two variances are estimated from the
    same runs, so the ratio is more stable than independence would suggest.
    Resampling replications captures that correlation for free.
    """
    rng = np.random.default_rng(seed)
    n = numerator.size
    ratios = np.empty(boot)
    for b in range(boot):
        idx = rng.integers(0, n, n)
        denom = denominator[idx].var(ddof=1)
        ratios[b] = numerator[idx].var(ddof=1) / denom
    return float(ratios.std(ddof=1))


def bootstrap_se_of_reduction_difference(
    a: np.ndarray,
    b: np.ndarray,
    baseline: np.ndarray,
    boot: int = 2_000,
    seed: int = 0,
) -> float:
    """Paired bootstrap SE of the difference in variance reduction between ``a`` and ``b``.

    Two reductions measured on the same replications are strongly correlated, so
    comparing them against independent standard errors would hide a real
    difference. Resampling whole replications keeps the pairing.
    """
    rng = np.random.default_rng(seed)
    n = a.size
    diffs = np.empty(boot)
    for i in range(boot):
        idx = rng.integers(0, n, n)
        base = baseline[idx].var(ddof=1)
        diffs[i] = (b[idx].var(ddof=1) - a[idx].var(ddof=1)) / base
    return float(diffs.std(ddof=1))
