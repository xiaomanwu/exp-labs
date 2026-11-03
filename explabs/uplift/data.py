"""Data for study 05: synthetic randomized experiments with a known effect,
plus loaders for the Criteo Uplift and Hillstrom email benchmarks.

The synthetic generator is the primary data source and doubles as the
ground-truth generator: the individual effect ``tau(x)`` is an explicit
function of the features, so oracle targeting, decile truth and allocation
arithmetic are exact rather than estimated.

Design constraints worth recording:

- Features are jointly Gaussian with unit marginals and a common correlation,
  so ``tau`` is built from *univariate* functions of the coordinates and its
  population mean (the ATE) is an exact one-dimensional quadrature per term.
- The response surface deliberately has interactions and curvature the
  response models must learn, while the effect surface is simpler than the
  response surface. Real uplift problems look like this: who converts is
  harder to predict than how much the treatment moves them.
- Assignment is randomized with a realistic treated share (0.85, Criteo-like),
  which is what makes Qini evaluation on a held-out slice unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.special import expit as sigmoid


# --------------------------------------------------------------------------
# Synthetic experiment
# --------------------------------------------------------------------------


@dataclass
class UpliftDGP:
    """A randomized experiment with a known heterogeneous effect.

    ``tau(x) = tau0 + tau1 * sigmoid(beta * x_a) - tau2 * 1{x_b > cut}``

    ``tau1``-term is a smooth "persuadable" segment along feature ``a``; the
    ``tau2``-term is a segment where the treatment *backfires* (fatigue). The
    response surface is a different, nonlinear function of other features.
    """

    n_features: int = 10
    rho: float = 0.3
    treat_share: float = 0.85
    a: int = 0          # persuadable-segment feature
    b: int = 5          # backfire-segment feature
    beta: float = 2.5
    cut: float = 0.8
    tau0: float = 0.015
    tau1: float = 0.060
    tau2: float = 0.025

    def __post_init__(self) -> None:
        # (1-rho) I + rho J: every marginal is N(0,1) and every pair of
        # distinct features correlates at exactly rho.
        self._chol = np.linalg.cholesky(
            (1.0 - self.rho) * np.eye(self.n_features)
            + self.rho * np.ones((self.n_features, self.n_features))
        )

    # -- truth --------------------------------------------------------------

    def tau(self, x: np.ndarray) -> np.ndarray:
        """Individual treatment effect on the probability scale (exact)."""
        return (self.tau0
                + self.tau1 * sigmoid(self.beta * x[:, self.a])
                - self.tau2 * (x[:, self.b] > self.cut).astype(float))

    def _p0(self, x: np.ndarray) -> np.ndarray:
        a0, a1, a2 = self.a, (self.a + 3) % self.n_features, (self.a + 2) % self.n_features
        lin = (-2.6 + 0.9 * x[:, a0] - 0.7 * x[:, a1] + 0.6 * x[:, a2]
               + 0.35 * x[:, a1] ** 2 - 0.5 * x[:, (a1 + 2) % self.n_features]
               + 0.4 * x[:, a0] * x[:, a2])
        return sigmoid(lin)

    def probs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Response probabilities under control and treatment (exact)."""
        p0 = self._p0(x)
        p1 = np.clip(p0 + self.tau(x), 1e-6, 1.0 - 1e-6)
        return p0, p1

    @property
    def ate(self) -> float:
        """Population ATE, exact by one-dimensional quadrature per term."""
        # E over a standard normal marginal: E[f(X)] = pi^-1/2 * sum w f(sqrt(2) t)
        t, w = np.polynomial.hermite.hermgauss(64)
        mean_sigmoid = float(np.sum(w * sigmoid(self.beta * np.sqrt(2.0) * t)) / np.sqrt(np.pi))
        p_backfire = float(ndtr(-self.cut))
        return self.tau0 + self.tau1 * mean_sigmoid - self.tau2 * p_backfire

    def backfired_share(self) -> float:
        return float(ndtr(-self.cut))

    def label(self) -> str:
        return (f"tau0={self.tau0:.3f}, persuadable x{self.a}, "
                f"backfire x{self.b}>{self.cut}")

    # -- sampling -----------------------------------------------------------

    def draw(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """One randomized experiment: features, assignment, outcomes."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.n_features)) @ self._chol.T
        t = (rng.random(n) < self.treat_share).astype(float)
        p0, p1 = self.probs(x)
        y = (rng.random(n) < np.where(t == 1.0, p1, p0)).astype(float)
        return {"x": x, "t": t, "y": y, "e": self.treat_share}


def heterogeneous_dgp() -> UpliftDGP:
    """Default world: persuadables, a flat majority, and a backfire segment."""
    return UpliftDGP()


def constant_dgp(tau: float = 0.02) -> UpliftDGP:
    """Constant-effect world: every individual effect equals ``tau``."""
    return UpliftDGP(tau0=tau, tau1=0.0, tau2=0.0)


@dataclass
class SegmentDGP(UpliftDGP):
    """Effect is ``seg_tau`` inside one segment (x_a > cut) and zero outside."""

    seg_tau: float = 0.10

    def tau(self, x: np.ndarray) -> np.ndarray:
        return self.seg_tau * (x[:, self.a] > self.cut).astype(float)

    @property
    def ate(self) -> float:
        return self.seg_tau * float(ndtr(-self.cut))

    def label(self) -> str:
        return f"single segment: tau={self.seg_tau:.2f} for x{self.a}>{self.cut}"


def single_segment_dgp(tau: float = 0.10) -> SegmentDGP:
    return SegmentDGP(seg_tau=tau, tau0=0.0, tau1=0.0, tau2=0.0, cut=0.8)


# --------------------------------------------------------------------------
# Benchmark loaders
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    """A ready-to-model randomized dataset.

    ``e`` is the assignment propensity. For a randomized experiment it is a
    constant; treating it as known is exactly what study 04 argued for.
    """

    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    e: float
    source: str
    truth: object | None = None      # UpliftDGP when synthetic
    feature_names: list[str] = field(default_factory=list)


_CRITEO_TREAT = "treatment"
_CRITEO_OUTCOMES = ("visit", "conversion")
_HILLSTROM_SEGMENTS = {"mens email": 1, "womens email": 1, "no email": 0}


def _load_criteo(df, path: Path) -> Dataset | None:
    if _CRITEO_TREAT not in df.columns:
        return None
    outcome = next((c for c in _CRITEO_OUTCOMES if c in df.columns), None)
    if outcome is None:
        return None
    features = [c for c in df.columns
                if c.startswith("f") and c[1:].isdigit()]
    if not features:
        return None
    x = df[features].to_numpy(dtype=float)
    t = df[_CRITEO_TREAT].to_numpy(dtype=float)
    y = df[outcome].to_numpy(dtype=float)
    return Dataset(x=x, t=t, y=y, e=float(t.mean()),
                   source=f"Criteo Uplift ({path.name}), outcome={outcome}",
                   feature_names=features)


def _load_hillstrom(df, path: Path) -> Dataset | None:
    seg = next((c for c in df.columns if c.lower() == "segment"), None)
    if seg is None:
        return None
    outcome = next((c for c in ("visit", "conversion", "spend")
                    if c in df.columns), None)
    if outcome is None:
        return None
    values = df[seg].astype(str).str.strip().str.lower()
    if not set(values.unique()) <= set(_HILLSTROM_SEGMENTS):
        return None
    features = [c for c in df.columns
                if c not in (seg, outcome) and df[c].dtype.kind in "if"]
    x = df[features].to_numpy(dtype=float)
    t = values.map(_HILLSTROM_SEGMENTS).to_numpy(dtype=float)
    y = (df[outcome].to_numpy(dtype=float) > 0).astype(float) \
        if outcome == "spend" else df[outcome].to_numpy(dtype=float)
    return Dataset(x=x, t=t, y=y, e=float(t.mean()),
                   source=f"Hillstrom email ({path.name}), outcome={outcome}",
                   feature_names=features)


def load_data(directory: str | Path = "data") -> Dataset:
    """Load a benchmark from ``data/`` if present, else the synthetic world.

    Recognised: a CSV whose name mentions ``criteo`` (columns f0..f11,
    treatment, visit/conversion) or ``hillstrom`` (segment, visit/conversion/
    spend). The fallback is the synthetic heterogeneous world, so the whole
    pipeline runs out of the box and the ground truth stays exact.
    """
    import pandas as pd

    root = Path(directory)
    if root.is_dir():
        for path in sorted(root.glob("*.csv")):
            name = path.name.lower()
            if "criteo" in name or "hillstrom" in name:
                df = pd.read_csv(path)
                data = (_load_criteo(df, path) if "criteo" in name
                        else _load_hillstrom(df, path))
                if data is not None:
                    return data
    dgp = heterogeneous_dgp()
    drawn = dgp.draw(100_000, seed=2025)
    return Dataset(x=drawn["x"], t=drawn["t"], y=drawn["y"], e=drawn["e"],
                   source="synthetic heterogeneous world (100,000 users)",
                   truth=dgp)
