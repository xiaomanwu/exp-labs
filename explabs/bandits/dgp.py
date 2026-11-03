"""Ground truth for the adaptive-allocation study.

A two-arm conversion problem with covariates. User ``i`` arrives with
``X_i ~ N(0, I_2)``; if shown arm ``a`` they convert with probability
``p_a(X)``. Two worlds are available, because the study asks two different
questions:

``shift`` mode
    ``p_1(X) = p_0(X) + shift``. The lift is the same number for every user, so
    the best fixed policy loses exactly ``shift / 2`` per user and regret has a
    closed-form baseline. This is the world used to compare allocation rules.

``target_ate`` / ``treat_intercept`` mode
    Both arms are logistic in ``X`` with *different* slopes, so the arms rank
    users differently. The average effect can be set to zero while the segment
    effects are far from zero. This is the world used to study inference,
    because it is the one where a targeting policy's allocation correlates with
    the outcome.

Everything the study calls "truth" is computed by quadrature over ``X``, not by
simulation: ``beta . X`` is normal, so a marginal rate is a one-dimensional
integral of a smooth function and a segment rate is the same integral over a
quadrant. Sixty-four Gauss-Hermite nodes per dimension agree with a
20-million-draw Monte Carlo to the fourth decimal, and they never move between
runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

# A quadrant grid gives four segments of equal probability. Two segments split
# on the first feature; one segment means the policy ignores context entirely.
SEGMENT_COUNTS = (1, 2, 4)


@lru_cache(maxsize=8)
def _grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nodes, weights and quadrant labels for ``E[g(X)]`` over ``N(0, I_2)``.

    Gauss-Hermite nodes are mirrored into the negative half-line so that
    quadrant expectations are exact rather than approximate: the standard rule
    integrates over both signs at once, and conditioning on a quadrant requires
    the weights to be split by sign. Doubling the nodes and halving the weights
    reproduces the same total mass while making that split explicit.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    nodes = np.concatenate([nodes, -nodes])
    weights = np.concatenate([weights, weights]) / np.sqrt(2.0 * np.pi) / 2.0
    order = np.argsort(nodes)
    nodes, weights = nodes[order], weights[order]

    g1, g2 = np.meshgrid(nodes, nodes, indexing="ij")
    w = np.outer(weights, weights)
    segment = (g1 > 0).astype(int) * 2 + (g2 > 0).astype(int)
    return g1.ravel(), g2.ravel(), w.ravel(), segment.ravel()


def _mean_rate(intercept: float, slope: tuple[float, float], n: int) -> float:
    """``E[sigmoid(intercept + slope . X)]`` by quadrature."""
    g1, g2, w, _ = _grid(n)
    return float(np.sum(w * expit(intercept + slope[0] * g1 + slope[1] * g2)))


@dataclass(frozen=True)
class ContextualDGP:
    """Two-arm conversion problem with a known average treatment effect.

    Parameters
    ----------
    control_intercept, control_slope:
        The control arm's logit. Defaults give a marginal control rate near
        0.12 with a wide spread across users (roughly 0.05 to 0.21).
    treat_slope:
        The treatment arm's slope in logit mode. Making it different from
        ``control_slope`` is what creates effect heterogeneity across segments
        -- and therefore what makes a targeting policy's allocation correlate
        with the outcome.
    treat_intercept:
        Explicit treatment intercept (logit mode). Mutually exclusive with
        ``target_ate`` and ``shift``.
    target_ate:
        Solve the treatment intercept so that ``p_1 - p_0`` equals this value
        *on average* (logit mode). ``0.0`` builds the world where the average
        effect is nil but the segment-level effects are not.
    shift:
        Additive-lift mode: ``p_1(X) = p_0(X) + shift``. The per-user effect is
        constant, so the average effect is ``shift`` by construction.
    grid:
        Gauss-Hermite nodes per dimension. 64 is plenty; the validation suite
        checks it against Monte Carlo.
    """

    control_intercept: float = -2.20
    control_slope: tuple[float, float] = (0.60, -0.40)
    treat_slope: tuple[float, float] = (0.60, 0.55)
    treat_intercept: float | None = None
    target_ate: float | None = None
    shift: float | None = None
    grid: int = 64

    def __post_init__(self) -> None:
        given = sum(v is not None for v in (self.treat_intercept, self.target_ate, self.shift))
        if given != 1:
            raise ValueError("pass exactly one of shift, target_ate or treat_intercept")

        if self.shift is not None:
            control = _mean_rate(self.control_intercept, self.control_slope, self.grid)
            if not 0.0 < control + self.shift < 1.0:
                raise ValueError("additive shift pushes the treatment rate outside (0, 1)")
            object.__setattr__(self, "treat_intercept", float("nan"))
            return

        control = _mean_rate(self.control_intercept, self.control_slope, self.grid)
        if self.target_ate is not None:
            target = control + self.target_ate
            lo, hi = -12.0, 6.0
            if not (_mean_rate(lo, self.treat_slope, self.grid) <= target
                    <= _mean_rate(hi, self.treat_slope, self.grid)):
                raise ValueError(f"target rate {target:.4f} is outside the reach of the slope")
            solved = brentq(
                lambda al: _mean_rate(al, self.treat_slope, self.grid) - target,
                lo, hi, xtol=1e-12,
            )
            object.__setattr__(self, "treat_intercept", float(solved))

    # -- truth ------------------------------------------------------------

    @property
    def additive(self) -> bool:
        return self.shift is not None

    @property
    def intercepts(self) -> tuple[float, float]:
        return (self.control_intercept, float(self.treat_intercept))

    @property
    def slopes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.control_slope, self.treat_slope)

    def _probs(self, g1: np.ndarray, g2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conversion probabilities on both arms at the given covariate values."""
        p0 = expit(self.control_intercept + self.control_slope[0] * g1
                   + self.control_slope[1] * g2)
        if self.additive:
            return p0, np.clip(p0 + float(self.shift), 0.0, 1.0)
        p1 = expit(self.intercepts[1] + self.treat_slope[0] * g1
                   + self.treat_slope[1] * g2)
        return p0, p1

    def arm_rate(self, arm: int) -> float:
        """Marginal conversion rate on one arm."""
        g1, g2, w, _ = _grid(self.grid)
        p0, p1 = self._probs(g1, g2)
        return float(np.sum(w * (p1 if arm else p0)))

    @property
    def rates(self) -> tuple[float, float]:
        return (self.arm_rate(0), self.arm_rate(1))

    @property
    def ate(self) -> float:
        """Average treatment effect over the arrival distribution."""
        return self.rates[1] - self.rates[0]

    def _segment_stats(self, n_segments: int, segment: int) -> tuple[float, float, float]:
        """Return ``(mass, rate_control, rate_treat)`` for one policy bucket."""
        g1, g2, w, seg4 = _grid(self.grid)
        if n_segments == 4:
            mask = seg4 == segment
        elif n_segments == 2:
            mask = (g1 > 0).astype(int) == segment
        else:
            mask = np.ones(g1.shape, dtype=bool)
        mass = float(np.sum(w[mask]))
        p0, p1 = self._probs(g1[mask], g2[mask])
        return mass, float(np.sum(w[mask] * p0) / mass), float(np.sum(w[mask] * p1) / mass)

    def segment_shares(self, n_segments: int = 4) -> np.ndarray:
        """Probability mass of each policy bucket, summing to one."""
        self._validate_segments(n_segments)
        return np.array([self._segment_stats(n_segments, s)[0] for s in range(n_segments)])

    def segment_rates(self, n_segments: int = 4) -> np.ndarray:
        """Per-segment ``(control, treatment)`` rates, shape ``(segments, 2)``."""
        self._validate_segments(n_segments)
        return np.array([[self._segment_stats(n_segments, s)[1],
                          self._segment_stats(n_segments, s)[2]]
                         for s in range(n_segments)])

    def segment_ates(self, n_segments: int = 4) -> np.ndarray:
        r = self.segment_rates(n_segments)
        return r[:, 1] - r[:, 0]

    @staticmethod
    def _validate_segments(n_segments: int) -> None:
        if n_segments not in SEGMENT_COUNTS:
            raise ValueError(f"n_segments must be one of {SEGMENT_COUNTS}")

    # -- sampling ---------------------------------------------------------

    def probs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conversion probabilities on both arms for each row of ``x``."""
        return self._probs(x[:, 0], x[:, 1])

    @staticmethod
    def segments(x: np.ndarray, n_segments: int) -> np.ndarray:
        """Bucket labels used by a policy that looks at the user's features."""
        if n_segments == 1:
            return np.zeros(x.shape[0], dtype=np.int64)
        if n_segments == 2:
            return (x[:, 0] > 0).astype(np.int64)
        if n_segments == 4:
            return (x[:, 0] > 0).astype(np.int64) * 2 + (x[:, 1] > 0).astype(np.int64)
        raise ValueError(f"n_segments must be one of {SEGMENT_COUNTS}")

    def label(self) -> str:
        return (f"control {self.rates[0]:.4f} treat {self.rates[1]:.4f} "
                f"ATE {self.ate:+.4f}")


def additive_dgp(lift: float, grid: int = 64) -> ContextualDGP:
    """Constant per-user lift. Regret has a closed-form baseline here."""
    return ContextualDGP(shift=lift, grid=grid)


def heterogeneous_dgp(target_ate: float = 0.0, grid: int = 64) -> ContextualDGP:
    """The study's inference world.

    The two arms rank the segments in opposite orders: arm 1 does best where
    ``x2`` is large and arm 0 does best where it is small. With
    ``target_ate = 0`` the average effect is exactly nil while the two halves of
    the population disagree in sign -- the setting where a targeting policy can
    manufacture an average effect that is not there.
    """
    return ContextualDGP(
        control_intercept=-2.20,
        control_slope=(0.60, -0.40),
        treat_slope=(0.60, 0.55),
        target_ate=target_ate,
        grid=grid,
    )
