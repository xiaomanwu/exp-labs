"""Traffic simulator for two-arm online experiments.

One call to :func:`simulate` returns ``reps`` complete experiment realizations,
each stored as cumulative per-day sufficient statistics. Every stopping policy
is then evaluated on *the same* traffic, which keeps comparisons low-variance
(common random numbers).

Per arm and per day ``d`` we keep

    ``n``   cumulative users
    ``s1``  cumulative sum of the metric
    ``s2``  cumulative sum of squared metric values

Both a conversion-rate metric (Bernoulli) and a revenue metric (log-normal)
reduce to these three numbers, so downstream code never needs to know which
metric it is looking at. Adding a third metric family means teaching this
module one new ``_draw_day`` branch and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPPORTED_METRICS = ("bernoulli", "lognormal")


@dataclass(frozen=True)
class ExperimentConfig:
    """Shape and ground truth of one simulated experiment.

    Parameters
    ----------
    baseline_rate:
        Control conversion rate. Ignored when ``metric="lognormal"``.
    relative_lift:
        True relative effect on the treatment arm. ``0.0`` means the null
        hypothesis is true, which is what the false-positive checks use.
    daily_traffic_per_arm:
        New users per arm per day. Constant in this version; real platforms
        ramp up, and that is a deliberate simplification to note in the README.
    horizon_days:
        Planned experiment length. Sequential policies may stop earlier.
    metric:
        ``"bernoulli"`` for conversion-style metrics, ``"lognormal"`` for
        heavy-tailed revenue per user.
    revenue_mean, revenue_sigma:
        Mean and log-scale standard deviation of the log-normal revenue
        distribution (only used when ``metric="lognormal"``).
    """

    baseline_rate: float = 0.10
    relative_lift: float = 0.0
    daily_traffic_per_arm: int = 1_000
    horizon_days: int = 30
    metric: str = "bernoulli"
    revenue_mean: float = 30.0
    revenue_sigma: float = 0.8

    def __post_init__(self) -> None:
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of {SUPPORTED_METRICS}, got {self.metric!r}")
        if not 0.0 < self.baseline_rate < 1.0:
            raise ValueError("baseline_rate must be strictly between 0 and 1")
        if self.relative_lift <= -1.0:
            raise ValueError("relative_lift must be greater than -1")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        if self.daily_traffic_per_arm < 2:
            raise ValueError("daily_traffic_per_arm must be at least 2")

    @property
    def treatment_rate(self) -> float:
        """Conversion rate on the treatment arm under the configured lift."""
        return self.baseline_rate * (1.0 + self.relative_lift)

    def label(self) -> str:
        """Short human-readable description, used in report tables."""
        return (
            f"{self.metric} lift={self.relative_lift:+.1%} "
            f"traffic={self.daily_traffic_per_arm}/day horizon={self.horizon_days}d"
        )


@dataclass
class Simulation:
    """A batch of experiment realizations, stored as cumulative statistics.

    All arrays have shape ``(reps, horizon_days)``. Column ``d`` holds the
    cumulative total from day 0 through day ``d``, so a policy that peeks on
    day ``d`` reads column ``d`` and never has to touch earlier columns.
    """

    config: ExperimentConfig
    c_n: np.ndarray
    c_s1: np.ndarray
    c_s2: np.ndarray
    t_n: np.ndarray
    t_s1: np.ndarray
    t_s2: np.ndarray

    @property
    def reps(self) -> int:
        return self.c_n.shape[0]

    @property
    def horizon(self) -> int:
        return self.c_n.shape[1]

    @property
    def daily_rates_c(self) -> np.ndarray:
        """Running conversion rate on control, shape ``(reps, horizon)``."""
        return self.c_s1 / self.c_n

    @property
    def daily_rates_t(self) -> np.ndarray:
        """Running conversion rate on treatment, shape ``(reps, horizon)``."""
        return self.t_s1 / self.t_n


def simulate(
    config: ExperimentConfig,
    reps: int = 2_000,
    seed: int = 0,
) -> Simulation:
    """Draw ``reps`` independent experiments from ``config``.

    The day loop is intentional: drawing one day at a time keeps memory flat
    even for large ``reps``, and it mirrors how a real experiment accumulates
    traffic.
    """
    if reps < 1:
        raise ValueError("reps must be at least 1")

    rng = np.random.default_rng(seed)
    horizon = config.horizon_days

    c_s1 = np.zeros((reps, horizon), dtype=float)
    c_s2 = np.zeros((reps, horizon), dtype=float)
    t_s1 = np.zeros((reps, horizon), dtype=float)
    t_s2 = np.zeros((reps, horizon), dtype=float)

    for day in range(horizon):
        c_s1[:, day], c_s2[:, day] = _draw_day(rng, config, 0.0, reps)
        t_s1[:, day], t_s2[:, day] = _draw_day(rng, config, config.relative_lift, reps)

    # Daily sums become cumulative sums.
    n_c = np.cumsum(np.full((reps, horizon), float(config.daily_traffic_per_arm)), axis=1)
    n_t = n_c.copy()

    return Simulation(
        config=config,
        c_n=n_c,
        c_s1=np.cumsum(c_s1, axis=1),
        c_s2=np.cumsum(c_s2, axis=1),
        t_n=n_t,
        t_s1=np.cumsum(t_s1, axis=1),
        t_s2=np.cumsum(t_s2, axis=1),
    )


def _draw_day(
    rng: np.random.Generator,
    config: ExperimentConfig,
    relative_effect: float,
    reps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sum, sum_of_squares)`` for one day on one arm.

    ``relative_effect`` is the lift applied to this arm (0.0 for control).
    """
    n = config.daily_traffic_per_arm

    if config.metric == "bernoulli":
        # A binomial draw is the exact aggregate of n Bernoulli users.
        rate = config.baseline_rate * (1.0 + relative_effect)
        conversions = rng.binomial(n, rate, size=reps).astype(float)
        return conversions, conversions

    # Log-normal revenue: the lift multiplies the mean, the shape is unchanged.
    meanlog = np.log(config.revenue_mean) - 0.5 * config.revenue_sigma**2
    meanlog += np.log1p(relative_effect)
    values = rng.lognormal(meanlog, config.revenue_sigma, size=(reps, n))
    return values.sum(axis=1), (values**2).sum(axis=1)


if __name__ == "__main__":
    # Smoke check: one experiment, day by day, using the same numbers a
    # policy would see. Helpful for eyeballing how fast p-values wander.
    from .sequential.policies import NaivePeek

    cfg = ExperimentConfig(relative_lift=0.0, horizon_days=10)
    sim = simulate(cfg, reps=1, seed=42)
    p_values = NaivePeek().p_values(sim)[0]

    print(cfg.label())
    print(f"{'day':>4} {'users':>7} {'ctrl':>7} {'treat':>7} {'p':>8}")
    for day in range(cfg.horizon_days):
        print(
            f"{day + 1:>4} {int(sim.c_n[0, day] + sim.t_n[0, day]):>7} "
            f"{sim.daily_rates_c[0, day]:>7.4f} {sim.daily_rates_t[0, day]:>7.4f} "
            f"{p_values[day]:>8.4f}"
        )
