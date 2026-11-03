"""A two-sided market in which treatment changes the market-clearing outcome.

Why this model
--------------
A rider subsidy raises treated riders' demand. Demand is served by a driver
pool that responds to *total* demand with elasticity ``supply_elasticity``, but
only partially within an experiment's horizon. Higher utilisation makes
matching slower for everyone in the city, so the per-request completion
probability falls with utilisation.

That one feedback term is enough to break SUTVA: a treated rider's outcome
depends on how many *other* riders in the same city are treated. A user-level
randomisation holds that number fixed at the experiment's share, while a
roll-out moves it to 100%, so the two ask different questions and get different
answers.

What the model is not
---------------------
It is not a structural model of a ride-hailing platform. There is no matching
queue, no spatial dimension, no dynamic pricing and no driver entry over weeks.
It is the smallest model with (a) a direct treatment effect, (b) a congestion
externality, and (c) a scarcity-allocation rule, which are the three ingredients
the assignment designs need to disagree about. Every number in Study 03 should
be read as "in this model", not "on Uber".

Units
-----
One city, one period. ``trips_per_rider`` is the number of ride requests a
rider would like to make; the outcome is completed trips, in [0, trips_per_rider].
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class MarketConfig:
    """Parameters of one city.

    The two dials that matter for the study:

    ``congestion`` (phi)
        How strongly completion degrades as utilisation rises above baseline.
        phi = 0 removes the externality entirely and restores SUTVA, which is
        the model's own control condition. It only restores SUTVA if capacity
        never binds -- see ``MarketConfig.strained``.
    ``priority`` (pi)
        How much the allocation of scarce capacity favours treated riders.
        pi = 0 is proportional rationing (a common shock, so it cancels in a
        within-city contrast); pi > 0 is crowd-out, which does not cancel.
        It has no effect unless capacity actually binds.
    """

    riders: int = 2_000
    drivers: float = 1_000.0
    trips_per_rider: int = 4
    trips_per_driver: float = 10.0

    subsidy: float = 0.25
    demand_elasticity: float = 0.8
    elasticity_sd: float = 0.25
    supply_elasticity: float = 0.15

    congestion: float = 1.0
    congestion_curve: float = 2.0
    friction: float = 0.08
    priority: float = 0.0

    @property
    def baseline_requests(self) -> float:
        return self.riders * self.trips_per_rider

    @property
    def baseline_capacity(self) -> float:
        return self.drivers * self.trips_per_driver

    @property
    def baseline_utilization(self) -> float:
        return self.baseline_requests / self.baseline_capacity

    @property
    def treated_multiplier(self) -> float:
        """Relative ride demand of a treated rider."""
        return 1.0 + self.demand_elasticity * self.subsidy

    @property
    def baseline_fill(self) -> float:
        """Completion probability at baseline utilisation (independent of phi)."""
        return 1.0 / (1.0 + self.friction)

    def strained(self, drivers: float = 830.0) -> "MarketConfig":
        """Same market, but tight enough that capacity binds at full treatment.

        At the default size the driver pool absorbs a 20% demand increase, so
        scarcity never has to be allocated and the ``priority`` channel is
        dormant. This variant is what makes that channel observable.
        """
        return replace(self, drivers=drivers)


@dataclass(frozen=True)
class MarketState:
    """Market clearing outcome for one city in one period."""

    riders: int
    requests_treated: float
    requests_control: float
    drivers: float
    capacity: float
    utilization: float
    fill_treated: float
    fill_control: float
    rides: float

    @property
    def rides_per_rider(self) -> float:
        return self.rides / self.riders

    @property
    def available_capacity(self) -> float:
        """Capacity after per-request friction: an upper bound on completed rides."""
        return self.capacity


def _allocate(
    requests_treated: float,
    requests_control: float,
    capacity: float,
    priority: float,
) -> tuple[float, float]:
    """Split capacity between arms, favouring treated riders by ``1 + priority``.

    Proportional rationing (``priority = 0``) gives every request the same fill
    rate, so a common capacity shortfall cancels out of a within-city contrast.
    With ``priority > 0`` the same shortfall lands mostly on control riders,
    which is the channel that does *not* cancel.

    Leftover capacity is redistributed, otherwise an arm whose own demand is
    already satisfied would leave capacity unused and quietly shrink the market.
    """
    if requests_treated <= 0 and requests_control <= 0:
        return 0.0, 0.0

    weight_treated = requests_treated * (1.0 + priority)
    weight_control = requests_control
    total_weight = weight_treated + weight_control

    allocated_treated = min(requests_treated, capacity * weight_treated / total_weight)
    allocated_control = min(requests_control, capacity * weight_control / total_weight)

    leftover = capacity - allocated_treated - allocated_control
    if leftover > 0:
        spare_treated = requests_treated - allocated_treated
        spare_control = requests_control - allocated_control
        total_spare = spare_treated + spare_control
        if total_spare > 0:
            share = min(1.0, leftover / total_spare)
            allocated_treated += spare_treated * share
            allocated_control += spare_control * share

    return allocated_treated, allocated_control


def market_state(cfg: MarketConfig, treated_share: float) -> MarketState:
    """Solve one period of the market for a given treated fraction.

    ``treated_share`` is the fraction of *riders* treated, in [0, 1]. The
    driver pool is always at its static equilibrium for this demand level:
    slower supply adjustment is handled in :mod:`designs` as carryover, where
    it belongs, rather than being smuggled in here as a second equilibrium.
    """
    if not 0.0 <= treated_share <= 1.0:
        raise ValueError(f"treated_share must be in [0, 1], got {treated_share}")

    multiplier = cfg.treated_multiplier
    requests_treated = cfg.riders * treated_share * cfg.trips_per_rider * multiplier
    requests_control = cfg.riders * (1.0 - treated_share) * cfg.trips_per_rider
    total_requests = requests_treated + requests_control

    if total_requests <= 0:
        drivers = cfg.drivers
    else:
        relative = total_requests / cfg.baseline_requests
        drivers = cfg.drivers * relative**cfg.supply_elasticity

    capacity = drivers * cfg.trips_per_driver
    utilization = total_requests / capacity if capacity > 0 else np.inf

    # Friction rises with utilisation *relative to baseline*, scaled by phi.
    # At phi = 0 this is the constant 1/(1+friction): no externality, SUTVA holds.
    relative_util = utilization / cfg.baseline_utilization
    degradation = 1.0 + cfg.congestion * (
        relative_util**cfg.congestion_curve - 1.0
    )
    fill_all = 1.0 / (1.0 + cfg.friction * max(degradation, 0.0))

    alloc_treated, alloc_control = _allocate(
        requests_treated, requests_control, capacity, cfg.priority
    )
    fill_treated = alloc_treated / requests_treated * fill_all if requests_treated > 0 else 0.0
    fill_control = alloc_control / requests_control * fill_all if requests_control > 0 else 0.0

    rides = alloc_treated * fill_all + alloc_control * fill_all
    return MarketState(
        riders=cfg.riders,
        requests_treated=requests_treated,
        requests_control=requests_control,
        drivers=drivers,
        capacity=capacity,
        utilization=utilization,
        fill_treated=fill_treated,
        fill_control=fill_control,
        rides=rides,
    )


def global_effect(cfg: MarketConfig) -> float:
    """Effect of treating everyone, in completed trips per rider.

    This is the estimand a roll-out decision needs: what happens to the average
    rider if the treatment goes to 100% of the market, against today's 0%.
    """
    all_treated = market_state(cfg, 1.0)
    none_treated = market_state(cfg, 0.0)
    return all_treated.rides_per_rider - none_treated.rides_per_rider


def direct_effect(cfg: MarketConfig, share: float = 0.5) -> float:
    """What a user-level experiment in a ``share``-treated market estimates.

    Not a biased version of :func:`global_effect` -- a different quantity. The
    contrast is measured inside a market frozen at ``share``, so it carries the
    congestion of that market and, when capacity binds, the crowd-out that
    treatment inflicts on control.
    """
    state = market_state(cfg, share)
    return cfg.trips_per_rider * (
        cfg.treated_multiplier * state.fill_treated - state.fill_control
    )


def _average_over_cities(cfg: MarketConfig, fn, n_nodes: int) -> float:
    """Integrate ``fn`` over the city-level elasticity distribution.

    Cities differ, so every estimand is a population average. Quadrature over
    quantiles is deterministic and cheaper than raising the replication count
    until the curvature stops mattering.
    """
    if cfg.elasticity_sd <= 0:
        return float(fn(cfg))

    from scipy.stats import norm

    quantiles = (np.arange(n_nodes) + 0.5) / n_nodes
    deltas = cfg.demand_elasticity + cfg.elasticity_sd * norm.ppf(quantiles)
    # An elasticity must stay positive; clipping keeps a tail draw from
    # inverting demand without truncating the rest of the distribution.
    deltas = np.clip(deltas, 0.05, None)
    return float(np.mean([fn(replace(cfg, demand_elasticity=d)) for d in deltas]))


def population_global_effect(cfg: MarketConfig, n_nodes: int = 401) -> float:
    """Expected global effect over the population of cities."""
    return _average_over_cities(cfg, global_effect, n_nodes)


def population_direct_effect(
    cfg: MarketConfig, share: float = 0.5, n_nodes: int = 401
) -> float:
    """Expected within-market contrast over the population of cities."""
    return _average_over_cities(cfg, lambda c: direct_effect(c, share), n_nodes)


def expected_trips(cfg: MarketConfig, treated_share: float) -> float:
    """Per-rider mean completed trips in a period with this treated share.

    A single number per period is all the switchback needs: carryover is a
    weighted average of two such means.
    """
    state = market_state(cfg, treated_share)
    if treated_share >= 1.0:
        return cfg.trips_per_rider * cfg.treated_multiplier * state.fill_treated
    if treated_share <= 0.0:
        return cfg.trips_per_rider * state.fill_control
    return cfg.trips_per_rider * (
        treated_share * cfg.treated_multiplier * state.fill_treated
        + (1.0 - treated_share) * state.fill_control
    )


def draw_city(
    cfg: MarketConfig,
    treated_share: float,
    rng: np.random.Generator,
    mean_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, MarketState]:
    """Simulate one city: returns ``(completed_trips, is_treated, state)``.

    A rider's demand is a Poisson count with mean ``trips_per_rider`` times the
    treatment multiplier, and each request is completed independently with the
    arm's fill probability. Poisson thinning makes the completed count exactly
    Poisson with the product as its mean, so the simulator needs no
    approximation and the outcome is an integer (completed trips).

    ``mean_override`` replaces the per-rider mean, which is how a switchback
    carries part of the previous period's treatment into this one.

    The market state is solved at the *realised* treated share, not the target,
    so a randomisation that lands on 51/49 is treated as 51/49.
    """
    n = cfg.riders
    if treated_share <= 0.0:
        is_treated = np.zeros(n, dtype=bool)
    elif treated_share >= 1.0:
        is_treated = np.ones(n, dtype=bool)
    else:
        is_treated = rng.random(n) < treated_share

    state = market_state(cfg, float(is_treated.mean()))
    fill = np.where(is_treated, state.fill_treated, state.fill_control)
    demand = np.where(is_treated, cfg.treated_multiplier, 1.0)
    mean = mean_override if mean_override is not None else cfg.trips_per_rider * demand * fill
    return rng.poisson(mean).astype(float), is_treated, state


def draw_city_effect(cfg: MarketConfig, rng: np.random.Generator) -> MarketConfig:
    """A city drawn from the population: same parameters, its own elasticity."""
    if cfg.elasticity_sd <= 0:
        return cfg
    delta = cfg.demand_elasticity + cfg.elasticity_sd * rng.standard_normal()
    return replace(cfg, demand_elasticity=max(delta, 0.05))


def _smoke() -> None:
    """Eyeball the market: the direct effect, the global effect, and the gap."""
    cfg = MarketConfig()
    print(
        f"baseline: util={cfg.baseline_utilization:.3f} "
        f"fill={cfg.baseline_fill:.3f} multiplier={cfg.treated_multiplier:.3f}"
    )
    print()
    header = (
        f"{'phi':>5} {'fill(x=0)':>10} {'fill(0.5)':>10} {'fill(1)':>9} "
        f"{'Y(0)':>7} {'Y(0.5)':>7} {'Y(1)':>7} {'global':>7} {'user-lvl':>8}"
    )
    print("Congestion-only market (capacity never binds):")
    print(header)
    for phi in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        c = replace(cfg, congestion=phi)
        s0, sh, s1 = market_state(c, 0.0), market_state(c, 0.5), market_state(c, 1.0)
        direct = cfg.trips_per_rider * (cfg.treated_multiplier - 1.0) * sh.fill_treated
        print(
            f"{phi:>5.1f} {s0.fill_control:>10.4f} {sh.fill_control:>10.4f} "
            f"{s1.fill_treated:>9.4f} {s0.rides_per_rider:>7.3f} "
            f"{sh.rides_per_rider:>7.3f} {s1.rides_per_rider:>7.3f} "
            f"{global_effect(c):>7.4f} {direct:>8.4f}"
        )
    print()
    print(f"population effect (elasticity sd={cfg.elasticity_sd}): "
          f"{population_global_effect(cfg):.4f}")

    tight = cfg.strained()
    print()
    print(f"Strained market (drivers={tight.drivers:.0f}, "
          f"util={tight.baseline_utilization:.3f}, capacity binds):")
    for pi in (0.0, 0.5, 1.0):
        c = replace(tight, priority=pi)
        sh = market_state(c, 0.5)
        print(
            f"  priority={pi:.1f}: util={sh.utilization:.3f} "
            f"fill_t={sh.fill_treated:.4f} fill_c={sh.fill_control:.4f} "
            f"global={global_effect(c):+.4f}"
        )
    print("  (the global effect is priority-invariant: with everyone treated "
          "there is no control arm to crowd out)")


if __name__ == "__main__":
    _smoke()
