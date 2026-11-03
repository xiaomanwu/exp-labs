# Roadmap — the five studies

Every study follows the same contract: known ground truth, common random
numbers across methods, a loud correctness suite, and results in decision
units. This document fixes the design before the code, because the hardest
mistakes in simulation studies are design mistakes.

---

## Study 01 — Sequential testing: five method families compared

> **Status: done.** Implemented as `explabs/sequential/` (simulator, six
> stopping policies, boundary machinery, evaluation, 16-check validation
> suite); report and figures in
> [`studies/01-sequential/`](../studies/01-sequential/README.md), method-by-
> method comparison in [`method_comparison.md`](method_comparison.md).
> Deviations from the spec below: the Bayesian posterior rule was added as a
> sixth policy after the five families were planned, because "Bayesian is
> immune to peeking" needed a measured answer (it is not: FPR 20.2%); and
> daily OBF looks were excluded by design — `alpha/30` underflows and the
> first boundary passes z = 8, which is the method's own message, not a bug.

**Question.** A dashboard is checked every day. Which stopping method keeps
the false-positive rate at 5%, what does each one cost in power, and what
does it save in users?

**Design.** Bernoulli conversions on simulated traffic: 3,000 replications per
scenario, 2,000 users/day/arm, 30-day horizon, 10% baseline, two-sided
alpha = 0.05. All policies are evaluated on identical traffic (common random
numbers), so differences are the method's, not the sample's.

| Method | Error guarantee | Expected failure mode |
|---|---|---|
| Fixed horizon | type-I error at one planned look | no early stopping, no savings |
| Naive peek | none | FPR grows toward ~4·alpha with daily looks |
| Bonferroni | FWER across K looks | holds alpha but flat alpha/K wastes power early *and* late |
| Group sequential (OBF via Lan-DeMets spending) | type-I at every look schedule | needs interim-analysis schedules; daily looks are infeasible |
| mSPRT (always-valid) | valid at every stopping time | very conservative at fixed horizons — boundary far above 1.96 |
| Bayesian posterior rule | none (frequentist) | repeated credible-interval checks behave like shrunk naive peeking |

**Outputs.** FPR under H0 per policy and look schedule; power, mean stopping
day and users saved vs the fixed horizon at a true +5% relative lift;
sensitivity of mSPRT and the Bayesian rule to their prior scales.

**Validation anchors.** (1) The simulator itself must be exact: recovered lift
and p-values must match an external implementation. (2) Boundary machinery
must reproduce independently published constants. (3) Measured FPR must be
nominal where theory says nominal, and must rise monotonically with peeking
frequency where no correction is applied.

**Anchors → measured outcome** (checks in `explabs/sequential/validate.py`,
16-check suite):

| Anchor | Check | Measured outcome |
|---|---|---|
| Simulator recovers the configured lift; p-values match an external implementation | `simulator/lift`, `pvalue/statsmodels` | 0.10499 vs 0.10500 (within 4 MC SE); max p-value difference vs `proportions_ztest` 2.8e-06 |
| Boundaries reproduce published constants | `boundaries/single-look`, `boundaries/published-obf-table`, `boundaries/pocock-constant`, `boundaries/spending-recovered` | K=1 boundary 1.9603; classic OBF K=5 table crosses at alpha = 0.04996; Pocock mean boundary 2.4111 vs published 2.413; per-look spend recovered to 1.8e-04 |
| FPR nominal where promised, monotone in peeking | `calibration/fixed-horizon`, `calibration/monotone-in-looks`, `calibration/peeking-is-costly` | fixed horizon 0.046; peeking 0.046 → 0.098 → 0.151 → 0.233 as looks go 1 → 3 → 6 → 30 |

**References.** Wald (1945) for the SPRT foundation; Armitage, McPherson &
Rowe (1969) for what peeking costs; Pocock (1977), O'Brien & Fleming (1979)
and Lan & DeMets (1983) for the boundary family; Johari, Koomen, Pekelis &
Walsh (2017) for mSPRT; Howard et al. (2021) and Waudby-Smith & Ramdas (2023)
for the always-valid machinery behind it; Kohavi, Tang & Xu (2020) for the
practitioner framing.

---

## Study 02 — Variance reduction: CUPED, CUPAC, post-stratification

**Question.** How much variance do covariate-adjusted estimators actually
remove, and how does that translate into MDE and required sample size?

**Design.** Semi-synthetic data: a pre-experiment covariate vector
`X ~ N(0, Sigma)` with a plausible factor structure, outcome
`Y = tau*T + f(X) + eps`, where `f` mixes linear and nonlinear components and
`eps` is scaled so that the marginal variance matches a real conversion or
revenue metric. The truth `tau` is known by construction.

| Method | Estimator | Notes |
|---|---|---|
| None (baseline) | difference in means | what the naive dashboard reports |
| CUPED | `Y_adj = Y - theta (X - X̄)`, theta = Cov(X,Y)/Var(X) from control arm only | the standard; theta must never be estimated on treated units |
| CUPAC | same form, but `theta * X` replaced by an ML prediction `m(X)` (sklearn `HistGradientBoosting`) | prediction-as-covariate; the risk is overfitting inflating apparent power |
| Post-stratification | stratum-weighted difference in means | nonparametric; needs a discrete covariate |

**Outputs.** Variance reduction ratio vs `rho^2` (theory says CUPED achieves
`rho^2`, where rho is the covariate-outcome correlation — this is the validation
anchor); MDE and required sample size before and after; CUPAC vs CUPED when the
truth is nonlinear (CUPAC should win, CUPED should degrade).

**Validation anchors.** (1) On a purely linear outcome, CUPED's realized
variance reduction must match `rho^2` within Monte Carlo error. (2) Estimates
must stay unbiased: `abs(E[tau_hat] - tau)` within 3 SE for every method.
(3) CUPAC fitted on treated units (the classic mistake) must visibly inflate
the estimated power — implemented as a demonstration, not an accident.

**Anchors → measured outcome** (checks in `explabs/variance/validate.py`,
39-check suite):

| Anchor | Check | Measured outcome |
|---|---|---|
| CUPED reduction matches `rho^2` on a linear outcome | `ceiling/cuped[linear]` | 0.472 measured vs 0.490 theory at rho = 0.7, within 3 bootstrap SE |
| Every estimator unbiased within 3 SE | `bias/<method>` | all methods within 3 MC SE on the interaction outcome |
| CUPAC pitfall visible, not accidental | `cupac/*`, `leakage/*` | in-sample fit understates its own SE (ratio 0.895–0.905 vs 1.000); permuting treated outcomes moves CUPED/CUPAC/oracle estimates by < 1e-8 |

**Status: done.** See [`studies/02-variance-reduction/README.md`](../studies/02-variance-reduction/README.md).
Notes on what changed relative to this plan:

- The "CUPAC fitted on treated units" anchor became a leakage *test* instead:
  permuting treated outcomes must leave the estimate exactly unchanged, which
  pins down where the coefficient may come from. The failure actually
  demonstrated is subtler and more common — a model fitted and applied
  in-sample, which understates its own SE by ~10%.
- An **oracle** estimator (adjusting by the true `f(X)`) was added; without it
  "CUPAC closes the gap" has no reference point to be measured against.
- A measurement-floor finding was not in the plan: the variance reduction is a
  difference of two nearly equal variances, so its Monte Carlo error is ~1
  variance point *in absolute terms*, which makes small correlations look
  systematically under-performing. Every reduction now carries a bootstrap
  interval, and the floor is documented in §7 of the study README.
- Model capacity was chosen by measurement (60 iterations / 31 leaves recovers
  `f` with correlation^2 = 0.90, better than a larger model that fits the
  noise), not by default.

**References.** Deng, Xu, Kohavi & Walker (2013) for CUPED; Miratrix, Sekhon &
Yu (2013) for post-stratification; DoorDash engineering (2020) for CUPAC as
practised; Kohavi, Tang & Xu (2020) ch. 17 for the power arithmetic.

---

## Study 03 — Interference: user-level vs cluster vs switchback

> **Status: done.** Implemented as `explabs/interference/` (market model, three
> designs, evaluation, 29-check validation suite); report and figures in
> [`studies/03-interference/`](../studies/03-interference/README.md).
> Deviations from the spec below: the carryover mechanism is the crossover
> form (`lambda` share of the previous period's outcome) rather than slow
> driver supply, because supply inertia alone moves drivers 2.8% between arms
> and is second-order at this calibration; and the study adds a scarcity-
> allocation channel (`priority`) the spec did not have, which separates the
> two bias mechanisms.

**Question.** In a two-sided market (ride-hailing), how biased is each design's
estimate of the *global* treatment effect?

**Design.** A stylized market simulator: `M` cities, each with riders and
drivers; treatment (a rider subsidy) raises demand, demand lowers pickup speed,
pickup speed lowers driver earnings and supply — a feedback loop that violates
SUTVA at the market level. The ground-truth global effect is computed by
comparing full-treatment to full-control equilibria.

| Design | Assignment | Expected failure mode |
|---|---|---|
| User-level | users randomized within market | control units are contaminated through the shared market; estimated effect biased *upward* |
| Cluster (city-level) | whole cities randomized | unbiased for the global effect, but effective N = number of cities; variance explodes and CIs must use cluster-level inference |
| Switchback | time periods randomized within city | captures equilibrium effects but risks carryover between periods — implemented and measured |

**Outputs.** Bias, variance and CI coverage per design as a function of the
strength of the externality; the variance-bias frontier between user-level and
cluster designs; effect of carryover on switchback estimates.

**Validation anchors.** (1) With the externality switched off, all three designs
must be unbiased and switchback/user-level must have similar variance. (2) As
the externality grows, user-level bias must rise monotonically. (3) Cluster
randomization must be unbiased with coverage near 95% once cluster-robust
standard errors are used — and below 95% with naive SEs, which is the point.

**Anchors → measured outcome** (checks in `explabs/interference/validate.py`,
29-check suite):

| Anchor | Check | Measured outcome |
|---|---|---|
| Externality off → all designs unbiased | `sutva/<design>` | user-level, cluster and switchback all recover the global effect within 4 MC SE; the effect itself is nonzero (so the check is not vacuous) |
| User-level bias rises with the externality | report §3 | gap vs roll-out effect: 18% at the base calibration, 78% under strong congestion |
| Cluster unbiased; robust SE ≈ 95%, naive below | `cluster/unbiased-*`, `cluster/robust-coverage-*`, `cluster/naive-coverage-*` | cluster and switchback recover the roll-out effect to within 0.01 everywhere tested; naive rider-level SE coverage < 0.85, t-corrected ≥ 0.88 (still short below ~10 clusters — kept as a finding) |

**References.** Hudgens & Halloran (2008) for the interference framework; Basse
& Airoldi (2018) for design under network interference; Karrer et al. (2021)
for cluster experiments at scale; Lyft and Uber engineering blogs for switchback
practice; Kohavi, Tang & Xu (2020) ch. 20 for the marketplace framing.

---

## Study 04 — Adaptive experiments: bandits vs fixed A/B

> **Status: done.** Implemented as `explabs/bandits/` (quadrature-exact DGP,
> five policies with recorded propensities, three estimators, evaluation,
> 40-check validation suite); report and figures in
> [`studies/04-bandits/`](../studies/04-bandits/README.md).
> Deviations from the spec below: the "naive estimates must show measurable
> bias under adaptive allocation" anchor turned out to need a *mechanism*, and
> the study found two — selection bias (a policy abandoning an arm after bad
> luck) and composition bias (a targeting policy sending different users to
> each arm) — which it separates with an exact allocation-based decomposition
> rather than attributing everything to context. The spec's "UCB1 must satisfy
> its logarithmic regret bound qualitatively" is checked as a bound (simulated
> regret stays under 14% of `8 log T / Delta`) because the growth *rate* is
> not logarithmic at experiment-scale horizons and small gaps: the bound holds
> and is never tight, which is the honest version of the claim. Two findings
> the spec did not anticipate: the naive standard error (not just the point
> estimate) is invalid under adaptation, and Thompson sampling's naive bias
> saturates rather than shrinks with the horizon while epsilon-greedy's decays.

**Question.** How much regret does Thompson sampling / UCB save over a fixed
A/B test, and what does adaptation do to the inference you want to run
afterwards?

**Design.** Two-armed Bernoulli bandits with known true rates. Fixed A/B
allocation vs Thompson sampling vs UCB1, over a horizon sweep. For each policy,
record the assignment propensity at every step — this is what makes post-hoc
inference possible.

| Estimator | Formula | Expected failure |
|---|---|---|
| Naive difference in arm means | `mean(treated) - mean(control)` | biased under adaptivity: allocation correlates with early luck |
| IPW | Horvitz-Thompson with true propensities | unbiased, high variance |
| AIPW | IPW plus an outcome model correction | unbiased with lower variance; the right tool |

**Outputs.** Regret curves (cumulative regret vs horizon, per policy and per
true-effect gap); post-hoc inference table: bias and CI coverage of naive vs
IPW vs AIPW after adaptive allocation; the trade-off surface "regret saved vs
inference cost".

**Validation anchors.** (1) Under fixed allocation, all three estimators must be
unbiased — any bias under fixed allocation is a bug. (2) Under adaptive
allocation, naive estimates must show measurable bias, IPW zero, AIPW zero with
smaller SE than IPW. (3) Thompson sampling regret must scale with the gap
between arm rates; UCB1 must satisfy its known logarithmic regret bound
qualitatively.

**Anchors → measured outcome** (checks in `explabs/bandits/validate.py`,
40-check suite):

| Anchor | Check | Measured outcome |
|---|---|---|
| Fixed allocation → all three estimators unbiased | `estimator/fixed-split-coverage` | naive, IPW and AIPW all unbiased under a fixed 50/50 split; coverage 0.933–0.940 |
| Adaptive: naive biased, IPW zero, AIPW zero with smaller SE | `adaptive/naive-is-biased`, `adaptive/aipw-is-tighter-than-ipw`, `adaptive/aipw-double-robust` | naive bias +30% of the true effect (3.4 MC SE), coverage 0.833; IPW/AIPW unbiased (bias ≤ 1.6 MC SE); AIPW SE 0.90–0.97 × IPW's |
| TS regret scales with the gap; UCB1 within its bound | `regret/*` | wasted users (regret/Δ) drop 10× as Δ goes 0.2 → 4 pts; UCB1 regret ≤ 14% of `8 log T / Δ` at every horizon up to 16k (bound holds, never tight) |

**References.** Thompson (1933); Russo et al. (2018) for the modern TS tutorial;
Lattimore & Szepesvári (2020) for UCB theory; Scott (2015) for bandits as
business practice; Horvitz & Thompson (1952) and Robins et al. for IPW/AIPW;
Hadad et al. (2021) for why adaptive data breaks standard SEs and what fixes it.

---

## Study 05 — Uplift modeling: T-learner, X-learner, causal forest

> **Status: done.** Implemented as `explabs/uplift/` (synthetic DGP with
> exact effect truth plus Criteo/Hillstrom loaders, four models, shared Qini
> evaluation, budget allocation); report and figures in
> [`studies/05-uplift/`](../studies/05-uplift/README.md).
> Deviations from the spec below: `CausalForestDML` lives in EconML, which is
> not a dependency here, so the honest causal forest is implemented by hand
> (Athey–Imbens honest splitting, disjoint leaf estimation) and validated
> against its dishonest twin. The Qini evaluation is the Radcliffe
> count-ratio curve on one shared code path, with paired bootstrap standard
> errors — and the study's main methodological finding is about evaluation
> *power*: at an 85% treated share, a 40k-user holdout cannot separate any
> model from any other; only model-vs-random comparisons clear 2 SE. The
> spec's anchor "on constant effects the models recover the same Qini curve"
> held, with the documented exception that the S-learner collapses to a
> zero-effect prediction (recorded as its own check).

**Question.** Which heterogeneous-effect model turns prediction quality into
better targeting, measured by Qini — and how should a fixed budget be allocated
given the estimate?

**Design.** Primary data: Criteo Uplift (Diemert et al. 2018) or Hillstrom
email data, loaded if present under `data/`; otherwise a synthetic dataset with
the same shape (heterogeneous effects, correlated features, realistic treatment
rate) so the pipeline runs out of the box. The simulator doubles as the
ground-truth generator for bias checks.

| Model | Construction | Known weakness |
|---|---|---|
| T-learner | separate response models per arm | differences of fits inherit both models' errors |
| X-learner | imputes individual effects, then models them | better in imbalanced treatment; more moving parts |
| Causal forest | honesty + splitting on effect (EconML `CausalForestDML`) | needs honest splitting; otherwise is a random forest in costume |

**Outputs.** Qini and cumulative-gain curves per model with auQC; uplift by
decile with ground truth overlaid (possible in the synthetic setting); budget
allocation table: incremental conversions captured vs fraction of population
targeted, per model.

**Validation anchors.** (1) On synthetic data with constant effects, all three
models must recover the same Qini curve — heterogeneity artifacts would show as
divergence. (2) On a known heterogeneous effect (a single strong segment), the
top-decile uplift must rank correctly. (3) Qini must be computed identically
across models (same evaluation code path), since Qini implementations differ and
differences there leak into conclusions.

**Anchors → measured outcome** (checks in `explabs/uplift/validate.py`,
24-check suite):

| Anchor | Check | Measured outcome |
|---|---|---|
| Constant effects → agreeing Qini | `constant/all-models-agree`, `constant/s-learner-shrinks-to-zero` | T/X/forest auQC agree within paired MC noise; the S-learner collapses to exactly 0.0000 (its known failure mode, recorded as its own check) |
| One strong segment → top decile ranks correctly | `segment/top-decile/<model>`, `segment/oracle-is-the-segment` | every model's top-decile true mean τ ≥ 1.5× population mean; oracle dominates |
| Qini on one shared code path | `qini/curve-formula`, `qini/random-ranker-is-null`, `qini/oracle-beats-random-beats-anti` | hand-computed curve matches; random ranking ≈ 0; oracle > random > anti-oracle beyond 2 paired SE |

**References.** Künzel, Sekhon, Bickel & Yu (2019) for T/S/X-learners; Athey &
Wager (2019) for causal forests and honesty; Radcliffe (2007) and Radcliffe &
Surry (2011) for Qini and uplift trees; Diemert et al. (2018) for the Criteo
benchmark.

---

## Sequencing

01 first: it establishes the fixed-horizon baseline and the measured cost of
peeking that every later study compares against, and its validation habits
(external cross-checks, published-constant reproduction) set the standard the
others inherit. 02 before 03: variance reduction vocabulary (MDE, sample size)
feeds directly into why cluster designs feel expensive. 04 before 05: IPW/AIPW
in 04 is the same machinery that makes 05's off-policy evaluation honest.
05 last: it is the only study whose data may not be self-contained, and its
evaluation code inherits habits from 02 and 04.
