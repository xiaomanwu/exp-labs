# exp-labs — Five studies in how online experiments actually behave

A portfolio of simulation studies on the gap between textbook experiment
methodology and what happens when a team runs experiments for real: people peek
at dashboards, variance is huge, users interfere with each other, treatment
allocation adapts, and effects differ across users.

| # | Study | Question | Status |
|---|---|---|---|
| 01 | [Sequential testing](studies/01-sequential/README.md) | Which stopping method keeps FPR at 5%, and what does each cost? | **Done** — 5 families, 16 validation checks |
| 02 | [Variance reduction](studies/02-variance-reduction/README.md) | How much do CUPED, CUPAC and post-stratification actually shrink variance, and what does that buy in MDE? | **Done** — 7 methods, 39 validation checks |
| 03 | [Interference](studies/03-interference/README.md) | How biased is a user-level A/B test in a two-sided market, vs cluster and switchback designs? | **Done** — 3 designs, 29 validation checks |
| 04 | [Adaptive experiments](studies/04-bandits/README.md) | Thompson sampling and UCB vs fixed A/B: regret, and unbiased inference after adaptation (IPW/AIPW) | **Done** — 6 policies, 40 validation checks |
| 05 | [Uplift modeling](studies/05-uplift/README.md) | T-learner vs X-learner vs causal forest: Qini curves and budget allocation | **Done** — 4 models, 24 validation checks |

## Why simulation first

All five studies are simulations before they are applications. That is
deliberate: in a simulation the ground truth is known, so bias, coverage and
error rates are *measurable* rather than assumed. Every study is built around
the same discipline:

1. a known ground truth,
2. methods compared on identical data (common random numbers),
3. a correctness suite that fails loudly (`validate.py` per study),
4. results reported in decision units — users, days, money — not just in
   statistical units.

## Repository layout

```
explabs/                     shared code package
  sim.py                     experiment simulator (Bernoulli / log-normal metrics,
                             cumulative sufficient statistics)
  checks.py                  shared validation plumbing (Checker, bootstrap SEs)
  sequential/                study 01: policies, boundaries, evaluation, validation
  variance/                  study 02: data generator, estimators, evaluation
  interference/              study 03: market model, designs, evaluation
  bandits/                   study 04: policies, DGP, inference, evaluation
  uplift/                    study 05: data, models, Qini evaluation
studies/                     one directory per study: README + figures + results
  01-sequential/  02-variance-reduction/  03-interference/  04-bandits/  05-uplift/
docs/
  roadmap.md                 per-study design notes and reference list
  methods.md                 how each method works: intuition, formulas, implementation choices
  method_comparison.md       sequential methods vs each other and vs open-source implementations
REFERENCES.md                annotated bibliography for all five studies
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m explabs.sequential.run       # study 01, ~5 s at 3,000 replications
python -m explabs.sequential.validate  # 16 checks, exits non-zero on failure

python -m explabs.variance.run         # study 02, ~13 min on 8 cores
python -m explabs.variance.validate    # 39 checks, exits non-zero on failure

python -m explabs.interference.run     # study 03, ~2.5 min
python -m explabs.interference.validate  # 29 checks, exits non-zero on failure

python -m explabs.bandits.run          # study 04, ~3 min
python -m explabs.bandits.validate     # 40 checks, exits non-zero on failure

python -m explabs.uplift.run           # study 05, ~3.5 min
python -m explabs.uplift.validate      # 24 checks, exits non-zero on failure
```

Python 3.11+; core dependencies are numpy, scipy, pandas, matplotlib,
scikit-learn (gradient-boosted covariates in study 02, response surfaces and
the causal forest in study 05); statsmodels is needed for the p-value
cross-check in study 01's validation suite.

## Design principles that apply across studies

- **Common random numbers.** All methods see the same simulated data, so
  differences between methods are signal, not noise.
- **Ground truth by construction.** Every headline number (FPR, bias, coverage,
  regret) is measured against a truth the simulator defines.
- **Validation against the published record.** Where closed-form or tabulated
  values exist (OBF boundary tables, fixed-sample 1.96, martingale expectations,
  statsmodels p-values), the code checks against them and fails loudly.
- **Censoring is reported, not hidden.** Methods that never stop pay full price
  in users; the tables show it.
- **Reproducibility.** Seeds are fixed; numpy RNG streams are not stable across
  major versions, which is documented where it matters.

## Headline findings so far

From study 01 (3,000 replications, 2,000 users/day/arm, 30-day horizon,
10% baseline, +5% lift):

- Daily dashboard peeking turns a 5% test into a **24%** test.
- **O'Brien-Fleming is the quiet winner**: holds alpha at every schedule and
  saves 23% of users with no meaningful power loss.
- **mSPRT implemented correctly is deeply conservative** here: FPR 1.0%, power
  52%, implied boundary z* = 3.08 — a ~2.4x sample multiplier. The martingale
  property E[Lambda] = 1 was verified, so this is the method's cost, not a bug.
- **Bayesian stopping does not fix peeking**: 20.2% FPR with a sensible prior.

From study 02 (400–1,500 replications, 5,000 units/arm, `rho^2 = 0.49`, +1% effect):

- **CUPED delivers exactly what the theory promises on a linear outcome** —
  0.472 measured against 0.490 — and at `rho = 0.7` that is a **47% smaller**
  experiment. It is the biggest lever in these two studies and it is free.
- **Nonlinearity caps it, not the method**: where `f(X)` has interactions and
  hinges, CUPED stops at `rho^2 R^2_lin = 0.28`; a gradient-boosted covariate
  (CUPAC) closes 88–92% of the remaining gap, and 20-bin post-stratification —
  which fits nothing — closes a third of it.
- **The in-sample mistake fails on both axes**: fitted-and-applied on the same
  units, CUPAC's standard error is ~10% too small (0.895–0.905 across six
  independent runs, target 1.000) *and* its actual variance is higher than the
  cross-fitted version's. Better-looking dashboard, worse experiment.
- **Variance reduction has an absolute noise floor** of ~1 variance point, so
  "CUPED gave us 6%" from a small experiment is indistinguishable from 1%.

From study 03 (400 replications per cell, 2,000 riders/city, 8–32 cities):

- **User-level randomisation is exact for the wrong quantity.** At the base
  calibration it lands 1 MC SE from the within-market effect and **22 SE** from
  the roll-out effect. The gap grows with the externality: 18% of the roll-out
  effect at the base, 78% when congestion is strong. Cluster and switchback
  stay within 0.01 of the truth everywhere.
- **Coverage fails twice, differently.** A rider-level SE on clustered data
  covers ~47% forever; a cluster-robust SE with the normal quantile covers 77%
  at four clusters. The t correction fixes most (90%) but not all of it — below
  ~10 clusters the honest answer is randomisation inference, which is stated
  rather than papered over.
- **Switchback attenuation has a closed form**, `1 - lambda(2k-1)/k`, and the
  simulation matches it to 4 SE. At 50% carryover a daily-alternation design
  reports 12% of the true effect; washout fixes the bias exactly and pays for
  it in traffic.
- **Switchback's precision is set by the number of cities, not the traffic.**
  With homogeneous cities all three designs deliver identical precision per
  unit of traffic. With city-level effect heterogeneity, switchback lands at
  nearly *twice* the standard deviation of the others at equal traffic — the
  periods average away rider noise but cannot average away city effects.

Details and figures: [`studies/03-interference/README.md`](studies/03-interference/README.md).

From study 04 (800 replications for regret, 300 for inference; 1,500 users,
12% base rate, effects spanning −8 to +13 points by segment in the inference
world):

- **Thompson sampling returns 35% of a fixed split's traffic cost at a 2-point
  lift and 62% at 4 points** — and the prize grows faster than the gap: 5.3
  conversions saved per 1,500 users at 2 points, 0.4 at half a point, where the
  whole prize is smaller than the propensity-logging pipeline an honest
  analysis of the bandit's data requires. UCB1, whose bonus never falls below
  a 2-point gap within 1,500 users, saves 14%.
- **The naive difference in means is wrong twice after a bandit ran.** Its
  bias is 30–41% of the true effect, and *independently* its standard error is
  half of the actual spread (coverage 62% after a contextual ε-greedy).
- **The bias decomposes exactly into two mechanisms.** Composition bias —
  different kinds of users sent to each arm — is computable in closed form
  from the realised allocation and explains 95–100% of the bias of every
  *contextual* policy and 0% of every pooled one; the rest is selection bias
  from abandoning an arm after bad luck.
- **In a world with zero true effect, the naive test "finds" one 37% of the
  time** after a contextual bandit. Recording propensities and running AIPW
  restores both the point estimate (bias ≈ 0) and the test (4.8% false
  positives) — at roughly 1.8× the standard error.
- **UCB1 has no valid inverse-propensity estimate at all** (propensities are
  exactly 0/1), and Thompson sampling's own bias does not shrink with the
  horizon — it saturates near half the effect, because an abandoned arm stops
  accumulating data. ε-greedy's forced exploration makes its naive estimator
  consistent again: exploration is an identification guarantee, not just a
  regret knob.

Details and figures: [`studies/04-bandits/README.md`](studies/04-bandits/README.md).

From study 05 (100,000-user randomized experiment, 85% treated, exact ground
truth; 60,000 training / 40,000 evaluation users):

- **The honest causal forest is the best model by every measure that
  resolves anything** — effect-recovery correlation 0.78 vs the truth
  (T-learner 0.46, X-learner 0.33, S-learner 0.05) and 89% of the oracle's
  incremental conversions captured at 10% targeting (best meta-learner 83%).
  Its auQC is statistically indistinguishable from the oracle's.
- **A Qini evaluation at an 85% treated share cannot tell good models
  apart.** Every model-vs-model auQC gap on the 40k-user holdout is inside
  1.5 paired standard errors; only model-vs-random clears 2 SE. The
  bottleneck is the control arm, and the same comparison in a balanced world
  separates oracle from random at 4 SE. Qini "wins" reported from an
  imbalanced holdout of this size are noise.
- **The S-learner does not participate.** One treatment column among ten
  means no split ever uses it: correlation with the true effect 0.05, and a
  mean predicted effect a quarter of the truth. In a constant-effect world
  it predicts exactly 0.0000.
- **Honesty is measurable, not decorative.** The same forest with in-sample
  leaf estimation (the "random forest in costume") has 31% worse fresh-data
  RMSE than the honest version.
- **The X-learner's imbalance claim is real but points the other way at
  Criteo-like shares.** At 10% treated, X beats T on RMSE (0.048 vs 0.063)
  exactly as Künzel et al. promise; at 85% treated the T-learner edges it.
- **Model choice matters least where budgets are loose.** Targeting 50% of
  users with the *worst* model captures 80% of the oracle's lift; the
  interesting case is a small budget against a concentrated effect.

Details and figures: [`studies/05-uplift/README.md`](studies/05-uplift/README.md).
