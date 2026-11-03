# Methods — how the machinery works

This document is the layer between the design specs
([`roadmap.md`](roadmap.md)) and the measured results (the study READMEs): it
explains how each method actually works — the intuition, the key formulas, and
the specific implementation choices this repo made, including the ones that
were wrong first. It is not a proof; for proofs and the primary literature see
[`REFERENCES.md`](../REFERENCES.md).

Chapters are ordered by study. Every claim marked *measured* points at a
number in the corresponding study README; every method described here is
implemented in this repo and exercised by its validation suite.

---

## 1. Sequential testing (study 01)

### 1.1 Why peeking breaks the error rate

A fixed-horizon test controls one probability: the chance of rejecting H0
when H0 is true, held at alpha for the single planned look. Peeking turns the
test into a union of rejection regions
— reject at day 5 *or* day 10 *or* ... *or* day 30 — and the union's false
positive rate is not alpha; it grows with the number of looks. Each look adds
a mostly-disjoint slice of H0 paths that got unlucky early and recovered late.

*Measured*: with daily looks at alpha = 0.05 the realized rate is 24.0%
(`studies/01-sequential/README.md` §1) — and the growth is monotone in the
number of looks, which is the anchor `calibration/monotone-in-looks` checks
(0.046 → 0.098 → 0.151 → 0.233 as looks go 1 → 3 → 6 → 30).

### 1.2 Bonferroni: valid, and the wrong shape

Splitting alpha evenly across K looks (alpha/K each) bounds the family-wise
error rate by alpha — a union bound, so it is valid but never tight. The real
cost is the shape: alpha/K is the same threshold on day 5 as on day 30, but
early looks have a fraction of the information, so the boundary is wildly far
out early (no early stopping is ever possible in practice) and needlessly
strict late. *Measured*: valid (FPR 1.2–3.6%) but 18 points of power gone at
K = 6.

The lesson Bonferroni teaches is the design requirement for everything that
follows: **the threshold must depend on how much information has accrued**,
not on the calendar.

### 1.3 Group sequential boundaries: the Armitage–McPherson recursion

O'Brien–Fleming (OBF) is the canonical "spend almost nothing early" schedule.
The machinery that makes it usable at *arbitrary* look times is Lan–DeMets
alpha spending: instead of computing boundaries for a fixed K, define a
budget function alpha\*(t) over information fraction t and require

```
P(cross at look k and not before) = alpha*(t_k) - alpha*(t_{k-1})
```

The boundaries follow from the canonical joint distribution of sequential
z-statistics: increments are independent, so `Cov(Z_i, Z_j) = sqrt(t_i/t_j)`
for i < j. The repo computes boundaries with the Armitage–McPherson recursion
(`explabs/sequential/boundaries.py`): on a 600-point grid, keep the sub-density
of Z_k restricted to paths that have *not* crossed an earlier boundary, then
integrate out the previous look:

```
f_k(z) = ∫_{abs(u) <= b_{k-1}} f_{k-1}(u) · phi((z − rho·u)/s) / s du,
         rho = sqrt(t_{k-1}/t_k),  s = sqrt(1 − rho²)
```

Implementation choices worth knowing:

- **Boundaries are computed, never simulated.** A boundary off by 0.01 in z
  shifts the type-I rate by a few percent — invisible at a few thousand Monte
  Carlo replications, which is exactly the precision the study runs at. So
  correctness is anchored against *published tables* instead: the classic OBF
  K = 5 vector (4.877 … 2.031) is pushed through this repo's recursion and
  must cross at alpha = 0.04996 (`boundaries/published-obf-table`).
- **Daily OBF is excluded by design.** At K = 30 the per-look spend alpha/30
  underflows and the first boundary runs past z = 8 — it can never fire. That
  is not a bug; it is the method saying group-sequential designs belong to
  interim analyses, not dashboards.

*Measured*: OBF holds alpha at every schedule tested (4.5–4.8%) and saves 23%
of users at a +5% lift with essentially no power loss (79.6% vs 82.0%).

### 1.4 mSPRT: always-valid, and honestly conservative

The mixture SPRT replaces SPRT's "two simple hypotheses" with a mixture over a
prior on the effect: `N(0, tau²)`. The test statistic is the mixture likelihood
ratio, which under H0 is a martingale with mean exactly 1 (checked in
simulation: E[Lambda] = 1.000 at day 1 for every tau tested). Ville's
inequality then gives P(cross at *any* time) <= 1/alpha·(threshold) — validity
at every stopping time, no schedule needed. The running form used here:

```
Lambda_t = exp(tau² V_t² d_t² / (2(1 + V_t tau²))) / sqrt(1 + V_t tau²)
```

with V_t the accumulated information and d_t the current mean difference.

The part the folklore gets wrong: **always-valid is not free**. The implied
final-look boundary solves a closed form (quoted in the study README) and at
the optimal tau lands at z\* = 3.08 — a 2.4× sample multiplier over the
fixed-horizon 1.96. No tau puts the boundary near 2.0 while keeping power at
the effect of interest, because information V grows two orders of magnitude
over the experiment and a fixed tau cannot be right at every level.
*Measured*: FPR 1.0% (not 5%), power 51.9%. Safe at every prior value tested,
underpowered everywhere — the Ville bound is valid but loose.

### 1.5 Bayesian monitoring does not fix peeking

A posterior is a stopping statistic like any other. Thresholding a two-sided
95% credible interval repeatedly — with a sensible prior (sd 1pp on a 10%
baseline) — inherits the peeking problem with a shrinkage step: the shrinkage
makes the statistic *stable*, which is not the same as making it *valid*.
*Measured*: FPR 20.2%, and 12.7–21.0% across every prior value tested. The
result is reported as a measured contradiction of a common claim, not as an
argument against Bayesian methods generally.

---

## 2. Variance reduction (study 02)

### 2.1 CUPED: the regression view of a covariate

CUPED adjusts each outcome by its pre-experiment covariate:

```
Y_adj = Y − theta' (X − X̄),   theta = Cov(X, Y) / Var(X)
```

Because X is pre-experiment data, it is independent of treatment assignment,
so the adjustment removes the share of Y's variance that X explains *without
touching the treatment contrast*: the adjusted treated mean minus the adjusted
control mean is
still tau, and Var(Y_adj) = Var(Y)·(1 − rho²) with rho the covariate–outcome
correlation. That (1 − rho²) — variance *reduction* rho² — is the entire
theory, and it is why the generator fixes rho by construction and the
validation suite checks the realized reduction against it
(*measured*: 0.472 vs 0.490 theory at rho = 0.7).

The one rule that matters: **theta must come from the control arm only**. Fit
on both arms, the coefficient soaks up treatment signal and the estimate is no
longer exact. The repo enforces this with an *exact* test rather than a
statistical one (`leakage/*`): permute outcomes within the treated arm — the
treated mean is untouched, so any control-only-fit estimator must return an
identical number to machine precision. CUPED, CUPAC and the oracle all pass
with change 0.00e+00. Post-stratification is deliberately excluded from this
check because it legitimately uses treated values *within* each stratum.

### 2.2 The ceiling: why CUPED cannot beat rho²·R²_lin

theta'X is a *linear* predictor. If the outcome's signal f(X) is not linear,
the best linear adjustment removes only the linearly-explained share:

```
CUPED reduction <= rho² · R²_lin,    R²_lin = R² of regressing f on X
```

The study builds three signal shapes (linear / interaction / threshold) with
known R²_lin precisely so this cap is a *measured* number, not an opinion —
and an **oracle** estimator (adjusting by the true f(X)) is included so
"CUPAC closes the gap" has a reference point. `theory/r2-linear[*]` recomputes
R²_lin on an independent draw to make sure the ceiling quoted in the report is
not an artefact of the calibration draw.

### 2.3 CUPAC: prediction as covariate, and the cross-fitting tax

CUPAC replaces theta'X with an ML prediction m(X) (here
`HistGradientBoosting`, 60 iterations / 31 leaves — capacity chosen by
measurement: it recovers f with correlation² = 0.90, better than a larger
model that fits noise). It works because m(X) is still pre-experiment data;
the danger is entirely in how m is fitted:

- **Cross-fitted (the version used)**: fit m on folds, predict out-of-fold.
  Honest, and it works — recovers 88% of the gap CUPED leaves on nonlinear
  signals.
- **In-sample (the classic mistake)**: fit and apply on the same data. The
  prediction overfits its own noise, which *understates the standard error*
  (ratio 0.895–0.905 vs 1.000 across six runs) while simultaneously being
  *less* efficient — the mistake costs on both axes while looking better on
  the dashboard. This is study 02's version of the Chernozhukov et al.
  cross-fitting requirement.

One honest measurement note: the variance *reduction* is a difference of two
nearly-equal variances, so its Monte Carlo error is ~1 variance point in
absolute terms regardless of rho — small correlations look systematically
under-performing at any affordable replication count. Every reduction in the
report carries a bootstrap interval for exactly this reason (§7 of the study
README).

### 2.4 Post-stratification

Stratify on a discrete covariate, difference within stratum, reweight by
stratum sizes. Nonparametric — no linearity assumption — but it needs a
discrete covariate and the estimator's reduction is monotone in the number of
strata while always sitting under the oracle ceiling (`poststrat/*` checks
both).

---

## 3. Interference (study 03)

### 3.1 A market where SUTVA is false by construction

The simulator (`explabs/interference/market.py`) is the smallest two-sided
market with the three channels that matter:

1. **Direct effect** — treatment (a rider subsidy) raises treated riders'
   request probability.
2. **Congestion** (phi) — more requests relative to driver capacity degrades
   the fill rate for *everyone in the market* (the spillover): degradation is
   convex in relative utilization.
3. **Scarcity allocation** (pi) — when capacity is short, it must be rationed
   between arms; proportional rationing (pi = 0) spreads the shortfall evenly,
   and pi > 0 lands it mostly on control riders.

The **ground truth** is not analytic: it is the difference between the
full-treatment and full-control *equilibria* of the same market — the effect a
100% roll-out would deliver. That distinction is the study's point.

### 3.2 User-level randomization: exact estimate of the wrong thing

Randomizing users within a market does not create two markets — both arms
share the same congestion. The difference-in-means therefore estimates the
*within-market* effect, not the roll-out effect, and it does so *precisely*:
*measured* 0.731 against a within-market effect of 0.730 (1 MC SE away) and
0.618 against the roll-out effect (22 SE away). User-level randomization is
not a noisy estimate of the roll-out effect; it is an exact estimate of a
different quantity, and the gap grows with the externality (18% of the
roll-out effect at the base calibration, 78% under strong congestion).

### 3.3 Cluster randomization: unbiased, and you pay in effective N

Randomizing whole cities breaks the spillover channel — control cities
generate their own equilibrium. The estimate is unbiased everywhere tested,
but effective sample size becomes the number of *cities*, and naive
rider-level standard errors are far too small: coverage < 0.85 (`cluster/
naive-coverage-*`). Cluster-robust SEs with the t-correction get close to
nominal — but the repo records the residual honestly: below ~10 clusters the
interval is still a little short, and the real answer at that scale is
randomisation inference, which is documented as out of scope.

### 3.4 Switchback: captures the equilibrium, exposed to carryover

Randomizing time periods within a city lets each period see the market-wide
equilibrium, so it recovers the roll-out effect to within 0.01 like cluster
randomization. The risk is carryover: a rider who rode last period is changed
by it. The implementation uses the crossover form — a share lambda of the
previous period's outcome persists — for which the attenuation has a closed
form:

```
measured / true  ->  1 − 2·lambda   for a long experiment
```

At lambda = 0.5 the switchback recovers *almost nothing* — which is why
"the switchback showed no effect" is a finding about the design, not the
market (`switchback/attenuation-lambda*` checks the curve).

---

## 4. Adaptive experiments (study 04)

### 4.1 What adaptivity does to an estimator

Under a fixed 50/50 split, the assignment indicator is independent of potential
outcomes, and every standard estimator is unbiased. A bandit breaks that:
allocation depends on *observed outcomes*, so early luck changes who ends up
in which arm. Study 04 separates two mechanisms that are usually conflated:

- **Selection bias** — a policy abandons an arm after bad luck, so that arm's
  remaining observations are conditioned on past disappointment. This happens
  even with *no* covariates.
- **Composition bias** — a targeting policy sends *different kinds of users*
  to each arm (persuadable segments toward treatment). This happens even with
  a "balanced" policy, and it is exactly the allocation-based quantity the
  study computes: the mean difference of realized per-segment treated shares
  applied to true segment rates (`composition_bias`, exact by construction —
  no Monte Carlo in it).

The decomposition matters because the fixes differ: selection bias needs
inverse-probability weighting; composition bias needs either weighting or
segment-level analysis.

### 4.2 Recording the propensity — and computing it exactly

Every policy records e_t, the probability of assigning treatment given the
history so far, at every step. That
record is what makes post-hoc inference possible at all (Li et al. 2010's
logging practice). For Thompson sampling the propensity is
P(theta_treated > theta_control) under two independent Beta posteriors, which
has no closed form — the repo computes it by Gauss–Legendre quadrature on a
per-row interval (mean ± 10 posterior SD, 48 nodes; agreement with a 96-node
reference at machine precision and with a 400k-draw Monte Carlo to 2 SE).

One trap the implementation sidesteps: Thompson sampling *sampled* propensities
would put noise in the denominator of the IPW weights. The policy implements
TS's *Bernoulli decision* against the exact propensity instead — the realized
assignment share matches the mean recorded propensity to 0.2 MC SE, and the
balance identity E[1{A=a}/e_a] = 1 holds to under 0.01.

UCB1 needs no propensity computation — it is deterministic given history —
which is exactly why its propensities hit exactly 0/1: with a deterministic
policy, inverse-probability weighting is *undefined*, not merely noisy. The
estimators flag this (`_identified`: > 50% degenerate steps ⇒ undefined)
rather than emitting a garbage number.

### 4.3 The three estimators

All on the same recorded data, T users, known propensities e_t:

- **Naive** — treated mean minus control mean. Biased under any
  adaptive policy: *measured* +30% of the true effect (3.4 MC SE) with
  coverage 0.833 — and its standard error is independently invalid too
  (claimed 0.019, actual 0.041), which naive CIs cannot reveal.
- **Horvitz–Thompson IPW** — weight each outcome by its inverse assignment
  probability:

  ```
  tau_hat = (1/T) · Σ [ A_t·Y_t/e_t  −  (1−A_t)·Y_t/(1−e_t) ]
  ```

  Unbiased by construction of the propensity record; high variance when e is
  extreme.
- **Hájek (self-normalised) IPW** — divide HT by the realized weight mass.
  Lower variance; SE by ratio linearisation, `psi = w·(y − arm mean)`. Under a
  constant propensity it reduces exactly to naive — checked to 0.00e+00,
  which is the cheapest way to verify both implementations at once.
- **AIPW** — IPW plus an outcome-model correction:

  ```
  tau_hat = (1/T) Σ [ m1(X) − m0(X)
                      + A(Y − m1(X))/e − (1−A)(Y − m0(X))/(1−e) ]
  ```

  with a cross-fitted logistic outcome model. Doubly robust: correct if the
  propensities are right (they are — they are recorded) *or* the outcome model
  is right (deliberately misspecified here, and the estimate stays unbiased —
  `adaptive/aipw-double-robust`). On binary outcomes the model buys little
  variance: AIPW's SE is 0.90–0.97 × IPW's, bounded below by the Bernoulli
  noise floor.

*Measured bottom line*: naive bias +30–41% of the effect under every adaptive
policy; IPW/AIPW unbiased (bias within 1.6 MC SE), coverage 0.88–0.93.

### 4.4 Regret accounting

Regret is measured against the *additive* world's closed-form baseline:
a fixed 50/50 split wastes `T·Δ/2` conversions on the inferior arm. The
study's regret numbers are expressed both as cumulative conversions and as
**wasted users** (regret/Δ) — the right unit for policy comparison, because
raw regret grows with Δ (each mistake costs more) while the *count of wrong
assignments* shrinks (the gap is easier to learn). *Measured*: wasted users
drop ~10× as Δ goes 0.2 → 4 points; Thompson sampling returns 62% of the
fixed split's waste at a 4-point lift and noise below half a point.

---

## 5. Uplift modeling (study 05)

### 5.1 The meta-learners

All three wrap the same base learner (gradient boosting, 300 estimators /
depth 2 / min leaf 200 — chosen by grid) into an effect estimator:

- **S-learner** — one response surface with treatment as a feature. When the
  treatment effect is small relative to response noise, no split ever uses the
  treatment column, and the model predicts "no effect anywhere". *Measured*
  exactly that: in a constant-effect world the S-learner collapses to 0.0000 —
  recorded as its own check (`constant/s-learner-shrinks-to-zero`), the
  failure mode Künzel et al. predicted.
- **T-learner** — two separate surfaces, differenced. Unbiased but the
  difference inherits both surfaces' errors: two overfits subtract to noise.
- **X-learner** — the construction that pays attention to imbalance:
  1. Fit m1 on treated, m0 on control (as the T-learner).
  2. **Impute individual effects**: treated units give d1 = Y − m0(X) (their
     outcome minus the *control* surface's guess), control units give
     d0 = m1(X) − Y. Each imputation leans on the other arm's surface.
  3. Fit effect surfaces tau1 on (X, d1) and tau0 on (X, d0).
  4. Combine with the propensity: `tau = e·tau0(X) + (1−e)·tau1(X)` — the
     *well-estimated* arm's effect surface gets the weight.

  *Measured*: at a 10% treated share the X-learner clearly beats the T-learner
  (RMSE 0.048 vs 0.063 — exactly Künzel's promise), and at 85% treated the
  ranking flips. The weighting is the whole trick.

### 5.2 The honest causal forest

A random forest in costume would split each node by fitting two surfaces and
taking the biggest mean difference in the *same* sample it then estimates
from — selection on noise, the tree- analogue of peeking. The hand-built
forest (`HonestCausalForest`, no EconML) does three things differently:

1. **Effect-heterogeneity split criterion** — at each node, split the parent's
   *splitting* sample to maximize the between-child variance of leaf effect
   estimates, not the response variance.
2. **Honesty (Athey–Imbens)** — disjoint samples: the tree structure is grown
   on one half-subsample, and each leaf's effect (mean treated − mean control)
   is estimated on the *other* half. The estimate in a leaf is therefore
   unaffected by which split created it.
3. Subsampling per tree, majority averaging over trees for the final tau.

The `honest=False` twin isolates what honesty buys on the same data:
*measured* fresh-data RMSE 0.0189 honest vs 0.0222 in-sample — ~15% for
honesty alone, and the honest forest's overall effect-recovery correlation
(0.777) beats every meta-learner (T 0.458, X 0.328, S 0.054).

A build note recorded here because it cost an afternoon: the first version
predicted a constant for every user. The leaf-value fallback had overwritten
the internal nodes' NaN markers, so traversal stopped at the root. Trees here
key leaves on `feature == -1` instead — the debug path (dump one tree's nodes,
check the traversal) is in the study's history and is the recommended first
move for any hand-rolled tree.

### 5.3 Qini: measuring targeting, not prediction

The Qini curve scores a ranking by incremental responses captured when
targeting the top-k, in Radcliffe's count-ratio form
(`explabs/uplift/qini.py`, one shared code path for every model):

```
Q(k) = Y_t(k) − Y_c(k) · N_t(k)/N_c(k)
```

cumulative treated responses minus control responses *scaled by the treated/
control count ratio* (the scaling makes Q immune to arm imbalance in counts;
the early steps before any control arrives fall back to raw Y_t). Area under
Q minus its diagonal — auQC — is the single number per model, with standard
errors from a **paired** bootstrap: all models are scored on the same
evaluation users, so resampling users once and re-scoring every model keeps
the correlation between models. Unpaired "SEs" would turn every comparison
into two guesses.

The study's main methodological finding lives here: **Qini comparisons have a
power budget**. At an 85% treated share, a 40k-user holdout cannot separate
any model from any other (paired gaps <= 1.5 SE); in a balanced world the
machinery separates oracle from random at 4 SE. "Our model won on Qini" from
an imbalanced holdout of that size is noise, and the machinery to know it is
the paired bootstrap, not the point estimate.

### 5.4 Budget allocation

On synthetic data the true tau(x) is known exactly, so the report also scores
each ranking by **exact incremental conversions captured**: sum of true tau
over the top-f users by each model's score, at f = 5/10/20/50%. This is the
decision-unit number the Qini curve only proxies. *Measured*: the honest
forest captures 89% of the oracle's incremental conversions at 10% targeting;
capture is monotone in f and equals N·ATE at f = 100% for every ranking
(`targeting/capture-*`).

---

## Where to go next

- Design contracts and anchor status: [`roadmap.md`](roadmap.md)
- Measured results and figures: `studies/01-sequential/` … `studies/05-uplift/`
- Which open-source implementations to compare against:
  [`method_comparison.md`](method_comparison.md)
- Primary literature per study: [`REFERENCES.md`](../REFERENCES.md)
