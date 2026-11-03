# Method comparison

Two comparisons: the statistical methods themselves, and the open-source
projects that implement them. Numbers labelled *measured* come from this repo
at 3,000 replications, 2,000 users/day/arm, 30-day horizon, 10% baseline;
numbers labelled *expected* are qualitative expectations from the literature
that this project has **not** yet verified — they are stated as things to check,
not as results.

---

## 1. Methods

The single most important column is the second one. "Sequential test" is used
loosely to mean at least three different guarantees, and the difference decides
whether a result survives being looked at tomorrow.

| Method | Error guarantee | Look schedule | Key parameter | Stops early? | Status here | Reference implementation |
|---|---|---|---|---|---|---|
| Fixed horizon | Type-I error at one planned look | Exactly one | `alpha` | No | **Implemented** | any two-sample test |
| Naive peek | Pointwise only — no global guarantee | Any | `alpha` per look | Yes, wrongly | **Implemented** (as the villain) | most dashboards |
| Bonferroni over looks | FWER ≤ α over K planned looks | K planned looks | `alpha / K` | Yes, conservatively | **Implemented** | `statsmodels.stats.multitest` |
| Wald SPRT | α and β exactly, simple vs simple | Continuous | `theta0, theta1, alpha, beta` | Optimally | Roadmap | classic; `statsmodels.stats.sequential` (not in 0.14.6) |
| Pocock | FWER over K looks | K planned looks | constant `c(K, alpha)` | Yes, generously | Roadmap | `gsDesign`, `rpact` |
| O'Brien–Fleming + Lan–DeMets spending | FWER over K, any timing | Any (spending by information fraction) | spending function | Yes, needs strong early evidence | Roadmap | `gsDesign`, `rpact` |
| mSPRT | Anytime-valid | Continuous, arbitrary | mixture prior `tau` | Yes | Roadmap | Johari et al. (2017) |
| Asymptotic confidence sequences | Anytime-valid, asymptotic | Continuous, arbitrary | tuning sample size `N*` | Yes | Roadmap | `confseq`; `gbstats` (GrowthBook) |
| Bayesian monitoring | Posterior, no frequentist α | Any | prior, threshold | Yes | Not planned | `gbstats` |

### Measured against each other

| Policy | Looks | FPR under H0 | Power at +5% lift | Mean users | Note |
|---|---:|---:|---:|---:|---|
| Fixed horizon | 1 | 4.8% | 82.0% | 120,000 | calibrated; the baseline |
| Bonferroni, every 5 days | 6 | 3.5% | 64.0% | 117,573 | valid, but 18 pts of power gone |
| Bonferroni, daily | 30 | 1.2% | 48.3% | 95,727 | valid and far too strict |
| Naive peek, every 5 days | 6 | 15.8% | 87.0%* | 109,327 | *contaminated by FPR |
| Naive peek, daily | 30 | 24.0% | 89.8%* | 52,355 | *contaminated by FPR |

The shape of the gap is the story: **Bonferroni buys validity with ~34 points of
power, while naive peeking buys speed with a 24% error rate.** Neither is
acceptable. OBF sits between them on purpose — conservative early, near-nominal
late — and mSPRT/confidence sequences give up the "planned K" requirement
entirely at the cost of slightly wider intervals. Those four numbers are the
targets the next implementation has to hit, and `evaluate.py` already measures
all four.

### Where the methods actually differ

1. **What must be fixed in advance.** Fixed horizon and Bonferroni need a
   planned look count; OBF needs only a spending function; mSPRT and confidence
   sequences need nothing. In practice look times are never what was planned,
   which is why the schedule-free methods win on operational grounds even when
   their nominal efficiency is slightly worse.
2. **What "efficiency" means.** SPRT minimizes expected sample size at two
   specified hypotheses. OBF minimizes the expected sample size subject to a
   spending function. Bonferroni optimizes nothing — it is a bound, not a
   design. mSPRT's efficiency depends on whether the mixing prior `tau` matches
   the true effect, which is why the tau-sensitivity sweep is on the roadmap
   rather than optional.
3. **The tail is the price.** Every early-stopping method trades a shorter
   typical run for a longer worst case. Reporting the mean alone hides this;
   that is why `evaluate.py` also reports the censored share.

---

## 2. Open-source projects

| Project | Language | Scope | Sequential methods | What this repo borrows | What it does that we do not |
|---|---|---|---|---|---|
| [`gsDesign`](https://cran.r-project.org/package=gsDesign) | R | Group sequential design and analysis | OBF, Pocock, Wang–Tsiatis, custom spending; boundary and sample-size computation | Boundary conventions, terminology (information fraction) | Correct, validated boundary numerics; sample-size re-estimation; survival endpoints |
| [`rpact`](https://cran.r-project.org/package=rpact) | R | Confirmatory adaptive designs | Group sequential, adaptive, blinded sample-size review | Design vocabulary; the "what a regulator accepts" bar | Validated adaptive designs, pipeline integration, regulatory documentation |
| [`confseq`](https://github.com/gostevehoward/confseq) | Python / C++ | Time-uniform inference | Confidence sequences (normal, empirical-Bernstein, betting), always-valid p-values | The confidence-sequence formulation | Real boundaries with proven coverage; quantile and ATE sequences; production-grade numerics |
| [`gbstats`](https://docs.growthbook.io/statistics/sequential) (GrowthBook) | Python | Production A/B analysis engine | Asymptotic confidence sequences (Waudby-Smith et al. 2023), tuned by `N*` | The `N*` tuning idea, and the framing of wider-but-always-valid intervals | CUPED/regression adjustment, ratio and quantile metrics, Bayesian and bandit engines, an actual UI |
| [`statsmodels`](https://www.statsmodels.org/) | Python | General statistics | Historically an SPRT helper; `sequential` is absent from the 0.14.6 wheel | `proportions_ztest` as an external check in `validation.py` | Mature inference library, wide test coverage |
| This repo (`seq-ab`) | Python | **Evaluation harness** for stopping policies | Fixed horizon, naive peek, Bonferroni (v1) | — | Please submit a PR with OBF and mSPRT |

### The one thing worth knowing about the projects

GrowthBook's documentation states that its sequential testing uses
**asymptotic confidence sequences** from Waudby-Smith et al. (2023), with a
tuning parameter `N*` described as "the sample size at which you tend to make
decisions". It is a common second-hand claim that GrowthBook implements mSPRT.
It does not — mSPRT is Johari et al. (2017), a different construction with a
different tuning parameter. Both give anytime-valid guarantees; they differ in
what you have to specify and how wide the intervals are. If you are comparing
implementations, comparing these two families is more informative than
comparing either to Bonferroni.

### What this project adds

Nothing to statistical theory, and it should not pretend otherwise. What it
adds is the piece the libraries above all leave to the user:

- **A policy abstraction** — `peekable()` × `alpha()` — that makes "how often
  do I look" and "how strict am I at each look" orthogonal. That is the axis the
  peeking problem lives on, and it is invisible in implementations that hardcode
  a stopping rule.
- **Cost accounting.** Users spent, decision days, censoring rate — measured on
  identical traffic for every policy. Library documentation reports power and
  interval width; teams decide on cost and speed.
- **A reproducible false-positive demonstration.** `figures/peeking_cost.png`
  regenerates in four seconds. It is the figure people ask for after the
  meeting, not during it.

### What this project deliberately does not do

- **No CUPED or covariate adjustment.** Variance reduction changes the
  comparison so much that including it partially would be worse than leaving it
  out. It is listed in the README roadmap as a separate axis.
- **No ratio metrics.** Delta-method variance interacts badly with
  always-valid guarantees if the variance estimate is not consistent at every
  look; this needs its own careful pass.
- **No regulatory-grade validation.** `gsDesign` and `rpact` are validated for
  submission; this is a teaching and evaluation tool.
- **No new method.** The contribution is measurement, not inference.

---

## 3. How to use these comparisons

- Before implementing OBF: read Jennison & Turnbull (2000) ch. 2–7, then check
  your boundary against `gsDesign` for the same `K`, `alpha`, and spending
  function. Anything more than a rounding difference is a bug.
- Before implementing mSPRT: read Johari et al. (2017), then verify that your
  FPR stays at 5% for a range of `tau` values, not just one. A method that only
  works at a hand-tuned `tau` has not been implemented, it has been overfitted.
- Before claiming a speed-up: report users spent, not days saved. Days saved is
  a function of traffic, which is not a property of the method.
