# References

Annotated bibliography for this project, organised by the five studies it
contains — the useful question is usually "what do I need to read before I
implement (or re-read) study N". Cross-cutting references are filed under
the study that leans on them hardest.

Two notes on citation hygiene:

- Volume and page numbers were checked against publisher records where
  possible. Before submitting anything academic, re-verify each entry against
  its DOI — citation metadata drifts, and the arXiv versions of several of
  these papers have different titles from the published ones.
- Method labels travel badly. "Sequential test" covers at least four different
  guarantees (see `docs/method_comparison.md`); always check which one a paper
  actually proves.

---

## Study 01 — Sequential testing (`explabs/sequential/`)

### Core papers

**Wald, A. (1945).** Sequential tests of statistical hypotheses.
*Annals of Mathematical Statistics*, 16(2), 117–186.
[doi:10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118)
The origin of the field: the SPRT, and the proof that a test can stop early
while keeping both error rates fixed. Every other method in this list relaxes
one of SPRT's assumptions.

**Armitage, P., McPherson, C. K. & Rowe, B. C. (1969).** Repeated significance
tests on accumulating data. *Journal of the Royal Statistical Society A*,
132(2), 235–244.
What repeated testing does to the error rate. The naive-peek problem is sixty
years old and still shows up in dashboards.

**Pocock, S. J. (1977).** Group sequential methods in the design and analysis of
clinical trials. *Biometrika*, 64(2), 191–199.
[doi:10.1093/biomet/64.2.191](https://doi.org/10.1093/biomet/64.2.191)
Constant critical values across looks — cheap, explainable, generous early.

**O'Brien, P. C. & Fleming, T. R. (1979).** A multiple testing procedure for
clinical trials. *Biometrics*, 35(3), 549–556.
[doi:10.2307/2530245](https://doi.org/10.2307/2530245)
Conservative at early looks, near the fixed-sample threshold at the end.

**Lan, K. K. G. & DeMets, D. L. (1983).** Discrete sequential boundaries for
clinical trials. *Biometrika*, 70(3), 659–663.
[doi:10.1093/biomet/70.3.659](https://doi.org/10.1093/biomet/70.3.659)
Alpha-spending: derive the boundary from an error budget spent over time.
What makes OBF usable when look times never go as planned.

**Jennison, B. L. & Turnbull, B. W. (2000).** *Group Sequential Methods with
Applications to Clinical Trials*. Chapman & Hall/CRC.
The reference text; read chapters 2–7 before implementing any spending
function — the boundary recursions are easy to get subtly wrong.

**Johari, R., Koomen, P., Pekelis, L. & Walsh, D. (2017).** Peeking at A/B
tests: why it matters, and what to do about it. *KDD '17*.
[doi:10.1145/3097983.3097994](https://doi.org/10.1145/3097983.3097994)
The mSPRT paper for the A/B-testing audience, and the closest published
analogue of what this repo's `evaluate.py` measures.

**Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. S. (2021).** Time-uniform,
nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*,
49(2), 1055–1080.
[doi:10.1214/20-AOS1991](https://doi.org/10.1214/20-AOS1991)
Confidence sequences that hold simultaneously over all sample sizes, with no
assumptions beyond boundedness or a known variance proxy.

**Waudby-Smith, I. & Ramdas, A. (2023).** Estimating means of bounded random
variables by betting. *Journal of the Royal Statistical Society B*.
[arXiv:2010.09686](https://arxiv.org/abs/2010.09686)
Hedged-capital betting martingales — the tightest known confidence sequences
for bounded outcomes, current state of the art for proportion-style metrics.

**Ramdas, A., Grünwald, P., Vovk, V. & Shafer, G. (2023).** Game-theoretic
statistics and safe anytime-valid inference. *Statistical Science*, 38(4),
576–601.
[doi:10.1214/23-STS901](https://doi.org/10.1214/23-STS901)
The unifying frame: e-values, e-processes, Ville's inequality — *why* all the
anytime-valid methods work.

### Practice

**Kohavi, R., Tang, D. & Xu, Y. (2020).** *Trustworthy Online Controlled
Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
Sequential testing and the organizational cost of peeking; the "why does this
matter commercially" framing. Reused by the later studies: ch. 17 supplies
study 02's MDE arithmetic, ch. 20 the marketplace framing for study 03.

### Software referenced or used

| Package | What it does | Link |
|---|---|---|
| `gsDesign` (R) | Group sequential boundaries; de facto standard for OBF. | [CRAN](https://cran.r-project.org/package=gsDesign) |
| `gsDesign2` (R) | Spending-function design, weighted log-rank. | [CRAN](https://cran.r-project.org/package=gsDesign2) |
| `rpact` (R) | Confirmatory adaptive designs, regulatory use. | [CRAN](https://cran.r-project.org/package=rpact) |
| `confseq` (Python/C++) | Time-uniform boundaries; implements Howard et al. (2021). | [GitHub](https://github.com/gostevehoward/confseq) |
| `gbstats` (Python) | GrowthBook's stats engine; sequential CIs via betting. | [Docs](https://docs.growthbook.io/statistics/sequential) |
| `statsmodels` (Python) | External p-value cross-check in `validation.py`. | [site](https://www.statsmodels.org/) |

### Further reading (one line each)

- **Wald, A. (1947).** *Sequential Analysis*. Wiley — OC curves, ASN, and the
  approximations an SPRT implementation should be validated against.
- **Wald, A. & Wolfowitz, J. (1948).** *Annals of Mathematical Statistics*,
  19(3), 326–339 — SPRT is optimal: minimizes expected sample size at both
  hypotheses. The theoretical justification for "sequential saves samples".
- **Robbins, H. (1970).** *Annals of Mathematical Statistics*, 41(5) — the
  mixture-martingale construction mSPRT later specializes.
- **Lai, T. L. (1988).** *Annals of Statistics*, 16(2) — the 2-SPRT: keeping
  SPRT's efficiency when the alternative is composite.
- **Johari, R., Pekelis, L. & Walsh, D. (2015/2022).** arXiv:1512.04922; later
  *Operations Research* — sequential p-values and confidence sequences.
- **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. S. (2021).**
  *Probability Surveys*, 18 — the constructive companion: how to build the
  boundaries, not just that they exist.
- **Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H. & Ramdas, A.
  (2023).** arXiv:2103.06476 — asymptotic confidence sequences; what GrowthBook
  ships in production (see `docs/method_comparison.md`).
- **Ville, J. (1939).** *Étude critique de la notion de collectif* — the
  maximal inequality behind every anytime-valid guarantee.
- **Kohavi, R., Longbotham, R., Sommerfield, D. & Henne, R. M. (2009).**
  *Data Mining and Knowledge Discovery*, 18(1) — what a real experiment
  platform looks like, and which mistakes recur.

---

## Study 02 — Variance reduction (`explabs/variance/`)

**Deng, A., Xu, Y., Kohavi, R. & Walker, T. (2013).** Improving the sensitivity
of online controlled experiments by utilizing pre-experiment data. *WSDM '13*.
CUPED — the paper this study reimplements. Variance reduction, not error
control — but it dominates the sample-size conversation in practice, so any
sequential method has to be compared against "fixed horizon + CUPED" rather
than the naive baseline.

**Lin, W. (2013).** Agnostic notes on regression adjustments to experimental
data. *Annals of Applied Statistics*, 7(1), 294–320.
When regression adjustment helps, when it hurts, and why the coefficients
should come from the control arm. The reason CUPED is specified the way it is.

**Miratrix, L. W., Sekhon, J. S. & Yu, B. (2013).** Adjusting treatment effect
estimates by post-stratification. *Journal of the Royal Statistical Society B*,
75(2), 369–386.
Theory for the nonparametric arm of the comparison.

**DoorDash Engineering (2020).** CUPAC — machine-learned predictions used as
covariates in place of a single pre-experiment metric (engineering blog).
The CUPAC reference is an engineering write-up rather than a paper; this study
treats it as an implementation pattern with known overfitting risks, not as
established theory.

**Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey,
W. & Robins, J. (2018).** Double/debiased machine learning for treatment and
structural parameters. *Econometrics Journal*, 21(1), C1–C68.
Cross-fitting. This is the reference for study 02's central methodological
claim: a learned nuisance function must be evaluated out-of-fold, or its own
fitting error contaminates the estimate. Study 02 measures what happens when it
is not (see `cupac/in-sample-understates-se`).

---

## Study 03 — Interference and marketplace experiments (`explabs/interference/`)

**Hudgens, M. G. & Halloran, M. E. (2008).** Toward causal inference with
interference. *Journal of the American Statistical Association*, 103(482),
832–842.
The vocabulary this study uses: direct effects, spillover effects, and the
stratified-interference framework the simulator's ground truth is written in.

**Basse, G. W. & Airoldi, E. M. (2018).** Model-assisted design of experiments
in the presence of network-correlated outcomes. *Biometrika*, 105(1), 199–209.
Why design, not estimation, is where interference is best handled.

**Karrer, B., Shi, L., Cole, B. et al. (2021).** Network experimentation at
scale. *KDD '21*.
Cluster randomization as practised at LinkedIn scale.

**Bojinov, I., Simchi-Levi, D. & Yin, R. (2022).** Design and analysis of
switchback experiments. arXiv:2009.00148; *Management Science*.
[arXiv:2009.00148](https://arxiv.org/abs/2009.00148)
The formal treatment of the switchback arm, including the carryover problem.

**Lyft / Uber engineering blogs.** Switchback and two-sided-market experiments
in ridesharing. Cited as practice documentation; exact titles vary and should be
looked up when citing formally.

---

## Study 04 — Adaptive experiments and post-hoc inference (`explabs/bandits/`)

**Thompson, W. R. (1933).** On the likelihood that one unknown probability
exceeds another in view of the evidence of two samples. *Biometrika*, 25(3–4),
285–294.
The 1933 origin of Thompson sampling.

**Russo, D., Van Roy, B., Kazerouni, A., Osband, I. & Wen, Z. (2018).** A
tutorial on Thompson sampling. *Foundations and Trends in Machine Learning*,
11(1).
The modern entry point.

**Lattimore, T. & Szepesvári, C. (2020).** *Bandit Algorithms*. Cambridge
University Press.
UCB1, its regret bound, and everything around it.

**Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002).** Finite-time analysis of
the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.
[doi:10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352)
The UCB1 paper: the algorithm this study implements, the `8 log T / Delta`
bound its regret is checked against, and the reason a tuning-free policy is
only tuning-free when the reward scale cooperates.

**Scott, S. L. (2015).** Multi-armed bandit experiments in the online service
economy. *Applied Stochastic Models in Business and Industry*, 31(1), 37–45.
The case for bandits as a business practice — and the inference costs it glosses
over, which are exactly what this study measures.

**Villar, S. S., Bowden, J. E. & Wason, J. (2015).** Multi-armed bandit
models for the optimal design of sequential clinical trials. *Journal of
Medical Statistics*, 47(1), 71–87.
Documents the selection bias that adaptive allocation inflicts on arm means —
the mechanism behind study 04's finding that Thompson sampling's naive bias
*saturates* rather than shrinks, because an abandoned arm stops accumulating
data.

**Li, L., Chu, W., Langford, J. & Schapire, R. E. (2010).** A contextual-bandit
approach to personalized news article recommendation. *WWW '10*.
[doi:10.1145/1772690.1772758](https://doi.org/10.1145/1772690.1772758)
The practice study 04 depends on: logging the propensity alongside every
assignment. Without that record, none of the post-hoc inference is possible
no matter which estimator you use.

**Horvitz, D. G. & Thompson, D. J. (1952).** A generalization of sampling
without replacement from a finite universe. *JASA*, 47(260), 663–685.
Inverse-probability weighting.

**Robins, J. M., Rotnitzky, A. & Zhao, L.-P. (1994).** Estimation of
regression coefficients when some regressors are not always observed.
*JASA*, 89(427), 846–866.
The augmentation step: doubly robust estimation, in which an outcome model
and an inverse-probability weight each rescue the other's failures. The reason
study 04's AIPW stays unbiased with a deliberately misspecified model.

**Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S. & Athey, S. (2021).**
Confidence intervals for policy evaluation in adaptive experiments.
*PNAS*, 118(15).
Why adaptive allocation breaks standard errors and how to fix them. The single
most relevant reference for the inference half of this study. Study 04
reproduces the failure it describes (a naive interval covering 62% of the
time) and stops short of its repair — adaptive confidence sequences — which
is stated as out of scope.

---

## Study 05 — Uplift modeling (`explabs/uplift/`)

**Künzel, S. R., Sekhon, J. S., Bickel, P. J. & Yu, B. (2019).** Metalearners
for heterogeneous treatment effects using observable data. *PNAS*, 116(10),
4156–4165.
T-learner and X-learner.

**Athey, S. & Imbens, G. (2016).** Recursive partitioning for heterogeneous
causal effects. *PNAS*, 113(27), 7353–7360.
The honest causal-tree construction this study implements by hand: split on
one subsample, estimate leaf effects on a disjoint one.

**Wager, S. & Athey, S. (2018).** Estimation and inference of heterogeneous
treatment effects using random forests. *JASA*, 113(523), 1228–1242.
Honesty, subsampling, and why causal forests are not just random forests.

**Radcliffe, N. (2007).** Using control groups to target on predicted lift.
Direct Marketing Analytics Journal.
The Qini metric.

**Radcliffe, N. & Surry, P. (2011).** Real-world uplift modelling with
significance-based uplift trees. Stochastic Solutions white paper.
Uplift trees and the reporting conventions the study adopts.

**Diemert, E., Betlei, A., Renaudin, C. & Amini, M.-R. (2018).** A large-scale
benchmark for uplift modeling. *AdKDD '18*.
The Criteo uplift dataset.
