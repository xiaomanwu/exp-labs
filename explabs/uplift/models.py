"""Uplift models for study 05.

All models share one interface: ``fit(x, t, y)`` then ``tau(x) -> scores``.
The score is only used for *ranking* users by expected effect, so models are
compared on what targeting does with their rankings (Qini, capture), not on
estimator elegance.

Implementations, following the constructions in the literature rather than a
library:

- ``SLearner`` / ``TLearner`` (Künzel et al. 2019): one response surface with
  treatment as a feature; two response surfaces, differenced.
- ``XLearner`` (Künzel et al. 2019): impute individual effects per arm from
  the *other* arm's response model, regress the imputations, combine with
  propensity weights ``g = e``. The weighting is the point: with a large
  treated share, ``tau_1`` inherits error from the small-sample control
  response model ``m0``, so the combination leans on ``tau_0``, whose errors
  come from the well-estimated ``m1``.
- ``HonestCausalForest`` (Athey & Imbens 2016; Athey & Wager 2019): trees
  split to maximise between-leaf effect differences on a *splitting* sample,
  and leaf effects are estimated on a *disjoint estimation* sample. Honesty
  is the difference between a causal forest and a random forest in costume,
  so the flag can be switched off for the comparison check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def _response_learner(seed: int) -> GradientBoostingRegressor:
    """Shared response-surface learner.

    Heavy leaf smoothing (``min_samples_leaf``) matters more than depth here:
    a T-learner differences two surfaces, so each surface's prediction noise
    enters the effect estimate twice. A sharp response model is a noisy
    uplift model.
    """
    return GradientBoostingRegressor(
        n_estimators=400, max_depth=2, learning_rate=0.02,
        subsample=0.8, min_samples_leaf=100, random_state=seed,
    )


# --------------------------------------------------------------------------
# Meta-learners
# --------------------------------------------------------------------------


class SLearner:
    """One response surface with treatment as a feature; effect is a difference."""

    key = "s"
    label = "S-learner"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray,
            e: float | None = None) -> "SLearner":
        self._model = _response_learner(self._seed)
        self._model.fit(np.column_stack([x, t]), y)
        return self

    def tau(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        ones = np.column_stack([x, np.ones(n)])
        zeros = np.column_stack([x, np.zeros(n)])
        return self._model.predict(ones) - self._model.predict(zeros)


class TLearner:
    """Two response surfaces, differenced. Inherits both models' errors."""

    key = "t"
    label = "T-learner"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray,
            e: float | None = None) -> "TLearner":
        self._m1 = _response_learner(self._seed + 1)
        self._m0 = _response_learner(self._seed + 2)
        self._m1.fit(x[t == 1.0], y[t == 1.0])
        self._m0.fit(x[t == 0.0], y[t == 0.0])
        return self

    def tau(self, x: np.ndarray) -> np.ndarray:
        return self._m1.predict(x) - self._m0.predict(x)


class XLearner:
    """Imputed-effect learner with propensity weighting (Künzel et al.)."""

    key = "x"
    label = "X-learner"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray,
            e: float | None = None) -> "XLearner":
        if e is None:
            e = float(t.mean())
        self._e = float(e)
        one, zero = t == 1.0, t == 0.0

        m1 = _response_learner(self._seed + 1).fit(x[one], y[one])
        m0 = _response_learner(self._seed + 2).fit(x[zero], y[zero])

        # Imputed effects: treated units give y - m0(x); control units give
        # m1(x) - y. Each imputation leans on the *other* arm's surface.
        d1 = y[one] - m0.predict(x[one])
        d0 = m1.predict(x[zero]) - y[zero]

        tau1 = _response_learner(self._seed + 3).fit(x[one], d1)
        tau0 = _response_learner(self._seed + 4).fit(x[zero], d0)

        # g = e: a large treated share makes d1's error (from small-sample m0)
        # the bigger risk, so the combination leans on tau0, whose error comes
        # from the well-estimated m1.
        self._tau1 = tau1
        self._tau0 = tau0
        return self

    def tau(self, x: np.ndarray) -> np.ndarray:
        return self._e * self._tau0.predict(x) + (1.0 - self._e) * self._tau1.predict(x)


# --------------------------------------------------------------------------
# Honest causal forest
# --------------------------------------------------------------------------


@dataclass
class _Tree:
    feature: list[int]
    threshold: list[float]
    left: list[int]
    right: list[int]
    value: list[float]          # nan for internal nodes

    def predict(self, x: np.ndarray) -> np.ndarray:
        feature = np.asarray(self.feature)
        threshold = np.asarray(self.threshold)
        left = np.asarray(self.left)
        right = np.asarray(self.right)
        value = np.asarray(self.value, dtype=float)
        # A leaf is a node whose feature was never set (-1); internal nodes
        # keep value = nan, which is never read.
        node = np.zeros(x.shape[0], dtype=int)
        at_leaf = feature[node] == -1
        while not at_leaf.all():
            idx = np.where(~at_leaf)[0]
            f = feature[node[idx]]
            go_left = x[idx, f] <= threshold[node[idx]]
            node[idx] = np.where(go_left, left[node[idx]], right[node[idx]])
            at_leaf = feature[node] == -1
        return value[node]


class HonestCausalForest:
    """Causal trees with honest two-sample leaf estimation, averaged.

    Each tree draws a subsample, splits it into a *splitting* half (structure:
    choose splits maximising ``n_L n_R (tau_L - tau_R)^2 / n``) and a
    *disjoint estimation* half (leaf effects are ``ybar1 - ybar0`` there).
    With ``honest=False`` the estimation set is the splitting set, which is
    exactly the "random forest in costume" the roadmap warns about.
    """

    key = "forest"
    label = "Causal forest (honest)"

    def __init__(self, n_trees: int = 200, max_depth: int = 3,
                 subsample: float = 0.5, min_leaf: int = 100,
                 honest: bool = True, seed: int = 0) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_leaf = min_leaf
        self.honest = honest
        self.seed = seed
        self.key = "forest" if honest else "forest_dishonest"

    # -- one tree -----------------------------------------------------------

    def _best_split(self, x: np.ndarray, t: np.ndarray, y: np.ndarray,
                    idx: np.ndarray, min_leaf: int):
        """Greedy effect-heterogeneity split, vectorised over positions."""
        n = idx.size
        best = (None, None, -np.inf)
        xi, ti, yi = x[idx], t[idx], y[idx]
        tot1, tot0 = ti.sum(), n - ti.sum()
        tot_y1, tot_y0 = (yi * ti).sum(), (yi * (1.0 - ti)).sum()
        for f in range(x.shape[1]):
            order = np.argsort(xi[:, f], kind="stable")
            xs, ts, ys = xi[order, f], ti[order], yi[order]
            c1 = np.cumsum(ts)[:-1]
            c0 = np.cumsum(1.0 - ts)[:-1]
            s1 = np.cumsum(ys * ts)[:-1]
            s0 = np.cumsum(ys * (1.0 - ts))[:-1]
            ok = ((c1 >= min_leaf) & (c0 >= min_leaf)
                  & (tot1 - c1 >= min_leaf) & (tot0 - c0 >= min_leaf)
                  & (xs[:-1] < xs[1:]))
            if not ok.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                tau_l = s1 / c1 - s0 / c0
                tau_r = (tot_y1 - s1) / (tot1 - c1) - (tot_y0 - s0) / (tot0 - c0)
            k = np.arange(1, n)
            score = k * (n - k) / n * (tau_l - tau_r) ** 2
            score = np.where(ok, score, -np.inf)
            j = int(np.argmax(score))
            if score[j] > best[2]:
                thr = 0.5 * (xs[j] + xs[j + 1])
                left = idx[order[:j + 1]]
                right = idx[order[j + 1:]]
                best = (f, thr, score[j], left, right)
        return best if best[0] is not None else None

    def _grow(self, x, t, y, split_idx, est_idx, min_leaf) -> _Tree:
        tree = _Tree(feature=[], threshold=[], left=[], right=[], value=[])

        def leaf(node_est: np.ndarray) -> float:
            t_e, y_e = t[node_est], y[node_est]
            n1, n0 = t_e.sum(), node_est.size - t_e.sum()
            if n1 < 5 or n0 < 5:
                return np.nan
            return float(y_e[t_e == 1.0].mean() - y_e[t_e == 0.0].mean())

        def build(split: np.ndarray, est: np.ndarray, depth: int) -> int:
            node = len(tree.feature)
            tree.feature.append(-1)
            tree.threshold.append(np.nan)
            tree.left.append(-1)
            tree.right.append(-1)
            tree.value.append(np.nan)
            min_leaf_here = min(min_leaf, max(10, split.size // 20))
            if depth >= self.max_depth or split.size < 4 * min_leaf_here:
                tree.value[node] = leaf(est)
                return node
            found = self._best_split(x, t, y, split, min_leaf_here)
            if found is None:
                tree.value[node] = leaf(est)
                return node
            f, thr, _, left_idx, right_idx = found
            # estimation set follows the same path as the splitting set
            go_left = x[est, f] <= thr
            tree.feature[node] = f
            tree.threshold[node] = thr
            tree.left[node] = build(left_idx, est[go_left], depth + 1)
            tree.right[node] = build(right_idx, est[~go_left], depth + 1)
            return node

        build(split_idx, est_idx, 0)
        # Leaves that lost their estimation sample: borrow the forest's mean
        # leaf value. Internal nodes keep nan -- predict() keys on feature==-1.
        arr = np.asarray(tree.value, dtype=float)
        is_leaf = np.asarray(tree.feature) == -1
        finite_leaves = is_leaf & np.isfinite(arr)
        fallback = float(np.nanmean(arr[finite_leaves])) if finite_leaves.any() else 0.0
        arr[is_leaf & ~np.isfinite(arr)] = fallback
        tree.value = arr.tolist()
        return tree

    # -- forest ---------------------------------------------------------------

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray,
            e: float | None = None) -> "HonestCausalForest":
        rng = np.random.default_rng(self.seed)
        n = x.shape[0]
        size = max(200, int(self.subsample * n))
        self._trees = []
        half = size // 2
        for _ in range(self.n_trees):
            picked = rng.choice(n, size=2 * half, replace=False)
            split_idx, est_idx = picked[:half], picked[half:]
            if not self.honest:
                est_idx = split_idx          # the "costume" variant
            min_leaf = self.min_leaf if half >= 40 * self.min_leaf else max(
                20, half // 40)
            self._trees.append(self._grow(x, t, y, split_idx, est_idx, min_leaf))
        return self

    def tau(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(x.shape[0])
        for tree in self._trees:
            scores += tree.predict(x)
        return scores / len(self._trees)
