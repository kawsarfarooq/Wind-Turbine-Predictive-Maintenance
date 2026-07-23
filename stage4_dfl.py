"""Stage 4 -- decision-focused learning (Lecture Block 8) + cost simulation.

Cost model (per run-to-failure episode):
  * maintain on day d before failure:  C_PREV + C_WASTE * remaining_life_days
  * never maintain before failure:     C_FAIL
Stochastic policy: p(maintain today) = sigmoid(w . x + b) on daily features.
Trained with the score-function (REINFORCE) gradient of expected episode
cost, with a moving-average baseline for variance reduction.

Baselines simulated with the same episode replay:
  reactive / anomaly-threshold / predict-then-optimize (classifier + threshold).
"""
import numpy as np

C_PREV = 50.0     # planned maintenance (k EUR)
C_FAIL = 300.0    # unplanned failure   (k EUR)
C_WASTE = 1.0     # wasted life per day of early action (k EUR/day)

FEAT_COLS = ["res_mean", "res_max", "res_mean_7d", "res_trend_7d"]


def episodes(feat):
    """Yield (X, rul_days) for observed run-to-failure episodes only.

    A turbine's final observation window is commonly right-censored. It must
    not be charged as a failure or used to train a failure-timing policy.
    """
    for _, g in feat.groupby("episode"):
        if "event_observed" in g and not bool(g["event_observed"].all()):
            continue
        if g["rul_days"].isna().any() or len(g) < 20:
            continue
        yield g[FEAT_COLS].values, g["rul_days"].values


def episode_cost(maintain_day, rul_days):
    if maintain_day is None:
        return C_FAIL
    return C_PREV + C_WASTE * rul_days[maintain_day]


# ---------------- baseline policies ----------------

def policy_reactive(X, rul):
    return None


def policy_threshold(X, rul, thr):
    """Act on first day the residual-mean feature exceeds thr."""
    hits = np.where(X[:, 0] > thr)[0]
    return int(hits[0]) if len(hits) else None


def policy_classifier(X, rul, clf, p_thr):
    p = clf.predict_proba(X)[:, 1]
    hits = np.where(p > p_thr)[0]
    return int(hits[0]) if len(hits) else None


# ---------------- decision-focused policy ----------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def dfl_sample_episode(w, b, X, rng):
    """Sample stop day + accumulate grad of log-prob of the trajectory."""
    glw, glb = np.zeros_like(w), 0.0
    for d in range(len(X)):
        p = _sigmoid(X[d] @ w + b)
        act = rng.random() < p
        # d log pi / dz  =  (1-p) if act else (-p)
        coef = (1.0 - p) if act else (-p)
        glw += coef * X[d]
        glb += coef
        if act:
            return d, glw, glb
    return None, glw, glb


def train_dfl(train_eps, n_iter=500, lr=0.05, batch=16, seed=0):
    """Batch REINFORCE: per step, sample `batch` episodes, normalize the
    advantage within the batch (scale-invariant in C_FAIL), average grads."""
    rng = np.random.default_rng(seed)
    dim = train_eps[0][0].shape[1]
    w, b = np.zeros(dim), -4.0        # start reluctant to act
    for it in range(n_iter):
        costs, grads = [], []
        for _ in range(batch):
            X, rul = train_eps[rng.integers(len(train_eps))]
            day, glw, glb = dfl_sample_episode(w, b, X, rng)
            costs.append(episode_cost(day, rul))
            grads.append((glw, glb))
        costs = np.asarray(costs)
        adv = (costs - costs.mean()) / (costs.std() + 1e-6)
        w -= lr * sum(a * g[0] for a, g in zip(adv, grads)) / batch
        b -= lr * sum(a * g[1] for a, g in zip(adv, grads)) / batch
    return w, b


def policy_dfl(X, rul, w, b):
    """Deterministic deployment: act on first day with p > 0.5."""
    p = _sigmoid(X @ w + b)
    hits = np.where(p > 0.5)[0]
    return int(hits[0]) if len(hits) else None


# ---------------- rolling evaluation ----------------

def evaluate(policies, test_eps):
    """policies: dict name -> callable(X, rul) -> maintain_day or None."""
    out = {}
    for name, pol in policies.items():
        costs = [episode_cost(pol(X, rul), rul) for X, rul in test_eps]
        out[name] = {
            "mean_cost": float(np.mean(costs)),
            "failures_avoided": float(np.mean([c < C_FAIL for c in costs])),
        }
    return out
