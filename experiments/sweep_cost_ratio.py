"""Experiment 1 -- cost-ratio sensitivity sweep (Block 8 headline figure).

Sweep C_FAIL over several values (C_PREV fixed at 50), retrain the DFL
policy for each cost setting, and compare against:
  * reactive (run-to-failure)
  * predict-then-optimize, done FAIRLY: the classifier is fixed, but its
    decision threshold is re-tuned on the TRAINING episodes for each cost
    setting (that is what "optimize" means once a cost model exists).
Outputs: sweep_results.csv + sensitivity_sweep.png

Run:  python sweep_cost_ratio.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_data import generate_farm
from stage1_anomaly import healthy_mask, fit_normal_behaviour, residuals
from stage3_rul import daily_features, train_classifier
import stage4_dfl as s4
from stage4_dfl import (episodes, train_dfl, policy_dfl, policy_reactive,
                        policy_classifier, episode_cost)

TRAIN_T, TEST_T = [0, 1, 2], [3, 4]
C_FAIL_GRID = [100.0, 300.0, 600.0, 1000.0]
N_SEEDS = 3          # DFL is stochastic -> average over training seeds


def build_features():
    df = generate_farm(n_turbines=5, n_days=730, seed=0)
    train_df = df[df.turbine.isin(TRAIN_T)]
    nb = fit_normal_behaviour(train_df[healthy_mask(train_df)])
    feats = {}
    for tid in TRAIN_T + TEST_T:
        dt = df[df.turbine == tid].reset_index(drop=True)
        feats[tid] = daily_features(dt, residuals(dt, nb))
    return feats


def tune_pto_threshold(clf, train_eps):
    """Pick the classifier threshold minimising mean cost on train episodes."""
    grid = np.linspace(0.05, 0.95, 19)
    costs = [np.mean([episode_cost(policy_classifier(X, r, clf, th), r)
                      for X, r in train_eps]) for th in grid]
    return float(grid[int(np.argmin(costs))])


def main():
    feats = build_features()
    feat_train = pd.concat([feats[t] for t in TRAIN_T], ignore_index=True)
    clf = train_classifier(feat_train.dropna(subset=["rul_days"]))

    train_eps = [e for t in TRAIN_T for e in episodes(feats[t])]
    test_eps = [e for t in TEST_T for e in episodes(feats[t])]
    allX = np.vstack([X for X, _ in train_eps])
    mu, sd = allX.mean(0), allX.std(0) + 1e-9
    train_s = [((X - mu) / sd, r) for X, r in train_eps]

    rows = []
    for c_fail in C_FAIL_GRID:
        s4.C_FAIL = c_fail                       # cost model for this setting

        # predict-then-optimize: same classifier, cost-tuned threshold
        p_thr = tune_pto_threshold(clf, train_eps)
        pto = [episode_cost(policy_classifier(X, r, clf, p_thr), r)
               for X, r in test_eps]

        # decision-focused: retrain per cost setting, average over seeds
        dfl_means = []
        for seed in range(N_SEEDS):
            w, b = train_dfl(train_s, seed=seed)
            dfl_means.append(np.mean(
                [episode_cost(policy_dfl((X - mu) / sd, r, w, b), r)
                 for X, r in test_eps]))

        rows.append({
            "C_fail": c_fail,
            "ratio": c_fail / s4.C_PREV,
            "reactive": float(np.mean(
                [episode_cost(policy_reactive(X, r), r) for X, r in test_eps])),
            "pto": float(np.mean(pto)),
            "pto_threshold": p_thr,
            "dfl": float(np.mean(dfl_means)),
            "dfl_std": float(np.std(dfl_means)),
        })
        print(f"C_fail={c_fail:6.0f}  ratio={rows[-1]['ratio']:4.0f}x  "
              f"PTO(thr={p_thr:.2f})={rows[-1]['pto']:6.1f}  "
              f"DFL={rows[-1]['dfl']:6.1f} (+/-{rows[-1]['dfl_std']:.1f})")

    res = pd.DataFrame(rows)
    res.to_csv("sweep_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["ratio"], res["reactive"], "o--", color="gray",
            label="reactive (run-to-failure)")
    ax.plot(res["ratio"], res["pto"], "s-", color="tab:blue",
            label="predict-then-optimize (cost-tuned threshold)")
    ax.errorbar(res["ratio"], res["dfl"], yerr=res["dfl_std"], fmt="^-",
                color="tab:red", capsize=4,
                label=f"decision-focused (mean of {N_SEEDS} seeds)")
    ax.set_xlabel("cost asymmetry  C_fail / C_prev")
    ax.set_ylabel("mean cost per episode (k EUR)")
    ax.set_title("Maintenance policy cost vs failure-cost asymmetry")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("sensitivity_sweep.png", dpi=150)
    print("\nsaved: sweep_results.csv, sensitivity_sweep.png")


if __name__ == "__main__":
    main()
