"""Experiment 2 -- signal-quality sweep (the DFL-vs-PTO hypothesis test).

Hypothesis (from Experiment 1): a cost-tuned predict-then-optimize policy
is near-optimal when the degradation signal is clean and monotone; DFL
should close the gap / pull ahead as the signal gets noisier.

Design: paired comparison -- the SAME 20 failures at the SAME times for
every noise level (noise uses a separate RNG stream); only observability
changes. Grid: noise in {0, 0.5, 1.0} x C_fail in {300, 1000} (C_prev=50).

Outputs: noise_sweep_results.csv + noise_sweep.png
Run:  python sweep_noise.py        (~5 min)
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
from stage4_dfl import (episodes, train_dfl, policy_dfl, policy_classifier,
                        episode_cost)

TRAIN_T, TEST_T = [0, 1, 2], [3, 4]
NOISE_GRID = [0.0, 0.5, 1.0, 1.5]
C_FAIL_GRID = [300.0, 1000.0]
N_SEEDS = 3


def build(noise):
    """Regenerate data at this noise level, rebuild the feature pipeline."""
    df = generate_farm(n_turbines=5, n_days=730, seed=0, noise_level=noise)
    train_df = df[df.turbine.isin(TRAIN_T)]
    nb = fit_normal_behaviour(train_df[healthy_mask(train_df)])
    feats = {}
    for tid in TRAIN_T + TEST_T:
        dt = df[df.turbine == tid].reset_index(drop=True)
        feats[tid] = daily_features(dt, residuals(dt, nb))
    feat_train = pd.concat([feats[t] for t in TRAIN_T], ignore_index=True)
    clf = train_classifier(feat_train.dropna(subset=["rul_days"]))
    train_eps = [e for t in TRAIN_T for e in episodes(feats[t])]
    test_eps = [e for t in TEST_T for e in episodes(feats[t])]
    allX = np.vstack([X for X, _ in train_eps])
    mu, sd = allX.mean(0), allX.std(0) + 1e-9
    return clf, train_eps, test_eps, mu, sd


def tune_pto_threshold(clf, train_eps):
    grid = np.linspace(0.05, 0.95, 19)
    costs = [np.mean([episode_cost(policy_classifier(X, r, clf, th), r)
                      for X, r in train_eps]) for th in grid]
    return float(grid[int(np.argmin(costs))])


def main():
    rows = []
    for noise in NOISE_GRID:
        clf, train_eps, test_eps, mu, sd = build(noise)
        train_s = [((X - mu) / sd, r) for X, r in train_eps]
        for c_fail in C_FAIL_GRID:
            s4.C_FAIL = c_fail

            p_thr = tune_pto_threshold(clf, train_eps)
            pto_c = [episode_cost(policy_classifier(X, r, clf, p_thr), r)
                     for X, r in test_eps]

            dfl_c, dfl_avoid = [], []
            for seed in range(N_SEEDS):
                w, b = train_dfl(train_s, seed=seed)
                cs = [episode_cost(policy_dfl((X - mu) / sd, r, w, b), r)
                      for X, r in test_eps]
                dfl_c.append(np.mean(cs))
                dfl_avoid.append(np.mean([c < c_fail for c in cs]))

            rows.append({
                "noise": noise, "C_fail": c_fail,
                "pto": float(np.mean(pto_c)), "pto_threshold": p_thr,
                "pto_avoided": float(np.mean([c < c_fail for c in pto_c])),
                "dfl": float(np.mean(dfl_c)), "dfl_std": float(np.std(dfl_c)),
                "dfl_avoided": float(np.mean(dfl_avoid)),
            })
            r = rows[-1]
            print(f"noise={noise:.1f} C_fail={c_fail:5.0f}  "
                  f"PTO(thr={r['pto_threshold']:.2f})={r['pto']:6.1f} "
                  f"(avoid {100*r['pto_avoided']:.0f}%)   "
                  f"DFL={r['dfl']:6.1f} (+/-{r['dfl_std']:.1f}, "
                  f"avoid {100*r['dfl_avoided']:.0f}%)")

    res = pd.DataFrame(rows)
    res.to_csv("noise_sweep_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, c_fail in zip(axes, C_FAIL_GRID):
        sub = res[res.C_fail == c_fail]
        ax.axhline(c_fail, ls="--", color="gray", label="reactive")
        ax.plot(sub["noise"], sub["pto"], "s-", color="tab:blue",
                label="predict-then-optimize (cost-tuned)")
        ax.errorbar(sub["noise"], sub["dfl"], yerr=sub["dfl_std"], fmt="^-",
                    color="tab:red", capsize=4, label="decision-focused")
        ax.set_xlabel("signal degradation level (0 = clean)")
        ax.set_title(f"C_fail/C_prev = {c_fail / s4.C_PREV:.0f}x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean cost per episode (k EUR)")
    axes[0].legend()
    fig.suptitle("Policy cost vs signal quality (same failures at all levels)")
    fig.tight_layout()
    fig.savefig("noise_sweep.png", dpi=150)
    print("\nsaved: noise_sweep_results.csv, noise_sweep.png")


if __name__ == "__main__":
    main()
