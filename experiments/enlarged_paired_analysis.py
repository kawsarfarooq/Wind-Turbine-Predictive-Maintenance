"""Enlarged-test-set paired analysis: tuned PTO vs DFL, more episodes.

Training is UNCHANGED: turbines 0-2, same seed, same models, same DFL
hyperparameters. The test set is enlarged from 2 to 12 turbines
(turbines 3-14). Because the generator draws each turbine sequentially
from one RNG stream, turbines 0-4 are bit-identical to all previous
experiments; turbines 5-14 are new held-out data. Cost model: C_prev=50,
C_fail=300. Noise levels {0.0, 1.0, 1.5}, paired by construction (same
failures at same times across noise levels).

Per noise level: DFL per-episode cost is averaged over 3 training seeds;
paired differences (PTO - DFL) are tested with a Wilcoxon signed-rank
test (scipy, if available).

Outputs: enlarged_paired_differences.csv (per episode)
         enlarged_paired_summary.csv     (per noise level)
Run:  python enlarged_paired_analysis.py     (~8-12 min)
"""
import numpy as np
import pandas as pd

from synth_data import generate_farm
from stage1_anomaly import healthy_mask, fit_normal_behaviour, residuals
from stage3_rul import daily_features, train_classifier
import stage4_dfl as s4
from stage4_dfl import (episodes, train_dfl, policy_dfl, policy_classifier,
                        episode_cost)

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

TRAIN_T = [0, 1, 2]
TEST_T = list(range(3, 15))          # enlarged: 12 held-out turbines
N_TURBINES = 15
NOISE_LEVELS = [0.0, 1.0, 1.5]
N_SEEDS = 3
s4.C_FAIL, C_FAIL = 300.0, 300.0


def build(noise):
    df = generate_farm(n_turbines=N_TURBINES, n_days=730, seed=0,
                       noise_level=noise)
    train_df = df[df.turbine.isin(TRAIN_T)]
    nb = fit_normal_behaviour(train_df[healthy_mask(train_df)])
    feats = {}
    for tid in TRAIN_T + TEST_T:
        dt = df[df.turbine == tid].reset_index(drop=True)
        feats[tid] = daily_features(dt, residuals(dt, nb))
    feat_train = pd.concat([feats[t] for t in TRAIN_T], ignore_index=True)
    clf = train_classifier(feat_train.dropna(subset=["rul_days"]))
    train_eps = [e for t in TRAIN_T for e in episodes(feats[t])]
    test_eps = [(t, i, X, r) for t in TEST_T
                for i, (X, r) in enumerate(episodes(feats[t]))]
    allX = np.vstack([X for X, _ in train_eps])
    mu, sd = allX.mean(0), allX.std(0) + 1e-9
    return clf, train_eps, test_eps, mu, sd


def tune_pto_threshold(clf, train_eps):
    grid = np.linspace(0.05, 0.95, 19)
    costs = [np.mean([episode_cost(policy_classifier(X, r, clf, th), r)
                      for X, r in train_eps]) for th in grid]
    return float(grid[int(np.argmin(costs))])


def main():
    detail_rows, summary_rows = [], []
    for noise in NOISE_LEVELS:
        clf, train_eps, test_eps, mu, sd = build(noise)
        train_s = [((X - mu) / sd, r) for X, r in train_eps]

        p_thr = tune_pto_threshold(clf, train_eps)
        pto = np.array([episode_cost(policy_classifier(X, r, clf, p_thr), r)
                        for _, _, X, r in test_eps])

        dfl_per_seed = []
        for seed in range(N_SEEDS):
            w, b = train_dfl(train_s, seed=seed)
            dfl_per_seed.append(
                [episode_cost(policy_dfl((X - mu) / sd, r, w, b), r)
                 for _, _, X, r in test_eps])
        dfl = np.mean(dfl_per_seed, axis=0)

        diff = pto - dfl
        for (tid, ei, _, _), p, d, dd in zip(test_eps, pto, dfl, diff):
            detail_rows.append({
                "noise": noise, "turbine": tid, "episode": ei,
                "pto_cost": p, "dfl_cost": round(float(d), 1),
                "diff_pto_minus_dfl": round(float(dd), 1),
                "cheaper": "DFL" if dd > 0 else ("PTO" if dd < 0 else "tie"),
            })

        n = len(diff)
        n_dfl = int((diff > 0).sum())
        n_pto = int((diff < 0).sum())
        if HAVE_SCIPY and np.any(diff != 0):
            try:
                p_val = float(wilcoxon(diff[diff != 0]).pvalue)
            except ValueError:
                p_val = np.nan
        else:
            p_val = np.nan
        summary_rows.append({
            "noise": noise, "n_episodes": n,
            "pto_mean": round(float(pto.mean()), 1),
            "dfl_mean": round(float(dfl.mean()), 1),
            "mean_diff_pto_minus_dfl": round(float(diff.mean()), 1),
            "std_diff": round(float(diff.std()), 1),
            "dfl_cheaper_on": n_dfl, "pto_cheaper_on": n_pto,
            "ties": n - n_dfl - n_pto,
            "wilcoxon_p": round(p_val, 4) if np.isfinite(p_val) else "n/a",
            "pto_threshold": p_thr,
        })
        r = summary_rows[-1]
        print(f"noise={noise:.1f}  n={n}  PTO={r['pto_mean']:6.1f}  "
              f"DFL={r['dfl_mean']:6.1f}  diff={r['mean_diff_pto_minus_dfl']:+6.1f} "
              f"+/-{r['std_diff']:5.1f}  DFL cheaper {n_dfl}/{n}, "
              f"PTO cheaper {n_pto}/{n}  Wilcoxon p={r['wilcoxon_p']}")

    pd.DataFrame(detail_rows).to_csv("enlarged_paired_differences.csv",
                                     index=False)
    pd.DataFrame(summary_rows).to_csv("enlarged_paired_summary.csv",
                                      index=False)
    print("\nsaved: enlarged_paired_differences.csv, "
          "enlarged_paired_summary.csv")


if __name__ == "__main__":
    main()
